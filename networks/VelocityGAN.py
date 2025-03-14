from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

NORM_LAYERS = {'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm}


class ConvBlock(nn.Module):
    """
    Basic network unit
    """
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        """
        constructor

        :param in_fea:              Number of input channels
        :param out_fea:             Number of output channels
        :param kernel_size:         Convolution kernel size
        :param stride:              The step size of the convolution displacement
        :param padding:             Outer border width (set to 0)
        :param norm:                normalization approach
        :param relu_slop:           Activation function parameter
        :param dropout:             Whether to use drop out (default is None)
        """
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea,
                            kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """

        :param x:                   Feature maps
        :return:
        """
        return self.layers(x)


class ConvBlockTanh(nn.Module):
    """
    Convolution at the end for normalization
    """
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        """
        constructor

        :param in_fea:              Number of input channels
        :param out_fea:             Number of output channels
        :param kernel_size:         Convolution kernel size
        :param stride:              The step size of the convolution displacement
        :param padding:             Outer border width (set to 0)
        :param norm:                normalization approach
        """
        super(ConvBlockTanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea,
                            kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """

        :param x:                   Feature maps
        :return:
        """
        return self.layers(x)


class DeconvBlock(nn.Module):
    """
    Upsampling component
    """
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size,
                                     stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """

        :param x:                   Feature maps
        :return:
        """
        return self.layers(x)


class VelocityGAN(nn.Module):
    """
    VelocityGAN architecture
    """

    def __init__(self, inch, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(VelocityGAN, self).__init__()
        self.convblock1 = ConvBlock(inch, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlockTanh(dim1, 1)

    def forward(self, x):
        """

        :param x:                   Feature maps
        :return:
        """
        # Encoder part
        x = self.convblock1(x)      # (None, 32, 500, 70)
        x = self.convblock2_1(x)    # (None, 64, 250, 70)
        x = self.convblock2_2(x)    # (None, 64, 250, 70)
        x = self.convblock3_1(x)    # (None, 64, 125, 70)
        x = self.convblock3_2(x)    # (None, 64, 125, 70)
        x = self.convblock4_1(x)    # (None, 128, 63, 70)
        x = self.convblock4_2(x)    # (None, 128, 63, 70)
        x = self.convblock5_1(x)    # (None, 128, 32, 35)
        x = self.convblock5_2(x)    # (None, 128, 32, 35)
        x = self.convblock6_1(x)    # (None, 256, 16, 18)
        x = self.convblock6_2(x)    # (None, 256, 16, 18)
        x = self.convblock7_1(x)    # (None, 256, 8, 9)
        x = self.convblock7_2(x)    # (None, 256, 8, 9)
        x = self.convblock8(x)      # (None, 512, 1, 1)

        # Decoder Part
        x = self.deconv1_1(x)       # (None, 512, 5, 5)
        x = self.deconv1_2(x)       # (None, 512, 5, 5)
        x = self.deconv2_1(x)       # (None, 256, 10, 10)
        x = self.deconv2_2(x)       # (None, 256, 10, 10)
        x = self.deconv3_1(x)       # (None, 128, 20, 20)
        x = self.deconv3_2(x)       # (None, 128, 20, 20)
        x = self.deconv4_1(x)       # (None, 64, 40, 40)
        x = self.deconv4_2(x)       # (None, 64, 40, 40)
        x = self.deconv5_1(x)       # (None, 32, 80, 80)
        x = self.deconv5_2(x)       # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)
                                    # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x)         # (None, 1, 70, 70)
        return x


class Discriminator(nn.Module):
    """
    Discriminator architecture
    """
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, **kwargs):
        super(Discriminator, self).__init__()
        self.convblock1_1 = ConvBlock(1, dim1, stride=2)
        self.convblock1_2 = ConvBlock(dim1, dim1)
        self.convblock2_1 = ConvBlock(dim1, dim2, stride=2)
        self.convblock2_2 = ConvBlock(dim2, dim2)
        self.convblock3_1 = ConvBlock(dim2, dim3, stride=2)
        self.convblock3_2 = ConvBlock(dim3, dim3)
        self.convblock4_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock4_2 = ConvBlock(dim4, dim4)
        self.convblock5 = ConvBlock(dim4, 1, kernel_size=5, padding=0)

    def forward(self, x):
        """

        :param x:                  Feature maps
        :return:
        """
        x = self.convblock1_1(x)
        x = self.convblock1_2(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5(x)
        x = x.view(x.shape[0], -1)
        return x


class WassersteinGP(nn.Module):
    """
    Discriminator Loss
    """
    def __init__(self, device, lambda_gp):
        """
        constructor

        :param device:              Device identification information
        :param lambda_gp:           The weight parameter of gradient penalty
        """
        super(WassersteinGP, self).__init__()
        self.device = device
        self.lambda_gp = lambda_gp

    def forward(self, real, fake, model):
        """

        :param real:                True velocity model target
        :param fake:                Velocity model for generator inversion
        :param model:               Discriminator network
        :return:
        """
        gradient_penalty = self.compute_gradient_penalty(model, real, fake)
        loss_real = torch.mean(model(real))
        loss_fake = torch.mean(model(fake))
        loss = -loss_real + loss_fake + gradient_penalty * self.lambda_gp
        return loss, loss_real-loss_fake, gradient_penalty

    def compute_gradient_penalty(self, model, real_samples, fake_samples):
        """
        Calculate the gradient penalty of the mixed velocity model

        :param model:               Discriminator network
        :param real_samples:        True velocity model target
        :param fake_samples:        Velocity model for generator inversion
        :return:
        """
        # real_samples.size(0) is the batch number, we need to set different random numbers for different batches
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = model(interpolates)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(real_samples.size(0), d_interpolates.size(1)).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        # The obtained gradients are consistent with the input dimensions, that is, (64x1x70x70).
        # After passing this line of code, the subsequent three dimensions are compressed to 4900, that is, (64x4900)
        gradients = gradients.view(gradients.size(0), -1)
        # dim = 1 is to find the 2-norm for each row of the matrix.
        # That is, each row is regarded as a vector, and the 2 norm of this vector is found.
        # Finally, each element in the vector is squared and added together to take the square root.
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        # Therefore, in the matrix of (64x4900), the vector of length 4900 in each row becomes a single value, that is,
        # the final matrix becomes (64x1).
        return gradient_penalty


class GeneratorLoss(nn.Module):
    """
    Generator Loss
    """
    def __init__(self, lambda_g1v, lambda_g2v, lambda_adv):
        """
        Constructor

        :param lambda_g1v:          The weight parameter of l1 loss
        :param lambda_g2v:          The weight parameter of l2 loss
        :param lambda_adv:          The weight parameter of adversarial loss
        """
        super(GeneratorLoss, self).__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_adv = lambda_adv

    def forward(self, pred, gt, model_d=None):
        """

        :param pred:                Velocity model for generator inversion
        :param gt:                  True velocity model target
        :param model_d:             Discriminator network
        :return:
        """

        l1loss = nn.L1Loss()
        l2loss = nn.MSELoss()
        loss_g1v = l1loss(pred, gt)
        loss_g2v = l2loss(pred, gt)
        loss = self.lambda_g1v * loss_g1v + self.lambda_g2v * loss_g2v
        if model_d is not None:
            loss_adv = -torch.mean(model_d(pred))
            loss += self.lambda_adv * loss_adv
        return loss, loss_g1v, loss_g2v


if __name__ == '__main__':
    x = torch.zeros((10, 5, 1000, 70))
    model = VelocityGAN(5)
    out = model(x)
    print("out: ", out.size())
