from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F

NORM_LAYERS = {'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }


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


class InversionNet(nn.Module):
    """
    InversionNet architecture
    """
    def __init__(self, inch, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(InversionNet, self).__init__()
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


if __name__ == '__main__':

    model = InversionNet(inch=7)
    device = torch.device('cuda:0')
    model.to(device)
    from torchsummary import summary

    summary(model, input_size=[(7, 1000, 70)])
