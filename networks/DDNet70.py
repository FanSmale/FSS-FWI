import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeismicRecordDownSampling(nn.Module):
    """
    Downsampling component for seismic records
    """
    def __init__(self, shot_num):
        """
        constructor

        :param shot_num:        Number of input channels
        """
        super().__init__()

        self.pre_dim_reducer1 = ConvBlock(shot_num, 8, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer2 = ConvBlock(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.pre_dim_reducer3 = ConvBlock(8, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer4 = ConvBlock(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.pre_dim_reducer5 = ConvBlock(16, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer6 = ConvBlock(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        """

        :param x:               Feature maps
        :return:
        """

        width = x.shape[3]
        new_size = [width * 8, width]
        dimred0 = F.interpolate(x, size=new_size, mode='bilinear', align_corners=False)

        dimred1 = self.pre_dim_reducer1(dimred0)
        dimred2 = self.pre_dim_reducer2(dimred1)
        dimred3 = self.pre_dim_reducer3(dimred2)
        dimred4 = self.pre_dim_reducer4(dimred3)
        dimred5 = self.pre_dim_reducer5(dimred4)
        dimred6 = self.pre_dim_reducer6(dimred5)

        return dimred6


class UnetConv2(nn.Module):
    """
    Basic network unit
    """
    def __init__(self, in_size, out_size, is_batchnorm, activ_fuc = nn.ReLU(inplace=True)):
        """
        constructor

        :param in_size:         Number of channels of input
        :param out_size:        Number of channels of output
        :param is_batchnorm:    Whether to use BN
        :param activ_fuc:       Activation function
        """
        super(UnetConv2, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       activ_fuc)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       activ_fuc)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       activ_fuc)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       activ_fuc)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetDown(nn.Module):
    """
    Downsampling component with skip connections
    """
    def __init__(self, in_size, out_size, is_batchnorm, activ_fuc=nn.ReLU(inplace=True)):
        """
        constructor

        :param in_size:         Number of channels of input
        :param out_size:        Number of channels of output
        :param is_batchnorm:    Whether to use BN
        :param activ_fuc:       Activation function
        """
        super(UnetDown, self).__init__()
        self.conv = UnetConv2(in_size, out_size, is_batchnorm, activ_fuc)
        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, inputs):
        skip_output = self.conv(inputs)
        outputs = self.down(skip_output)
        return outputs


class UNetUp(nn.Module):
    """
    Upsampling component with skip connections
    """
    def __init__(self, in_size, out_size, output_lim, is_deconv, activ_fuc=nn.ReLU(inplace=True)):
        """
        constructor

        :param in_size:         Number of channels of input
        :param out_size:        Number of channels of output
        :param is_deconv:       Whether to use deconvolution
        :param activ_fuc:       Activation function
        """
        super(UNetUp, self).__init__()
        self.output_lim = output_lim
        self.conv = UnetConv2(in_size, out_size, True, activ_fuc)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, input1, input2):
        """

        :param input1:          Feature maps from shallow layers
        :param input2:          Feature maps from deep layers
        :return:
        """
        input2 = self.up(input2)
        input2 = F.interpolate(input2, size=self.output_lim, mode='bilinear', align_corners=False)
        return self.conv(torch.cat([input1, input2], 1))


class NetUp(nn.Module):
    """
    Upsampling component
    """
    def __init__(self, in_size, out_size, output_lim, is_deconv):
        """
        constructor

        :param in_size:         Number of channels of input
        :param out_size:        Number of channels of output
        :param output_lim:      Input data dimensions
        :param is_deconv:       Whether to use deconvolution
        """
        super(NetUp, self).__init__()
        self.output_lim = output_lim
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, input):
        """

        :param input:           Feature maps
        :return:
        """
        input = self.up(input)
        output = F.interpolate(input, size=self.output_lim, mode='bilinear', align_corners=False)
        return output


class ConvBlock(nn.Module):
    """
    Non-square convolution with flexible definition
    """
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, activ_fuc = nn.ReLU(inplace=True)):
        """
        constructor

        :param in_fea:          Number of channels for convolution layer input
        :param out_fea:         Number of channels for convolution layer output
        :param kernel_size:     Size of the convolution kernel
        :param stride:          Convolution stride
        :param padding:         Convolution padding
        :param activ_fuc:       Activation function
        """
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea,
                            kernel_size=kernel_size, stride=stride, padding=padding)]
        layers.append(nn.BatchNorm2d(out_fea))
        layers.append(activ_fuc)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """

        :param x:               Feature maps
        :return:
        """
        return self.layers(x)


class ConvBlockTanh(nn.Module):

    """
    Convolution at the end for normalization
    """
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1):
        """
        constructor

        :param in_fea:          Number of channels for convolution layer input
        :param out_fea:         Number of channels for convolution layer output
        :param kernel_size:     Size of the convolution kernel
        :param stride:          Convolution stride
        :param padding:         Convolution padding
        """
        super(ConvBlockTanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea,
                            kernel_size=kernel_size, stride=stride, padding=padding)]
        layers.append(nn.BatchNorm2d(out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LossDDNet:
    """
    Joint loss function for DDNet and DDNet70
    """
    def __init__(self, weights=[1, 1], entropy_weight=[1, 1]):
        """
        constructor

        :param weights:         The weights of the two decoders in the calculation of the loss value.
        :param entropy_weight:  The weights of the two output channels in the second decoder.
        """
        self.mse = None
        self.cross = None
        self.criterion1 = nn.MSELoss()
        ew = torch.from_numpy(np.array(entropy_weight).astype(np.float32)).cuda()
        self.criterion2 = nn.CrossEntropyLoss(weight=ew)
        self.weights = weights

    def __call__(self, outputs1, outputs2, targets1, targets2):
        """
        :param outputs1:        Output of the first decoder
        :param outputs2:        Velocity model
        :param targets1:        Output of the second decoder
        :param targets2:        Profile of the speed model
        :return:
        """
        self.mse = self.criterion1(outputs1, targets1)
        self.cross = self.criterion2(outputs2, torch.squeeze(targets2).long())

        criterion = (self.weights[0] * self.mse + self.weights[1] * self.cross)

        return criterion


class DDNet70Model(nn.Module):
    """
    DD-Net70 Architecture
    """
    def __init__(self, in_channels, n_classes=1, is_deconv=True, is_batchnorm=True):
        """
        constructor

        :param n_classes:       Number of channels of output (any single decoder)
        :param in_channels:     Number of channels of network input
        :param is_deconv:       Whether to use deconvolution
        :param is_batchnorm:    Whether to use BN
        """

        super(DDNet70Model, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        self.pre_seis_conv = SeismicRecordDownSampling(self.in_channels)

        # Intrinsic UNet section
        self.down3 = UnetDown(32, 64, self.is_batchnorm)
        self.down4 = UnetDown(64, 128, self.is_batchnorm)
        self.down5 = UnetDown(128, 256, self.is_batchnorm)

        self.center = UnetDown(256, 512, self.is_batchnorm)

        self.up5 = UNetUp(512, 256, output_lim=[9, 9], is_deconv=self.is_deconv)
        self.up4 = UNetUp(256, 128, output_lim=[18, 18], is_deconv=self.is_deconv)
        self.up3 = NetUp(128, 64, output_lim=[35, 35], is_deconv=self.is_deconv)
        self.up2 = NetUp(64, 32, output_lim=[70, 70], is_deconv=self.is_deconv)

        self.dc1_final = ConvBlockTanh(32, self.n_classes)
        self.dc2_final = ConvBlockTanh(32, 2)

    def forward(self, inputs):
        """
        :param inputs:          Input Image
        :returns:               The output of both decoders
        """

        compress_seis = self.pre_seis_conv(inputs)

        down3 = self.down3(compress_seis)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        center = self.center(down5)

        decoder1_image = center
        decoder2_image = center

        # decoder1
        dc1_up5 = self.up5(down5, decoder1_image)
        dc1_up4 = self.up4(down4, dc1_up5)
        dc1_up3 = self.up3(dc1_up4)
        dc1_up2 = self.up2(dc1_up3)

        # decoder2
        dc2_up5 = self.up5(down5, decoder2_image)
        dc2_up4 = self.up4(down4, dc2_up5)
        dc2_up3 = self.up3(dc2_up4)
        dc2_up2 = self.up2(dc2_up3)

        return [self.dc1_final(dc1_up2), self.dc2_final(dc2_up2)]


if __name__ == '__main__':

    model = DDNet70Model(n_classes=1, in_channels=7, is_deconv=True, is_batchnorm=True)
    device = torch.device('cuda:0')
    model.to(device)
    from torchsummary import summary
    summary(model, input_size=[(7, 1000, 70)])
