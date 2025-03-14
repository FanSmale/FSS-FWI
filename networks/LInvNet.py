import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic network unit
    """
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, act_fuc=nn.ReLU(inplace=True)):
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
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea,
                            kernel_size=kernel_size, stride=stride, padding=padding)]

        layers.append(nn.BatchNorm2d(out_fea))
        layers.append(act_fuc)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """

        :param x:                   Feature maps
        :return:
        """
        return self.layers(x)

class ConvDownBlock2(nn.Module):
    """
    A Downsampling component composed of two convolutional blocks
    """
    def __init__(self, in_ch, out_ch):
        """
        constructor

        :param in_ch:               Number of input channels
        :param out_ch:              Number of output channels
        """
        super(ConvDownBlock2, self).__init__()

        self.maxp = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True))

    def forward(self, inputs):
        """

        :param inputs:              Feature maps
        :return:
        """
        inputs = self.maxp(inputs)
        inputs = self.conv1(inputs)
        outputs = self.conv2(inputs)

        return outputs

class ConvDownBlock3(nn.Module):
    """
    Downsampling component composed of three convolutional blocks
    """

    def __init__(self, in_ch, out_ch):
        """
        constructor

        :param in_ch:               Number of input channels
        :param out_ch:              Number of output channels
        """
        super(ConvDownBlock3, self).__init__()

        self.maxp = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True))

    def forward(self, inputs):
        """

        :param inputs:              Feature maps
        :return:
        """
        inputs = self.maxp(inputs)
        inputs = self.conv1(inputs)
        inputs = self.conv2(inputs)
        outputs = self.conv3(inputs)

        return outputs


class ConvUpBlock2(nn.Module):
    """
    Upsampling component composed of two convolutional blocks
    """
    def __init__(self, in_ch, out_ch, interpolate_size):
        """

        :param in_ch:               Number of input channels
        :param out_ch:              Number of output channels
        :param interpolate_size:    Dimension size after feature map interpolation
        """
        super(ConvUpBlock2, self).__init__()

        self.cov_list = []
        self.interpolate_size = interpolate_size

        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True))

    def forward(self, inputs):
        """

        :param inputs:              Feature maps
        :return:
        """
        inputs = F.interpolate(inputs, size=self.interpolate_size, mode='bilinear', align_corners=False)
        inputs = self.conv1(inputs)
        outputs = self.conv2(inputs)

        return outputs

class ConvUpBlock2for2Input(nn.Module):
    """
    Upsampling component with skip connections consisting of two convolutional blocks
    """

    def __init__(self, in_ch, out_ch, interpolate_size):
        """

        :param in_ch:               Number of input channels
        :param out_ch:              Number of output channels
        :param interpolate_size:    Dimension size after feature map interpolation
        """
        super(ConvUpBlock2for2Input, self).__init__()

        self.cov_list = []
        self.interpolate_size = interpolate_size

        self.conv1 = nn.Sequential(nn.Conv2d(in_ch + out_ch, out_ch, 3, 1, 1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True))

    def forward(self, inputs, pre_skip_part):
        """

        :param inputs:              Feature maps from deep layer
        :param pre_skip_part:       Feature maps from shallow layer
        :return:
        """

        inputs = F.interpolate(inputs, size=self.interpolate_size, mode='bilinear', align_corners=False)
        if pre_skip_part.size()[2] != self.interpolate_size[0] or pre_skip_part.size()[3] != self.interpolate_size[1]:
            pre_skip_part = F.interpolate(pre_skip_part, size=self.interpolate_size, mode='bilinear', align_corners=False)

        inputs = self.conv1(torch.cat([inputs, pre_skip_part], 1))
        outputs = self.conv2(inputs)

        return outputs


class LInvNet(nn.Module):
    """
    LInvNet architecture
    """
    def __init__(self, input_channel):
        """
        constructor

        :param input_channel:       Number of input channels
        """
        super(LInvNet, self).__init__()

        self.pre_dim_reducer1 = ConvBlock(input_channel, 8, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer2 = ConvBlock(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.pre_dim_reducer3 = ConvBlock(8, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer4 = ConvBlock(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.pre_dim_reducer5 = ConvBlock(16, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer6 = ConvBlock(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.down_block1 = ConvDownBlock2(32, 64)

        self.down_block2 = ConvDownBlock2(64, 128)

        self.down_block3 = ConvDownBlock3(128, 256)

        self.up_block1_2input = ConvUpBlock2for2Input(256, 128, interpolate_size=[16, 18])

        self.up_block2_2input = ConvUpBlock2for2Input(128, 64, interpolate_size=[35, 35])

        self.up_block3 = ConvUpBlock2(64, 32, interpolate_size=[70, 70])

        self.end = ConvBlock(32, 1, act_fuc=nn.Softsign())

    def forward(self, input):
        """

        :param input:               Feature maps
        :return:
        """
        orig_size = input.shape[2:]
        new_size = [orig_size[0] // 2, orig_size[1]]
        dimred0 = F.interpolate(input, size=new_size, mode='bilinear', align_corners=False)

        dimred1 = self.pre_dim_reducer1(dimred0)
        dimred2 = self.pre_dim_reducer2(dimred1)
        dimred3 = self.pre_dim_reducer3(dimred2)
        dimred4 = self.pre_dim_reducer4(dimred3)
        dimred5 = self.pre_dim_reducer5(dimred4)
        dimred6 = self.pre_dim_reducer6(dimred5)

        down1 = self.down_block1(dimred6)
        down2 = self.down_block2(down1)
        center = self.down_block3(down2)

        up1 = self.up_block1_2input(center, pre_skip_part=down2)
        up2 = self.up_block2_2input(up1, pre_skip_part=down1)
        up3 = self.up_block3(up2)

        output = self.end(up3)

        return output


if __name__ == '__main__':
    net = LInvNet(7)
    device = torch.device('cuda:0')
    net.to(device)

    from torchsummary import summary
    summary(net.cuda(), input_size=(7, 1000, 70))



