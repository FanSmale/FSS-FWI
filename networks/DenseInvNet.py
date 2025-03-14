import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class ConvUpBlock(nn.Module):
    """
    Upsampling component composed
    """
    def __init__(self, in_ch, out_ch, interpolate_size):
        """
        constructor

        :param in_ch:               Number of input channels
        :param out_ch:              Number of output channels
        :param interpolate_size:    Dimension size after feature map interpolation
        """
        super(ConvUpBlock, self).__init__()

        self.cov_list = []
        self.interpolate_size = interpolate_size

        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(True))

        self.conv2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(True))

    def forward(self, inputs):
        """

        :param inputs:              Feature maps
        :return:
        """

        inputs = F.interpolate(inputs, size=self.interpolate_size, mode='bilinear', align_corners=False)
        inputs = self.conv1(inputs)
        outputs = self.conv2(inputs)

        return outputs


class SeismicRecordDownSampling(nn.Module):
    """
    Downsampling component for seismic records
    """
    def __init__(self, shot_num):
        """
        constructor

        :param shot_num:            Number of input channels
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

        :param x:                   Feature maps
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


class Bottleneck(nn.Module):
    """
    Basic components of the Dense module
    """
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate=0):
        """

        :param in_channels:         Number of input channels
        :param growth_rate:         Number of output channels
        :param bn_size:             The product of this parameter and the number of output channels
                                    is the number of channels in the middle
        :param drop_rate:           The rate of "drop up" operation
        """
        super(Bottleneck, self).__init__()
        self.drop_rate = drop_rate
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, bn_size*growth_rate, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(bn_size*growth_rate),
            nn.ReLU(True),
            nn.Conv2d(bn_size*growth_rate, growth_rate, 3, stride=1, padding=1, bias=False)
        )
        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, x):
        """

        :param x:                   Feature maps
        :return:
        """
        y = self.bottleneck(x)
        if self.drop_rate > 0:
            y = self.dropout(y)
        output = torch.cat([x, y], 1)
        return output


class Transition(nn.Module):
    """
    The middle layer between two bottlenecks
    """
    def __init__(self, in_channels, channels):
        """
        constructor
        (in_channels / 2 = channel)

        :param in_channels:         Number of input channels
        :param channels:            Number of output channels
        """
        super(Transition, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, channels, 1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        """

        :param x:                   Feature maps
        :return:
        """
        output = self.transition(x)
        return output


class DenseBlock(nn.Module):
    """
    Dense block
    """
    def __init__(self, layers_num, in_channels, growth_rate, bn_size, drop_rate=0):
        """
        constructor
        (channel x 4, but size unchanged)

        :param layers_num:          Number of convolutional layers for a Dense block
        :param in_channels:         Number of input channels
        :param growth_rate:         The growth rate of the input channel
        :param bn_size:             The product of this parameter and the number of output channels
                                    is the number of channels in the middle
        :param drop_rate:           The rate of "drop up" operation
        """
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(layers_num):
            layers.append(Bottleneck(in_channels+i*growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """

        :param x:                   Feature maps
        :return:
        """
        output = self.layers(x)
        return output


class ConvBNReLU(nn.Module):
    """
    Basic network unit: Conv + BN + ReLU
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        """
        constructor

        :param in_ch:               Number of input channels
        :param out_ch:              Number of output channels
        :param kernel_size:         Convolution kernel size
        :param dilation:            Dilated convolution parameters
        """
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        """

        :param x:
        :return:
        """
        return self.relu(self.bn(self.conv(x)))


class DenseInvNet(nn.Module):
    """
    DenseInvNet architecture
    """
    def __init__(self, in_ch=7, growth_rate=32, blocks=[6, 12]):
        """
        constructor

        :param in_ch:               Number of input channels
        :param growth_rate:         The growth rate of the input channel
        :param blocks:              List to configure the number of convolutional layers in different Dense blocks
        """
        super(DenseInvNet, self).__init__()
        bn_size = 4
        drop_rate = 0

        self.pre_seis_conv = SeismicRecordDownSampling(shot_num=in_ch)

        self.seis_conv = nn.Sequential(
            nn.Conv2d(32, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        num_features = 64
        self.seis_dense_block1 = DenseBlock(blocks[0], num_features, growth_rate, bn_size, drop_rate)

        num_features = num_features + blocks[0] * growth_rate
        self.seis_dense_transition = Transition(num_features, num_features // 2)

        num_features = num_features // 2
        self.seis_dense_block2 = DenseBlock(blocks[1], num_features, growth_rate, bn_size, drop_rate)

        self.up_block1 = ConvUpBlock(512, 256, interpolate_size=[18, 18])

        self.up_block2 = ConvUpBlock(256, 128, interpolate_size=[35, 35])

        self.up_block3 = ConvUpBlock(128, 64, interpolate_size=[70, 70])

        self.end1 = ConvBlock(64, 1, act_fuc=nn.Softsign())

        self.end2 = ConvBlock(64, 2, act_fuc=nn.Softsign())

    def forward(self, lowres_vms, seis):
        """

        :param lowres_vms:          Low-resolution velocity model from LResInvNet inversion
        :param seis:                Seismic records without direct waves
        :return:
        """

        compress_seis = self.pre_seis_conv(seis)

        seis_fea = self.seis_conv(compress_seis)

        seis_db_fea1 = self.seis_dense_block1(seis_fea)
        seis_dt_fea = self.seis_dense_transition(seis_db_fea1)
        seis_db_fea2 = self.seis_dense_block2(seis_dt_fea)

        decoder1 = seis_db_fea2
        decoder2 = seis_db_fea2

        # decoder1
        up_fea1 = self.up_block1(decoder1)
        up_fea2 = self.up_block2(up_fea1)
        up_fea3 = self.up_block3(up_fea2)
        output1 = self.end1(up_fea3)

        # decoder2
        up_fea1 = self.up_block1(decoder2)
        up_fea2 = self.up_block2(up_fea1)
        up_fea3 = self.up_block3(up_fea2)
        output2 = self.end2(up_fea3)

        return output1 + lowres_vms, output2


class MixLoss:
    """
    The joint loss function with \mathcal{L}_{main} and \mathcal{L}_{cont}
    """
    def __init__(self, weights: list = [0.5, 0.5], entropy_weight: list = [0.5, 0.5]):
        """
        constructor

        :param weights:             List of weights for the two losses
        :param entropy_weight:      The weight of two classes in cross entropy
        """

        self.weights = weights
        ew = torch.from_numpy(np.array(entropy_weight).astype(np.float32)).cuda()

        self.criterion1 = None
        self.criterion2 = None

        self.cross = 0
        self.mse = 0

        if self.weights[0] != 0.0:
            self.criterion1 = nn.MSELoss()

        if self.weights[1] != 0.0:
            self.criterion2 = nn.CrossEntropyLoss(weight=ew)

    def __call__(self, outputs1, outputs2, targets1, targets2):
        """

        :param outputs1:            Inverted velocity model
        :param outputs2:            True velocity model target
        :param targets1:            Inverted velocity model contour
        :param targets2:            True velocity model contour target
        :return:
        """

        if self.criterion1 is not None:
            self.mse = self.criterion1(outputs1, targets1)

        if self.criterion2 is not None:
            self.cross = self.criterion2(outputs2, torch.squeeze(targets2).long())

        self.base_loss = (self.weights[0] * self.mse + self.weights[1] * self.cross)

        return self.base_loss


if __name__ == '__main__':

    net = DenseInvNet()

    device = torch.device('cuda:0')
    net.to(device)

    from torchsummary import summary
    summary(net, input_size=[(1, 70, 70), (7, 1000, 70)])


