import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from torch.utils.serialization import load_lua


class Gram(nn.Module):

    def forward(self, x):
        (b, ch, h, w) = x.size()
        feature = x.view(b, ch, h * w)
        feature_t = feature.transpose(1, 2)
        return feature.bmm(feature_t) / (ch * h * w)


class InstanceNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super(InstanceNorm, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(dim))
        self.bias = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_params()

    def _reset_params(self):
        self.weight.data.uniform_()
        self.bias.data.uniform_()

    def forward(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).expand_as(x)
        var = torch.var(t, 2).unsqueeze(2).expand_as(x) * (n - 1) / float(n)
        scale_broadcast = self.weight.unsqueeze(
            1).unsqueeze(1).unsqueeze(0).expand_as(x)
        shift_broadcast = self.bias.unsqueeze(
            1).unsqueeze(1).unsqueeze(0).expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        self.reflection_padding = int(np.floor(kernel_size / 2.))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        return out


class ResidualBasicblock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(ResidualBasicblock, self).__init__()
        self.downsample = downsample
        if self.downsample is not None:
            self.residual_layer = nn.Conv2d(in_channels, out_channels,
                                            kernel_size=1, stride=stride)
        conv_block = []
        conv_block += [norm_layer(in_channels),
                       nn.ReLU(inplace=True),
                       ConvLayer(in_channels, out_channels,
                                 kernel_size=3, stride=stride),
                       norm_layer(out_channels),
                       nn.ReLU(inplace=True),
                       ConvLayer(out_channels, out_channels,
                                 kernel_size=3, stride=1),
                       norm_layer(out_channels)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, input):
        if self.downsample is not None:
            residual = self.residual_layer(input)
        else:
            residual = input
        return residual + self.conv_block(input)


class UpsampleConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample is not None:
            self.unsample_layer = nn.UpsamplingNearest2d(upsample)
        reflection_padding = int(np.floor(kernel_size / 2.))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample_layer(x)
        out = self.reflection_pad(x)
        out = self.conv(out)
        return out


class UpBasicblock(nn.Module):
    """ Up-sample residual block (from MSG-Net paper)
    Enables passing identity all the way through the generator
    ref https://arxiv.org/abs/1703.06953
    """

    def __init__(self, inplanes, planes, stride=2, norm_layer=nn.BatchNorm2d):
        super(UpBasicblock, self).__init__()
        self.residual_layer = UpsampleConvLayer(inplanes, planes,
                                                kernel_size=1, stride=1, upsample=stride)
        conv_block = []
        conv_block += [norm_layer(inplanes),
                       nn.ReLU(inplace=True),
                       UpsampleConvLayer(
                           inplanes, planes, kernel_size=3, stride=1, upsample=stride),
                       norm_layer(planes),
                       nn.ReLU(inplace=True),
                       ConvLayer(planes, planes, kernel_size=3, stride=1)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.residual_layer(x) + self.conv_block(x)


class TransformNet(nn.Module):

    def __init__(self):
        super(TransformNet, self).__init__()

        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = InstanceNorm(32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = InstanceNorm(64)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = InstanceNorm(128)

        self.res1 = ResidualBasicblock(128, 128)
        self.res2 = ResidualBasicblock(128, 128)
        self.res3 = ResidualBasicblock(128, 128)
        self.res4 = ResidualBasicblock(128, 128)
        self.res5 = ResidualBasicblock(128, 128)

        self.deconv1 = UpsampleConvLayer(
            128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = InstanceNorm(64)
        self.deconv2 = UpsampleConvLayer(
            64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = InstanceNorm(64)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        in_X = x
        y = self.relu(self.in1(self.conv1(in_X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class Vgg16(nn.Module):

    def __init__(self):
        super(Vgg16, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        relu4_3 = h

        return [relu1_2, relu2_2, relu3_3, relu4_3]


def init_vgg16(model_folder):
    if not os.path.exists(os.path.join(model_folder, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_folder, 'vgg16.t7')):
            os.system(
                'wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_folder, 'vgg16.t7'))
        vgglua = load_lua(os.path.join(model_folder, 'vgg16.t7'))
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(
            model_folder, 'vgg16.weight'))


def train(style_image, img_size, dataset, vgg_model_dir, style_size, batch_size=8, lr=0.001):

    img_transforms = transforms.Compose([transforms.Scale(img_size),
                                         transforms.CenterCrop(img_size),
                                         transforms.ToTensor,
                                         transforms.Lambda(
                                             lambda x: x.mul(255))
                                         ])

    train_dataset = datasets.ImageFolder(dataset, img_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    transformer = TransformNet()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    vgg = Vgg16()
    init_vgg16(vgg_model_dir)
    vgg.load_state_dict(torch.load(
        os.path.join(vgg_model_dir, "vgg16.weight")))

    