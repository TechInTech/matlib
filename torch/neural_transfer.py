import torch
import torch.nn as nn
import numpy as np
from torch.nn import ConvL
class Gram(nn.Module):

    def forward(self, x):
        (b, ch, h, w) = x.size()
        feature = x.view(b, ch, h*w)
        feature_t = feature.transpose(1,2)
        return feature.bmm(feature_t) / (ch*h*w)

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
        var = torch.var(t, 2).unsqueeze(2).expand_as(x) * (n-1) / float(n)
        scale_broadcast = self.weight.unsqueeze(1).unsqueeze(1).unsqueeze(0).expand_as(x)
        shift_broadcast = self.bias.unsqueeze(1).unsqueeze(1).unsqueeze(0).expand_as(x)
        out = (x - mean) / torch.sqrt(var+self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        self.reflection_padding = int(np.floor(kernel_size/2.))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        return out

class Basicblock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
		super(Basicblock, self).__init__()
		self.downsample = downsample
		if self.downsample is not None:
			self.residual_layer = nn.Conv2d(in_channels, out_channels,
														kernel_size=1, stride=stride)
		conv_block=[]
		conv_block+=[norm_layer(in_channels),
                    nn.ReLU(inplace=True),
                    ConvLayer(in_channels, out_channels, kernel_size=3, stride=stride),
                    norm_layer(out_channels),
                    nn.ReLU(inplace=True),
                    ConvLayer(out_channels, out_channels, kernel_size=3, stride=1),
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
        reflection_padding = int(np.floor(kernel_size/2.))
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
		conv_block=[]
		conv_block+=[norm_layer(inplanes),
								nn.ReLU(inplace=True),
								UpsampleConvLayer(inplanes, planes, kernel_size=3, stride=1, upsample=stride),
								norm_layer(planes),
								nn.ReLU(inplace=True),
								ConvLayer(planes, planes, kernel_size=3, stride=1)]
		self.conv_block = nn.Sequential(*conv_block)
	
	def forward(self, input):
		return self.residual_layer(input) + self.conv_block(input)