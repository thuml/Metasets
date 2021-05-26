import torch
import torch.nn as nn
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d, MetaBatchNorm1d, MetaBatchNorm2d, MetaLinear)
from torchmeta.modules.utils import get_subdict


class conv_2d(MetaModule):
    def __init__(self, in_ch, out_ch, kernel, activation='relu'):
        super(conv_2d, self).__init__()
        if activation == 'relu':
            self.conv = MetaSequential(
                MetaConv2d(in_ch, out_ch, kernel_size=kernel),
                MetaBatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        elif activation == 'tanh':
            self.conv = MetaSequential(
                MetaConv2d(in_ch, out_ch, kernel_size=kernel),
                MetaBatchNorm2d(out_ch),
                nn.Tanh()
            )
        elif activation == 'leakyrelu':
            self.conv = MetaSequential(
                MetaConv2d(in_ch, out_ch, kernel_size=kernel),
                MetaBatchNorm2d(out_ch),
                nn.LeakyReLU()
            )


    def forward(self, x, params=None):
        x = self.conv(x, params=get_subdict(params, 'conv'))
        return x


class fc_layer(MetaModule):
    def __init__(self, in_ch, out_ch, bn=True, activation='leakyrelu'):
        super(fc_layer, self).__init__()
        if activation == 'relu':
            self.ac = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.ac = nn.LeakyReLU()
        if bn:
            self.fc = MetaSequential(
                MetaLinear(in_ch, out_ch),
                MetaBatchNorm1d(out_ch),
                self.ac
            )
        else:
            self.fc = MetaSequential(
                MetaLinear(in_ch, out_ch),
                self.ac
            )

    def forward(self, x, params):
        x = self.fc(x, params=get_subdict(params, 'fc'))
        return x



class transform_net(MetaModule):
    def __init__(self, in_ch, K=3):
        super(transform_net, self).__init__()
        self.K = K
        self.conv2d1 = conv_2d(in_ch, 64, 1)
        self.conv2d2 = conv_2d(64, 128, 1)
        self.conv2d3 = conv_2d(128, 1024, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(512, 1))
        self.fc1 = fc_layer(1024, 512)
        self.fc2 = fc_layer(512, 256)
        self.fc3 = MetaLinear(256, K*K)
    


    def forward(self, x, params=None):
        x = self.conv2d1(x, params=get_subdict(params, 'conv2d1'))
        x = self.conv2d2(x, params=get_subdict(params, 'conv2d2'))
        x = self.conv2d3(x, params=get_subdict(params, 'conv2d3'))
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = x.view(x.size(0), -1).contiguous()
        x = self.fc1(x, params=get_subdict(params, 'fc1'))
        x = self.fc2(x, params=get_subdict(params, 'fc2'))
        x = self.fc3(x, params=get_subdict(params, 'fc3'))

        iden = torch.eye(self.K).view(1,self.K * self. K).repeat(x.size(0),1)
        iden = iden.to(device='cuda') 
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x

