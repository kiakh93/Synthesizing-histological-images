"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from IPython.core.debugger import set_trace


class SPADEGenerator(BaseNetwork):


    def __init__(self, num_class = 4):
        super().__init__()
        nf = 64

        self.sw, self.sh = self.compute_latent_vector_size()

        self.fc = nn.Linear(256, 16 * nf * 8*8)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, num_class)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, num_class)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, num_class)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, num_class)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, num_class)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, num_class)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, num_class)

        final_nc = nf


        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self):
        num_up_layers = 5


        sw = 256 // (2**num_up_layers)
        sh = round(sw / 1)

        return sw, sh

    def forward(self, input, z=None):
        seg = input
        flag = 1
        if flag==1:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), 256,
                                dtype=torch.float32, device=input.get_device())
                
            x = self.fc(z)
            x = x.view(-1, 16 * 64, self.sh, self.sw)
#            x = x.view(-1, 16 * 64, 8, 8)
#            x = F.interpolate(x, size=(32, 32))
            
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(input, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)


        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x
