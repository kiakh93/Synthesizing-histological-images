"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from spec import *
from IPython.core.debugger import set_trace

class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = 64
        
        self.layer1 = nn.Sequential((SpectralNorm(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))),nn.InstanceNorm2d(ndf))
        self.layer2 = nn.Sequential((SpectralNorm(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))),nn.InstanceNorm2d(ndf*2))
        self.layer3 = nn.Sequential((SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))),nn.InstanceNorm2d(ndf*4))
        self.layer4 = nn.Sequential((SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))),nn.InstanceNorm2d(ndf*8))
        self.layer5 = nn.Sequential((SpectralNorm(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))),nn.InstanceNorm2d(ndf*8))        
        self.layer6 = nn.Sequential((SpectralNorm(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))),nn.InstanceNorm2d(ndf*8))
     

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, False)

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        x = self.layer6(self.actvn(x))

        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar
