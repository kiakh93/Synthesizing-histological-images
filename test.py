# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:30:47 2019

@author: kf4
"""

import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import scipy.io
from sync_batchnorm import convert_model
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from KiaNetMain import *

from datasetsT import *
from models.networks import *


import torch.nn as nn
import torch.nn.functional as F
import torch

from IPython.core.debugger import set_trace


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs,1), targets)


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=8500, help='number of epochs of training')
    parser.add_argument('--dataset_name', type=str, default="facades", help='name of the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.599, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--decay_epoch', type=int, default=1, help='epoch from which to start lr decay')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threaDSA to use during batch generation')
    parser.add_argument('--img_height', type=int, default=256, help='size of image height')
    parser.add_argument('--img_width', type=int, default=256, help='size of image width')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=3000, help='interval between sampling of images from generators')
    parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
    opt = parser.parse_args()
    print(opt)

#    writer = SummaryWriter()
    os.makedirs('imagesT/%s' % opt.dataset_name, exist_ok=True)
    os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)
    
    cuda = True if torch.cuda.is_available() else False
    

#    criterion_Class = IW_MaxSquareloss(ignore_index= 0, num_class=11, ratio=0.2) 
    # Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.img_height//2**4, opt.img_width//2**4)
    

    
    netS_A = SPADEGenerator()

#    netD_B = Dis2(in_channels=3)

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
 # model = nn.DataParallel(model)

        netS_A = nn.DataParallel(netS_A)
        netS_A.to(device)
        netS_A.load_state_dict(torch.load('saved_models/generator_80000.pth'))
        torch.set_grad_enabled(False)
        torch.cuda._lazy_init()
        netS_A = netS_A.eval() 
#        torch.cuda.set_device(0)
#        torch.distributed.init_process_group(
#    'nccl',
#    init_method='env://',
#    world_size=2,
#    rank=0,
#)
#        netG_A2B = nn.SyncBatchNorm.convert_sync_batchnorm(netG_A2B)
#        netG_B2A = nn.SyncBatchNorm.convert_sync_batchnorm(netG_B2A)
#        netS_B = nn.SyncBatchNorm.convert_sync_batchnorm(netS_B)
#        netS_A = nn.SyncBatchNorm.convert_sync_batchnorm(netS_A)


        print("Let's use", torch.cuda.device_count(), "GPUs!")
#    netD_B = Discriminator()
    
#    if opt.epoch == 0:
#        # Load pretrained models
#        set_trace()
#        netG_A2B.load_state_dict(torch.load('saved_models/netG_A2B60000.pth'))
#        netG_B2A.load_state_dict(torch.load('saved_models/netGB2A60000.pth'))
#        netS_A.load_state_dict(torch.load('saved_models/netS_A60000.pth'))
#        netD_A.apply(weights_init_normal)
#        
#        netD_B.apply(weights_init_normal)
#        netS_B.apply(weights_init_normal)
#        
#        netD_SA.apply(weights_init_normal)
#        netD_SB.apply(weights_init_normal)
#    else:
#        # Initialize weights
#        netG_A2B.apply(weights_init_normal)
#        netG_B2A.apply(weights_init_normal)
#        netS_A.apply(weights_init_normal)
#        netD_A.apply(weights_init_normal)
#        
#        netD_B.apply(weights_init_normal)
#        netS_B.apply(weights_init_normal)
#        
#        netD_SA.apply(weights_init_normal)
#        netD_SB.apply(weights_init_normal)

   

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    Tensor2 = torch.cuda.LongTensor if cuda else torch.FloatTensor
    CK = 100
    temploss = 1
    count = 0
    Step = 1
    prev_time = time.time()
    loss_D = Tensor(1)   
    prev_time = time.time()
    loss_DA = Tensor(1)
    loss_DB = Tensor(1)
    loss_DSA = Tensor(1)
    loss_DSB = Tensor(1)

    dataloader = DataLoader(ImageDataset("../testa3r2/", lr_transforms=None, hr_transforms=None),
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    for i, batch in enumerate(dataloader):
        


            # Optimizers
#            optimizer_DA = torch.optim.Adam(itertools.chain(netD_A.parameters()), lr=LR*1, betas=(opt.b1, opt.b2))
#            optimizer_DB = torch.optim.Adam(itertools.chain(netD_B.parameters()), lr=LR*1, betas=(opt.b1, opt.b2))
        he = (batch['he'].type(Tensor))
        cond = (batch['cond'].type(Tensor))
        name = (batch['name'][0])
#            cond = (batch['cond'].type(Tensor2))

        if count>=0:
            z = torch.randn(1, 256,
                                dtype=torch.float32, device=device)
            fake_ir = netS_A(cond,z=z)




           # for ii in range(1):
               # m,label = torch.max(F.softmax(seg_he[:,1:,:,:],1), 1, keepdim=False, out=None)
                
               # cond2 = cond*0
               # cond2[m>.9] = label[m>.9]
               # cond2 = cond2.detach()

            

            

#                set_trace()
            


        # If at sample interval save image
        adict = {}
  
        adict['fake_ir'] = fake_ir.data.cpu().numpy()
        adict['cond'] = cond.data.cpu().numpy()
        adict['he'] = he.data.cpu().numpy()
 

        aa = 'imagesT/'+name[:-4]+'.mat'
        scipy.io.savemat(aa, adict)

    
     #   if epoch==299 or epoch==499 or epoch==699:
            # Save model checkpoints
#                torch.save(netG_A2B.module.state_dict(), 'saved_models/netG_A2B%d.pth' % batches_done)
#                torch.save(discriminator.state_dict(), 'saved_models/%s/discriminator_%d.pth' % batches_done)
#                torch.save(netG_B2A.module.state_dict(), 'saved_models/netGB2A%d.pth' % batches_done)
#                
if __name__ == '__main__':
    
    main()
