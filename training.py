import argparse
import os
import time
import datetime
import sys

import numpy as np
import scipy.io

from sync_batchnorm import convert_model

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from datasets import *
from models.networks import *


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=2700, help='number of epochs of training')
    parser.add_argument('--dataset_name', type=str, default="synth", help='name of the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--decay_epoch', type=int, default=1, help='epoch from which to start lr decay')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--num_class', type=int, default=4, help='size of image height')
    parser.add_argument('--img_width', type=int, default=128, help='size of image width')
    parser.add_argument('--channels', type=int, default=10, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=3000, help='interval between sampling of images from netGs')
    parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
    opt = parser.parse_args()
    print(opt)

    os.makedirs('images/', exist_ok=True)
    os.makedirs('saved_models/', exist_ok=True)
    
    cuda = True if torch.cuda.is_available() else False
    
    # Defines the GAN loss which uses either LSGAN or the regular GAN.
    # Borrowed from https://github.com/NVlabs/SPADE/
    class GANLoss(nn.Module):
        def __init__(self, gan_mode = 'hinge', target_real_label=1.0, target_fake_label=0.0,
                     tensor=torch.cuda.FloatTensor, opt=None):
            super(GANLoss, self).__init__()
            self.real_label = target_real_label
            self.fake_label = target_fake_label
            self.real_label_tensor = None
            self.fake_label_tensor = None
            self.zero_tensor = None
            self.Tensor = tensor
            self.gan_mode = 'ls'
            if gan_mode == 'ls':
                pass
            elif gan_mode == 'original':
                pass
            elif gan_mode == 'w':
                pass
            elif gan_mode == 'hinge':
                pass
            else:
                raise ValueError('Unexpected gan_mode {}'.format(gan_mode))
    
        def get_target_tensor(self, input, target_is_real):
            if target_is_real:
                if self.real_label_tensor is None:
                    self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                    self.real_label_tensor.requires_grad_(False)
                return self.real_label_tensor.expand_as(input)
            else:
                if self.fake_label_tensor is None:
                    self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                    self.fake_label_tensor.requires_grad_(False)
                return self.fake_label_tensor.expand_as(input)
    
        def get_zero_tensor(self, input):
            if self.zero_tensor is None:
                self.zero_tensor = self.Tensor(1).fill_(0)
                self.zero_tensor.requires_grad_(False)
            return self.zero_tensor.expand_as(input)
    
        def loss(self, input, target_is_real, for_discriminator=True):
            if self.gan_mode == 'original':  # cross entropy loss
                target_tensor = self.get_target_tensor(input, target_is_real)
                loss = F.binary_cross_entropy_with_logits(input, target_tensor)
                return loss
            elif self.gan_mode == 'ls':
                target_tensor = self.get_target_tensor(input, target_is_real)
                return F.mse_loss(input, target_tensor)
            elif self.gan_mode == 'hinge':
                if for_discriminator:
                    if target_is_real:
                        minval = torch.min(input - 1, self.get_zero_tensor(input))
                        loss = -torch.mean(minval)
                    else:
                        minval = torch.min(-input - 1, self.get_zero_tensor(input))
                        loss = -torch.mean(minval)
                else:
                    assert target_is_real, "The generator's hinge loss must be aiming for real"
                    loss = -torch.mean(input)
                return loss
            else:
                # wgan
                if target_is_real:
                    return -input.mean()
                else:
                    return input.mean()
    
        def __call__(self, input, target_is_real, for_discriminator=True):
            # computing loss is a bit complicated because |input| may not be
            # a tensor, but list of tensors in case of multiscale discriminator
            if isinstance(input, list):
                loss = 0
                for pred_i in input:
                    if isinstance(pred_i, list):
                        pred_i = pred_i[-1]
                    loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                    bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                    new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                    loss += new_loss
                return loss / len(input)
            else:
                return self.loss(input, target_is_real, for_discriminator)
      
    # Loss function
    gan = GANLoss() 
       
    # Defininjg the generator
    netG = SPADEGenerator(num_class = opt.num_class)
    
    # Defininjg the  discriminator
    netD = MultiscaleDiscriminator()
    
    # Sync_batchnorm
    netD = convert_model(netD)
    netG = convert_model(netG)
    
    # initializng the models
    netD.init_weights('xavier',.02)
    netG.init_weights('xavier',.02)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Parallel computing for more than 1 GPUs
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        

        netD = nn.DataParallel(netD)
        netD.to(device)
        netG = nn.DataParallel(netG)
        netG.to(device)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    count = 0
    prev_time = time.time()
    dataloader = DataLoader(ImageDataset("../dataset/", num_class = opt.num_class),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
    
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            # learning rate decay
            LR =  opt.lr*(.97 ** (count // 1000))
            count = count + 1
         
            # Optimizers
            optimizer_G = torch.optim.Adam(netG.parameters(), lr=LR, betas=(opt.b1, opt.b2))
            optimizer_D = torch.optim.Adam(netD.parameters(), lr=LR, betas=(opt.b1, opt.b2))

            he = (batch['he'].type(Tensor))
            labels = (batch['labels'].type(Tensor))

            # Normalizing H&E images to [-1 1]
            he = (he - .5)/.5

            optimizer_G.zero_grad()
            
            # Synthesizing H&E images
            Synth_he = netG(labels)
            
            # adversarial loss
            pred_fake = netD(Synth_he)
            loss_gan_a = gan(pred_fake,target_is_real=True, for_discriminator=False)
            
            loss_G = loss_gan_a*10
            loss_G.backward()
    
            optimizer_G.step()

            optimizer_D.zero_grad()
            loss_real = gan(netD(he),target_is_real=True, for_discriminator=True)
            loss_fake = gan(netD(Synth_he.detach()),target_is_real=False, for_discriminator=True)
            loss_D = (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------
    
            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            
            # If at sample interval save images in a .mat format
            if batches_done % opt.sample_interval == 0:
                torch.save(netG.state_dict(), 'saved_models/%s/netG_%d.pth' % (opt.dataset_name, batches_done))
                adict = {}
                adict['he'] = he.data.cpu().numpy()
                adict['Synth_he'] = (Synth_he).data.cpu().numpy()
                adict['labels'] = labels.data.cpu().numpy()
                aa = 'images/'+str(batches_done)+'.mat'
                scipy.io.savemat(aa, adict)
            
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] ETA: %s lr: %f CR: %f" %
                                                            (epoch, opt.n_epochs,
                                                            i, len(dataloader),
                                                            loss_D.item(), 
                                                            time_left,LR,LR))

if __name__ == '__main__':
    
    main()
