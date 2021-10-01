# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:17:27 2019

@author: kf4
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 23:35:38 2019

@author: kf4
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 22:23:23 2018

@author: kf4
"""

import glob
import random
import os
import torch.nn as nn

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from IPython.core.debugger import set_trace
from AffineT import *

class ImageDataset(Dataset):
    def __init__(self, root, lr_transforms=None, hr_transforms=None):
#        RAND = random.randint(24,48)

        self.lr_transform = transforms.Compose(lr_transforms)
        self.hr_transform = transforms.Compose(hr_transforms)

#        self.files = sorted(glob.glob(root + '/*.*'))
        self.filesI = os.path.join(root, 'ir')
        self.list_I = os.listdir(self.filesI)
        self.filesT = os.path.join(root, 'he')
        self.list_T = os.listdir(self.filesT)
        self.filesC = os.path.join(root, 'cond')
        self.list_C = os.listdir(self.filesC)

        
    def __getitem__(self, index):
        
                
#        I_name = os.path.join(self.filesI,self.list_I[index%len(self.list_I)])
#        #img = Image.open(I_name)
#        img = np.load(I_name)
##        img = im['xx']
        
        
        T_name = os.path.join(self.filesT,self.list_I[index%len(self.list_T)])
        #img = Image.open(I_name)
        img_T = np.load(T_name)
#        img = im['xx']

        C_name = os.path.join(self.filesC,self.list_I[index%len(self.list_T)])
        #img = Image.open(I_name)
        img_C = np.load(C_name)
        
#        img_C = np.squeeze(img_C, axis=0)
        CRx = 0
        CRy = 0
        fact = 1024
        img_T = img_T[CRx:CRx+fact,CRy:CRy+fact,:]
        img_C = img_C[CRx:CRx+fact,CRy:CRy+fact]
#        img_C[img_C==10] = 0
        
#        set_trace()
       # img_C = img_C + 1
       
        
        img_C[np.isnan(img_C)]=0

        img_T[img_T==0] = 1        
   
        trans1 = transforms.ToTensor()
        img_C = img_C.astype(float)
        #img_C = np.squeeze(img_C, axis=0)
        img_C = np.expand_dims(img_C, axis=2)
        
        L = img_C*0
        temp = np.zeros((1024,1024,11))
        for i in range(11):
            L[img_C==i] = 1
#            temp[:,:,i:i+1] = np.expand_dims(L, axis=2)
            temp[:,:,i:i+1] = L
        
#        temp2[1:7,:,:] = temp2[1:7,:,:]*img_lr[-1,:,:].repeat((6,1,1))
        
        temp = trans1(temp)
        
        img_C = trans1(img_C) 
        img_T = trans1(img_T) -1
        
        
#        shear = random.uniform(-.35,0.35)
#        translation = (random.uniform(-.25,0.25),random.uniform(-.25,0.25))
#        rotation = random.uniform(0,180)
#        scale = (random.uniform(.65,1.35),random.uniform(.65,1.35))
#        S = random.randint(16,33)
#        affine_transform = Affine(rotation_range=rotation, translation_range=translation, shear_range = shear)
#        
#        
#        

        
        name = self.list_I[index%len(self.list_I)];
        img_T = (img_T)
        img_T = img_T+1
        img_C = (img_C)
#        img_C = img_C.long()

        



       # V = img_V.repeat(3,1,1)
#        
        return {'he': img_T, 'cond': temp[:,:,:], 'name': name}
        #return {'lr': img_lr, 'name': name}

    def __len__(self):
        return len(self.list_I)
