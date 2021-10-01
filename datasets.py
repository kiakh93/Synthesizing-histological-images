import os
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, num_class = 4):

        self.num_class = num_class
        
        # H&E images directory (Inouts)
        self.filesI = os.path.join(root, 'he')
        self.list_I = os.listdir(self.filesI)[:]
        
        # Labels directory (Targets)
        self.filesT = os.path.join(root, 'ln')
        self.list_T = os.listdir(self.filesT)[:]
        

        
    def __getitem__(self, index):

        I_name = os.path.join(self.filesI,self.list_I[index%len(self.list_I)])
        img = sio.loadmat(I_name)
        img = img['xx']/256
        # Extracting background
        Background = np.sum(img,axis=2)>.91*3 
        
        trans1 = transforms.ToTensor()
        img_he = trans1(img)
        
        name = self.list_I[index%len(self.list_I)];

        T_name = os.path.join(self.filesT,self.list_I[index%len(self.list_I)])
        img = sio.loadmat(T_name)
        img = img['rr']
        
        # correct the background in labels
        img[Background] = 0

        # creating a seperate channel for each class (each channel would be a binary class)
        labels = np.zeros((512,512,self.num_class))
        for i in range(self.num_class):
            Binary = img*0
            Binary[img==i] = 1
            labels[:,:,i:i+1] = np.expand_dims(Binary, axis=2)
        
        labels = trans1(labels)

        return {'he': img_he, 'labels': labels[:,:,:], 'name': name}

    def __len__(self):
        return len(self.list_I)
