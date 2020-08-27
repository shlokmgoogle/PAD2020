from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
#import config as cf #write separate config.py file to take the inputs

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime

#from networks import *
from torch.autograd import Variable

from data_loader import SfSDataset, ToTensor
from sfsnet import SfSNet
from sfs_loss import SfSLoss


use_cuda = torch.cuda.is_available()

print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([ToTensor()
]) # meanstd transformation

transform_test = transforms.Compose([ToTensor(),
])

traindata =  SfSDataset(csv_file='/scratch2/photometric_stereo/Data_Generation/Train_Data_syn.csv',transform=transform_train)  #Write a dataloader function that can read the database provided by .csv file
#testdata =  SfSDataset(csv_file='/scratch2/photometric_stereo/Data_Generate_real/Test_Data.csv',transform=transform_test)  #Write a dataloader function that can read the database provided by .csv file

train_loader = torch.utils.data.DataLoader(traindata, batch_size=10, shuffle=False, num_workers=4)
#test_loader = torch.utils.data.DataLoader(testdata, batch_size=100, shuffle=False, num_workers=4)

sample=iter(train_loader).next()


net=SfSNet()

#Use GPU
#net.cuda()
#net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
#cudnn.benchmark = True
image, mask, normal, albedo, light = sample['image'], sample['mask'], sample['normal'], sample['albedo'], sample['light']
#image=image.unsqueeze(0)
#image, normal, albedo, light = image.cuda(), normal.cuda(), albedo.cuda(), light.cuda() # GPU settings
image, mask, normal, albedo, light = Variable(image), Variable(mask), Variable(normal), Variable(albedo), Variable(light)
nout, aout, lout = net(image)


criterion=SfSLoss()
loss,_,_,_,recloss=criterion(image,mask,normal,albedo,light, nout, aout,lout)
print(recloss.data[0])




#train_loader = torch.utils.data.DataLoader(traindata, batch_size=10, shuffle=False, num_workers=4)
#test_loader = torch.utils.data.DataLoader(testdata, batch_size=100, shuffle=False, num_workers=4)

#data=iter(train_loader).next() #data is a dictionary with whole minibatch
#image=data['image'] #Tensor of mini-batchx3x128x128
 #other variables can be called in the similar way



#for index,item in enumerate(train_loader): #to enumerate this

