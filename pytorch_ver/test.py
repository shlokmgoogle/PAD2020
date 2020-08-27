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
import matplotlib.pyplot as plt

#from networks import *
from torch.autograd import Variable

from data_loader import SfSDataset, ToTensor
from sfsnet import SfSNet
from sfs_loss import SfSLoss


use_cuda = torch.cuda.is_available()


# DATA LOADING

print('\n[Phase 1] : Data Preparation')

transform_test = transforms.Compose([ToTensor(),
])

testdata =  SfSDataset(csv_file='/scratch2/photometric_stereo/Data_Generate_real/Test_Data.csv',transform=transform_test)  #Write a dataloader function that can read the database provided by .csv file

test_loader = torch.utils.data.DataLoader(testdata, batch_size=20, shuffle=False, num_workers=4)


#Load the model
net=SfSNet()
net.load_state_dict(torch.load('net_epoch_8.pth'))
#net=torch.load('net_epoch_0.pt')
print('Network Initialized')

#Use GPU
net.cuda(1)
#net = torch.nn.DataParallel(net, device)
cudnn.benchmark = True
net.eval()

#Loss function
criterion = SfSLoss() #Write a loss function for SfS
criterion.cuda(1)
print('Loss Initialized')

def visualize(rec0):

    rec0=((rec0.data).cpu()).numpy()
    rec0=rec0.transpose((1,2,0))

    rec0[rec0>1]=1
    rec0[rec0<0]=0

    #print(rec0.shape)
    plt.imshow(rec0)



print('Visualization')
#Visualization

data=iter(test_loader).next() #data is a dictionary with whole minibatch
image=data['image']
image=Variable(image.cuda(1))
nout, aout, lout = net(image)

for i in range(0,nout.size(0)):

    ax = plt.subplot(1,3,1)
    plt.tight_layout()
    ax.axis('off')
    visualize(image[i,...])

    ax = plt.subplot(1,3,2)
    plt.tight_layout()
    ax.axis('off')
    visualize(nout[i,...])

    ax = plt.subplot(1,3,3)
    plt.tight_layout()
    ax.axis('off')
    visualize(aout[i,...])

    plt.show()


print('Starting Validation')

#VALIDATION
net_loss=0.0
#norm_loss=0.0
#alb_loss=0.0
#light_loss=0.0
for i,data in enumerate(test_loader):
    image=data['image']
    normal=data['normal']
    mask=data['mask']
    albedo=data['albedo']
    light=data['light']


    image, mask, normal, albedo, light = image.cuda(1), mask.cuda(1), normal.cuda(1), albedo.cuda(1), light.cuda(1) # GPU settings
    image, mask, normal, albedo, light = Variable(image,volatile=True), Variable(mask,requires_grad=False,volatile=True), Variable(normal,requires_grad=False,volatile=True), Variable(albedo,requires_grad=False,volatile=True), Variable(light,requires_grad=False,volatile=True)

    nout, aout, lout = net(image)

    loss=criterion(image,mask,normal,albedo,light, nout, aout,lout)

    net_loss+=loss.data[0]
    #norm_loss+=nloss.data[0]
    #alb_loss+=aloss.data[0]
    #light_loss+=lloss.data[0]


print('Net loss: %.3f' %((net_loss)/len(test_loader)))


