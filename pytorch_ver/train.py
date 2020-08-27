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
import subprocess
import gc

#from networks import *
from torch.autograd import Variable

from data_loader import SfSDataset, ToTensor
from sfsnet import SfSNet, conv_init
import sfsnet
from sfs_loss import SfSLoss


use_cuda = torch.cuda.is_available()

# DATA LOADING

print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([ToTensor()
]) # meanstd transformation

transform_test = transforms.Compose([ToTensor()])



traindata =  SfSDataset(csv_file='/scratch2/photometric_stereo/Data_Generate_real/Train_Data.csv',transform=transform_train)  #Write a dataloader function that can read the database provided by .csv file
testdata =  SfSDataset(csv_file='/scratch2/photometric_stereo/Data_Generate_real/Test_Data.csv',transform=transform_test)  #Write a dataloader function that can read the database provided by .csv file


train_loader = torch.utils.data.DataLoader(traindata, batch_size=10, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(testdata, batch_size=50, shuffle=True, num_workers=4)


#Load the model
net=SfSNet()
net.apply(conv_init)
print('Network Initialized')

#Use GPU
net.cuda()
#net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True

#Loss function
criterion = SfSLoss() #Write a loss function for SfS
criterion.cuda()
print('Loss Initialized')

optimizer = optim.Adam(net.parameters(), lr=0.01)

print('Starting Training')
step=100

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])#, encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


for epoch in range(12):
    net.train()
    running_loss=0.0  #AVG loss over a certain minibatch-group
    nor_loss=0.0
    alb_loss=0.0
    li_loss=0.0
    recon_loss=0.0
    for i,data in enumerate(train_loader):
        image=data['image']
        normal=data['normal']
        mask=data['mask']
        albedo=data['albedo']
        light=data['light']


        image, mask, normal, albedo, light = image.cuda(), mask.cuda(), normal.cuda(), albedo.cuda(), light.cuda() # GPU settings
        image, mask, normal, albedo, light = Variable(image), Variable(mask,requires_grad=False), Variable(normal,requires_grad=False), Variable(albedo,requires_grad=False), Variable(light,requires_grad=False)

        optimizer.zero_grad()

        nout, aout, lout = net(image)

        loss, nloss, aloss, lloss, recloss = criterion(image,mask,normal,albedo,light, nout, aout,lout)

        loss.backward()

        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        nor_loss += nloss.data[0]
        alb_loss += aloss.data[0]
        li_loss += lloss.data[0]
        recon_loss += recloss.data[0]

        if i % step == (step-1):   #PRINT AVG LOSS over certain minibatch size 
            print('[%d, %5d] loss: %.3f, norm-loss: %.3f, alb-loss: %.3f, light-loss: %.3f Recon-loss: %.3f' %(epoch + 1, i + 1, running_loss / step, nor_loss / step, alb_loss / step, li_loss / step, recon_loss / step))
            
            mem=get_gpu_memory_map()
            #print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / step))
            #print(mem)

            running_loss = 0.0
            nor_loss=0.0
            li_loss=0.0
            alb_loss=0.0
            recon_loss=0.0

             # do checkpointing
            torch.save(net.state_dict(), 'net_epoch_r5_%d.pth' %(epoch))

        del image, mask, normal, albedo, light
        gc.collect()



    print('DONE 1 EPOCH')

    print('VALIDATING')
    #VALIDATION
    net.eval()
    net_loss=0.0 
    norm_loss=0.0 
    alb_loss=0.0  
    light_loss=0.0
    recon_loss=0.0
    for i,data in enumerate(test_loader):
        image=data['image']
        normal=data['normal']
        mask=data['mask']
        albedo=data['albedo']
        light=data['light']


        image, mask, normal, albedo, light = image.cuda(), mask.cuda(), normal.cuda(), albedo.cuda(), light.cuda() # GPU settings
        image, mask, normal, albedo, light = Variable(image,volatile=True), Variable(mask,requires_grad=False,volatile=True), Variable(normal,requires_grad=False,volatile=True), Variable(albedo,requires_grad=False,volatile=True), Variable(light,requires_grad=False,volatile=True)
        optimizer.zero_grad()

        nout, aout, lout = net(image)

        loss, nloss, aloss, lloss, recloss=criterion(image,mask,normal,albedo,light, nout, aout,lout)

        net_loss+=loss.data[0]
        norm_loss+=nloss.data[0]
        alb_loss+=aloss.data[0]
        light_loss+=lloss.data[0]
        recon_loss += recloss.data[0]


    print('EPOCH: %d loss: %.3f, norm-loss: %.3f, alb-loss: %.3f, light-loss: %.3f recon-loss: %.3f' %(epoch+1,net_loss/(i+1),norm_loss/(i+1),alb_loss/(i+1),light_loss/(i+1),recon_loss/(i+1)))




print('Finished Training')


