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
import numpy as np
import cv2

#from networks import *
from torch.autograd import Variable

from test_dataset import TestDataset, ToTensor
from sfsnet import SfSNet
from sfs_loss import SfSLoss



def Recon_loss(im,n,a,light,mask,id,path):
	n=normalize(n)

	att = np.pi*np.array([1, 2.0/3, 0.25])

	c1=att[0]*(1.0/np.sqrt(4*np.pi))
	c2=att[1]*(np.sqrt(3.0/(4*np.pi)))
	c3=att[2]*0.5*(np.sqrt(5.0/(4*np.pi)))
	c4=att[2]*(3.0*(np.sqrt(5.0/(12*np.pi))))
	c5=att[2]*(3.0*(np.sqrt(5.0/(48*np.pi))))

	c=torch.from_numpy(np.array([c1,c2,c3,c4,c5]))

	L1_err=0
	L2_err=0
	for i in range(0,n.size(0)):
		nx=n[i,0,...]
		ny=n[i,1,...]
		nz=n[i,2,...]

		
		H1=c[0]*Variable((torch.ones((n.size(2),n.size(3)))).cuda(1))
		H2=c[1]*nz
		H3=c[1]*nx
		H4=c[1]*ny
		H5=c[2]*(2*nz*nz - nx*nx -ny*ny)
		H6=c[3]*nx*nz
		H7=c[3]*ny*nz
		H8=c[4]*(nx*nx - ny*ny)
		H9=c[3]*nx*ny

		shd=Variable((torch.zeros(a.size())).cuda(1))



		for j in range(0,3):
			Lo=light[i,j*9:(j+1)*9]

			shd[i,j,...]=Lo[0]*H1+Lo[1]*H2+Lo[2]*H3+Lo[3]*H4+Lo[4]*H5+Lo[5]*H6+Lo[6]*H7+Lo[7]*H8+Lo[8]*H9

		

		err=255*(torch.sum((shd[i,...]*a[i,...]*mask[i,...] - im[i,...]*mask[i,...])**2,dim=0)**0.5)
		nele=torch.sum(mask[i,0,...])

		L1_err+=torch.sum(err)/nele
		L2_err+=(torch.norm(err)**2/nele)**0.5


		save_images(im[i,...],mask[i,...],light[i,...],n[i,...],a[i,...],shd[i,...],str(id)+str(i),path)
	
	return L1_err/(i+1), L2_err/(i+1)

def save_images(im,mask,l,n,a,shd,name_id,path):
	rec=(shd*a)*mask+(1-mask)*im
	im, mask,n, a, shd, rec = normalize_to_im(im), normalize_to_im(mask), normalize_to_im(n), normalize_to_im(a), normalize_to_im(shd), normalize_to_im(rec)
	
	n=(1+n)/2 #Convert back to normal [0,1] format

	im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
	n=cv2.cvtColor(n,cv2.COLOR_RGB2BGR)
	a=cv2.cvtColor(a,cv2.COLOR_RGB2BGR)
	mask=cv2.cvtColor(mask,cv2.COLOR_RGB2BGR)
	shd=cv2.cvtColor(shd,cv2.COLOR_RGB2BGR)
	rec=cv2.cvtColor(rec,cv2.COLOR_RGB2BGR)
	


	#plt.imshow(shd)
	#plt.show()

	cv2.imwrite(path+name_id+'_image.png',(255*im).astype(np.uint8))
	cv2.imwrite(path+name_id+'_mask.png',(255*mask).astype(np.uint8))
	cv2.imwrite(path+name_id+'_normal.png',(255*n).astype(np.uint8))
	cv2.imwrite(path+name_id+'_albedo.png',(255*a).astype(np.uint8))
	cv2.imwrite(path+name_id+'_shading.png',(255*shd).astype(np.uint8))
	cv2.imwrite(path+name_id+'_recon.png',(255*rec).astype(np.uint8))
	

	l=((l.data).cpu()).numpy()
	np.savetxt(path+name_id+'_light.txt',l,fmt='%f')


	nt=(255*n*mask).astype(np.uint8)
	at=(255*a*mask).astype(np.uint8)
	shdt=(255*shd*mask).astype(np.uint8)
	rect=(255*rec*mask).astype(np.uint8)

	cv2.imwrite(path+name_id+'_normal_mask.png',nt+(1-mask)*255*np.ones(n.shape))
	cv2.imwrite(path+name_id+'_albedo_mask.png',at+(1-mask)*255*np.ones(n.shape))
	cv2.imwrite(path+name_id+'_shading_mask.png',shdt+(1-mask)*255*np.ones(n.shape))
	cv2.imwrite(path+name_id+'_recon_mask.png',rect+(1-mask)*255*np.ones(n.shape))
	

def normalize_to_im(rec0): #Takes a GPU variable and converts it back to image format
	rec0=((rec0.data).cpu()).numpy()
	rec0=rec0.transpose((1,2,0))
	rec0[rec0>1]=1
	rec0[rec0<0]=0
	return rec0


def normalize(n):
	n=2*n-1
	norm=torch.norm(n,2,1,keepdim=True)
	norm=norm.repeat(1,3,1,1)
	return (n/norm)



use_cuda = torch.cuda.is_available()


# DATA LOADING

print('\n[Phase 1] : Data Preparation')

transform_test = transforms.Compose([ToTensor(),
])

#Replace Dataset name here
dataset='Data.csv'
save_dir='/scratch2/photometric_stereo/pytorch_ver/Evaluations/Adobe/'
testdata =  TestDataset(csv_file=save_dir+dataset,transform=transform_test)  #Write a dataloader function that can read the database provided by .csv file
print(len(testdata))
test_loader = torch.utils.data.DataLoader(testdata, batch_size=20, shuffle=False, num_workers=1)

#Load the model
net=SfSNet()
net.load_state_dict(torch.load('net_epoch_r5_5.pth'))
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

#Pass data through network
L1_err=0
L2_err=0
for i,data in enumerate(test_loader):
	image=data['image']
	image=Variable(image.cuda(1),volatile=True)
	mask=data['mask']
	mask=Variable(mask.cuda(1),volatile=True)
	nout, aout, lout = net(image)

	print(image.size(0))
	l1err, l2err = Recon_loss(image, nout, aout, lout, mask, i, save_dir+'Our_r5/')

	L1_err+=l1err
	L2_err+=l2err



print('Total Loss for the experiemnt')
print('L1 Loss: %.3f L2 Loss: %.3f' %(L1_err/(i+1),L2_err/(i+1)))
print(i)
