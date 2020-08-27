'''
Code of 'Searching Central Difference Convolutional Networks for Face Anti-Spoofing' 
By Zitong Yu & Zhuo Su, 2019

If you use the code, please cite:
@inproceedings{yu2020searching,
    title={Searching Central Difference Convolutional Networks for Face Anti-Spoofing},
    author={Yu, Zitong and Zhao, Chenxu and Wang, Zezheng and Qin, Yunxiao and Su, Zhuo and Li, Xiaobai and Zhou, Feng and Zhao, Guoying},
    booktitle= {CVPR},
    year = {2020}
}

Only for research purpose, and commercial use is not allowed.

MIT License
Copyright (c) 2020 
'''

from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
mtcnn = MTCNN(image_size=128 , margin=0)

frames_total = 8    # each video 8 uniform samples
 
face_scale = 1.3  #default for test and val 
#face_scale = 1.1  #default for test and val

def crop_face_from_scene(image,face_name_full, scale,image_id):
    # f=open(face_name_full,'r')
    f = open(face_name_full, 'r')
    lines = f.readlines()
    # lines[0].split(',')
    lines = lines[image_id].split(',')
    y1, x1, w, h = [float(ele) for ele in lines[1:5]]
    # print(y1,x1,w,h)

    if y1 == 0 and x1 == 0 and w == 0 and h == 0:
        y1 = 381
        x1 = 849
        w = 614
        h = 866


    f.close()
    y2=y1+w
    x2=x1+h

    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    h_img, w_img = image.shape[0], image.shape[1]
    #w_img,h_img=image.size
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=max(math.floor(y1),0)
    x1=max(math.floor(x1),0)
    y2=min(math.floor(y2),w_img)
    x2=min(math.floor(x2),h_img)

    #region=image[y1:y2,x1:x2]
    region=image[x1:x2,y1:y2]
    return region




class Normaliztion_valtest(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, val_map_x, spoofing_label = sample['image_x'],sample['val_map_x'] ,sample['spoofing_label']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        return {'image_x': new_image_x, 'val_map_x': val_map_x , 'spoofing_label': spoofing_label}


class ToTensor_valtest(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, val_map_x, spoofing_label = sample['image_x'],sample['val_map_x'] ,sample['spoofing_label']
        
        # swap color axis because    BGR2RGB
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W
        image_x = image_x[:,:,:,::-1].transpose((0, 3, 1, 2))
        image_x = np.array(image_x)
        
        val_map_x = np.array(val_map_x)
                        
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'val_map_x': torch.from_numpy(val_map_x.astype(np.float)).float(),'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.long)).long()} 


# /home/ztyu/FAS_dataset/OULU/Train_images/          6_3_20_5_121_scene.jpg        6_3_20_5_121_scene.dat
# /home/ztyu/FAS_dataset/OULU/IJCB_re/OULUtrain_images/        6_3_20_5_121_depth1D.jpg
class Spoofing_valtest(Dataset):

    def __init__(self, info_list, root_dir, val_map_dir,  transform=None):
        print('in val test')

        self.landmarks_frame = pd.read_csv(info_list, delimiter=',', header=None)
        self.root_dir = root_dir
        self.val_map_dir = val_map_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):
        #print(self.landmarks_frame.iloc[idx, 0])
        # videoname = str(self.landmarks_frame.iloc[idx, 1])
        # image_path = os.path.join(self.root_dir, videoname)
        videoname = str(self.landmarks_frame.iloc[idx, 1])
        image_path = os.path.join(self.root_dir)
        # map_path = os.path.join(self.map_dir, videoname)
        self.cnt = 0

        # val_map_path = os.path.join(self.val_map_dir, videoname)

        image_x = self.get_single_image_without_map(image_path, videoname)

        # image_x,_ = self.get_single_image_x(image_path, videoname)
        spoofing_label = self.landmarks_frame.iloc[idx, 0]
        if spoofing_label == 1:
            spoofing_label = 1  # real
        else:
            spoofing_label = 0
            map_x = np.zeros((32, 32))  # fake

        # sample = {'image_x': image_x, 'map_x': map_x, 'spoofing_label': spoofing_label}
        sample = {'image_x': image_x, 'map_x': image_x, 'spoofing_label': spoofing_label}
        # sample = {image_x,spoofing_label}
        if self.transform:
            sample = self.transform(sample)

        return sample['image_x'],spoofing_label

    def get_single_image_without_map(self, image_path, videoname):
        image_id = 30
        image_id = 30
        s = "_%06d" % image_id
        image_name = videoname + s + '.jpg'
        bbox_name = videoname + '.txt'

        image_path_temp = os.path.join(image_path, image_name)
        testing_single_image = False
        if testing_single_image:
            print("Trueeee")
            image_path_temp = '/vulcanscratch/shlokm/Chalearn_challenge/SupContrast/3d_masks_kuntal/spoof3d2/3d_spoof2/13.png'
            print(image_path_temp)
            # image_path_temp = '/vulcanscratch/shlokm/Chalearn_challenge/SupContrast/3d_masks_kuntal/spoof3d/5.png'
        image_x_temp = cv2.imread(image_path_temp)

        bbox_name = os.path.join(image_path,bbox_name)
        # gray-map
        # map_path = os.path.join(map_path, map_name)
        # map_x_temp = cv2.imread(map_path, 0)
        generate_label = False

        if generate_label:

            cropped_image = cv2.resize(crop_face_from_scene(image_x_temp, bbox_name, face_scale,image_id), (256, 256))
            # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
            # image_x_aug = seq.augment_image(image_x)

            # map_x = cv2.resize(crop_face_from_scene(map_x_temp, bbox_path, face_scale), (32, 32))
            width, height = cropped_image.shape[:2]
            # print(width, height)
            img_pil = Image.open(image_path_temp)
            use_mtcnn = True
            # print(image_path_temp,width)
            if width > 0:

                image_x_original = cv2.resize(cropped_image, (256, 256))
                image_x = image_x_original
                if use_mtcnn:
                    image_x_original = cv2.resize(cropped_image, (128, 128))
                    # print('/vulcanscratch/shlokm/Chalearn_challenge/OULU/Test_files_Images/'+image_name)

                    image_x = mtcnn(img_pil,
                                    save_path='/vulcanscratch/shlokm/Chalearn_challenge/OULU/Test_files_Images/'+image_name)
                    #
                    if image_x is None:
                        print('None')
                        image_x = image_x_original
                    path = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge/OULU/temp5/' + str(self.cnt) + '.jpg'

                    image_x = cv2.imread(path)
                    image_x = cv2.resize(image_x, (256, 256))
                    self.cnt = self.cnt + 1

                # cv2.imwrite('temp_OULU.jpg', image_x)
                # image_x_mask = create_mask(image_x)
                # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
                # print(image_x.shape)
                # import pdb
                # pdb.set_trace()
                # image_x_aug = seq.augment_image(image_x)
                # import pdb
                # pdb.set_trace()

                # map_x = cv2.resize(crop_face_from_scene(map_x_temp, bbox_path, face_scale), (32, 32))
            # return image_x_aug, map_x, image_x, image_x_aug
        else:
            image_x = cv2.resize(image_x_temp, (128, 128))
            # cropped_image = cv2.resize(crop_face_from_scene(image_x_temp, bbox_name, face_scale, image_id), (256, 256))

        return image_x

    def get_single_image_x(self, image_path, val_map_path, videoname):

        files_total = len([name for name in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, name))])//3
        interval = files_total//10
        
        image_x = np.zeros((frames_total, 256, 256, 3))
        val_map_x = np.ones((frames_total, 32, 32))
        
        # random choose 1 frame
        for ii in range(frames_total):
            image_id = ii*interval + 1 
            
            for temp in range(50):
                s = "_%03d_scene" % image_id
                s1 = "_%03d_depth1D" % image_id
                image_name = videoname + s + '.jpg'
                map_name = videoname + s1 + '.jpg'
                bbox_name = videoname + s + '.dat'
                bbox_path = os.path.join(image_path, bbox_name)
                val_map_path2 = os.path.join(val_map_path, map_name)
                val_map_x_temp2 = cv2.imread(val_map_path2, 0)
            
                if os.path.exists(bbox_path) & os.path.exists(val_map_path2)  :    # some scene.dat are missing
                    if val_map_x_temp2 is not None:
                        break
                    else:
                        image_id +=1
                else:
                    image_id +=1
                    
            # RGB
            image_path2 = os.path.join(image_path, image_name)
            image_x_temp = cv2.imread(image_path2)
            
            
            
            # gray-map
            val_map_x_temp = cv2.imread(val_map_path2, 0)

            image_x[ii,:,:,:] = cv2.resize(crop_face_from_scene(image_x_temp, bbox_path, face_scale), (256, 256))
            # transform to binary mask --> threshold = 0 
            temp = cv2.resize(crop_face_from_scene(val_map_x_temp, bbox_path, face_scale), (32, 32))
            np.where(temp < 1, temp, 1)
            val_map_x[ii,:,:] = temp
            
			
        return image_x, val_map_x



            
 

