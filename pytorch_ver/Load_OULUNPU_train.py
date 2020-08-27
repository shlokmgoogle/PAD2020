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
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 
import imgaug.augmenters as iaa

import cv2
import mahotas
# import numpy as np
import dlib

from facenet_pytorch import MTCNN, InceptionResnetV1
from OULU_extract_Face import create_mask as create_mask_default

from PIL import Image
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#face_scale = 1.3  #default for test, for training , can be set from [1.2 to 1.5]

# data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Add color 
    iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast with a gamma of 0.5 to 1.5
])

mtcnn = MTCNN(image_size=128 , margin=0)

path = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge/facenet_pytorch/temp_OULU_1.jpg'
                    # path_mask = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge/facenet_pytorch/temp_OULU_mask.jpg'
def valid(x, y):
    if x >= 224:
        x = 223
    if y >= 224:
        y = 223
    return (x, y)


def create_mask(frame):
    points_fidducial = []

    gray = cv2.cvtColor((frame), cv2.COLOR_BGR2GRAY)
    #
    print(gray.shape)

    mask = np.zeros(shape=frame.shape)
    faces = detector(gray)

    if len(faces) > 0:
        face_temp = faces
    print(len(faces))
    if len(faces) == 0:
        return None

    # The face landmarks code begins from here

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # Then we can also do cv2.rectangle function (frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        landmarks = predictor(gray, face)
        # print(landmarks.part)
        # We are then accesing the landmark points
        for n in range(0, 17):

            x = landmarks.part(n).x
            y = landmarks.part(n).y
            if x >= 224:
                x = 223
            if y >= 224:
                y = 223
            # cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
            points_fidducial.append((y, x))
            #
            # points_fidducial.append((y,x))
            # if  n == 18:
            #     points_fidducial.append((landmarks.part(0).y, landmarks.part(0).x))
            # if n == 28:
            #     points_fidducial.append((landmarks.part(17).y, landmarks.part(17).x))
            # if n == 25:
            #     points_fidducial.append((landmarks.part(16).y, landmarks.part(16).x))
            # if n == 21:
            #     points_fidducial.append((landmarks.part(16).y, landmarks.part(16).x))

    # c1 = [(landmarks.part(0).y,landmarks.part(0).x),(landmarks.part(2).y,landmarks.part(1).x)]
    # c2 = [(landmarks.part(16).y, landmarks.part(0).x), (landmarks.part(7).y, landmarks.part(1).x)]
    # eye = (norm(points_fidducial[18:23] - points_fidducial[23:26]))
    # c3 = points_fidducial[0:3]
    # c3 =

    points_fidducial.append(valid(landmarks.part(26).y, landmarks.part(26).x))
    points_fidducial.append(valid(landmarks.part(25).y, landmarks.part(25).x))
    points_fidducial.append(valid(landmarks.part(24).y, landmarks.part(24).x))

    points_fidducial.append(valid(landmarks.part(19).y, landmarks.part(19).x))
    points_fidducial.append(valid(landmarks.part(18).y, landmarks.part(18).x))
    points_fidducial.append(valid(landmarks.part(17).y, landmarks.part(17).x))
    # import pdb
    # pdb.set_trace()
    # p = Path(points_fidducial)
    mahotas.polygon.fill_polygon(points_fidducial, mask)
    # import pdb
    # pdb.set_trace()
    mask = np.round((mask * 255))
    # cv2.imwrite(path_mask, mask)
    return mask

image_x_temp = cv2.imread(path)
# img_cropped = cv2.resize(image_x_temp, (256, 256))
default_mask = create_mask_default(image_x_temp)
default_mask = cv2.resize(default_mask, (128, 128))
default_mask = np.asarray( default_mask, dtype="int32" )

def crop_face_from_scene(image,face_name_full, scale,image_id):
    # import pdb
    # pdb.set_trace()
    f=open(face_name_full,'r')
    lines=f.readlines()
    # lines[0].split(',')

    lines = lines[image_id].split(',')
    # print(lines)

    y1,x1,w,h=[float(ele) for ele in lines[1:5]]
    # print(y1,x1,w,h)

    if y1==0 and x1==0 and w==0 and h==0:
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





# array
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.01, sh = 0.05, r1 = 0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, sample):
        img, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        
        if random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]
           
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)
    
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
    
                if w < img.shape[1] and h < img.shape[0]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)

                    img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                    img[x1:x1+h, y1:y1+w, 1] = self.mean[1]
                    img[x1:x1+h, y1:y1+w, 2] = self.mean[2]
                    
        return {'image_x': img, 'map_x': map_x, 'spoofing_label': spoofing_label}


# Tensor
class Cutout(object):
    def __init__(self, length=50):
        self.length = length

    def __call__(self, sample):
        img, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        h, w = img.shape[1], img.shape[2]    # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)
        
        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return {'image_x': img, 'map_x': map_x, 'spoofing_label': spoofing_label}


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        new_map_x = map_x/255.0                 # [0,1]
        return {'image_x': new_image_x, 'map_x': new_map_x, 'spoofing_label': spoofing_label}



class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        
        new_image_x = np.zeros((256, 256, 3))
        new_map_x = np.zeros((32, 32))

        p = random.random()
        if p < 0.5:
            #print('Flip')

            new_image_x = cv2.flip(image_x, 1)
            new_map_x = cv2.flip(map_x, 1)

                
            return {'image_x': new_image_x, 'map_x': new_map_x, 'spoofing_label': spoofing_label}
        else:
            #print('no Flip')
            return {'image_x': image_x, 'map_x': map_x, 'spoofing_label': spoofing_label}



class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        
        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        # image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = image_x.transpose((2, 0, 1))
        image_x = np.array(image_x)
        
        map_x = np.array(map_x)
        
                        
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'map_x': torch.from_numpy(map_x.astype(np.float)).float(), 'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.long)).long()}


# /home/ztyu/FAS_dataset/OULU/Train_images/          6_3_20_5_121_scene.jpg        6_3_20_5_121_scene.dat
# /home/ztyu/FAS_dataset/OULU/IJCB_re/OULUtrain_images/        6_3_20_5_121_depth1D.jpg
class Spoofing_train(Dataset):

    def __init__(self, info_list, root_dir, map_dir,  transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=',', header=None)
        self.root_dir = root_dir
        self.map_dir = map_dir
        self.transform = transform
        self.cnt = 0

    def __len__(self):
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):
        #print(self.landmarks_frame.iloc[idx, 0])
        videoname = str(self.landmarks_frame.iloc[idx, 1])
        image_path = os.path.join(self.root_dir)
        map_path = os.path.join(self.map_dir, videoname)
        generate_labels = False
             
        image_x, map_x = self.get_single_image_without_map_for_generating_labels(image_path, videoname)
        # image_x = self.get_single_image_without_map(image_path,videoname)
        if generate_labels:
            # print('true') image_x_aug, map_x, image_x,
            image_x,image_x_mask = self.get_single_image_without_map_for_generating_labels(image_path,videoname)
            # return image_x, image_x_mask
            # return  image_x

		    
        spoofing_label = self.landmarks_frame.iloc[idx, 0]
        # print(image_x.shape)
        if spoofing_label == 1:
            spoofing_label = 1            # real
        else:
            spoofing_label = 0
            map_x = np.zeros((32, 32))    # fake


        # sample = {'image_x': image_x, 'map_x': map_x, 'spoofing_label': spoofing_label}
        sample = {'image_x': image_x,'map_x': image_x, 'spoofing_label': spoofing_label}
        # sample = {image_x,spoofing_label}
        if self.transform:
            sample = self.transform(sample)

        return sample['image_x'],spoofing_label

    def valid(self,x, y):
        if x >= 224:
            x = 223
        if y >= 224:
            y = 223
        return (x, y)

    def create_mask(self,frame):
        points_fidducial = []

        gray = cv2.cvtColor(cv2.UMat(np.float32(frame)), cv2.COLOR_RGB2GRAY)
        # gray = cv2.cvtColor((frame), cv2.COLOR_BGR2GRAY)
        #
        print(gray.shape)

        mask = np.zeros(shape=np.array(frame).shape)
        faces = detector(gray)

        if len(faces) > 0:
            face_temp = faces
        print(len(faces))
        if len(faces)==0:
            return None


        # The face landmarks code begins from here

        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            # Then we can also do cv2.rectangle function (frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            landmarks = predictor(gray, face)
            # print(landmarks.part)
            # We are then accesing the landmark points
            for n in range(0, 17):

                x = landmarks.part(n).x
                y = landmarks.part(n).y
                if x >= 128:
                    x = 127
                if y >= 128:
                    y = 127
                # cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
                points_fidducial.append((y, x))
                #
                # points_fidducial.append((y,x))
                # if  n == 18:
                #     points_fidducial.append((landmarks.part(0).y, landmarks.part(0).x))
                # if n == 28:
                #     points_fidducial.append((landmarks.part(17).y, landmarks.part(17).x))
                # if n == 25:
                #     points_fidducial.append((landmarks.part(16).y, landmarks.part(16).x))
                # if n == 21:
                #     points_fidducial.append((landmarks.part(16).y, landmarks.part(16).x))

        # c1 = [(landmarks.part(0).y,landmarks.part(0).x),(landmarks.part(2).y,landmarks.part(1).x)]
        # c2 = [(landmarks.part(16).y, landmarks.part(0).x), (landmarks.part(7).y, landmarks.part(1).x)]
        # eye = (norm(points_fidducial[18:23] - points_fidducial[23:26]))
        # c3 = points_fidducial[0:3]
        # c3 =

        points_fidducial.append(self.valid(landmarks.part(26).y, landmarks.part(26).x))
        points_fidducial.append(self.valid(landmarks.part(25).y, landmarks.part(25).x))
        points_fidducial.append(self.valid(landmarks.part(24).y, landmarks.part(24).x))

        points_fidducial.append(self.valid(landmarks.part(19).y, landmarks.part(19).x))
        points_fidducial.append(self.valid(landmarks.part(18).y, landmarks.part(18).x))
        points_fidducial.append(self.valid(landmarks.part(17).y, landmarks.part(17).x))
        # import pdb
        # pdb.set_trace()
        # p = Path(points_fidducial)
        mahotas.polygon.fill_polygon(points_fidducial, mask)
        # import pdb
        # pdb.set_trace()
        mask = np.round((mask * 255))
        # cv2.imwrite(path_mask, mask)
        return mask

    def get_single_image_without_map(self,image_path,videoname):
            # frames_total = len([name for name in os.listdir(map_path) if os.path.isfile(os.path.join(map_path, name))])
            # for temp in range(500):
            #     image_id = np.random.randint(1, frames_total - 1)
            #
            #     s = "_%03d_scene" % image_id
            #     image_name = videoname + s + '.jpg'
            #     bbox_name = videoname + s + '.dat'
            #     bbox_path = os.path.join(image_path, bbox_name)
            #     s = "_%03d_depth1D" % image_id
            #     map_name = videoname + s + '.jpg'
            #     map_path2 = os.path.join(map_path, map_name)
            #
            #     # some .dat & map files have been missing
            #     if os.path.exists(bbox_path) & os.path.exists(map_path2):
            #         map_x_temp2 = cv2.imread(map_path2, 0)
            #         if map_x_temp2 is not None:
            #             break

            # random scale from [1.2 to 1.5]

            image_id = 30
            s = "_%06d" % image_id
            image_name = videoname + s + '.jpg'
            bbox_name = videoname + '.txt'
            face_scale = np.random.randint(12, 15)
            face_scale = face_scale / 10.0

            image_x = np.zeros((256, 256, 3))
            map_x = np.zeros((32, 32))

            # RGB
            # import pdb
            # pdb.set_trace()
            image_path_temp = os.path.join(image_path, image_name)
            print(image_path_temp)
            image_x_temp = cv2.imread(image_path_temp)
            img_pil = Image.open(image_path)
            bbox_name = os.path.join(image_path,bbox_name)

            # gray-map
            # map_path = os.path.join(map_path, map_name)
            # map_x_temp = cv2.imread(map_path, 0)
            cropped_image = crop_face_from_scene(image_x_temp, bbox_name, face_scale, image_id)
            width, height = cropped_image.shape[:2]
            # print(width, height)
            if width>0:

                image_x = cv2.resize(cropped_image, (128, 128))
                # cv2.imwrite('temp_OULU.jpg', image_x)
                # image_x_mask = create_mask(image_x)
                # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
                # image_x_aug = seq.augment_image(image_x)

                # map_x = cv2.resize(crop_face_from_scene(map_x_temp, bbox_path, face_scale), (32, 32))
                return image_x_aug, map_x, image_x, img_pil

    def to_tensor(self,pic):
        """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
        See ``ToTensor`` for more details.
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        # if not(_is_pil_image(pic) or _is_numpy_image(pic)):
        #    raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            # handle numpy array
            # import pdb
            # pdb.set_trace()
            # pic = pic.resize((128,128))transpose((2, 0, 1))
            img = torch.from_numpy(pic)
            # backward compatibility
            return img.float()

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


    def get_single_image_x(self, image_path, map_path, videoname):

            frames_total = len([name for name in os.listdir(map_path) if os.path.isfile(os.path.join(map_path, name))])

            # random choose 1 frame
            for temp in range(500):
                image_id = np.random.randint(1, frames_total-1)

                s = "_%03d_scene" % image_id
                image_name = videoname + s + '.jpg'
                bbox_name = videoname + s + '.dat'
                bbox_path = os.path.join(image_path, bbox_name)
                s = "_%03d_depth1D" % image_id
                map_name = videoname + s + '.jpg'
                map_path2 = os.path.join(map_path, map_name)

                # some .dat & map files have been missing
                if os.path.exists(bbox_path) & os.path.exists(map_path2):
                    map_x_temp2 = cv2.imread(map_path2, 0)
                    if map_x_temp2 is not None:
                        break


            # random scale from [1.2 to 1.5]
            face_scale = np.random.randint(12, 15)
            face_scale = face_scale/10.0


            image_x = np.zeros((256, 256, 3))
            map_x = np.zeros((32, 32))


            # RGB
            image_path = os.path.join(image_path, image_name)
            print(image_path)
            image_x_temp = cv2.imread(image_path)
            img_pil = Image.open(image_path)

            # gray-map
            map_path = os.path.join(map_path, map_name)
            map_x_temp = cv2.imread(map_path, 0)

            image_x = cv2.resize(crop_face_from_scene(image_x_temp, bbox_path, face_scale), (256, 256))


            # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
            image_x_aug = seq.augment_image(image_x)

            map_x = cv2.resize(crop_face_from_scene(map_x_temp, bbox_path, face_scale), (32, 32))


            return image_x_aug, map_x,image_x,img_pil



    def get_single_image_without_map_for_generating_labels(self,image_path,videoname):
            # frames_total = len([name for name in os.listdir(map_path) if os.path.isfile(os.path.join(map_path, name))])
            # for temp in range(500):
            #     image_id = np.random.randint(1, frames_total - 1)
            #
            #     s = "_%03d_scene" % image_id
            #     image_name = videoname + s + '.jpg'
            #     bbox_name = videoname + s + '.dat'
            #     bbox_path = os.path.join(image_path, bbox_name)
            #     s = "_%03d_depth1D" % image_id
            #     map_name = videoname + s + '.jpg'
            #     map_path2 = os.path.join(map_path, map_name)
            #
            #     # some .dat & map files have been missing
            #     if os.path.exists(bbox_path) & os.path.exists(map_path2):
            #         map_x_temp2 = cv2.imread(map_path2, 0)
            #         if map_x_temp2 is not None:
            #             break

            # random scale from [1.2 to 1.5]

            image_id = 30
            s = "_%06d" % image_id
            image_name = videoname + s + '.jpg'
            bbox_name = videoname + '.txt'
            face_scale = np.random.randint(12, 15)
            face_scale = face_scale / 10.0

            image_x = np.zeros((256, 256, 3))
            map_x = np.zeros((32, 32))

            # RGB
            # import pdb
            # pdb.set_trace()
            image_path_temp = os.path.join(image_path, image_name)
            image_x_temp = cv2.imread(image_path_temp)
            image_x_temp = cv2.cvtColor(image_x_temp, cv2.COLOR_BGR2RGB)

            img_pil = Image.open(image_path_temp)
            bbox_name = os.path.join(image_path,bbox_name)
            generate_label = False
            if generate_label:

                # gray-map
                # map_path = os.path.join(map_path, map_name)
                # map_x_temp = cv2.imread(map_path, 0)
                cropped_image = crop_face_from_scene(image_x_temp, bbox_name, face_scale, image_id)
                # cropped_image = image_x_temp
                width, height = cropped_image.shape[:2]
                # print(width, height)
                if width>0:

                    image_x_original = cv2.resize(cropped_image, (128, 128))
                    #

                    image_x = mtcnn(img_pil,
                                  save_path='/vulcanscratch/shlokm/Chalearn_challenge/OULU/Train_files_images/'+image_name)


                    if image_x is None:
                        print('None')
                        image_x = image_x_original
                    path = '/vulcanscratch/shlokm/Chalearn_challenge/OULU/Train_files_images/'+image_name
                    img_cropped = cv2.imread(path)
                    print(path)
                    print(img_cropped.shape)
                    image_x_mask = self.create_mask(img_cropped)
                    image_x = image_x_original
                    # else:
                    #
                    #     image_x = cv2.resize(np.float32(image_x), (256, 256))
                    # print(image_x.shape)
                    # cv2.imwrite('temp_OULU_2.jpg', image_x)
                    # import pdb
                    # pdb.set_trace()
                    # image_x_for_mask = float(image_x.reshape(3, 256, 256))
                    # image_x_for_mask = image_x.permute(1, 2, 0).numpy()
                    # image_x_for_mask = cv2.cvtColor(image_x_for_mask, cv2.COLOR_RGB2BGR)
                    # import pdb
                    # pdb.set_trace()
                    # image_x_for_mask = Image.fromarray(image_x_for_mask.astype(np.uint8))
                    # image_x_mask = self.create_mask(image_x_for_mask)
                    # print(image_x_mask.shape)
                    # print(image_x.shape)
                    # print(image_x_mask.shape)
                    self.cnt = self.cnt + 1
                    # if image_x_mask is None:


                    image_x_mask = default_mask


                    # faces = face_temp
                # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
                # image_x_aug = seq.augment_image(image_x)

                # map_x = cv2.resize(crop_face_from_scene(map_x_temp, bbox_path, face_scale), (32, 32))
            # return image_x_aug, map_x, image_x, img_pil
            # image_x = cv2.resize((image_x), (128, 128))
            # image_x_mask = cv2.resize((image_x_mask), (128, 128))

                image_x = self.to_tensor(image_x.cpu().detach().numpy())
                image_x_mask = self.to_tensor(image_x_mask.transpose((2, 0, 1)))
                print(image_x.shape)
                print(image_x_mask.shape)
            else:
                image_x_aug = seq.augment_image(image_x_temp)
                image_x = cv2.resize(image_x_aug, (128, 128))
            return image_x,image_x
    #out_name="${out_video_dir}/${video_name}_%06d.jpg"
      #ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"


