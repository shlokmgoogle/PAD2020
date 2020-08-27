from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tr, utils
import glob
import cv2
import lycon
from PIL import Image
from torchvision import transforms as tr
import numpy as np

class SfSDataset(Dataset):
    def __init__(self,csv_file,transform=None):
        self.frames = pd.read_csv(csv_file, header=None)
        self.transform = transform
        # data_list = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge/data/lists/folds_by_fakes/fold3/val_list.txt'
        data_list = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge/celeba/real/real.txt'
        # data_list = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge/data/lists/phase1_list.txt'

        self.df = pd.read_csv(data_list)
        self.transform_test = tr.transforms.Compose([
            tr.Resize((128, 128)),
            # transforms.RandomResizedCrop(224)
            # transforms.CustomRotate(0),
            # transforms.CustomRandomHorizontalFlip(p=0),
            # transforms.CustomCrop((112,112), crop_index=0),
            tr.ToTensor(),
            # tr.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self,idx):


        # print('hi')
        image = lycon.load(self.frames.iloc[idx, 0])
        mask = lycon.load(self.frames.iloc[idx, 1])
        normal = lycon.load(self.frames.iloc[idx, 2])
        albedo = lycon.load(self.frames.iloc[idx, 3])

        light = self.frames.iloc[idx, 4:].values #a 1x27 arra

        casia = True
        if casia:
            if idx >= self.df.shape[0]:
                idx = np.random.randint(0, self.df.shape[0]-1)
            #     while self.df.label.iloc[idx] == 1:
            #         idx = idx + 1
            #     # idx = 100
            # while self.df.label.iloc[idx] == 1:
            #     idx = idx + 1
        # print('label '+str(self.df.label.iloc[idx]))
        # print('image' + self.df.rgb.iloc[idx])

        if casia:
            # rgb_path = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge/CASIA-SURF/valid/' + self.df.rgb.iloc[idx]
            rgb_path = self.df.rgb.iloc[idx]
            # rgb_path = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge/casia_real_v8.1.png'
            save_path = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge/temp/'+str(idx)+'.png'
            save_path_normal = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge/temp/'+str(idx)+'normal.png'
            # print(rgb_path)
            image = lycon.load(rgb_path)
            image = Image.fromarray(np.uint8(image))
            image = self.transform_test(image)
            image = image.cpu().detach().numpy()
            image = image * 255
            # print(image.shape)
            # print(albedo.shape)
            image = np.transpose(image,[1,2,0])
        # print(image.shape)
            # image = image.reshape(3,128,128)

            # temp_img = Image.fromarray(np.uint8(image))
            # temp_img.save(save_path)
            # temp_img = Image.fromarray(np.uint8(albedo))
            # temp_img.save(save_path_normal)

            # lycon.save(normal, save_path_normal)


        # print(image.shape)
        # print(normal.shape)


        sample = {'image': image, 'mask': mask, 'normal': normal, 'albedo': albedo, 'light': light,'label':self.df.label.iloc[idx]}
        # print(image.shape)
        if self.transform:
                sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, mask, normal, albedo, light = sample['image'], sample['mask'], sample['normal'], sample['albedo'], sample['light']
        light=light.astype(float)

        return {'image': to_tensor(image),
                'mask': to_tensor(mask),
                'normal': to_tensor(normal),
                'albedo': to_tensor(albedo),
                'light': torch.from_numpy(light),
                'label':sample['label']}



def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    See ``ToTensor`` for more details.
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """
    #if not(_is_pil_image(pic) or _is_numpy_image(pic)):
    #    raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        # print(torch.min(img))
        # print(torch.max(img))
        # return img.float()
        return img.float().div(255)

    # if accimage is not None and isinstance(pic, accimage.Image):
    #     nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
    #     pic.copyto(nppic)
    #     return torch.from_numpy(nppic)

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

# real_train_dataset = SfSDataset(csv_file='/vulcan/scratch/koutilya/CelebA_train.csv', transform=tr.Compose([ToTensor()]))
# print(len(real_train_dataset))
# real_train_dataset[10]
# syn_train_dataset = SfSDataset(csv_file='/vulcan/scratch/koutilya/Syn_train.csv', transform=tr.Compose([ToTensor()]))
# print(len(syn_train_dataset))
# real_test_dataset = SfSDataset(csv_file='/vulcan/scratch/koutilya/CelebA_test.csv', transform=tr.Compose([ToTensor()]))
# print(len(real_test_dataset))

