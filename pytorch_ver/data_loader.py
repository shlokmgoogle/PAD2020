from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class SfSDataset(Dataset):
    def __init__(self,csv_file,transform=None):
        self.frames = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self,idx):

        image = io.imread(self.frames.iloc[idx, 0])


        testing_single_image = True
        if testing_single_image:
            print("Trueeee")
            path ='/vulcanscratch/shlokm/Chalearn_challenge/SupContrast/3d_masks_kuntal/spoof3d2/3d_spoof2/5.png'
            print(path)
            image = io.imread(path)
        mask = io.imread(self.frames.iloc[idx, 1])
        normal = io.imread(self.frames.iloc[idx, 2])
        albedo = io.imread(self.frames.iloc[idx, 3])

        light = self.frames.iloc[idx, 4:].as_matrix() #a 1x27 array


        sample = {'image': image, 'mask': mask, 'normal': normal, 'albedo': albedo, 'light': light}

        if self.transform:
            sample = self.transform(sample)
        return sample


# class ToTensor(object):
# 	#def __init__(self):
# 	#	pass

# 	def __call__(self, sample):
# 		image, mask, normal, albedo, light = sample['image'], sample['mask'], sample['normal'], sample['albedo'], sample['light']
# 		image = image.transpose((2, 0, 1))
# 		mask = mask.transpose((2, 0, 1))
# 		normal = normal.transpose((2, 0, 1))
# 		albedo = albedo.transpose((2, 0, 1))

# 		#light.type()

# 		light=light.astype(float)

# 		return {'image': torch.from_numpy(image),
#                 'mask': torch.from_numpy(mask),
#                 'normal': torch.from_numpy(normal),
#                 'albedo': torch.from_numpy(albedo),
#                 'light': torch.from_numpy(light)}


class ToTensor(object):
    def __call__(self, sample):
        image, mask, normal, albedo, light = sample['image'], sample['mask'], sample['normal'], sample['albedo'], sample['light']
        light=light.astype(float)

        return {'image': to_tensor(image),
                'mask': to_tensor(mask),
                'normal': to_tensor(normal),
                'albedo': to_tensor(albedo),
                'light': torch.from_numpy(light)}



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
        return img.float().div(255)

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

