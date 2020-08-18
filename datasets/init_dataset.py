import torch.utils.data as data
import torch
from PIL import Image
import pandas as pd
import numpy as np
import os

import torch.utils.data as data

def rgb_loader(path):
    return Image.open(path)


class ImageListDataset(data.Dataset):
    """
    Builds a dataset based on a list of images.
    data_root - image path prefix
    data_list - annotation list location
    """
    def __init__(self, data_root, data_list, transform=None):
        self.data_root = data_root
        #self.df = pd.read_csv(data_list)
        self.df = pd.read_csv(data_list)
        if 'label' not in self.df.columns:
            self.df['label'] = -1
        self.transform = transform
        self.loader = rgb_loader
        self.real = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (rgb_img, ir_img, depth_img, target) 
        """
        dict_elem = self.__get_simple_item__(index)

        dict_elem['meta'] = {
            'idx': index,
            'max_idx': len(self.df),
            'get_item_func': self.__get_simple_item__
        }
        dict_elem_original = dict_elem
        dict_elem_copy = self.__get_simple_item__(index)
        dict_elem_copy['meta'] = {
            'idx': index,
            'max_idx': len(self.df),
            'get_item_func': self.__get_simple_item__
        }

        if self.transform is not None:
            dict_elem_original = self.transform(dict_elem_original)
            dict_elem2 = self.transform(dict_elem_copy)
            
        return dict_elem_original['rgb'],dict_elem2['rgb'], dict_elem_original['depth'], dict_elem_original['ir'], dict_elem_original['label']

    def __get_simple_item__(self, index):
        # if (index%100==True):
        #     print(self.real)
        if self.real:
            while self.df.label.iloc[index] == 1 and index<len(self.df.label)-1:
                index = index + 1

        rgb_path = self.data_root + self.df.rgb.iloc[index]
        ir_path = self.data_root + self.df.ir.iloc[index]
        depth_path = self.data_root + self.df.depth.iloc[index]
        target = self.df.label.iloc[index]
        ir_path = ir_path.replace('_resized','')
        rgb_path = rgb_path.replace('_resized', '')
        depth_path = depth_path.replace('_resized', '')
        # print(rgb_path)
        rgb_img = self.loader(rgb_path)
        ir_img = self.loader(ir_path)
        depth_img = self.loader(depth_path)
        
        dict_elem = {
            'rgb': rgb_img,
            'ir': ir_img,
            'depth': depth_img,
            'label': target
        }
        return dict_elem

    def __len__(self):
        return len(self.df)