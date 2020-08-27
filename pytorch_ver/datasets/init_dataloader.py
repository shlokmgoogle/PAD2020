from .init_dataset import ImageListDataset
import torch.utils.data
import os
import pandas as pd
from Load_OULUNPU_train import Spoofing_train
from Load_OULUNPU_valtest import Spoofing_valtest
from torchvision import transforms
from Load_OULUNPU_train import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from Load_OULUNPU_valtest import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest

def generate_loader(opt, split, inference_list = None):
    
    if split == 'train':
        current_transform = opt.train_transform
        current_shuffle = True
        sampler = None
        drop_last = False
        
    else:
        current_transform = opt.test_transform
        current_shuffle = False
        sampler = None
        drop_last = False
        
    # import pdb
    # pdb.set_trace()
    train_image_dir = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge/OULU/Train_files_images/'
    val_image_dir = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge/OULU/Train_files_images/'
    test_image_dir = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge/OULU/Test_files_Images//'

    map_dir = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge'
    val_map_dir = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge'
    test_map_dir = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge'

    train_list = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge/OULU/Protocols/Protocol_1/Train.txt'
    val_list = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge/OULU/Protocols/Protocol_1/Dev.txt'
    test_list = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge/OULU/Protocols/Protocol_1/Test.txt'
    if inference_list:
        data_list = inference_list
        data_root= '/path/to/val/and/test/data'
        current_shuffle = False
    else:
        data_list = os.path.join(opt.data_list, split + '_list.txt')
        data_root = opt.data_root
    if opt.dataset == 'OULU':
        if split == 'train':
            train_data = Spoofing_train(train_list, train_image_dir, map_dir, transform=transforms.Compose(
                [RandomErasing(), RandomHorizontalFlip(), ToTensor(), Cutout(), Normaliztion()]))
            dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=0)
            return dataloader_train
        if split == 'val':
            train_data = Spoofing_valtest(test_list, test_image_dir, map_dir, transform=transforms.Compose(
                [RandomErasing(), RandomHorizontalFlip(), ToTensor(), Cutout(), Normaliztion()]))
            dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=0)
            return dataloader_train
    else:
        dataset = ImageListDataset(data_root = data_root,  data_list = data_list, transform=current_transform)


    assert dataset
    if split == 'train' and opt.fake_class_weight != 1:
        weights = [opt.fake_class_weight if x != 1 else 1.0 for x in dataset.df.label.values]
        num_samples = len(dataset)
        replacement = True
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement)
        current_shuffle = False
    if split == 'train' and len(dataset) % (opt.batch_size // opt.ngpu) < 32:
        drop_last = True

    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, shuffle = current_shuffle,
                                                 num_workers = int(opt.nthreads),sampler = sampler, pin_memory=True,
                                                 drop_last = drop_last)
    return dataset_loader
