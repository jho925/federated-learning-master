from __future__ import print_function
import torch
from torch.utils.data import Dataset
import glob
import os
import numpy as np
import csv
from collections import Counter
from torchvision import transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import random


class DatasetBoneAgeDist(Dataset):
    def __init__(self, opt, labels, edge_case = False):
        super(DatasetBoneAgeDist, self).__init__()
        self.img_paths = list({line.strip().split(',')[0] for line in open(opt.single_cite)})
        if edge_case:
            pass
        else:
            print('loading', len(self.img_paths), 'training images from file', os.path.basename(opt.single_cite),
                  '------')
            print('Initial with BoneAge dataset---', os.path.basename(opt.single_cite))
            print('We use DatasetBoneAgeDist---for ---', opt.phase )

        self.opt = opt
        self.labels = labels
        if opt.phase == 'train':
            data_transforms = transforms.Compose(
                [#transforms.ColorJitter(brightness=.05, contrast=.05, saturation=.05, hue=.05),
                 transforms.RandomHorizontalFlip(p=0.5),
                 transforms.RandomRotation(30),
                 transforms.Grayscale(),
                 transforms.RandomCrop([opt.fineSize_w, opt.fineSize_h]),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5])
                 #    transforms.Normalize([0.485], [0.229])
                 # transforms.Normalize((0.485, 0.456), (0.229, 0.224))
                ])
        else:
            data_transforms = transforms.Compose(
                [transforms.CenterCrop([opt.fineSize_w, opt.fineSize_h]),
                 transforms.Grayscale(),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5])
                 # transforms.Normalize([0.485], [0.229])
                 # transforms.Normalize((0.485, 0.456), (0.229, 0.224))
                 ])
        # if opt.resolution_variation and self.opt.phase == 'train':
        #     print('Using image aquistion variation with dataset', opt.data_file)



        self.transform = data_transforms

    def __getitem__(self, index):
        try:
            path = os.path.join(self.opt.data_path, self.img_paths[index])
            print(img_paths[index])
            if index == 0:
                print('Loading data from ', self.opt.csv_train)
                print(img_paths)
            # else:
            #     path = os.path.join(self.opt.data_path, self.opt.data_file, self.img_paths[index])
                # path = os.path.join(self.opt.data_path, 'train-all', self.img_paths[index])
        except:
            path = os.path.join(self.opt.data_path, self.img_paths[index])
            # path = os.path.join(self.opt.data_path, 'train-all', self.img_paths[index])
        # name = os.path.basename(path)
        name = self.img_paths[index]
        Img = Image.open(path).convert('RGB')

        input = self.transform(Img)

        if input.dim() == 2:
            input = torch.stack([input, input, input])
        elif input.shape[0] == 1:
            input = torch.cat([input, input, input])
        tmp_label = self.labels[name]

        if self.opt.regression:


            label = torch.FloatTensor(1)

            label[0] = tmp_label
            # label[1] = tmp_label
            # label[2] = tmp_label

        else:

            label = torch.LongTensor(3)

            label[0] = tmp_label
            label[1] = tmp_label
            label[2] = tmp_label
        return {'input': input, 'label': label, 'Img_paths': path}

    def __len__(self):
        return len(self.img_paths)