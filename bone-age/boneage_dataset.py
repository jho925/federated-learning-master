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


def get_transform(name):

    if 'train' in name:
        data_transforms = transforms.Compose(
            [#transforms.RandomHorizontalFlip(p=0.5),
             transforms.Resize([224, 224]),
             #transforms.RandomCrop([224, 224]),
             transforms.ToTensor(),
             transforms.Normalize([0.485], [0.229])
             ])
    else:
        data_transforms = transforms.Compose(
            [transforms.Resize([224, 224]),
            # transforms.CenterCrop([224, 224]),
             transforms.ToTensor(),
             transforms.Normalize([0.485], [0.229])
             ])
    return data_transforms

class Boneage_Dataset(Dataset):
    def __init__(self, name, args, data_batch,switch):
        super(Boneage_Dataset, self).__init__()
        assert name in ['train', 'val','test','test_final_loader','male_test','female_test']
        self.name = name
        self.args = args
        self.transform = get_transform(name)

        # Loading labels
        df = pd.read_csv('data/boneage-training-dataset/labels_reg.csv',header=None)
        df[0] = df[0].apply(str)
        self.labels = dict(zip(df[0], df[1]))


                
    
        if name == 'test_final_loader' or name == "male_test" or name == "female_test":
            data = list(pd.read_csv("data/test.csv", header=None, names=['img'])['img'])
        else:
            data = list(pd.read_csv(args.data_dir + "/total_train.csv", header=None)[0])

        data = [str(image) for image in data]
        random.Random(args.seed).shuffle(data)






        # Loading images
        files = glob.glob(os.path.join('data/boneage-training-dataset', "*"))
        self.images = {}
        for file in files:
            filename = os.path.basename(os.path.splitext(file)[0])

            if filename in data:                 
                self.images[filename] = Image.open(file)
        
        labels = {k: v for k,v in self.labels.items() if k in data}


        print("Label balance for " + name, Counter(labels.values()))
        self.set = list(self.images.keys())

    def __getitem__(self, idx):
        key = self.set[idx]
        return {#'image': np.stack((np.squeeze(np.array(self.transform(self.images[key]))),) * 3, axis=1).reshape(3,224,224),
        'image': self.transform(self.images[key]).view(224, 224, 1).expand(-1,-1,3).reshape(3,224,224),
                'label':  np.array([self.labels[key]]),
                'img_name': key}

    def __len__(self):
        return len(self.set)
