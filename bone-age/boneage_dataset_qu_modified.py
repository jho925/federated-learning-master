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
            [#transforms.ColorJitter(brightness=.05, contrast=.05, saturation=.05, hue=.05),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.RandomRotation(30),
             transforms.Grayscale(),
             transforms.RandomCrop([224, 224]),
             transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])
             #    transforms.Normalize([0.485], [0.229])
             # transforms.Normalize((0.485, 0.456), (0.229, 0.224))
            ])
    else:
        data_transforms = transforms.Compose(
            [transforms.CenterCrop([224 ,224]),
             transforms.Grayscale(),
             transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])
             # transforms.Normalize([0.485], [0.229])
             # transforms.Normalize((0.485, 0.456), (0.229, 0.224))
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
        df = pd.read_csv('data/total_labels.csv',header = None,names = ['id','boneage','male'])
        df['id'] = df['id'].apply(str)
        self.labels = dict(zip(df.id + '.png', df.boneage))
        label_gender = dict(zip(df.id + '.png',df.male))

                
    
        if name == 'test_final_loader' or name == "male_test" or name == "female_test":
            data = list(pd.read_csv(args.data_dir + "/total_test.csv", header=None)[0])
        else:
            data = list(pd.read_csv(args.data_dir + "/total_train.csv", header=None)[0])


        data = [str(image) for image in data]
        random.Random(args.seed).shuffle(data)



        j = data_batch
                


        positives = [a for a in data if label_gender[str(a)] == True]
        negatives = [a for a in data if label_gender[str(a)] != False]

        if name == 'male_test':
            data = positives

        if name == 'female_test':
            data = negatives


        if name == 'train':
            data = list(pd.read_csv(args.data_dir + "/total_train.csv", header=None)[0])
        if name == 'val':
            data = list(pd.read_csv(args.data_dir + "/total_val.csv", header=None)[0]) 

        random.shuffle(data)

        # Loading images
        files = glob.glob(os.path.join('data/boneage-training-dataset', "*"))
        self.images = {}
        for file in files:
            filename = os.path.basename(os.path.splitext(file)[0])
            if filename + '.png' in data:                 
                Img = Image.open(file).convert('RGB')

                input = self.transform(Img)

                if input.dim() == 2:
                    input = torch.stack([input, input, input])
                elif input.shape[0] == 1:
                    input = torch.cat([input, input, input])

                self.images[filename + '.png'] = input
        
        labels = {k: v for k,v in self.labels.items() if k in data}


        print("Label balance for " + name, Counter(labels.values()))
        self.set = list(self.images.keys())


    def __getitem__(self, idx):
        key = self.set[idx]
        return {#'image': np.stack((np.squeeze(np.array(self.transform(self.images[key]))),) * 3, axis=1).reshape(3,224,224),
        'image': self.images[key],
                'label':  np.array([self.labels[key]]),
                'img_name': key}

    def __len__(self):
        return len(self.set)
