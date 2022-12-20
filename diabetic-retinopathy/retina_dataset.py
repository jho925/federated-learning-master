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
import pandas as pd
import random

def get_transform(name):

    if 'train' in name:
        data_transforms = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
             transforms.Resize([256, 256]),
             transforms.RandomCrop([224, 224]),
             transforms.ToTensor(),
             # transforms.Normalize([0.485], [0.229])
             ])
    else:
        data_transforms = transforms.Compose(
            [transforms.Resize([256, 256]),
             transforms.CenterCrop([224, 224]),
             transforms.ToTensor(),
             # transforms.Normalize([0.485], [0.229])
             ])
    return data_transforms

class Retina_Dataset(Dataset):
    def __init__(self, name, args, data_batch,switch):
        super(Retina_Dataset, self).__init__()
        assert name in ['train', 'val','test','test_final_loader']
        self.name = name
        self.args = args
        self.transform = get_transform(name)

        # Loading labels
        df = pd.read_csv('data/total_labels.csv',  names=['image', 'label'], skiprows=1)
        self.labels = df.set_index('image')['label'].to_dict()
                
    
        if name == 'test_final_loader':
            data = list(pd.read_csv(args.data_dir + "/total_test.csv", header=None)[0])
        else:
            data = list(pd.read_csv(args.data_dir + "/total_train.csv", header=None)[0])



        random.Random(args.seed).shuffle(data)



        j = data_batch
        if args.class_incremental == 'no':
            self.labels.update({a:1 for a in self.labels if self.labels[a] >= 1})


            positives = [a for a in data if self.labels[a] ==1]
            negatives = [a for a in data if self.labels[a] ==0]




            if args.positive_percent == 0.5:
                train_size = int(args.train_size*0.8/(2*args.sites))
                val_size = int(train_size/4)
                if name == 'train':
                    data = positives[(train_size+val_size)*(j-1):(train_size+val_size)*(j-1)+train_size] + negatives[(train_size+val_size)*(j-1):(train_size+val_size)*(j-1)+train_size]
                
                if name == 'test':
                    data = positives[int(args.train_size/2)*j:int(args.train_size/2)*(j+1)] + negatives[int(args.train_size/2)*j:int(args.train_size/2)*(j+1)] 

                if name == 'val': 
                    data = positives[(train_size+val_size)*(j-1)+train_size:(train_size+val_size)*(j-1)+train_size+val_size] + negatives[(train_size+val_size)*(j-1)+train_size:(train_size+val_size)*(j-1)+train_size+val_size]
                
            else:

                train_size = int(args.train_size*0.8/(args.sites))
                val_size = int(train_size/4)

                positive_train_reg = int(train_size*args.positive_percent)
                negative_train_reg = train_size-positive_train_reg
                positive_val_reg = int(positive_train_reg/4)
                negative_val_reg = int(negative_train_reg/4)

                negative_train_switch = int(train_size*args.positive_percent)
                positive_train_switch = train_size-negative_train_switch
                positive_val_switch = int(positive_train_switch/4)
                negative_val_switch = int(negative_train_switch/4)

                if (int((j-1)/args.sites))%2 == 0:
                    positive_train = positive_train_reg
                    negative_train = negative_train_reg
                    positive_val = positive_val_reg
                    negative_val = negative_val_reg                
                else:
                    positive_train = positive_train_switch
                    negative_train = negative_train_switch
                    positive_val = positive_val_switch
                    negative_val = negative_val_switch

                rounds_completed = int((j-1)/args.sites)
                sites_completed = (j-1)%args.sites
                mod = sites_completed

                positive_add = 0
                negative_add = 0

                for i in range(rounds_completed):
                    if i%2 == 0:
                        positive_add += (positive_train_reg+positive_val_reg)*args.sites
                        negative_add += (negative_train_reg+negative_val_reg)*args.sites
                    else:
                        positive_add += (positive_train_switch+positive_val_switch)*args.sites
                        negative_add += (negative_val_switch+negative_val_switch)*args.sites
        

                if name == 'train':
                    data = positives[positive_add+(mod)*(positive_train+positive_val):positive_add+mod*(positive_train+positive_val)+positive_train] + negatives[negative_add+mod*(negative_train+negative_val):negative_add+mod*(negative_train+negative_val) + negative_train]
                
                if name == 'test':
                    data = positives[int(args.train_size/2)*j:int(args.train_size/2)*(j+1)] + negatives[int(args.train_size/2)*j:int(args.train_size/2)*(j+1)]
                if name == 'val': 
                    data = positives[positive_add+mod*(positive_train+positive_val)+positive_train:positive_add+(mod+1)*(positive_train+positive_val)] + negatives[negative_add+mod*(negative_train+negative_val) + negative_train:negative_add+(mod+1)*(negative_train+negative_val)]

     
        
        else:

            label_0 = []
            label_1 = []
            label_2 = []
            label_3 = []
            label_4 = []


            total = []
            
            label_0 = [a for a in data if self.labels[a] ==0]
            label_1 = [a for a in data if self.labels[a] ==1]
            label_2 = [a for a in data if self.labels[a] ==2]
            label_3 = [a for a in data if self.labels[a] ==3]
            label_4 = [a for a in data if self.labels[a] ==4]



            total.append(label_1)
            total.append(label_2)
            total.append(label_3)
            total.append(label_4)

            self.labels.update({a:1 for a in self.labels if self.labels[a] == 2})
            self.labels.update({a:2 for a in self.labels if self.labels[a] == 4})

            data = []    
            rounds_completed = (j-1)//args.sites
            sites_completed = (j-1)%args.sites
            
            train_size = int(args.train_size*0.8/(2*args.sites))
            val_size = int(train_size* 0.25)
            if name == "train":
                index = 0
                data += label_0[int((args.train_size/2)*rounds_completed+(train_size+val_size)*sites_completed):int((args.train_size/2)*rounds_completed+(train_size+val_size)*sites_completed+train_size)] 
                data += total[3-rounds_completed*2][sites_completed*(train_size+val_size):(sites_completed)*(train_size+val_size)+train_size]
                
            if name == "val":
                data += label_0[int((args.train_size/2)*rounds_completed+(train_size+val_size)*sites_completed+train_size):int((args.train_size/2)*rounds_completed+(train_size+val_size)*sites_completed+train_size+val_size)] 
                data += total[3-rounds_completed*2][sites_completed*(train_size+val_size)+train_size:(sites_completed)*(train_size+val_size)+train_size+val_size]

            
            if name == "test":
                data += label_0[int(args.train_size/2)*rounds_completed:int(args.train_size/2)*(rounds_completed+1)] 
                data += total[2-rounds_completed*2][0:int(args.train_size/2)]
            if name == 'test_final_loader':
                 data = list(pd.read_csv(args.data_dir + "/total_test.csv", header=None)[0])
                 data = label_0[0:len(label_4)*2]
                 for i in range (0,4,2):
                    data += total[3-i][0:len(label_4)]

        random.shuffle(data)

        # Loading images
        files = glob.glob(os.path.join(args.data_dir, 'combined', "*"))
        self.images = {}

        for file in files:
            filename = os.path.basename(os.path.splitext(file)[0])


            if filename in data:
                self.images[filename] = Image.fromarray(np.load(file))

        labels = {k: v for k,v in self.labels.items() if k in data}

        print("Label balance for " + name, Counter(labels.values()))
        self.set = list(self.images.keys())

    def __getitem__(self, idx):
        key = self.set[idx]
        return {'image': self.transform(self.images[key]),
                'label':  np.array([self.labels[key]]),
                'img_name': key}

    def __len__(self):
        return len(self.set)
