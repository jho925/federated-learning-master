import numpy as np
import argparse
import torch
import random
from torch.utils.data import DataLoader
from retina_dataset import Retina_Dataset
from torchvision.models import resnet34
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import torch.nn.functional as F
from torchsampler import ImbalancedDatasetSampler
import sklearn
from sklearn.metrics import confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--dataloader', type=str, default="Retina_Dataset")
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999999))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--classification',type=str,default = 'normal',help = "how many classes (binary or normal)")
    parser.add_argument('--sites',type=int,default = 1,help = "how many sites")
    parser.add_argument('--data',type=int,default = 1000, help = "how much data")
    parser.add_argument('--positive_percent',type=float,default = 0.5,help = "what fraction of training data is positive")
    parser.add_argument('--switch_distribution',type=str,default = "no",help = "whether to switch the data distribution each round")
    parser.add_argument('--rounds',type=int,default = 1,help = "how many training rounds")
    parser.add_argument('--distillation_loss',type=str,default = 'no',help = "use distillation loss or not")
    parser.add_argument('--epochs_per',type=int,default=10,help = 'how many epochs per round')
    parser.add_argument('--model_save_path',type=str, default = 'model.pth',help = "where to save your model")
    parser.add_argument('--split',type=int, default = 0,help = "which split to use")

    args = parser.parse_args()
    return args

args = parse_args()



net = torch.load("best_model.pth")
test2_loader = DataLoader(eval(args.dataloader)('test2', args,0,0), 32, num_workers=8, pin_memory=True)


accuracy = []
for iteration, data in enumerate(test2_loader):
        inputs = data['image']
        labels = data['label']
        pred = net(inputs).cpu().data.numpy()
        labels = labels.cpu().data.numpy()
        accuracy += list(np.argmax(pred, axis=1) == labels.flatten())
print(100 * np.mean(np.array(accuracy)))
net.train(True)