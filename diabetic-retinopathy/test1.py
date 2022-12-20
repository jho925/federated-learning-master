import numpy as np
import argparse
import torch
import random
from torch.utils.data import DataLoader
from retina_dataset import Retina_Dataset
from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import squeezenet1_0
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import torch.nn.functional as F
from torchsampler import ImbalancedDatasetSampler
import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

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
    parser.add_argument('--train_size',type=int,default = 1000, help = "how much train data each round")
    parser.add_argument('--positive_percent',type=float,default = 0.5,help = "what fraction of training data is positive")
    parser.add_argument('--switch_distribution',type=str,default = "no",help = "whether to switch the data distribution each round")
    parser.add_argument('--rounds',type=int,default = 1,help = "how many training rounds")
    parser.add_argument('--distillation_loss',type=str,default = 'no',help = "use distillation loss or not")
    parser.add_argument('--epochs_per',type=int,default=10,help = 'how many epochs per round')
    parser.add_argument('--model_save_path',type=str, default = 'model.pth',help = "where to save your model")
    parser.add_argument('--split',type=int, default = 0,help = "which split to use")
    parser.add_argument('--weighted_loss',type=str, default = 'no',help = "use weighted_loss or not")    
    parser.add_argument('--class_incremental',type=str, default = 'no',help = "class_incremental training or not")
    parser.add_argument('--val_auc',type=str, default = 'yes',help = "auc or not")

    args = parser.parse_args()
    return args




args = parse_args()

test_final_loader = DataLoader(eval(args.dataloader)('test_final_loader', args,0,0), args.batch_size, num_workers=8, pin_memory=True)
net = torch.load('round_binary.pth')
net.eval()
net.cuda()


def plot_roc(labels,pred,auc,auc_low,auc_high):

    fpr, tpr, _ = metrics.roc_curve(labels,pred)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f [%0.2f - %0.2f]' % (auc,auc_low,auc_high) , color="blue")


    plt.title("Round Dynamic Class Incremental",fontdict = {'fontsize':20})
    plt.legend(loc = 'upper left')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate',fontdict = {'fontsize':18})
    plt.xlabel('False Positive Rate',fontdict = {'fontsize':18})
    plt.tick_params(labelsize=14)
    plt.savefig('round_binary.svg',dpi =300,saveformat = 'svg')

def test_round(test_loader,net):
    full_pred = []
    full_labels = []
    unthreshold_pred = []
    for iteration, data in enumerate(test_loader):
        inputs = data['image'].cuda()
        labels = data['label'].cuda().cpu().data.numpy()
        pred = net(inputs).cpu().data.numpy()
        full_pred += list(np.argmax(pred, axis=1))
        unthreshold_pred += list(pred)
        full_labels += list(labels.flatten())

    unthreshold_pred = [p[1] for p in unthreshold_pred]
    return full_pred, full_labels,unthreshold_pred

def get_accuracy(full_pred, full_labels,unthreshold_pred):
    # tn, fp, fn, tp = confusion_matrix(full_labels,full_pred).ravel()
    # print("True negative: " + str(tn))
    # print("False positive: " + str(fp))
    # print("False negative: " + str(fn))
    # print("True positive: " + str(tp))

    all_predictions = np.asarray(full_pred)
    all_labels = np.asarray(full_labels)
    all_unthreshold_pred = np.asarray(unthreshold_pred)

    accuracy = np.nanmean(all_labels == all_predictions)
    auc = roc_auc_score(all_labels,all_unthreshold_pred)

    
    accuracy_list = []
    auc_list = []
    for i in range(1000):
        indices = np.random.randint(0, len(all_labels)-1, len(all_labels))
        accuracy_test = np.nanmean(all_labels[indices] == all_predictions[indices])
        accuracy_list.append(accuracy_test)

        auc_test = roc_auc_score(all_labels[indices],all_unthreshold_pred[indices])
        auc_list.append(auc_test)


    auc_list.sort()
    accuracy_list.sort()

    plot_roc(all_labels,unthreshold_pred,auc,auc_list[25],auc_list[-25])


    return accuracy, accuracy_list[25], accuracy_list[-25],auc,auc_list[25],auc_list[-25]


full_pred, full_labels,unthreshold_pred = test_round(test_final_loader, net)
accuracy, ci_low, ci_high,auc,auc_low,auc_high = get_accuracy(full_pred, full_labels,unthreshold_pred)

