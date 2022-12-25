import numpy as np
import argparse
import torch
import random
from torch.utils.data import DataLoader
from boneage_dataset import Boneage_Dataset
from torchvision.models import resnet18
from torchvision.models import resnet34
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import torch.nn.functional as F
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import math
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from elastic_weight_consolidation import ElasticWeightConsolidation


# import ray
# from ray import tune
# from ray.tune import track
# from ray.tune.schedulers import AsyncHyperBandScheduler

def train_epoch(ewc,args,train_loader,rounds,epoch,prev_model):
    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    for iteration, data in enumerate(train_loader):
        inputs = data['image'].cuda()
        #.cuda()

        labels = data['label'].cuda()
        #.cuda()
        ewc.forward_backward_update(inputs, labels,train_loader,args.batch_size,args.lr,epoch,iteration,prev_model,rounds)
    return ewc



def train_site(args, train_loader, eval_loader_dict, ewc, round_num,best_ewc,best_performance,epoch,prev_model):
    ewc.model.train(True)
    ewc= train_epoch(ewc, args, train_loader, round_num, epoch,prev_model)
    ewc.model.train(False)
    net = ewc.model
    
    full_labels = []
    unthreshold_pred =[]


    error = []
    for j in range(round_num*args.sites+1,(round_num+1)*args.sites+1):     
        for iteration, data in enumerate(eval_loader_dict['val_loader' + str(j)]):
            inputs = data['image'].cuda()
            labels = data['label'].cuda()
            pred = net(inputs).cpu().data.numpy()
            labels = labels.cpu().data.numpy()
            error += list(np.absolute(pred - labels))
    
    if  np.nanmean(np.array(error)) < best_performance:
        best_performance = np.nanmean(np.array(error))
        best_net = copy.deepcopy(net)
    print(np.nanmean(np.array(error)))
        
    return ewc,best_ewc,best_performance
    

# def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

#     #creating a set of all the unique classes using the actual class list
#     unique_class = set(actual_class)
#     roc_auc_dict = {}
#     for per_class in unique_class:
#         print(per_class)
#         #creating a list of all the classes except the current class
#         other_class = [x for x in unique_class if x != per_class]

#         #marking the current class as 1 and all other classes as 0
#         new_actual_class = [0 if x in other_class else 1 for x in actual_class]
#         new_pred_class = [0 if x in other_class else 1 for x in pred_class]
#         print(new_actual_class)
#         print(new_pred_class)

#         #using the sklearn metrics method to calculate the roc_auc_score
#         roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
#         roc_auc_dict[per_class] = roc_auc

#     return roc_auc_dict


def test_round(test_loader,ewc):
    r = nn.ReLU()
    net = ewc.model
    full_pred = []
    full_labels = []
    for iteration, data in enumerate(test_loader):
        inputs = data['image'].cuda()
        labels = data['label'].cuda().cpu().data.numpy()
        pred = r(net(inputs).cpu().data.numpy())
        full_pred += list(pred)
        full_labels += list(labels.flatten())

    return full_pred, full_labels

def get_error(full_pred, full_labels):
    # tn, fp, fn, tp = confusion_matrix(full_labels,full_pred).ravel()
    # print("True negative: " + str(tn))
    # print("False positive: " + str(fp))
    # print("False negative: " + str(fn))
    # print("True positive: " + str(tp))

    all_predictions = np.asarray(full_pred)
    all_labels = np.asarray(full_labels)

    error = np.nanmean(np.absolute(all_labels - all_predictions))

    
    error_list = []
    for i in range(1000):
        indices = np.random.randint(0, len(all_labels)-1, len(all_labels))
        error_test = np.nanmean(np.absolute(all_labels[indices] - all_predictions[indices]))
        error_list.append(error_test)

    error_list.sort()
    return error, error_list[25], error_list[-25]


def loss_fn_kd(outputs, labels, teacher_outputs):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = 0.5
    T = 3
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.mse_loss(outputs, labels) * (1. - alpha)

    return KD_loss

def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--dataloader', type=str, default="Boneage_Dataset")
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999999))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--positive_percent',type=float,default = 0.5,help = "what fraction of training data is positive")
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--sites',type=int,default = 1,help = "how many sites")
    parser.add_argument('--train_size',type=int,default = 1000, help = "how much train data each round")
    parser.add_argument('--rounds',type=int,default = 1,help = "how many training rounds")
    parser.add_argument('--distillation_loss',type=str,default = 'no',help = "use distillation loss or not")
    parser.add_argument('--epochs_per',type=int,default=10,help = 'how many epochs per round')
    parser.add_argument('--model_save_path',type=str, default = 'model.pth',help = "where to save your model")
    parser.add_argument('--split',type=int, default = 0,help = "which split to use")
    parser.add_argument('--weighted_loss',type=str, default = 'no',help = "use weighted_loss or not")    
    parser.add_argument('--class_incremental',type=str, default = 'no',help = "class_incremental training or not")


    args = parser.parse_args()
    return args

