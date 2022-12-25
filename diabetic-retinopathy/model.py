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


def train_epoch(ewc,args,train_loader,rounds,epoch,prev_model):
    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    for iteration, data in enumerate(train_loader):
        inputs = data['image'].cuda()
        #.cuda()

        labels = data['label'].cuda().flatten()
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


    accuracy = []
    for j in range(round_num*args.sites+1,(round_num+1)*args.sites+1):     
        for iteration, data in enumerate(eval_loader_dict['val_loader' + str(j)]):
            inputs = data['image'].cuda()
            labels = data['label'].cuda()
            pred = net(inputs).cpu().data.numpy()
            labels = labels.cpu().data.numpy()
            unthreshold_pred += list(pred)
            full_labels += list(labels.flatten())
            accuracy += list(np.argmax(pred, axis=1) == labels.flatten())
    
    if args.val_auc == 'yes':
        unthreshold_pred = [p[1] for p in unthreshold_pred]
        auc = roc_auc_score(full_labels,unthreshold_pred)
        if auc > best_performance:
            best_performance = auc
            best_ewc = copy.deepcopy(ewc)


        print(auc)
    else:
        if 100 * np.nanmean(np.array(accuracy)) > best_performance:
            best_performance = 100 * np.nanmean(np.array(accuracy))
            best_ewc = copy.deepcopy(ewc)

        
        print(100 * np.nanmean(np.array(accuracy)))
        
    return ewc,best_ewc,best_performance

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

    #creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        print(per_class)
        #creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]
        print(new_actual_class)
        print(new_pred_class)

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict


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

def plot_roc(labels,pred,auc,round_num,auc_low,auc_high):

    fpr, tpr, _ = metrics.roc_curve(labels,pred)
    plt.plot(fpr, tpr, 'b', label = 'Federated Incremental Learning Homogeneous = %0.2f [%0.2f - %0.2f]' % (auc,auc_low,auc_high) , color="blue")



    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('round' + str(round_num+1) + '.png')




def get_accuracy(full_pred, full_labels,unthreshold_pred,round_num):
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

    plot_roc(all_labels,unthreshold_pred,auc,round_num,auc_list[25],auc_list[-25])

    return accuracy, accuracy_list[25], accuracy_list[-25],auc,auc_list[25],auc_list[-25]


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
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss

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
    parser.add_argument('--round',type=str, default = 'yes',help = "round or site dynamic (no for site)")

    args = parser.parse_args()
    return args

