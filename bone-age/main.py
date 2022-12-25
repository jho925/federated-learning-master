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
from model import train_epoch,train_site,test_round,get_error,loss_fn_kd,parse_args

# import ray
# from ray import tune
# from ray.tune import track
# from ray.tune.schedulers import AsyncHyperBandScheduler

def main():
    if args.class_incremental = "yes":
        args.dataloader = "Boneage_Class_Dataset"


    args = parse_args()


    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader_dict = {}
    for j in range(1,args.rounds*args.sites+1):
        key_train_loader = "train_loader" + str(j)
        dataset = eval(args.dataloader)('train', args, j,0)
        if args.weighted_loss == 'yes':
            data_loader = DataLoader(dataset, args.batch_size,num_workers=8, pin_memory=True)
            labels = []
            for iteration, data in enumerate(data_loader):
                labels.extend(list(data['label'].data.cpu().numpy()))
            labels = np.array(labels)
            class_sample_count = np.array(
                [len(np.where(labels == t)[0]) for t in np.unique(labels)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in labels])
            samples_weight= np.array([s for p in samples_weight for s in p])
            sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))  
            train_loader_dict[key_train_loader] = DataLoader(dataset, args.batch_size,sampler=sampler,num_workers=8, pin_memory=True)
        else:
            train_loader_dict[key_train_loader] = DataLoader(dataset, args.batch_size,num_workers=8, pin_memory=True)


    val_loader_dict = {}
    for j in range(1,args.rounds*args.sites+1):
        key_val_loader = "val_loader" + str(j)
        dataset = eval(args.dataloader)('val', args, j,0)
        val_loader_dict[key_val_loader] = DataLoader(dataset, args.batch_size,num_workers=8, pin_memory=True)

    test_loader_dict = {}
    for j in range(1,args.rounds):
        key_test_loader = "test_loader" + str(j)
        dataset = eval(args.dataloader)('test', args, j,0)
        test_loader_dict[key_test_loader] = DataLoader(dataset, args.batch_size,num_workers=8, pin_memory=True)

    test_final_loader = DataLoader(eval(args.dataloader)('test_final_loader', args,0,0), args.batch_size, num_workers=8, pin_memory=True)
    male_test = DataLoader(eval(args.dataloader)('male_test', args,0,0), args.batch_size, num_workers=8, pin_memory=True)

    female_test = DataLoader(eval(args.dataloader)('female_test', args,0,0), args.batch_size, num_workers=8, pin_memory=True)


    net = eval(args.model)(pretrained=True)
    net.fc = nn.Linear(512, 1)
    net.cuda()

    ewc= ElasticWeightConsolidation(net, nn.L1Loss(reduction="mean").cuda() , lr=args.lr, weight=10000) 
    for j in range(args.rounds):
        best_ewc = None
        best_performance = 0
        ewc.model.train(True)
        prev_model = copy.deepcopy(net)
        print("Starting Training Round on Set " + str(j+1) + "...")
        for k in range(args.epochs_per):
            ewc,best_ewc,best_performace = train_site(args, train_loader_dict['train_loader' + str(args.sites*j+k%args.sites+1)], val_loader_dict, ewc, j,best_ewc,best_performance,k,prev_model)



        for i in range(args.sites):
            ewc.register_ewc_params(train_loader_dict['train_loader' + str((j)*args.sites +i + 1)].dataset,args.batch_size)

        prev_model = copy.deepcopy(ewc.model)
        ewc.model.train(False)

        if j != args.rounds - 1:
            full_pred, full_labels = test_round(test_loader_dict['test_loader' + str(j+1)], net)
            error, ci_low, ci_high = get_error(full_pred, full_labels)
            print("Round " + str(j+1) + " Error on Test " + str(j+1) + ": {} ({} - {})".format(error, ci_low, ci_high))


        full_pred, full_labels = test_round(test_final_loader, net)
        error, ci_low, ci_high = get_error(full_pred, full_labels)
        print("Round " + str(j+1) + " Error on Final Test: {} ({} - {})".format(error, ci_low, ci_high))

        full_pred, full_labels = test_round(male_test, net)
        error, ci_low, ci_high = get_error(full_pred, full_labels)
        print("Round " + str(j+1) + " Error on Male Test: {} ({} - {})".format(error, ci_low, ci_high))

        full_pred, full_labels = test_round(female_test, net)
        error, ci_low, ci_high = get_error(full_pred, full_labels)
        print("Round " + str(j+1) + " Error on Female Test: {} ({} - {})".format(error, ci_low, ci_high))










