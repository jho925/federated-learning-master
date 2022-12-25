import numpy as np
import argparse
import torch
import random
from torch.utils.data import DataLoader
from retina_dataset import Retina_Dataset
from retina_class import Retina_Class_Dataset
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
from model import train_epoch,train_site,roc_auc_score_multiclass,plot_roc,test_round,get_accuracy,loss_fn_kd,parse_args


def main():
    args = parse_args()

    if args.class_incremental = "yes":
        args.dataloader = "Retina_Class_Dataset"

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader_dict = {}
    for j in range(1,args.rounds*args.sites+1):
        key_train_loader = "train_loader" + str(j)
        dataset = eval(args.dataloader)('train', args, j,0)
        print(len(dataset))
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
            print(class_sample_count)
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

    net = eval(args.model)(pretrained=True)
    net.fc = nn.Linear(512, 2)
    #net.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.4, training=m.training))
    net.cuda()
    ewc= ElasticWeightConsolidation(net, nn.CrossEntropyLoss(reduction="sum").cuda(), lr=args.lr, weight=10000) 

    for j in range(args.rounds-1,-1,-1):
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

        if j != 0:
            full_pred, full_labels,unthreshold_pred = test_round(test_loader_dict['test_loader' + str(j)], ewc)
            accuracy, ci_low, ci_high,auc,auc_low,auc_high = get_accuracy(full_pred, full_labels,unthreshold_pred,j)
            print("Round " + str(j+1) + " Accuracy on Test " + str(j+1) + ": {} ({} - {})".format(accuracy, ci_low, ci_high))
            print("Round " + str(j+1) + " AUC_ROC on Test " + str(j+1) + ": {} ({} - {})".format(auc, auc_low, auc_high))


        full_pred, full_labels,unthreshold_pred = test_round(test_final_loader, ewc)
        accuracy, ci_low, ci_high,auc,auc_low,auc_high = get_accuracy(full_pred, full_labels,unthreshold_pred,j)
        print("Round " + str(j+1) + " Accuracy on Final Test: {} ({} - {})".format(accuracy, ci_low, ci_high))
        print("Round " + str(j+1) + " AUC_ROC on Final Test: {} ({} - {})".format(auc, auc_low, auc_high))

    torch.save(ewc.model, args.model_save_path)


if __name__ == '__main__':
    main()
