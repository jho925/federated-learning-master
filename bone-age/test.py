import numpy as np
import argparse
import torch
import random
from torch.utils.data import DataLoader
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
from dataset import DatasetBoneAgeDist
import os
from dataset import DatasetBoneAgeDist as DatasetDistribution


parser = argparse.ArgumentParser()

# CENTRAL PARAMETER SERVER INFO
parser = argparse.ArgumentParser(description="PyTorch DDCNN")
parser.add_argument("--batchSize", type=int, default=32, help="Training batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=0.1")
parser.add_argument("--fineSize_w", type=int, default=224, help="Learning Rate. Default=0.1")
parser.add_argument("--fineSize_h", type=int, default=224, help="Learning Rate. Default=0.1")
parser.add_argument("--loadSize_L", type=int, default=256, help="Learning Rate. Default=0.1")
parser.add_argument("--loadSize", type=int, default=256, help="Learning Rate. Default=0.1")
parser.add_argument("--save_freq", type=int, default=5, help="Learning Rate. Default=0.1")
# parser.add_argument("--data_path", type=str, default='/home/liangqiong/Research/Deep_Learning/Pytorch/Data_Distribution/Data/pre-processed/datadist/Retina/post/', )
parser.add_argument("--data_path", type=str,default='/home/liangqiong/Research/Deep_Learning/Pytorch/Data_Distribution/Data/pre-processed/datadist/ADNI/NPY/', )
# parser.add_argument("--data_path", type=str,default='/home/liangqiong/Research/Deep_Learning/Pytorch/Data_Distribution/Data/pre-processed/datadist/rsna-bone-age/pre_processed/', )
parser.add_argument("--num_classes", type=int, default=2, help="Learning Rate. Default=0.1")
parser.add_argument("--phase", type=str, default="train", help="Learning Rate. Default=0.1")
parser.add_argument("--continue_train", action='store_true', default=False, help='refine cerebellum regions or not')
parser.add_argument("--epoch", type=str, default="55", help="Learning Rate. Default=0.1")
parser.add_argument("--dis_mode_name", type=str, default="ADNI_central_imvar_real_vendor_v", help="model name")  # ResNet_ADNI_ASGD_4_sites_125 ADNI_single_site_in_real_reg
parser.add_argument("--csv_train", action='store_true', default = 'True', help='using csv to indicate the train set or not')
parser.add_argument("--train_file", type=str, default="train_net", help="Learning Rate. Default=0.1")
parser.add_argument("--crop", action='store_true', default='True', help="Learning Rate. Default=0.1")
parser.add_argument("--regression", action='store_true', default = True,  help="Indicating using regression model or not")
parser.add_argument("--segmentation", action='store_true', default = False,  help="Segmentation task")
parser.add_argument("--group_norm", action='store_true', default = False, help="Indicating using regression model or not")

# FOLLOWING ARGUMENTS MUST BE THE SAME FOR ALL PARTICIPATING INSTITUTIONS DURING TRAINING

# SELECT TRUE FOR ONE OF THE FOLLOWING TO ADDRESS LABEL IMBALANCE


if __name__ == '__main__':
    args = parser.parse_args()





opt = parser.parse_args()
net = resnet34(pretrained=True)
net.load_state_dict(torch.load('ResNet_Boneage_epoch55_all.pth'))
#net.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.4, training=m.training))


labels = None
if not opt.regression:
    if opt.num_classes > 1:
        tot_labels = {line.strip().split(',')[0] + '.png': float(line.strip().split(',')[1]) for line in
                  open(os.path.join(opt.data_path, 'labels.csv'))}
    print('NOTE------------Train with classfication-----------')
else:
    opt.num_classes = 1
    tot_labels = {line.strip().split(',')[0] + '.png': float(line.strip().split(',')[1]) for line in
              open(os.path.join(opt.data_path, 'labels_reg.csv'))}
    print('NOTE------------Train with regression-----------')

opt.single_cite = os.path.join(opt.data_path, opt.train_file + '.csv')
test_set = DatasetDistribution(opt, tot_labels)
test_final_loader = DataLoader(dataset=test_set, num_workers=16, batch_size=32, shuffle=True)


def test_round(test_loader,net):
    full_pred = []
    full_labels = []
    r = nn.ReLU()
    for iteration, data in enumerate(test_loader):
        inputs = data['input']
        labels = data['label'].cpu().data.numpy()
        pred = r(net(inputs)).cpu().data.numpy()
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


full_pred, full_labels = test_round(test_final_loader, net)
error, ci_low, ci_high = get_error(full_pred, full_labels)
print("Round " + str(1) + " Error on Final Test: {} ({} - {})".format(error, ci_low, ci_high))



