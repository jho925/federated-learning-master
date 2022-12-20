import os
import time
import sys
import numpy as np

import torch.utils.data as data
import torch
import numpy as np
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn

import os.path
from torch.utils.data import DataLoader
import argparse
import random
import cwt_models
from dataset import DatasetBoneAgeDist
# from torch.utils.tensorboard import SummaryWriter


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

    # Training settings

    def save_networks(net, opt, epoch, name):
        save_file_name = os.path.join('%s_%s.pth' % (name, epoch))
        torch.save(net.state_dict(), save_file_name)

    opt = parser.parse_args()
    opt.log_name = opt.dis_mode_name + '.txt'
    if 'Bone' in opt.dis_mode_name:
        from dataset import DatasetBoneAgeDist as DatasetDistribution

    log_name = os.path.join(os.getcwd(), 'log', opt.log_name)
    now = time.strftime("%c")
    # log_file.write('================ Training Loss (%s) ================\n' % now)

    dir_paths = opt.data_path + 'train/'

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

    opt.tot_labels = tot_labels
    if 'Bone' in opt.dis_mode_name:
        opt.single_cite = os.path.join(opt.data_path, opt.train_file + '.csv')
        train_set = DatasetDistribution(opt, tot_labels)
    else:
        train_set = DatasetDistribution(opt, tot_labels)
    train_set_loader = DataLoader(dataset=train_set, num_workers=16, batch_size=32, shuffle=True)
    print('Loading total', len(train_set), 'training images--------')
    print(opt.csv_train)

    #........................ set the model..................................
    # net = models.inception_v3(pretrained=True)
    # net.fc = nn.Linear(2048, opt.num_classes)
    tmp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.device = tmp_device
    if opt.segmentation:
        net = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                               in_channels=3, out_channels=1, init_features=32, pretrained=True)

    else:
        if opt.group_norm:  ## if we use group-normalization
            from utils.resnet import resnet34

            net = resnet34(pretrained=False, **{'group_norm': 32, 'num_classes': 2})
            net = net.to(opt.device)
            net = net.train()

        else:
            net = models.resnet34(pretrained=True)
            net.fc = nn.Linear(512, opt.num_classes)

    # net = models.resnet50(pretrained=True)
    # net.fc = nn.Linear(2048, opt.num_classes)
    net = net.to(tmp_device)
    SEED = 15
    # SEED = 35

    np.random.seed(SEED)  # if numpy is used
    torch.manual_seed(SEED)
    if not opt.device == 'cpu':
        torch.cuda.manual_seed(SEED)
    net = models.resnet34(pretrained=True)
    net = net.train()
    net.fc = nn.Linear(512, 1)
    net.cuda()

    if opt.regression:
        if 'Bone' in opt.dis_mode_name:
            criterion = torch.nn.L1Loss().cuda().to(tmp_device)
            # criterion = torch.nn.MSELoss().to(tmp_device)
        else:
            criterion = torch.nn.MSELoss().cuda().to(tmp_device)
    else:
        criterion = nn.CrossEntropyLoss().cuda().to(tmp_device)
    # criterion = nn.CrossEntropyLoss().to(tmp_device)
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)
    running_loss = 0
    # writer = SummaryWriter('/home/liangqiong/Research/Deep_Learning/Pytorch/Data_Distribution/Result/log_Bone_Central_single_val/')

    for epoch in range(0,60):
        mean1 = []
        std1 = []
        # if 'Bone' in opt.dis_mode_name and epoch == 60:
        #     criterion = torch.nn.MSELoss().to(tmp_device)

        for iteration, data in enumerate(train_set_loader):
            # print(iteration)
            inputs = data['input'].cuda().to(tmp_device)
            labels = data['label'].cuda().to(tmp_device)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            if opt.regression:
                # print('We use regression with label of size', labels.shape)
                labels1 = labels
            else:
                labels1 = labels[:, 1]
            loss = criterion(outputs, labels1)
            loss.backward()
            optimizer.step()
            # print statistics

            # tot_epoch_loss += loss.item()
            running_loss += loss.item()
            current_epoch = epoch * len(train_set)/opt.batchSize + iteration
            # writer.add_scalar('Train/Loss', loss.item(), current_epoch)

            current_epoch = epoch * len(train_set)/opt.batchSize + iteration
            # writer.add_scalar('Train/Loss', loss.item(), current_epoch)

            if iteration % 10 == 9:  # print every 2000 mini-batches
                # print(iteration)
                message = '(epoch: %d, iters: %d, time: %.3f, with loss: %.3f) ' % (epoch, iteration, time.time(), running_loss / 10)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, iteration + 1, running_loss / 10))
                running_loss = 0.0
                # log_file.write('%s\n' % message)

        if epoch % opt.save_freq == 0 and epoch >20:
            # if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d) of learning rate %f' % (epoch, opt.lr ))
            # save_networks(net, epoch, opt.dis_mode_name)
            save_networks(net, opt, 'epoch' + str(epoch) + '_' + 'all', opt.dis_mode_name)


        if epoch % 40 == 39:
            opt.lr = opt.lr / 10
            optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)
    # log_file.close()
