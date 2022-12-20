import os
import time
import sys
import numpy as np
import torchvision.models as models
import torch.utils.data as data
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

import os.path
from torch.utils.data import DataLoader
import argparse
import random
# from models import save_networks, set_up_model
from boneage_dataset_qu import Boneage_Dataset
from torchvision.models import resnet34

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
parser.add_argument("--data_path", type=str,default='data/', )
# parser.add_argument("--data_path", type=str,default='/home/liangqiong/Research/Deep_Learning/Pytorch/Data_Distribution/Data/pre-processed/datadist/rsna-bone-age/pre_processed/', )
parser.add_argument("--num_classes", type=int, default=2, help="Learning Rate. Default=0.1")
parser.add_argument("--phase", type=str, default="train", help="Learning Rate. Default=0.1")
parser.add_argument("--continue_train", action='store_true', default=False, help='refine cerebellum regions or not')
parser.add_argument("--epoch", type=str, default="55", help="Learning Rate. Default=0.1")
parser.add_argument("--dis_mode_name", type=str, default="ADNI_central_imvar_real_vendor_v", help="model name")  # ResNet_ADNI_ASGD_4_sites_125 ADNI_single_site_in_real_reg
parser.add_argument("--csv_train", action='store_true', default = 'True', help='using csv to indicate the train set or not')
parser.add_argument("--train_file", type=str, default="train_net", help="Learning Rate. Default=0.1")
parser.add_argument("--crop", action='store_true', default='True', help="Learning Rate. Default=0.1")
parser.add_argument("--regression", action='store_true', default = False,  help="Indicating using regression model or not")
parser.add_argument("--segmentation", action='store_true', default = False,  help="Segmentation task")
parser.add_argument("--group_norm", action='store_true', default = False, help="Indicating using regression model or not")
parser.add_argument('--model', type=str, default="resnet18")
parser.add_argument('--dataloader', type=str, default="Boneage_Dataset")
parser.add_argument('--seed', type=int, default=random.randint(0, 9999999))
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data_dir', type=str, default="data")
parser.add_argument('--sites',type=int,default = 1,help = "how many sites")
parser.add_argument('--positive_percent',type=float,default = 0.5,help = "what fraction of training data is positive")
parser.add_argument('--train_size',type=int,default = 1000, help = "how much train data each round")
parser.add_argument('--rounds',type=int,default = 1,help = "how many training rounds")
parser.add_argument('--distillation_loss',type=str,default = 'no',help = "use distillation loss or not")
parser.add_argument('--epochs_per',type=int,default=10,help = 'how many epochs per round')
parser.add_argument('--model_save_path',type=str, default = 'model.pth',help = "where to save your model")
parser.add_argument('--split',type=int, default = 0,help = "which split to use")
parser.add_argument('--weighted_loss',type=str, default = 'no',help = "use weighted_loss or not")    
parser.add_argument('--class_incremental',type=str, default = 'no',help = "class_incremental training or not")

# FOLLOWING ARGUMENTS MUST BE THE SAME FOR ALL PARTICIPATING INSTITUTIONS DURING TRAINING

# SELECT TRUE FOR ONE OF THE FOLLOWING TO ADDRESS LABEL IMBALANCE


if __name__ == '__main__':
    opt = parser.parse_args()


    # Training settings
    def test_round(test_loader,net):
        full_pred = []
        full_labels = []
        r = nn.ReLU()
        for iteration, data in enumerate(test_loader):
            inputs = data['image'].cuda()
            labels = data['label'].cuda().cpu().data.numpy()
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


    labels = None

    opt.num_classes = 1
    tot_labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
              open(os.path.join(opt.data_path, 'total_labels.csv',))}
    print('NOTE------------Train with regression-----------')

    opt.tot_labels = tot_labels

    train_set = Boneage_Dataset('train', opt, 1,0)
    train_set_loader = DataLoader(train_set, batch_size=32,num_workers=8, pin_memory=True)
    test_final_loader = DataLoader(Boneage_Dataset('test_final_loader', opt,0,0), batch_size=32, num_workers=8, pin_memory=True)
    male_test = DataLoader(Boneage_Dataset('male_test', opt,0,0), batch_size=32, num_workers=8, pin_memory=True)
    female_test = DataLoader(Boneage_Dataset('female_test', opt,0,0), batch_size=32, num_workers=8, pin_memory=True)
    print('Loading total', len(train_set_loader), 'training images--------')
    print(opt.csv_train)

    #........................ set the model..................................
    # net = models.inception_v3(pretrained=True)
    # # net.fc = nn.Linear(2048, opt.num_classes)
    # else:
    #     if opt.group_norm:  ## if we use group-normalization
    #         from utils.resnet import resnet34

    #         net = resnet34(pretrained=False, **{'group_norm': 32, 'num_classes': 2})
    #         net = net.to(opt.device)
    #         # set_up_model(net, opt)
    #         net = net.train()

    net = models.resnet34(pretrained=True)
    net.fc = nn.Linear(512, 1)
    net.cuda()

    # net = models.resnet50(pretrained=True)
    # net.fc = nn.Linear(2048, opt.num_classes)
    # SEED = 35


    torch.cuda.manual_seed(opt.seed)
    # set_up_model(net, opt)
    net = net.train()


    criterion = torch.nn.L1Loss()

    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)
    running_loss = 0

    for epoch in range(0,60):
        mean1 = []
        std1 = []
        # if 'Bone' in opt.dis_mode_name and epoch == 60:
        #     criterion = torch.nn.MSELoss().to(tmp_device)

        for iteration, data in enumerate(train_set_loader):
            # print(iteration)
            inputs = data['image'].cuda()
            labels = data['label'].cuda()
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics

            # tot_epoch_loss += loss.item()
            running_loss += loss.item()
            current_epoch = epoch * len(train_set)/opt.batchSize + iteration

            current_epoch = epoch * len(train_set)/opt.batchSize + iteration
            # writer.add_scalar('Train/Loss', loss.item(), current_epoch)

            if iteration % 10 == 9:  # print every 2000 mini-batches
                # print(iteration)
                message = '(epoch: %d, iters: %d, time: %.3f, with loss: %.3f) ' % (epoch, iteration, time.time(), running_loss / 10)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, iteration + 1, running_loss / 10))
                running_loss = 0.0

        if epoch % opt.save_freq == 0 and epoch >20:
            # if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d) of learning rate %f' % (epoch, opt.lr ))
            # save_networks(net, epoch, opt.dis_mode_name)
            # save_networks(net, opt, 'epoch' + str(epoch) + '_' + 'all', opt.dis_mode_name)


        if epoch % 40 == 39:
            opt.lr = opt.lr / 10
            optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)

    torch.save(net, "qu_model.pth")

    full_pred, full_labels = test_round(test_final_loader, net)
    error, ci_low, ci_high = get_error(full_pred, full_labels)
    print("Round " + str(1) + " Error on Final Test: {} ({} - {})".format(error, ci_low, ci_high))

    full_pred, full_labels = test_round(male_test, net)
    error, ci_low, ci_high = get_error(full_pred, full_labels)
    print("Round " + str(1) + " Error on Male Test: {} ({} - {})".format(error, ci_low, ci_high))

    full_pred, full_labels = test_round(female_test, net)
    error, ci_low, ci_high = get_error(full_pred, full_labels)
    print("Round " + str(1) + " Error on Female Test: {} ({} - {})".format(error, ci_low, ci_high))




