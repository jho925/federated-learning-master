import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
import numpy as np
from torch.utils.data import DataLoader
    
class ElasticWeightConsolidation:

    def __init__(self, model, crit, lr=0.001, weight=1000000):
        self.model = model
        self.weight = weight
        self.crit = crit
        self.optimizer = optim.Adam(self.model.parameters(), lr)
    

    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())

    def loss_fn_kd(self,outputs, labels, teacher_outputs):

        alpha = 0.5
        T = 3
        KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                                 F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
                  F.cross_entropy(outputs, labels) * (1. - alpha)

        return KD_loss

    # def _update_fisher_params(self, ds,batch_size):
    #     dl = DataLoader(ds, batch_size, shuffle=True)



    #     log_liklihoods = 0
    #     for i,data in enumerate(dl):
    #         inputs = data['image'].cuda()
    #         target = data['label'].cuda()
    #         # if len(data) != 8:
    #         #     continue
            
    #         output = F.log_softmax(self.model(inputs), dim=1)
    #         if len(output[:, target]) != 8:
    #             continue
    #         log_liklihoods = ((i+1) * log_liklihoods + output[:, target]) / (i+2)     
    #     log_likelihood = torch.cat(log_liklihoods).mean()
    #     grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters(),allow_unused=True)
    #     _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
    #     for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
    #         if param == None:
    #             continue
    #         self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)

    def _update_fisher_params(self, current_ds, batch_size, num_batch):
        dl = DataLoader(current_ds, batch_size, shuffle=True)
        log_liklihoods = []
        for i, (input, target) in enumerate(dl):
            if i > num_batch:
                break
            output = F.log_softmax(self.model(input), dim=1)
            log_liklihoods.append(output[:, target])
        log_likelihood = torch.cat(log_liklihoods).mean()
        grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters())
        _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)

    def register_ewc_params(self, ds,batch_size):
        self._update_fisher_params(ds,batch_size)
        self._update_mean_params()

    def _compute_consolidation_loss(self, weight):
        try:
            losses = []
            for param_name, param in self.model.named_parameters():
                _buff_param_name = param_name.replace('.', '__')
                estimated_mean = getattr(self.model, '{}_estimated_mean'.format(_buff_param_name))
                estimated_fisher = getattr(self.model, '{}_estimated_fisher'.format(_buff_param_name))
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
            return (weight / 2) * sum(losses)
        except AttributeError:
            return 0

    def forward_backward_update(self,input,target,train_loader,batch_size,lr,epoch,iteration,prev_model,rounds):
        output = self.model(input)
        if rounds >= 1: 
            teacher_outputs = prev_model(input)   
            loss = self._compute_consolidation_loss(self.weight) + self.loss_fn_kd(output, target, teacher_outputs)
        else:
            loss = self._compute_consolidation_loss(self.weight) + self.crit(output,target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        print("\r[Epoch %2d][Step %4d/%4d] Loss: %.4f, Lr: %.2e" % (
                  epoch+1,
                  iteration,
                  int(len(train_loader.dataset) / batch_size),
                  loss / batch_size,
                  lr,# *[group['lr'] for group in optim.param_groups],
              ), end='          ')

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)