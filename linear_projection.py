# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 20:16:44 2021

@author: tekin.evrim.ozmermer
"""
import torchvision
import torch
from torch import nn

from classifiers.MLP_ExactSolution import Model as mlpes

class LinearProjection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.model == "resnet18":
            self.model = torchvision.models.resnet18(zero_init_residual=True)#pretrained=True)#
        elif cfg.model == "resnet34":
            self.model = torchvision.models.resnet34(zero_init_residual=True)#pretrained=True)#
        elif cfg.model == "resnet50":
            self.model = torchvision.models.resnet50(zero_init_residual=True)#pretrained=True)#
        elif cfg.model == "resnet101":
            self.model = torchvision.models.resnet101(zero_init_residual=True)#pretrained=True)#
        else:
            print("Model architecture is given wrong, default is being used\n DEFAULT: RESNET50")
            self.model = torchvision.models.resnet50(pretrained=True)
        
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)
        
        self.model.embedding = nn.Sequential(nn.Linear(self.model.fc.in_features,
                                                       cfg.embedding_size, bias = False))
        
        self.linear_projection = mlpes(cfg)
        
    def forward_conv_layers(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x

    def forward_pooling(self, x):
        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)
        return avg_x+max_x
    
    def flatten(self, x):
        return x.view(x.size(0), -1)
    
    def l2_norm(self, x):
        input_size = x.size()
        buffer = torch.pow(x, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp).detach()
        _output = torch.div(x, norm.view(-1, 1).expand_as(x))
        output = _output.view(input_size)
        return output
    
    def criterion_negative(self, sims, alpha, mrg):
        shape = sims.shape[0]
        neg_exp_sum = torch.exp(alpha * (sims + mrg))
        neg_term = torch.log(1 + neg_exp_sum).sum()/shape
        return neg_term
    
    def criterion_positive(self, sims, alpha, mrg):
        shape = sims.shape[0]
        pos_exp_sum = torch.exp(-alpha * (sims - mrg))
        pos_term = torch.log(1 + pos_exp_sum).sum()/shape
        return pos_term
    
    def forward(self, x):
        if type(x) == tuple:
            x0 = self.forward_conv_layers(x[0])
            x0 = self.forward_pooling(x0)
            x0 = self.flatten(x0)
            z0 = self.model.embedding(x0)
            
            x1 = self.forward_conv_layers(x[1])
            x1 = self.forward_pooling(x1)
            x1 = self.flatten(x1)
            z1 = self.model.embedding(x1)
            
            # calculate loss
            self.linear_projection.create_collection(backbone = None,
                                                     dl_coll = None, input_batch = z0)
            self.linear_projection.solve_exact()
            loss = self.linear_projection.calculate_loss(z1)
            return loss
        
        else:
            x = self.forward_conv_layers(x)
            x = self.forward_pooling(x)
            x = self.flatten(x)
            z = self.model.embedding(x)
            
            return z
    
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()