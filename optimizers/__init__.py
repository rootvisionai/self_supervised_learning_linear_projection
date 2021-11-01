# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 13:54:04 2021

@author: tekin.evrim.ozmermer
"""
import torch
from .lars import LARS
from sys import exit as EXIT
import math

def load(cfg, param_groups):
    
    if cfg.optimizer == "lars":
        opt = LARS(param_groups, lr=float(cfg.learning_rate),
                   weight_decay=cfg.weight_decay,
                   weight_decay_filter=True,
                   lars_adaptation_filter=True)
    elif cfg.optimizer == "sgd":
        opt = torch.optim.SGD(param_groups, lr = float(cfg.learning_rate),
                              weight_decay = cfg.weight_decay,
                              momentum = 0.9,
                              nesterov=True)
    elif cfg.optimizer == 'adam': 
        opt = torch.optim.Adam(param_groups, lr = float(cfg.learning_rate),
                               weight_decay = cfg.weight_decay)
        
        
    elif cfg.optimizer == 'rmsprop':
        opt = torch.optim.RMSprop(param_groups, lr = float(cfg.learning_rate),
                                  alpha=0.9,
                                  weight_decay = cfg.weight_decay, momentum = 0.9)
        
    elif cfg.optimizer == 'adamw':
        opt = torch.optim.AdamW(param_groups, lr = float(cfg.learning_rate))
        
        
    else:
        print('Given optimizer is wrong. Choose one of lars, sgd, adam, rmsprop, adamw')
        EXIT(0)
        
    return opt

def adjust_learning_rate(cfg, optimizer, loader, step, scheduler = None):
    if cfg.optimizer == "lars":
        max_steps = cfg.epochs * len(loader)
        warmup_steps = 10 * len(loader)
        base_lr = cfg.batch_size / 256
        if step < warmup_steps:
            lr = base_lr * step / warmup_steps
        else:
            step -= warmup_steps
            max_steps -= warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
            end_lr = base_lr * 0.001
            lr = base_lr * q + end_lr * (1 - q)
        optimizer.param_groups[0]['lr'] = lr * cfg.learning_rate_weights
        optimizer.param_groups[1]['lr'] = lr * cfg.learning_rate_biases
    else:
        scheduler.step()