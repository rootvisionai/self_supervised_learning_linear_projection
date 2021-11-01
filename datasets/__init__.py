# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 20:19:52 2021

@author: tekin.evrim.ozmermer
"""
import torch 
from torchvision import datasets
from augmentations import TransformTrain, TransformEvaluate

def load(cfg):
    if cfg.dataset == "cifar10":
        dl_tr, dl_ev, dl_coll = cifar10(cfg)
        
    elif cfg.dataset == "cifar100":
        dl_tr, dl_ev, dl_coll = cifar100(cfg)
        
    return dl_tr, dl_ev, dl_coll

def cifar10(cfg):
    data_root = cfg.data_root
    input_size = cfg.input_size
    batch_size = cfg.batch_size
    
    trn_dataset = datasets.CIFAR10(root=data_root,
                                   train=True,
                                   transform=TransformTrain(input_size = input_size),
                                   download=True)
    
    dl_tr = torch.utils.data.DataLoader(trn_dataset,
                                        batch_size = batch_size,
                                        shuffle = True,
                                        num_workers = 0,
                                        drop_last = True,
                                        pin_memory = True)
    
    # Build evaluation set <<
    ev_dataset = datasets.CIFAR10(root=data_root,
                                  train=False,
                                  transform=TransformEvaluate(input_size = input_size),
                                  download=True)
    
    dl_ev = torch.utils.data.DataLoader(ev_dataset,
                                        batch_size = 1,
                                        shuffle = False,
                                        num_workers = 0,
                                        pin_memory = True)
    
    # Build collection set <<
    collection_dataset = datasets.CIFAR10(root=data_root,
                                          train=True,
                                          transform=TransformEvaluate(input_size = input_size),
                                          download=True)
    
    dl_coll = torch.utils.data.DataLoader(collection_dataset,
                                          batch_size = 32,
                                          shuffle = False,
                                          num_workers = 0,
                                          pin_memory = True)
    
    return dl_tr, dl_ev, dl_coll

def cifar100(cfg):
    data_root = cfg.data_root
    input_size = cfg.input_size
    batch_size = cfg.batch_size
    
    trn_dataset = datasets.CIFAR10(root=data_root,
                                   train=True,
                                   transform=TransformTrain(input_size = input_size),
                                   download=True)
    
    dl_tr = torch.utils.data.DataLoader(trn_dataset,
                                        batch_size = batch_size,
                                        shuffle = True,
                                        num_workers = 0,
                                        drop_last = True,
                                        pin_memory = True)
    
    # Build evaluation set <<
    ev_dataset = datasets.CIFAR10(root=data_root,
                                  train=False,
                                  transform=TransformEvaluate(input_size = input_size),
                                  download=True)
    
    dl_ev = torch.utils.data.DataLoader(ev_dataset,
                                        batch_size = batch_size,
                                        shuffle = False,
                                        num_workers = 0,
                                        pin_memory = True)
    
    # Build collection set <<
    collection_dataset = datasets.CIFAR10(root=data_root,
                                          train=True,
                                          transform=TransformEvaluate(input_size = input_size),
                                          download=True)
    
    dl_coll = torch.utils.data.DataLoader(collection_dataset,
                                        batch_size = batch_size,
                                        shuffle = False,
                                        num_workers = 0,
                                        pin_memory = True)
    
    return dl_tr, dl_ev, dl_coll