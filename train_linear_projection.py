# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 21:20:51 2021

@author: tekin.evrim.ozmermer
"""

from linear_projection import LinearProjection
import datasets
import optimizers
import classifiers
from utils import *

import config

import torch
import os
import tqdm
import json

cfg = config.load("./config/config_linear_projection.json")
start_epoch = 0
cfg.resume = "./checkpoints/{0}-{1}-{2}-{3}-{4}-{5}".format(cfg.__dict__["dataset"],
                                                        "LinearProjection",
                                                        cfg.__dict__["model"],
                                                        cfg.__dict__["embedding_size"],
                                                        cfg.__dict__["input_size"],
                                                        cfg.__dict__["optimizer"])
# import dataset
os.chdir("datasets")
cfg.data_root = os.getcwd()
dl_tr, dl_ev, dl_coll = datasets.load(cfg)

# import model
model = LinearProjection(cfg)
model.to(cfg.device)
model.train()

# import optimizer
if cfg.optimizer == "lars":
    param_weights = []
    param_biases = []
    for param in model.model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    param_groups = [{'params': param_weights}, {'params': param_biases}]
    opt = optimizers.load(cfg, param_groups)
    scheduler = None
else:
    param_groups = model.model.parameters()
    opt = optimizers.load(cfg, param_groups)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.learning_rate_decay_interval, gamma=cfg.learning_rate_decay)

os.chdir("..")
# resume
if os.path.isfile("{}.pth".format(cfg.resume)):
    print('=> loading checkpoint:\n{}.pth'.format(cfg.resume))
    checkpoint = torch.load("{}.pth".format(cfg.resume),torch.device(cfg.device))
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict = True)
    except:
        model.model.load_state_dict(checkpoint['model_state_dict'], strict = False)
    opt.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']

# model_cls = classifiers.load(cfg, model, dl_coll)
# precision = evaluate(cfg, model, model_cls, dl_ev)

# print("Precision: {:.2f}".format(precision*100))

# try:
#     with open("./results/{}.json".format(cfg.resume.split("/")[-1]), "r") as fp:
#         last_results = json.load(fp)
# except:
#     last_results = {}
    
# last_results["{}".format(-1)] = float("{:.2f}".format(precision*100))
# with open("./results/{}.json".format(cfg.resume.split("/")[-1]), "w") as fp:
#     json.dump(last_results, fp, indent = 2)

for epoch in range(start_epoch, cfg.epochs):
    pbar = tqdm.tqdm(enumerate(dl_tr, start = 1))
    for step, ((y1, y2), _) in pbar:
        for _ in range(1):
            
            try:
                del model.linear_projection.collection
                del model.linear_projection.labels
                del model.linear_projection.labels_int
                torch.cuda.empty_cache()
            except:
                pass
            
            if cfg.optimizer == "lars":
                optimizers.adjust_learning_rate(cfg, opt, dl_tr, step, scheduler)
            
            y1 = y1.to(cfg.device)
            y2 = y2.to(cfg.device)
            
            opt.zero_grad()
            loss = model.forward((y1, y2))
            loss.backward()
            opt.step()
            pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, step + 1, len(dl_tr),
                    100. * (step+1) / len(dl_tr),
                    loss.item()))

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer': opt.state_dict(),
                'epoch': epoch},
               '{}.pth'.format(cfg.resume))

    if (epoch+1)%cfg.evaluation_interval == 0:
        model_cls = classifiers.load(cfg, model, dl_coll)
        precision = evaluate(cfg, model, model_cls, dl_ev)
        del model_cls
        torch.cuda.empty_cache()
        print("Precision: {:.2f}".format(precision*100))
        
        try:
            with open("./results/{}.json".format(cfg.resume.split("/")[-1]), "r") as fp:
                last_results = json.load(fp)
        except:
            last_results = {}

        save_best(cfg,
                  last_result = precision,
                  results = last_results, 
                  package = {"model": model, "opt": opt, "epoch": epoch})

        last_results["{}".format(epoch)] = float("{:.2f}".format(precision*100))
        with open("./results/{}.json".format(cfg.resume.split("/")[-1]), "w") as fp:
            json.dump(last_results, fp, indent = 2)
    
    if cfg.optimizer != "lars" and epoch<100:
        scheduler.step()

generate_embedding_sets(cfg, model, dl_coll, dl_ev)


        
        
