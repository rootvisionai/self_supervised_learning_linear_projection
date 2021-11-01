# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 17:32:40 2021

@author: tekin.evrim.ozmermer
"""
from .MLP_ExactSolution import Model as mlpes
from .KNN import Model as knn
from sys import exit as EXIT

def load(cfg, backbone, dl_coll):
    if cfg.classifier == "MLP_ExactSolution":
        model_cls = mlpes(cfg)
        model_cls.create_collection(backbone, dl_coll = dl_coll, input_batch = None)
        model_cls.solve_exact()
    elif cfg.classifier == "KNN":
        if cfg.neighbors:
            pass
        else:
            cfg.neighbor = 8
        model_cls = knn(cfg)
        model_cls.create_collection(backbone, dl_coll)
    else:
        print("Classifier parameter given in the config.json is wrong.\n Choose one of KNN,MLP_ExactSolution")
        EXIT(0)
    return model_cls

        