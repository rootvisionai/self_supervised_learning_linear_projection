# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 11:35:32 2021

@author: tekin.evrim.ozmermer
"""
import torch
import numpy as np
import tqdm

class Model():
    def __init__(self, cfg):        
        self.cfg = cfg
        self.nb_classes = cfg.nb_classes
        self.K = cfg.neighbors
    
    def binarize_label(self, labels_int):
        labels_bin = torch.zeros(labels_int.shape[0], self.nb_classes)
        for i, x in enumerate(labels_int):
            labels_bin[i,x] = 1
        return labels_bin
    
    def create_collection(self, backbone, dl_coll):
        backbone.eval()
        pbar = tqdm.tqdm(enumerate(dl_coll))
        for step, (x, label_int) in pbar:
            with torch.no_grad():
                if step == 0:
                    collection = backbone(x.to(self.cfg.device)).cpu()
                    labels_bin = self.binarize_label(label_int)
                    labels_int = label_int
                else:
                    collection = torch.cat((collection,
                                            backbone(x.to(self.cfg.device)).cpu()), dim=0)
                    labels_bin = torch.cat((labels_bin,
                                            self.binarize_label(label_int)), dim=0)
                    labels_int = torch.cat((labels_int,
                                            label_int), dim=0)
            text = "[{}/{} ({:.1f}%)]".format(step, len(dl_coll), 100*step/len(dl_coll))
            pbar.set_description(text)
        
        self.collection = collection.cpu()
        self.labels_bin = labels_bin
        self.labels_int = labels_int
        backbone.train()
            
    def l2_norm(self, x):
        input_size = x.size()
        buffer = torch.pow(x, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        _output = torch.div(x, norm.view(-1, 1).expand_as(x))
        output = _output.view(input_size)
        return output
    
    def forward(self,embedding):
        if len(embedding.shape)<2:
            embedding = embedding.unsqueeze(0)
        embedding_norm = self.l2_norm(embedding)
        
        cos_sim = torch.nn.functional.linear(self.l2_norm(self.collection), embedding_norm.squeeze(0))
        cos_sim_topK = cos_sim.topk(1 + self.K)
        
        indexes = cos_sim_topK[1][1:self.K+1].numpy().tolist()
        
        preds_int = np.array([self.labels_int[i] for i in indexes])
        unqs, counts = np.unique(preds_int, return_counts = True)
        pred = unqs[np.argmax(counts)]
        
        return pred