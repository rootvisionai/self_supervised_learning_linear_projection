# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 11:33:23 2021

@author: tekin.evrim.ozmermer
"""

import torch
import tqdm

class Model(torch.nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        
        self.cfg = cfg
        self.nb_classes = cfg.nb_classes
        self.linear = torch.nn.Linear(in_features  = cfg.embedding_size,
                                      out_features = cfg.batch_size, bias = False)
        self.labels = None
    
    def binarize_label(self, labels_int):
        labels_bin = torch.zeros(labels_int.shape[0], self.nb_classes)
        for i, x in enumerate(labels_int):
            labels_bin[i,x] = 1
        return labels_bin
    
    def create_collection(self, backbone, dl_coll = None, input_batch = None):
        if dl_coll is not None:
            backbone.eval()
            pbar = tqdm.tqdm(enumerate(dl_coll))
            for step, (x, labels_int) in pbar:
                with torch.no_grad():
                    if step == 0:
                        collection = backbone(x.to(self.cfg.device)).cpu()
                        labels_bin = self.binarize_label(labels_int)
                    else:
                        collection = torch.cat((collection,
                                                backbone(x.to(self.cfg.device)).cpu()), dim=0)
                        labels_bin = torch.cat((labels_bin,
                                                self.binarize_label(labels_int)), dim=0)
                text = "[{}/{} ({:.1f}%)]".format(step, len(dl_coll), 100*step/len(dl_coll))
                pbar.set_description(text)
            
            self.collection = collection
            self.labels_bin = labels_bin
            backbone.train()
        else:
            self.collection = input_batch.to(self.cfg.device).detach()
            self.labels_bin = torch.diag_embed(torch.ones(self.collection.shape[0],
                                                          self.collection.shape[0]))[0].to(self.cfg.device)
            self.labels = self.labels_bin.argmax(dim=0)
            
    def l2_norm(self, x):
        input_size = x.size()
        buffer = torch.pow(x, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        _output = torch.div(x, norm.view(-1, 1).expand_as(x))
        output = _output.view(input_size)
        return output
    
    def solve_exact(self):
        if self.labels is None:          
            with torch.no_grad():
                FCpi = torch.pinverse(self.l2_norm(self.collection))
                self.W = torch.matmul(FCpi, self.labels_bin)
                self.linear.weight = torch.nn.Parameter(self.W.T)
        else:
            FCpi = torch.pinverse(self.l2_norm(self.collection))
            self.W = torch.matmul(FCpi, self.labels_bin)
            self.linear.weight = torch.nn.Parameter(self.W.T)
            
    def forward(self, embedding):
        if self.labels is None:
            with torch.no_grad():
                if len(embedding.shape)<2:
                    embedding = embedding.unsqueeze(0)
                embedding_norm = self.l2_norm(embedding)
                out = self.linear(embedding_norm)
                out = torch.nn.functional.softmax(out, dim=1)
            
                pred = out.argmax().cpu()
                return pred
        else:
            if len(embedding.shape)<2:
                embedding = embedding.unsqueeze(0)
            embedding_norm = self.l2_norm(embedding)
            out = self.linear(embedding_norm)
            out = torch.where(out>1, 2-out, out)
            out = torch.nn.functional.softmax(out, dim=1)
            return out
    
    def calculate_loss(self,embedding):
        if len(embedding.shape)<2:
            embedding = embedding.unsqueeze(0)
        embedding_norm = self.l2_norm(embedding)
        out = self.linear(embedding_norm)
        loss = torch.nn.functional.cross_entropy(input = out, target = self.labels)
        return loss