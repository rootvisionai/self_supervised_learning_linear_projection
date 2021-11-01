# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 18:14:46 2021

@author: tekin.evrim.ozmermer
"""
import torch
import tqdm
import os

def evaluate(cfg, model, model_cls, dl_ev):
    model.eval()
    
    if cfg.classifier == "MLP_ExactSolution":
        model_cls = model_cls.to(cfg.device)
        model_cls.eval()
    
    max_label = 0
    results = []
    
    pbar = tqdm.tqdm(enumerate(dl_ev))    
    for step, (x, labels_int) in pbar:
        if step>cfg.test_range and cfg.test_range != 0:
            break
        with torch.no_grad():
            embedding = model(x.to(cfg.device)).cpu()
            if cfg.classifier == "MLP_ExactSolution":
                out = model_cls(embedding)
            elif cfg.classifier == "KNN":
                out = model_cls.forward(embedding)
            results.append({"target": labels_int, "prediction": out,
                            "outcome": labels_int == out})
            text = "[{}/{} ({:.1f}%)]".format(step, len(dl_ev), 100*step/len(dl_ev))
            pbar.set_description(text)
            if labels_int>max_label:
                max_label = labels_int

    
    trues  = len([elm for elm in results if elm["outcome"]])
    falses = len([elm for elm in results if not elm["outcome"]])
    precision = trues/(trues+falses)
    
    return precision

def save_best(cfg, last_result, results, package):
    last_result = last_result*100
    try:
        os.mkdir("best_models")
    except:
        pass
    print("---", results.values(), "---")
    print("---", last_result, "---")
    # try:
    
    if results != {}:
        if last_result>max(results.values()):
            torch.save({'model_state_dict': package["model"].state_dict(),
                        'optimizer': package["opt"].state_dict(),
                        'epoch': package["epoch"]},
                       './best_models/{}.pth'.format(cfg.resume.split("/")[-1]))
            print("Saved model with precision: ", last_result)
    else:
        torch.save({'model_state_dict': package["model"].state_dict(),
                    'optimizer': package["opt"].state_dict(),
                    'epoch': package["epoch"]},
                    './best_models/{}.pth'.format(cfg.resume.split("/")[-1]))
        print("Saved model with precision: ", last_result)
        
def generate_embedding_sets(cfg, model, dl_coll, dl_ev):
    model.eval()
    
    def generate_embeddings(data_loader):
        pbar = tqdm.tqdm(enumerate(data_loader))  
        for step, (x, label_int) in pbar:
            with torch.no_grad():
                if step == 0:
                    collection = model(x.to(cfg.device)).cpu()
                    labels_int = label_int
                else:
                    collection = torch.cat((collection,
                                            model(x.to(cfg.device)).cpu()), dim=0)
                    labels_int = torch.cat((labels_int,
                                            label_int), dim=0)
            text = "[{}/{} ({:.1f}%)]".format(step, len(data_loader), 100*step/len(data_loader))
            pbar.set_description(text)
        
        return collection, labels_int
    
    def save_embedding_set(trn_collection, trn_labels_int, test_collection, test_labels_int):
        try:
            os.mkdir("dataset_embeddings")
        except:
            pass
        
        torch.save({'model_state_dict': model.cpu().state_dict(),
                    'train_embeddings': trn_collection.cpu(),
                    'train_labels': trn_labels_int.cpu(),
                    'test_embeddings': test_collection.cpu(),
                    'test_labels': test_labels_int.cpu()},
                    './dataset_embeddings/{}.pth'.format(cfg.resume.split("/")[-1]))
        print("Saved --->",'./dataset_embeddings/{}.pth'.format(cfg.resume.split("/")[-1]))   
    
    trn_collection, trn_labels_int = generate_embeddings(dl_coll)
    test_collection, test_labels_int = generate_embeddings(dl_ev)
    save_embedding_set(trn_collection, trn_labels_int, test_collection, test_labels_int)

    model.train()
    model.to(cfg.device)
    
    