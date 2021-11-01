# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 21:29:35 2021

@author: tekin.evrim.ozmermer
"""

from argparse import Namespace
import json

def load(config_path = "./config/config.json"):
    with open(config_path, "r") as fp:
        config_obj = json.load(fp, object_hook = lambda x: Namespace(**x))
    return config_obj