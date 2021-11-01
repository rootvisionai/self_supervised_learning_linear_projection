# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:56:50 2021

@author: tekin.evrim.ozmermer
"""
from torchvision import transforms
import random
from PIL import Image, ImageOps, ImageFilter

class CustomRotation(object):
    def __init__(self, fill=128,
                 padding_mode=None):
        
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric',None]
        self.fill = fill
        self.padding_mode = padding_mode
    
    def __call__(self, img, angle):
        """
        Args:img (PIL Image): Image to be rotated.
        Returns:PIL Image: Padded(if padding_mode is not None) randomly rotated and centercropped image.
        """
        if self.padding_mode:
            centercrop_size = img.size
            pad_size_r = centercrop_size[1]//2
            pad_size_c = centercrop_size[0]//2
            img = transforms.functional.pad(img, (pad_size_r,pad_size_c), fill = self.fill, padding_mode = self.padding_mode)
        else:
            centercrop_size = img.size[0]/2
        img = transforms.functional.rotate(img, angle = angle)
        img = transforms.functional.center_crop(img, centercrop_size)
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + '(fill={0}, padding_mode={1})'.format(self.fill, self.padding_mode)
    
class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class TransformTrain:
    def __init__(self, input_size = 224, scale = (0.5, 1.0)):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=scale, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=scale, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2
    
class TransformEvaluate:
    def __init__(self, input_size = 224):
        self.transform = transforms.Compose([transforms.Resize(input_size),
                                             transforms.CenterCrop(input_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    def __call__(self, x):
        y = self.transform(x)
        return y