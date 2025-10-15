import sys
sys.path.append('..')

import json
import traceback
import glob
import math
import os
import random
from io import BytesIO
import pathlib

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode, transforms
from dataset.augmentation import random_crop_arr,center_crop_arr
import os,glob 
import numpy as np
from PIL import Image
import re
import pdb
import logging
# import cv2 
# print(cv2.__file__)

def extract_selected_classes(file_path):
    categories = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                first_column = line.split()[0]  # Split by spaces and take the first element
                categories.append(first_column)
    return categories

logging.basicConfig(level=logging.INFO)

def find_consecutive(string):
    pattern = r'(\w)\1{4,}'
    result = re.search(pattern, string)
    if result:
        return True
    else:
        return False 
    
# random_crop_arr(pil_image, img_size)
def get_transform_new(img_size,crop_range=1.1):
    crop_size = int(img_size * crop_range)#crop_range=1.1
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    return transform


def get_resize_height_width(img_h, img_w, bucket_h, bucket_w):
    new_h = bucket_h 
    new_w = int(new_h/img_h*img_w)
    if new_w < bucket_w:
        new_w_ = bucket_w
        new_h_ = int(new_w_/new_w*new_h)
        res = (new_w_,new_h_)
    else:
        res = (new_w,new_h)
    return res 


class ImgDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        imgnet_root,
        imgnet_subdir,
        selected_file_list,
        resolution=256,
        classifier_free_training_prob=0.1,
    ):

        self.classifier_free_training_prob = classifier_free_training_prob
        self.selected_classes=None
        print('selected classes: ',self.selected_classes)
        print("classifier-free training prob: ",classifier_free_training_prob)

        self.transform=get_transform_new(img_size=resolution)
        train_images = []
        train_classes = []
        with open(os.path.join(imgnet_root,selected_file_list), 'r') as file:
            for line in file:
                curr_class=int(line.strip().split(' ')[-1])
                cur_file_key=line.strip().split(' ')[0]
                if self.selected_classes!=None:
                    if curr_class in self.selected_classes:
                        train_images.append(cur_file_key)
                        train_classes.append(curr_class)
                else:
                    train_images.append(cur_file_key)
                    train_classes.append(curr_class)
        self.train_images=train_images
        self.train_classes=train_classes
        self.negative_tag=len(set(train_classes))
        self.imgnet_root=os.path.join(imgnet_root,imgnet_subdir)

        self.num_instance_images = len(train_images)
        print('==>effective images', self.num_instance_images)
        self._length = self.num_instance_images


    def __len__(self):
        return self._length
    

    def get_image_from_file(self,file_key):
        image_path=os.path.join(self.imgnet_root,file_key)
        return Image.open(image_path)


    def __getitem__(self, index):
        example = {}
        file_key = self.train_images[index]

        instance_image = self.get_image_from_file(file_key)
        # print('instance image size: ',instance_image.size)
       
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        img_w, img_h = instance_image.size
        class_B = self.train_classes[index]

        if self.classifier_free_training_prob > 0.0 and random.random() < self.classifier_free_training_prob:
            class_B = self.negative_tag

        example["class_B"] = class_B
        example["file_key"] = file_key

        instance_image = self.transform(instance_image)


        example["image"] = instance_image
        return example
