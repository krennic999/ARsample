# import datasets as hf_datasets
from torchvision import transforms
import pdb
import numpy as np
import random
import torch
import json
import os
from PIL import Image
import shutil

def deduplicate_annotations(meta_json):
    unique_annotations = {}
    for annotation in meta_json['annotations']:
        unique_annotations[annotation['image_id']] = annotation
    meta_json['annotations'] = list(unique_annotations.values())

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)

def transform_image(image, size=512):
    transform = transforms.Compose([
        transforms.Resize(size, max_size=None),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ])
    return transform(image)

def batched_iterator_coco17val(data_root, batch_size, select_size, seed=42, imsize=512):
    random.seed(seed)
    with open(os.path.join(data_root,'annotations/captions_val2017.json'),'r') as f: meta_json=json.load(f)
    deduplicate_annotations(meta_json)
    total_samples = len(meta_json['annotations'])
    print('total samples: ',total_samples)

    images = []
    captions = []
    names=[]
    for item_annotations in meta_json['annotations']:
        impath=os.path.join(data_root,'val2017',f"{str(item_annotations['image_id']).zfill(12)}.jpg")
        images.append(transform_image(Image.open(impath).convert("RGB"),size=imsize))
        captions.append(item_annotations['caption'])
        names.append(f"{str(item_annotations['image_id']).zfill(12)}.jpg")

        if len(captions) == batch_size:
            indices = random.sample(range(batch_size), select_size)
            selected_images = [images[i] for i in indices]
            selected_captions = [captions[i] for i in indices]
            selected_names = [names[i] for i in indices]
            yield {'image': torch.stack(selected_images, dim=0), 'caption': selected_captions, 'name':selected_names}

            images = []
            captions = []
            names = []

    if images:
        if len(images) >= select_size:
            indices = random.sample(range(len(images)), select_size)
            selected_images = [images[i] for i in indices]
            selected_captions = [captions[i] for i in indices]
            selected_names = [names[i] for i in indices]
            yield {'image': torch.stack(selected_images, dim=0), 'caption': selected_captions, 'name':selected_names}
        else:
            yield {'image': torch.stack(images, dim=0), 'caption': captions, 'name':names}



def batched_iterator_MJHQ(data_root, batch_size, select_size, category=[], seed=42, imsize=512):
    random.seed(seed)
    with open(os.path.join(data_root,'meta_data.json'),'r') as f: meta_json=json.load(f)
    if len(category)>0:
        meta_json={key:value for key,value in meta_json.items() if value['category'] in category}

    images = []
    captions = []
    names=[]
    categories=[]
    for item_name,item in meta_json.items():
        impath=os.path.join(data_root,'data_meta',item['category'],'%s.jpg'%item_name)
        images.append(transform_image(Image.open(impath),size=imsize))
        captions.append(item['prompt'])
        names.append(item_name)
        categories.append(item['category'])

        if len(captions) == batch_size:
            indices = random.sample(range(batch_size), select_size)
            selected_images = [images[i] for i in indices]
            selected_captions = [captions[i] for i in indices]
            selected_names = [names[i] for i in indices]
            selected_categories = [categories[i] for i in indices]
            yield {'image': torch.stack(selected_images, dim=0), 'caption': selected_captions, 'name':selected_names, 'category':selected_categories}

            images = []
            captions = []
            names = []
            categories = []

    if images:
        if len(images) >= select_size:
            indices = random.sample(range(len(images)), select_size)
            selected_images = [images[i] for i in indices]
            selected_captions = [captions[i] for i in indices]
            selected_names = [names[i] for i in indices]
            selected_categories = [categories[i] for i in indices]
            yield {'image': torch.stack(selected_images, dim=0), 'caption': selected_captions, 'name':selected_names, 'category':selected_categories}
        else:
            yield {'image': torch.stack(images, dim=0), 'caption': captions, 'name':names, 'category':categories}


def batched_iterator(dataset, batch_size, select_size, seed=42):
    random.seed(seed)
    images = []
    captions = []
    for item in dataset:
        if item['image'].mode == 'RGB':
            images.append(transform_image(item['image']))
            captions.append(item['caption'])

            if len(captions) == batch_size:
                indices = random.sample(range(batch_size), select_size)
                selected_images = [images[i] for i in indices]
                selected_captions = [captions[i] for i in indices]

                yield {'image': torch.stack(selected_images, dim=0), 'caption': selected_captions}

                images = []
                captions = []

    if images:
        if len(images) >= select_size:
            indices = random.sample(range(len(images)), select_size)
            selected_images = [images[i] for i in indices]
            selected_captions = [captions[i] for i in indices]
            yield {'image': torch.stack(selected_images, dim=0), 'caption': selected_captions}
        else:
            yield {'image': torch.stack(images, dim=0), 'caption': captions}