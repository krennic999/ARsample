import sys
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import argparse
from collections import OrderedDict
import hpsv2
import random
import pdb
import json
import torch
from torchvision import transforms
from src.transformer import Transformer2DModel
from src.pipeline import Pipeline
from src.scheduler import Scheduler
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import VQModel

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)

def transform_image(image, size=512):
    transform = transforms.Compose([
        transforms.Resize(size, max_size=None),  # 等比缩放，最小边对齐 size
        transforms.CenterCrop(size),  # 居中裁剪成 size x size
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ])
    return transform(image)

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """销毁分布式进程"""
    dist.destroy_process_group()
    
def deduplicate_annotations(meta_json):
    unique_annotations = {}
    for annotation in meta_json['annotations']:
        unique_annotations[annotation['image_id']] = annotation  # 覆盖旧的，保留最新的
    meta_json['annotations'] = list(unique_annotations.values())

def get_prompt_data_for_rank(hps_prompts, rank, world_size, batch_size):
    # 收集所有 prompt 条目
    all_items = []
    for prompt_type, prompt_list in hps_prompts.items():
        all_items.extend([
            {'caption': p, 'type': prompt_type, 'idx': i}
            for i, p in enumerate(prompt_list)
        ])

    total_samples = len(all_items)
    if rank == 0:
        print(f"Total prompt samples: {total_samples}")

    # 分配数据到每个 rank
    samples_per_rank = total_samples // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank if rank != world_size - 1 else total_samples
    rank_items = all_items[start_idx:end_idx]

    # 按 batch 分组，返回格式为 dict of lists
    batched_data = []
    for i in range(0, len(rank_items), batch_size):
        batch = rank_items[i:i+batch_size]
        batch_dict = {
            'caption': [x['caption'] for x in batch],
            'type': [x['type'] for x in batch],
            'idx': [x['idx'] for x in batch],
        }
        batched_data.append(batch_dict)

    return batched_data
    
    
def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    return new_state_dict

def main(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # seed=20
    batch_size=16
    enable_entropy_filtering=args.enable_entropy_filtering
    
    setup(rank, world_size)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # 多GPU时
    print('seed: ',args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    
    device = 'cuda'
    dtype = torch.bfloat16
    model_path = "/your-model-path"
    model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
    vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", torch_dtype=dtype)
    # text_encoder = CLIPTextModelWithProjection.from_pretrained(model_path,subfolder="text_encoder", torch_dtype=dtype)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(   #using original text enc for stable sampling
                "/your-ckpt/ckpts/laion/CLIP-ViT-H-14-laion2B-s32B-b79K",torch_dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", torch_dtype=dtype)
    scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler")
    pipe=Pipeline(vq_model, tokenizer=tokenizer,text_encoder=text_encoder,transformer=model,scheduler=scheduler)
    pipe = pipe.to(device)

    if rank == 0:
        print("VQ model loaded")

    coco_dataset='/your-coco-dataset'
    imsize=1024
    steps = 64
    CFG = 9
    negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"
    # json文件有四个keys，只有俩是有用的
    # images：list
    # {'license': 4, 'file_name': '000000397133.jpg', 'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg', 'height': 427, 'width': 640, 'date_captured': '2013-11-14 17:02:52', 'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg', 'id': 397133}
    # annotations：list
    # {'image_id': 179765, 'id': 38, 'caption': 'A black Honda motorcycle parked in front of a garage.'}
    with open(os.path.join(coco_dataset,'annotations/captions_val2017.json'),'r') as f: meta_json=json.load(f)
    all_prompts = hpsv2.benchmark_prompts('all')
    batched_data=get_prompt_data_for_rank(all_prompts,rank,world_size,batch_size)
    print('enable_entropy_filtering: ',enable_entropy_filtering)
    args.save_root=os.path.join(args.save_root,'seed%d_entropy%s'%(args.seed,args.enable_entropy_filtering))

    # 处理数据集
    for item in batched_data:
        t2=time.time()
        print(item)
        text_prompts = item['caption']
        type = item['type']
        name = item['idx']
        print('rank %d: '%rank,type)

        samples = pipe(
            prompt=text_prompts, 
            negative_prompt=[negative_prompt] * len(text_prompts),
            height=imsize,
            width=imsize,
            guidance_scale=CFG,
            enable_entropy_filtering=enable_entropy_filtering,
            num_inference_steps=steps
            ).images
        
        for i, (fname, style, text_prompt) in enumerate(zip(name, type, text_prompts)):
            img_pred = samples[i]

            try:
                os.makedirs(os.path.join(args.save_root, style), exist_ok=True)
                img_pred.save(os.path.join(args.save_root, style, f"{fname:05d}.jpg"))
            except Exception as e:
                log_path = os.path.join("log.txt")  # 定义日志文件路径
                error_msg = f"Failed to save {fname}, error: {str(e)}\n"
                print(error_msg)
                with open(log_path, "a") as log_file:
                    log_file.write(error_msg)

            print(f"Image {fname} saved. to {args.save_root}")

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--enable_entropy_filtering", type=bool, default=False, help="entropy to sample with")
    parser.add_argument("--save_root", type=str, default='/your-save-root')
    parser.add_argument("--seed", type=int, default=20)
    args = parser.parse_args()
    main(args)