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
import random
import datasets as hf_datasets
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
    os.environ['MASTER_PORT'] = '29500'
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

def get_data_for_rank(dataset, rank, world_size, batch_size, imsize):
    """
    dataset: IterableDataset，元素是 dict，包含 'image' (PIL.Image), 'caption' (str)
    自动为每个样本生成唯一 name（5位整数字符串）

    返回：list，每个元素是 dict，包含 'image' (List[Tensor]), 'caption' (List[str]), 'name' (List[str])
    """
    images, captions, names = [], [], []
    batches = []
    local_idx = 0  # 每个 rank 内的样本计数

    for idx, item in enumerate(dataset):
        if idx % world_size != rank:
            continue  # 只处理当前 rank 的数据

        if item['image'].mode != 'RGB':
            continue  # 忽略非 RGB 图片

        img_tensor = transform_image(item['image'], size=imsize)
        images.append(img_tensor)
        captions.append(item['caption'])
        names.append(f"{rank:02d}{local_idx:05d}.png")  # 全局唯一 name
        local_idx += 1

        if len(images) == batch_size:
            batches.append({
                'image': images,
                'caption': captions,
                'name': names
            })
            images, captions, names = [], [], []

    # 收尾：最后一批不足 batch_size 也要保存
    if images:
        batches.append({
            'image': images,
            'caption': captions,
            'name': names
        })

    return batches
    
    
def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    return new_state_dict

def main(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    seed=20
    batch_size=16
    enable_entropy_filtering=args.enable_entropy_filtering
    
    setup(rank, world_size)
    torch.manual_seed(seed)
    random.seed(seed)
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

    coco_dataset = hf_datasets.load_dataset("/your-coco14-root/coco14val", split="train", streaming=True)
    imsize=1024
    steps = 64
    CFG = args.cfg_scale
    negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"
    # json文件有四个keys，只有俩是有用的
    # images：list
    # {'license': 4, 'file_name': '000000397133.jpg', 'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg', 'height': 427, 'width': 640, 'date_captured': '2013-11-14 17:02:52', 'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg', 'id': 397133}
    # annotations：list
    # {'image_id': 179765, 'id': 38, 'caption': 'A black Honda motorcycle parked in front of a garage.'}
    batched_data = get_data_for_rank(coco_dataset, rank, world_size, batch_size, args.image_size)
    print('enable_entropy_filtering: ',enable_entropy_filtering)

    # 处理数据集
    for item in batched_data:
        t2=time.time()
        imgs_B3HW = item['image']
        text_prompts = item['caption']
        name = item['name']
        print('rank %d: '%rank,name)

        samples = pipe(
            prompt=text_prompts, 
            negative_prompt=[negative_prompt] * len(text_prompts),
            height=imsize,
            width=imsize,
            guidance_scale=CFG,
            top_k=args.top_k,
            enable_entropy_filtering=enable_entropy_filtering,
            num_inference_steps=steps
            ).images

        savedir_pred = os.path.join(args.save_root, 'prediction_cfg%2f_topk%d_topp%2f_temp%2f'%(args.cfg_scale,args.top_k,args.top_p,args.temperature))
        savedir_gt = os.path.join(args.save_root, 'reference')
        os.makedirs(savedir_pred, exist_ok=True)
        os.makedirs(savedir_gt, exist_ok=True)
        
        for i, (fname, text_prompt) in enumerate(zip(name, text_prompts)):
            img_gt = (imgs_B3HW[i].permute(1, 2, 0).add_(1).mul_(0.5).clamp_(0, 1).cpu() * 255.).numpy().astype(np.uint8)
            img_pred = samples[i]

            try:
                img_pred.save(os.path.join(savedir_pred, fname))
                Image.fromarray(img_gt).save(os.path.join(savedir_gt, fname))
            except Exception as e:
                log_path = os.path.join("log.txt")  # 定义日志文件路径
                error_msg = f"Failed to save {fname} - img_gt shape: {img_gt.shape}, dtype: {img_gt.dtype}, error: {str(e)}\n"
                print(error_msg)
                with open(log_path, "a") as log_file:
                    log_file.write(error_msg)

            print(f"Image {fname} saved. to {savedir_pred}")

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512, 1024], default=1024)
    parser.add_argument("--enable_entropy_filtering", type=bool, default=False, help="entropy to sample with")
    parser.add_argument("--save_root", type=str, default='/your-save-root/meissonic_arsample_coco14/baseline')
    parser.add_argument("--cfg_scale", type=float, default=9)
    parser.add_argument("--top_k", type=int, default=0, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    args = parser.parse_args()
    main(args)