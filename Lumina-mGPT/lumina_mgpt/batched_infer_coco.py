import sys
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import GenerationConfig, TextStreamer
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from inference_solver import FlexARInferenceSolver
from sample_iteration import renew_pipeline_sampler
from typing import List
import time
import argparse
from collections import OrderedDict
import random
import pdb
import json
import torch
from torchvision import transforms

from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import VQModel


def process_text_tokens(image_size,prompts,tokenizer,device,sep_token='<reserved08706>',pad_token_id=8710):
    text_tokens=[]
    prompt_len_list=[]
    max_len=0
    formatted_prompts=[f"Generate an image of {image_size}x{image_size} according to the following prompt:\n{prompt}" for prompt in prompts]
    for prompt in formatted_prompts:
        # _prompt = inference_solver.item_processor.process_item(item)
        tokens=tokenizer.encode(prompt+sep_token, bos=True, eos=False)
        text_tokens.append(tokens)
        prompt_len_list.append(len(tokens))
        max_len=max(max_len,len(tokens))
    text_tokens_tensor = torch.zeros((len(text_tokens),max_len), dtype=torch.int64, device=device)
    for i in range(len(text_tokens)):
        text_tokens_tensor[i,-len(text_tokens[i]):]=torch.tensor(text_tokens[i],dtype=torch.int64, device=device)
    return text_tokens_tensor,prompt_len_list

def generate(
        prompts: List,
        inference_solver: FlexARInferenceSolver,
        max_gen_len: int,
        temperature: float,
        args=None,
        streamer=None,
        ):

    text_tokens,prompt_len_list=process_text_tokens(args.image_size,
                                                    prompts,
                                                    inference_solver.item_processor.tokenizer,
                                                    device=inference_solver.model.device)

    generation_config = GenerationConfig(
        max_new_tokens=max_gen_len,
        max_length=inference_solver.model.config.max_position_embeddings,
        temperature=temperature,
        top_k=None,
        do_sample=True,
        eos_token_id=[8710],
    )

    logits_processor=inference_solver.create_logits_processor(cfg=args.cfg_scale, image_top_k=2000, enable_entropy_filtering=args.enable_entropy_filtering)
    
    oup_images=[]
    effective_idx=[]
    with torch.no_grad():
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=inference_solver.dtype):
                generation_result = inference_solver.model.generate(
                    text_tokens, generation_config, logits_processor=logits_processor, streamer=streamer
                )
                for idx in range(text_tokens.shape[0]):
                    generation_result_=generation_result[idx][max(prompt_len_list):].tolist()#2个h/w indicator，每行的eol和boi/eoi
                    print(len(generation_result_))
                    if len(generation_result_) > 0 and generation_result_[-1] == 8710:
                        generation_result_ = generation_result_[:-1]
                    # with torch.cuda.amp.autocast(dtype=torch.float32):
                    #     oup_images.append(inference_solver.decode_ids(generation_result_)[1][0])
                        
                    with torch.cuda.amp.autocast(dtype=torch.float32):
                        try:
                            decoded = inference_solver.decode_ids(generation_result_)[1][0]
                        except Exception as e:
                            # log the error if you like
                            print(f"Decoding failed for one image: {e}")
                            oup_images.append('None')
                            continue
                        oup_images.append(decoded)
                        effective_idx.append(idx)

    return oup_images,effective_idx

    return inference_solver.decode_ids(generation_result)[1]

def decode_ids(self, tokens: List[int]):
    generated_images = []
    generation_result_processed = []
    i = 0
    while i < len(tokens):
        token_id = tokens[i]
        if token_id == self.item_processor.token2id(self.item_processor.image_start_token):
            cache = []
            for j in range(i + 1, len(tokens)):
                if tokens[j] != self.item_processor.token2id(self.item_processor.image_end_token):
                    cache.append(tokens[j])
                    i = j + 1
                else:
                    image = self.decode_image(cache)
                    generated_images.append(image)
                    generation_result_processed.append(self.item_processor.token2id("<|image|>"))
                    i = j + 1
                    break
        else:
            generation_result_processed.append(token_id)
            i += 1

    generated = self.item_processor.tokenizer.decode(generation_result_processed)

    return generated, generated_images


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
    os.environ['MASTER_PORT'] = '29502'
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

def get_data_for_rank(meta_json, coco_dataset, rank, world_size, batch_size, imsize,savedir_pred):
    deduplicate_annotations(meta_json)  # 先去重
    total_samples = len(meta_json['annotations'])
    if rank == 0:
        print(f"Total unique samples: {total_samples}")
        
    alredy_existed_file=os.listdir(savedir_pred)
    alredy_existed_idx=[str_.replace('.jpg','') for str_ in alredy_existed_file]

    samples_per_rank = total_samples // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank if rank != world_size - 1 else total_samples

    images, captions, names = [], [], []
    for item_annotations in meta_json['annotations'][start_idx:end_idx]:
        # print('existed:', item_annotations['image_id'],alredy_existed_idx)
        if str(item_annotations['image_id']).zfill(12) in alredy_existed_idx:
            print(item_annotations['image_id'],' exists')
        else:
            impath = os.path.join(coco_dataset, 'val2017', f"{str(item_annotations['image_id']).zfill(12)}.jpg")
            images.append(transform_image(Image.open(impath).convert("RGB"), size=imsize))
            captions.append(item_annotations['caption'])
            names.append(f"{str(item_annotations['image_id']).zfill(12)}.jpg")

    return [{'image': images[i:i+batch_size], 'caption': captions[i:i+batch_size], 'name': names[i:i+batch_size]} 
            for i in range(0, len(images), batch_size)]
    
    
def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    return new_state_dict

def main(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    seed=0
    batch_size=6
    enable_entropy_filtering=args.enable_entropy_filtering
    
    setup(rank, world_size)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    
    device = 'cuda'
    if args.image_size==512:
        model_path_='/your-ckpt/Alpha-VLLM/Lumina-mGPT-7B-512'
    elif args.image_size==768:
        model_path_='/your-ckpt/Alpha-VLLM/Lumina-mGPT-7B-768'
    elif args.image_size==1024:
        model_path_='/your-ckpt/Alpha-VLLM/Lumina-mGPT-7B-1024'
    inference_solver = FlexARInferenceSolver(
        model_path=model_path_,
        precision="bf16",
        target_size=args.image_size,
    )
    inference_solver = renew_pipeline_sampler(
        inference_solver,
        cfg=0.0
    )
    # inference_solver=inference_solver.to(device)

    if rank == 0:
        print("VQ model loaded")

    coco_dataset = '/your-coco-root/dataset'
    
    savedir_pred = os.path.join(args.save_root, 'prediction')
    savedir_gt = os.path.join(args.save_root, 'reference')
    os.makedirs(savedir_pred, exist_ok=True)
    os.makedirs(savedir_gt, exist_ok=True)
    # json文件有四个keys，只有俩是有用的
    # images：list
    # {'license': 4, 'file_name': '000000397133.jpg', 'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg', 'height': 427, 'width': 640, 'date_captured': '2013-11-14 17:02:52', 'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg', 'id': 397133}
    # annotations：list
    # {'image_id': 179765, 'id': 38, 'caption': 'A black Honda motorcycle parked in front of a garage.'}
    with open(os.path.join(coco_dataset,'annotations/captions_val2017.json'),'r') as f: meta_json=json.load(f)
    batched_data = get_data_for_rank(meta_json, coco_dataset, rank, world_size, batch_size, args.image_size,savedir_pred)
    print('enable_entropy_filtering: ',enable_entropy_filtering)

    # 处理数据集
    for item in batched_data:
        t2=time.time()
        imgs_B3HW = item['image']
        text_prompts = item['caption']
        name = item['name']
        print('rank %d: '%rank,name)

        samples,effective_idx = generate(
            text_prompts,inference_solver,max_gen_len=8192,temperature=1.0,args=args
        )
        
        for idx in effective_idx:#enumerate(zip(name, text_prompts)):
            fname=name[idx]
            img_gt = (imgs_B3HW[idx].permute(1, 2, 0).add_(1).mul_(0.5).clamp_(0, 1).cpu() * 255.).numpy().astype(np.uint8)
            img_pred = samples[idx]

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
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512, 768, 1024], default=1024)
    parser.add_argument("--cfg_scale", type=float, default=4)
    parser.add_argument("--enable_entropy_filtering", type=bool, default=True, help="entropy to sample with")
    parser.add_argument("--save_root", type=str, default='/your-save-root/entropy_cfg4_top2000_1024')
    args = parser.parse_args()
    main(args)