import sys
sys.path.append("./lumina_mgpt/")
sys.path.append("./")
print(sys.path)
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from llamagen.tokenizer.tokenizer_image.vq_model import VQ_models
from llamagen.language.t5 import T5Embedder
from llamagen.llamagen import GPT_models
from llamagen.llamagen_solver import LlamaGenSolver, renew_llamagen, generate
from scheduler.jacobi_iteration_lumina_mgpt import renew_sampler
import time
import argparse
from collections import OrderedDict
import random
import pdb
import json
import torch
from torchvision import transforms


def get_jacobi_param_dict(args):
    target_size = 512

    seeds = [args.seed, ]
    max_num_new_tokens =16 
    multi_token_init_scheme = 'repeat_horizon'
    image_top_k = 5000#1000
    text_top_k = 10
    guidance_scale = 7.5
    prefix_token_sampler_scheme = 'speculative_jacobi' # 'jacobi', 'speculative_jacobi'
    
    is_entropy_acc=args.is_entropy_acc
    save_NFE_dir=args.save_NFE_dir
    print(args.is_entropy_acc)

    jacobi_param_dict = dict(
        jacobi_loop_interval_l = 1,
        jacobi_loop_interval_r = (target_size // 16)**2 - max_num_new_tokens - 2, 
        max_num_new_tokens = max_num_new_tokens,
        guidance_scale = guidance_scale,
        if_entropy_acc = is_entropy_acc,
        save_NFE_dir = save_NFE_dir,
        model_type_2='llamagen',
        seed = seeds[0],
        multi_token_init_scheme = multi_token_init_scheme,
        do_cfg=  True,
        image_top_k=image_top_k, 
        text_top_k=text_top_k,
        prefix_token_sampler_scheme = prefix_token_sampler_scheme,
    )
    return jacobi_param_dict


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
    seed=20
    batch_size=1
        
    setup(rank, world_size)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    
    device = 'cuda'

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    print(gpt_model.__class__)

    jacobi_param_dict = get_jacobi_param_dict(args)
    image_top_k = jacobi_param_dict['image_top_k']

    gpt_model.__class__ = renew_llamagen(gpt_model.__class__)
    gpt_model._init_new_params(**jacobi_param_dict)
    gpt_model.__class__ = renew_sampler(gpt_model.__class__)
    gpt_model._init_new_params(**jacobi_param_dict)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
 
    if "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")


    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 
    
    if not os.path.exists(args.t5_path):
        os.makedirs(args.t5_path)

    assert os.path.exists(args.t5_path), f"t5 model path {args.t5_path} does not exist"
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )
    
    coco_dataset = '/your-coco-dataset/dataset'
    
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
    print('enable_entropy_filtering: ',args.enable_entropy_filtering)
    
    solver = LlamaGenSolver(
        model = gpt_model,
        image_top_k=image_top_k,
        image_top_p=args.top_p,
    )


    # 处理数据集
    for item in batched_data:
        t2=time.time()
        imgs_B3HW = item['image']
        text_prompts = item['caption']
        name = item['name']
        print('rank %d: '%rank,name)
        
        caption_embs, emb_masks = t5_model.get_text_embeddings(text_prompts)

        if not args.no_left_padding:
            print(f"processing left-padding...")    
            # a naive way to implement left-padding
            new_emb_masks = torch.flip(emb_masks, dims=[-1])
            new_caption_embs = []
            for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
                valid_num = int(emb_mask.sum().item())
                print(f'  prompt {idx} token len: {valid_num}')
                new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
                new_caption_embs.append(new_caption_emb)
            new_caption_embs = torch.stack(new_caption_embs)
        else:
            new_caption_embs, new_emb_masks = caption_embs, emb_masks
        c_indices = new_caption_embs * new_emb_masks[:,:, None]
        c_emb_masks = new_emb_masks

        qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
        index_sample = solver.generate(
            c_indices, latent_size ** 2, 
            c_emb_masks, 
            cfg_scale=args.cfg_scale,
            temperature=args.temperature, top_k=image_top_k,
            top_p=args.top_p, sample_logits=True, 
        )
        samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
            
        for i, (fname, text_prompt) in enumerate(zip(name, text_prompts)):
            img_gt = (imgs_B3HW[i].permute(1, 2, 0).add_(1).mul_(0.5).clamp_(0, 1).cpu() * 255.).numpy().astype(np.uint8)
            img_pred = (samples[i].permute(1, 2, 0).add_(1).mul_(0.5).clamp_(0, 1).cpu() * 255.).numpy().astype(np.uint8)

            try:
                Image.fromarray(img_pred).save(os.path.join(savedir_pred, fname))
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
    parser.add_argument("--t5-path", type=str, default='/your-ckpt/ckpts/google/')
    parser.add_argument("--t5_model_type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5_feature_max_len", type=int, default=120)
    parser.add_argument("--t5_feature_dim", type=int, default=2048)
    parser.add_argument("--no_left_padding", action='store_true', default=False)
    parser.add_argument("--gpt_model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default='/your-ckpt/ckpts/LlamaGen/t2i_XL_stage2_512.pt')
    parser.add_argument("--gpt_type", type=str, choices=['c2i', 't2i'], default="t2i", help="class->image or text->image")  
    parser.add_argument("--cls_token_num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq_model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default='/your-ckpt/ckpts/LlamaGen/vq_ds16_t2i.pt', help="ckpt path for vq model")
    parser.add_argument("--codebook_size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook_embed_dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image_size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample_size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--enable_entropy_filtering", type=bool, default=False, help="entropy to sample with")
    parser.add_argument("--is_entropy_acc", type=bool, default=True, help="entropy to sample with")
    parser.add_argument("--save_root", type=str, default='/your-save-root/llamagen_arsample/entropy_ab1_sjd_top5000_512')
    parser.add_argument("--save_NFE_dir", type=str, default='/your-save-root/llamagen_arsample/entropy_ab1_sjd_top5000_512_nfe.txt')
    args = parser.parse_args()
    main(args)