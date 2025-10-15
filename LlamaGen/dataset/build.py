import os
from dataset.imagenet import build_imagenet, build_imagenet_code
from dataset.coco import build_coco
from dataset.openimage import build_openimage
from dataset.pexels import build_pexels
from dataset.t2i import build_t2i, build_t2i_code, build_t2i_image
from dataset.imgnet_dataset import ImgDataset
from dataset.t2i import MJDataset


def build_dataset(args, **kwargs):
    # images
    if args.dataset == 'imagenet':
        return build_imagenet(args, **kwargs)
    if args.dataset == 'imagenet_code':
        return build_imagenet_code(args, **kwargs)
    if args.dataset == 'coco':
        return build_coco(args, **kwargs)
    if args.dataset == 'openimage':
        return build_openimage(args, **kwargs)
    if args.dataset == 'pexels':
        return build_pexels(args, **kwargs)
    if args.dataset == 't2i_image':
        return build_t2i_image(args, **kwargs)
    if args.dataset == 't2i':
        return build_t2i(args, **kwargs)
    if args.dataset == 't2i_code':
        return build_t2i_code(args, **kwargs)
    
    raise ValueError(f'dataset {args.dataset} is not supported')


def build_dataset_imgnet(args):
    train_set = ImgDataset(imgnet_root=args.imgnet_root,
                           imgnet_subdir='train',
                           selected_file_list='train.txt',
                           resolution=args.image_size)
    val_set=None
    return train_set,val_set


def build_dataset_mj(args):
    train_set = MJDataset(mj_root=args.mj_root,
                           mj_classes=[],
                           resolution=args.image_size)
    val_set=None
    return train_set,val_set