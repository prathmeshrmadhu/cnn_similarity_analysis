import numpy as np
import torch
from torch.utils.data import DataLoader
from lib.io import read_config
from src.lib.siamese.dataset import generate_train_dataset, get_transforms, add_file_list


def gem_npy(x, p=3, eps=1e-6):
    x = np.clip(x, a_min=eps, a_max=np.inf)
    x = x ** p
    x = x.mean(axis=0)
    return x ** (1. / p)


def train(args):
    if args.device == "gpu":
        print("hardware_image_description:", torch.cuda.get_device_name(0))

    if args.dataset == "image_collation":
        d1_images = [args.d1 + 'illustration/' + l.strip() for l in open(args.d1 + 'files.txt', "r")]
        d2_images = [args.d2 + 'illustration/' + l.strip() for l in open(args.d2 + 'files.txt', "r")]
        d3_images = [args.d3 + 'illustration/' + l.strip() for l in open(args.d3 + 'files.txt', "r")]

        train_images = d1_images + d2_images + d3_images



