import os
import glob
import time

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
sys.path.append('/cluster/yinan/cnn_similarity_analysis/')

from src.lib.loss import ContrastiveLoss, TripletLoss
from src.lib.siamese.args import siamese_args
from src.lib.siamese.dataset import generate_extraction_dataset, generate_validation_dataset, get_transforms
from src.lib.augmentations import *
from src.data.siamese_dataloader import ImageList, ContrastiveValList
from src.lib.siamese.model import TripletSiameseNetwork
from lib.utils import imshow
from lib.io import *


def generate_features(args, net, image_names, data_loader, save_path):
    features_list = list()
    images_list = list()
    t0 = time.time()
    with torch.no_grad():
        for no, data in enumerate(data_loader):
            images = data
            images = images.to(args.device)
            feats = net.forward_once(images)
            features_list.append(feats.cpu().numpy())
            images_list.append(images.cpu().numpy())
    t1 = time.time()
    features = np.vstack(features_list)
    # TODO: Maybe replace next line by dumping to pkl file/ or replace the pkl file dumping in
    # 02_create_archdata_retrieval by hdf5 descriptors.
    write_pickle_descriptors(features, image_names, save_path)
    print(f"writing descriptors to {save_path}")
    print(f"image_description_time: {(t1 - t0) / len(image_names):.5f} s per image")


def extract_features(args):
    # TODO: Returning the ground truth labels for a given dataset
    # groundtruth_list = read_ground_truth(args.gt_list)
    # query_list = [l.strip() for l in open(args.query_list, "r")]
    # database_list = [l.strip() for l in open(args.db_list, "r")]
    # train_list = [l.strip() for l in open(args.train_list, "r")]
    # query_ind = list(range(1300, 1500))
    # query = [str(l) + '00' for l in query_ind]
    # ref = [str(l) + '01' for l in query_ind]
    p1_images = [args.p1 + 'illustration/' + l.strip() for l in open(args.p1 + 'files.txt', "r")]
    p2_images = [args.p2 + 'illustration/' + l.strip() for l in open(args.p2 + 'files.txt', "r")]
    p3_images = [args.p3 + 'illustration/' + l.strip() for l in open(args.p3 + 'files.txt', "r")]
    d1_images = [args.d1 + 'illustration/' + l.strip() for l in open(args.d1 + 'files.txt', "r")]
    d2_images = [args.d2 + 'illustration/' + l.strip() for l in open(args.d2 + 'files.txt', "r")]
    d3_images = [args.d3 + 'illustration/' + l.strip() for l in open(args.d3 + 'files.txt', "r")]

    # creating the dataset
    #query_images, database_images, _ = generate_extraction_dataset(query, ref, ref)

    # defining the transforms
    transforms = get_transforms(args)

    print("computing features")
    # query_dataset = ImageList(query_images, transform=transforms)
    # database_dataset = ImageList(database_images, transform=transforms)
    p1_dataset = ImageList(p1_images, transform=transforms)
    p2_dataset = ImageList(p2_images, transform=transforms)
    p3_dataset = ImageList(p3_images, transform=transforms)
    d1_dataset = ImageList(d1_images, transform=transforms)
    d2_dataset = ImageList(d2_images, transform=transforms)
    d3_dataset = ImageList(d3_images, transform=transforms)


    # Loading the loaders/dataset
    # query_loader = DataLoader(dataset=query_dataset, shuffle=False, num_workers=args.num_workers,
    #                                       batch_size=args.batch_size)
    # db_loader = DataLoader(dataset=database_dataset, shuffle=False, num_workers=args.num_workers,
    #                                       batch_size=args.batch_size)
    p1_loader = DataLoader(dataset=p1_dataset, shuffle=False, num_workers=args.num_workers,
                                            batch_size=args.batch_size)
    p2_loader = DataLoader(dataset=p2_dataset, shuffle=False, num_workers=args.num_workers,
                                            batch_size=args.batch_size)
    p3_loader = DataLoader(dataset=p3_dataset, shuffle=False, num_workers=args.num_workers,
                                            batch_size=args.batch_size)
    d1_loader = DataLoader(dataset=d1_dataset, shuffle=False, num_workers=args.num_workers,
                                            batch_size=args.batch_size)
    d2_loader = DataLoader(dataset=d2_dataset, shuffle=False, num_workers=args.num_workers,
                                            batch_size=args.batch_size)
    d3_loader = DataLoader(dataset=d3_dataset, shuffle=False, num_workers=args.num_workers,
                                            batch_size=args.batch_size)

    # Loading the pretrained siamese model
    net = TripletSiameseNetwork(args.model)
    state_dict = torch.load(args.net + args.checkpoint)
    net.load_state_dict(state_dict)
    net.eval()
    net.to(args.device)
    print("checkpoint {} loaded\n".format(args.checkpoint))
    print("test model\n")


    # query_features = generate_features(args, net, query, query_loader, args.query_f)
    # database_features = generate_features(args, net, ref, db_loader, args.db_f)
    generate_features(args, net, p1_images, p1_loader, args.p1_f)
    generate_features(args, net, p2_images, p2_loader, args.p2_f)
    generate_features(args, net, p3_images, p3_loader, args.p3_f)
    generate_features(args, net, d1_images, d1_loader, args.d1_f)
    generate_features(args, net, d2_images, d2_loader, args.d2_f)
    generate_features(args, net, d3_images, d3_loader, args.d3_f)

if __name__ == "__main__":
    
    siamese_args = siamese_args()
    if siamese_args.device == "cuda:0":
        print("hardware_image_description:", torch.cuda.get_device_name(0))
        
    extract_features(siamese_args)
    
    
    

