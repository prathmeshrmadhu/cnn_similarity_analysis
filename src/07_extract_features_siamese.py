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


def generate_features(args, net, image_names, data_loader, query=True):
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
    if query:
        write_pickle_descriptors(features, image_names, args.query_f)
        print(f"writing descriptors to {args.query_f}")
    else:
        write_pickle_descriptors(features, image_names, args.db_f)
        print(f"writing descriptors to {args.db_f}")
    print(f"image_description_time: {(t1 - t0) / len(image_names):.5f} s per image")
    return features


def extract_features(args, visualization=False):
    # TODO: Returning the ground truth labels for a given dataset
    # groundtruth_list = read_ground_truth(args.gt_list)
    # query_list = [l.strip() for l in open(args.query_list, "r")]
    # database_list = [l.strip() for l in open(args.db_list, "r")]
    # train_list = [l.strip() for l in open(args.train_list, "r")]
    query_ind = list(range(1300, 1500))
    query = [str(l) + '00' for l in query_ind]
    ref = [str(l) + '01' for l in query_ind]

    # creating the dataset
    query_images, database_images, _ = generate_extraction_dataset(query, ref, ref)

    # defining the transforms
    transforms = get_transforms(args)

    print("computing features")
    query_dataset = ImageList(query_images, transform=transforms)
    database_dataset = ImageList(database_images, transform=transforms)

    # Loading the loaders/dataset
    query_loader = DataLoader(dataset=query_dataset, shuffle=False, num_workers=args.num_workers,
                                          batch_size=args.batch_size)
    db_loader = DataLoader(dataset=database_dataset, shuffle=False, num_workers=args.num_workers,
                                          batch_size=args.batch_size)

    # Loading the pretrained siamese model
    net = TripletSiameseNetwork(args.model)
    state_dict = torch.load(args.net + args.checkpoint)
    net.load_state_dict(state_dict)
    net.eval()
    net.to(args.device)
    print("checkpoint {} loaded\n".format(args.checkpoint))
    print("test model\n")


    # if visualization:
    #     test_list = generate_validation_dataset(query_images, groundtruth_list, train_images, 50)
    #     test_data = ContrastiveValList(test_list, transform=transforms, imsize=args.imsize)
    #     test_loader = DataLoader(dataset=test_data, shuffle=True, num_workers=args.num_workers,
    #                              batch_size=1)
    #     with torch.no_grad():
    #         distance_p = []
    #         distance_n = []
    #         for i, data in enumerate(test_loader, 0):
    #             img_name = 'test_{}.jpg'.format(i)
    #             img_pth = args.images + img_name
    #             query_img, reference_img, label = data
    #             concatenated = torch.cat((query_img, reference_img), 0)
    #             query_img = query_img.to(args.device)
    #             reference_img = reference_img.to(args.device)
    #             score = net(query_img, reference_img).cpu()
    #
    #             if label == 0:
    #                 label = 'matched'
    #                 distance_p.append(score.item())
    #                 print('matched with distance: {:.4f}\n'.format(score.item()))
    #             if label == 1:
    #                 label = 'not matched'
    #                 distance_n.append(score.item())
    #                 print('not matched with distance: {:.4f}\n'.format(score.item()))
    #
    #             imshow(torchvision.utils.make_grid(concatenated),
    #                    'Dissimilarity: {:.2f} Label: {}'.format(score.item(), label), should_save=True, pth=img_pth)
    #     mean_distance_p = torch.mean(torch.Tensor(distance_p))
    #     mean_distance_n = torch.mean(torch.Tensor(distance_n))
    #     print('-------------------------------------------------------------')
    #     print('not matched mean distance: {:.4f}\n'.format(mean_distance_n))
    #     print('matched mean distance: {:.4f}\n'.format(mean_distance_p))

    query_features = generate_features(args, net, query, query_loader, query=True)
    database_features = generate_features(args, net, ref, db_loader, query=False)

    return query_features, database_features

if __name__ == "__main__":
    
    siamese_args = siamese_args()
    if siamese_args.device == "cuda:0":
        print("hardware_image_description:", torch.cuda.get_device_name(0))
        
    query_features, database_features = extract_features(siamese_args)
    
    
    

