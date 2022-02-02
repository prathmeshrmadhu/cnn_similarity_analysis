import os
import glob
import json
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from lib.io import read_config, generate_train_list, generate_val_list
import sys
sys.path.append('/cluster/yinan/yinan_cnn/cnn_similarity_analysis/')

from src.lib.loss import TripletLoss, CustomLoss, CustomLoss_vgg
from src.lib.siamese.args import siamese_args
from src.lib.siamese.dataset import generate_train_dataset, get_transforms, add_file_list
from src.lib.augmentations import *
from src.data.siamese_dataloader import ImageList_with_label
from src.lib.siamese.model import TripletSiameseNetwork, TripletSiameseNetwork_custom, VGG16FC7
from pytorch_metric_learning import losses


def train(args, augmentations_list):
    transforms = get_transforms(args)
    if args.train_dataset == "artdl":
        val_file = args.data_path + args.val_list
        val_df = pd.read_csv(val_file)
        train_file = args.data_path + args.train_list
        train_df = pd.read_csv(train_file)
        train_images = []
        for ite in list(train_df['item']):
            train_images.append(args.database_path + ite + '.jpg')
        train_labels = list(train_df['label_encoded'])
    train_list = ImageList_with_label(train_images, train_labels, transform=transforms)
    dataloader = DataLoader(dataset=train_list, shuffle=True, num_workers=args.num_workers,
                                  batch_size=args.batch_size)
    model = VGG16FC7()
    loss_func = losses.TripletMarginLoss()
    model.to(args.device)
    loss_func.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for i, (data, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        data.cuda()
        labels.cuda()
        embeddings = model(data)
        loss = loss_func(embeddings, labels)
        loss.backward()
        optimizer.step()
    best_model_name = 'Triplet_kevin.pth'
    model_full_path = args.net + best_model_name
    torch.save(model.state_dict(), model_full_path)
    print('best model updated\n')


if __name__ == "__main__":

    siamese_args = siamese_args()

    # defining augmentations
    augmentations_list = [
        VerticalFlip(probability=0.25),
        HorizontalFlip(probability=0.25),
        AuglyRotate(0.25),
    ]

    train(siamese_args, augmentations_list)


