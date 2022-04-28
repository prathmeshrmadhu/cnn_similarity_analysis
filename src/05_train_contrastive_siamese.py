import os
import glob
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np
sys.path.append('/cluster/yinan/yinan_cnn/cnn_similarity_analysis/')

from lib.loss import FocalLoss
from lib.siamese.args import  siamese_args
from lib.siamese.dataset import generate_extraction_dataset, get_transforms
from lib.augmentations import *
from data.siamese_dataloader import ContrastiveValList
from lib.siamese.model import ContrastiveSiameseNetwork
from lib.io import read_config, generate_focal_train_list, generate_focal_val_list


def train(args, augmentations_list):
    if args.device == "cuda:0":
        print("hardware_image_description:", torch.cuda.get_device_name(0))
    # Getting the query, train, ref.. lists
    if args.train_dataset == "photoart50":
        print("Used dataset:{}".format(args.train_dataset))
        if args.mining_mode == "online":
            print("Used dataset:{}".format(args.train_dataset))
            val = generate_focal_val_list(args)
            query_val = list(val['query'])
            db_val = list(val['reference'])
            label_val = list(val['label'])
    transforms = get_transforms(args)
    val_list = []
    for j in range(len(query_val)):
        val_list.append((query_val[j], db_val[j], label_val[j]))

    val_pairs = ContrastiveValList(val_list, transform=transforms, imsize=args.imsize, augmentation=None)
    val_dataloader = DataLoader(dataset=val_pairs, shuffle=True, num_workers=args.num_workers,
                                batch_size=args.batch_size)

    print("loading siamese model")
    net = ContrastiveSiameseNetwork(args.model)
    if not args.start:
        state_dict = torch.load(args.net + args.checkpoint)
        net.load_state_dict(state_dict)
    net.to(args.device)

    # Defining the criteria for training
    criterion = FocalLoss()
    criterion.to(args.device)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                     lr=args.lr, weight_decay=args.weight_decay)

    loss_history = list()
    epoch_losses = list()
    train_losses = list()
    best_val_loss = np.inf
    for epoch in range(args.num_epochs):
        if args.mining_mode == "online":
            train = generate_focal_train_list(args)
            query_train = list(train['query'])
            db_train = list(train['reference'])
            label_train = list(train['label'])
            train_list = []
            for j in range(len(query_train)):
                train_list.append((query_train[j], db_train[j], label_train[j]))

        train_pairs = ContrastiveValList(train_list, transform=transforms, imsize=args.imsize,
                                         augmentation=augmentations_list)
        train_dataloader = DataLoader(dataset=train_pairs, shuffle=True, num_workers=args.num_workers,
                                      batch_size=args.batch_size)
        # Training over batches
        torch.autograd.set_detect_anomaly(True)
        for i, batch in enumerate(train_dataloader, 0):
            query_img, reference_img, label = batch
            query_img = query_img.to(args.device)
            reference_img = reference_img.to(args.device)
            label = label.to(args.device)

            p_score = net(query_img, reference_img)
            optimizer.zero_grad()
            loss = criterion(p_score, label, 0.5, 0.5)
            loss.backward()
            optimizer.step()
            loss_history.append(loss)

        mean_loss = torch.mean(torch.Tensor(loss_history))
        loss_history.clear()

        print("Epoch:{},  Current training loss {}\n".format(epoch, mean_loss))
        train_losses.append(mean_loss)  # Q: Does this only store the mean loss of the last 10 iterations/batches?

        # Validating over batches
        val_loss = []
        with torch.no_grad():
            for j, batch in enumerate(val_dataloader, 0):
                query_img, reference_img, label = batch
                query_img = query_img.to(args.device)
                reference_img = reference_img.to(args.device)
                label = label.to(args.device)

                p_score = net(query_img, reference_img)
                val_loss.append(criterion(p_score, label, 0.5, 0.5))
            val_loss = torch.mean(torch.Tensor(val_loss))
        print("Epoch:{},  Current validation loss {}\n".format(epoch, val_loss))
        epoch_losses.append(val_loss.cpu())

        # This re-write the model if validation loss is lower
        if val_loss.cpu() <= best_val_loss:
            best_val_loss = val_loss.cpu()
            best_model_name = 'Siamese_best.pth'
            model_full_path = args.net + best_model_name
            torch.save(net.state_dict(), model_full_path)
            print('best model updated\n')

        # # This saves model at each epoch - then finds the best model
        # # Replace this by saving only when best model for validation loss, re-write the model
        # trained_model_name = 'Siamese_Epoch_{}.pth'.format(epoch)
        # model_full_path = args.net + trained_model_name
        # torch.save(net.state_dict(), model_full_path)
        # print('model saved as: {}\n'.format(trained_model_name))

    epoch_losses = np.asarray(epoch_losses)
    train_losses = np.asarray(train_losses)
    epochs = np.asarray(range(args.num_epochs))

    # Loss plot
    plt.title('Loss Visualization')
    plt.plot(epochs, train_losses, color='blue', label='training loss')
    plt.plot(epochs, epoch_losses, color='red', label='validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(args.plots + 'loss.png')
    plt.show()

    # # Saving model with best validation loss
    # best_epoch = np.argmin(epoch_losses)
    # best_model_name = 'Siamese_Epoch_{}.pth'.format(best_epoch)
    # pth_files = glob.glob(args.net + '*.pth')
    # pth_files.remove(args.net + best_model_name)
    # for file in pth_files:
    #     os.remove(file)
    # print("best model is: {} with validation loss {}\n".format(best_model_name, epoch_losses[best_epoch]))


if __name__ == "__main__":

    siamese_args = siamese_args()

    # defining augmentations
    augmentations_list = [
        VerticalFlip(probability=0.25),
        HorizontalFlip(probability=0.25),
        AuglyRotate(0.25),
    ]

    train(siamese_args, augmentations_list)
