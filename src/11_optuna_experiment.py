import os
import glob
import json
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import optuna
import sys
sys.path.append('/cluster/yinan/yinan_cnn/cnn_similarity_analysis/')

from src.lib.io import read_config, generate_train_list, generate_val_list
from src.lib.loss import TripletLoss, CustomLoss, CustomLoss_vgg, SimCLR_Loss
from src.lib.siamese.args import siamese_args
from src.lib.siamese.dataset import generate_train_dataset, get_transforms, add_file_list
from src.lib.augmentations import *
from src.data.siamese_dataloader import TripletTrainList, TripletValList, ImageList
from src.lib.siamese.model import TripletSiameseNetwork, TripletSiameseNetwork_custom
from src.lib.metrics import *


def generate_features(args, net, data_loader):
    features_list = list()
    with torch.no_grad():
        if args.model == 'resnet50':
            for no, data in enumerate(data_loader):
                images = data
                images = images.to(args.device)
                feats1, feats2, feats3, feats4 = net.forward_once(images)
                features_list.append(feats3.cpu().numpy())
        elif args.model == 'vgg' or args.model == 'vgg_fc7':
            for no, data in enumerate(data_loader):
                images = data
                images = images.to(args.device)
                _, _, _, _, _, feats6 = net.forward_once(images)
                features_list.append(feats6.cpu().numpy())
    features = torch.Tensor(np.vstack(features_list))
    return features


def train(args, augmentations_list, lam):
    if args.device == "gpu":
        print("hardware_image_description:", torch.cuda.get_device_name(0))
        # defining the transforms
    transforms = get_transforms(args)

    if args.train_dataset == "artdl":

        if args.mining_mode == "offline":
            print("Used dataset: artdl")
            val_file_path = args.data_path + args.val_list
            val_file = pd.read_csv(val_file_path)
            val_list = []
            for ite in list(val_file['item']):
                val_list.append(args.database_path + ite + '.jpg')
            val_labels = list(val_file['label_encoded'])
            gt_array = np.array(val_labels)

            train_list = args.data_path + args.train_list
            train_origin = pd.read_csv(train_list)
            query_train_o = list(train_origin['anchor_query'])
            p_train_o = list(train_origin['ref_positive'])
            n_train_o = list(train_origin['ref_negative'])
            # train_list = []
            # for i in range(len(query_train)):
            #     train_list.append((query_train[i], p_train[i], n_train[i]))
            # num_triplets = len(train_list)

        elif args.mining_mode == "online":
            print("Used dataset: artdl")
            val_file_path = args.data_path + args.val_list
            val_file = pd.read_csv(val_file_path)
            val_list = []
            for ite in list(val_file['item']):
                val_list.append(args.database_path + ite + '.jpg')
            val_labels = list(val_file['label_encoded'])
            gt_array = np.array(val_labels)

    val_pairs = ImageList(val_list, transform=transforms, imsize=args.imsize)
    val_dataloader = DataLoader(dataset=val_pairs, shuffle=True, num_workers=args.num_workers,
                                batch_size=args.batch_size)

    print("loading siamese model")
    if args.loss == "normal":
        net = TripletSiameseNetwork(args.model, args.method)
        # Defining the criteria for training
        criterion = TripletLoss()
        criterion.to(args.device)
    elif args.loss == "simclr":
        net = TripletSiameseNetwork(args.model, args.method)
        # Defining the criteria for training
        criterion = SimCLR_Loss(args.batch_size)
        criterion.to(args.device)
    elif args.loss == "custom":
        net = TripletSiameseNetwork_custom(args.model)
        # Defining the criteria for training
        if args.model == 'resnet50':
            criterion = CustomLoss()
        elif args.model == 'vgg' or args.model == 'vgg_fc7':
            criterion = CustomLoss_vgg()
        criterion.to(args.device)
    if not args.start:
        print('load trained model:{}'.format(args.checkpoint))
        state_dict = torch.load(args.net + args.checkpoint)
        net.load_state_dict(state_dict)
    net = net.cuda()

    if args.optimizer == "adam":
        if args.model == 'resnet50':
            optimizer = torch.optim.Adam([{'params': net.head.conv1.parameters(), 'lr': args.lr * 0.25},
                                         {'params': net.head.layer1.parameters(), 'lr': args.lr * 0.25},
                                         {'params': net.head.layer2.parameters(), 'lr': args.lr * 0.5},
                                         {'params': net.head.layer3.parameters(), 'lr': args.lr * 0.75},
                                         {'params': net.head.layer4.parameters(), 'lr': args.lr}], lr=args.lr,
                                         weight_decay=args.weight_decay)
        elif args.model == 'vgg' or args.model == 'vgg_fc7':
            optimizer = torch.optim.Adam([{'params': net.head.features[:3].parameters(), 'lr': args.lr},
                                          {'params': net.head.features[4:9].parameters(), 'lr': args.lr},
                                          {'params': net.head.features[9:16].parameters(), 'lr': args.lr},
                                          {'params': net.head.features[16:23].parameters(), 'lr': args.lr},
                                          {'params': net.head.features[23:].parameters(), 'lr': args.lr},
                                          {'params': net.head.avgpool.parameters(), 'lr': args.lr},
                                          {'params': net.head.classifier.parameters(), 'lr': args.lr}], lr=args.lr,
                                          weight_decay=args.weight_decay)

    elif args.optimizer == "sgd":
        if args.model == 'resnet50':
            optimizer = torch.optim.SGD([{'params': net.head.conv1.parameters(), 'lr': args.lr * 0.25},
                                         {'params': net.head.layer1.parameters(), 'lr': args.lr * 0.25},
                                         {'params': net.head.layer2.parameters(), 'lr': args.lr * 0.5},
                                         {'params': net.head.layer3.parameters(), 'lr': args.lr * 0.75},
                                         {'params': net.head.layer4.parameters(), 'lr': args.lr}], lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        elif args.model == 'vgg' or args.model == 'vgg_fc7':
            optimizer = torch.optim.SGD([{'params': net.head.features[:3].parameters(), 'lr': args.lr},
                                         {'params': net.head.features[4:9].parameters(), 'lr': args.lr},
                                         {'params': net.head.features[9:16].parameters(), 'lr': args.lr},
                                         {'params': net.head.features[16:23].parameters(), 'lr': args.lr},
                                         {'params': net.head.features[23:].parameters(), 'lr': args.lr},
                                         {'params': net.head.avgpool.parameters(), 'lr': args.lr},
                                         {'params': net.head.classifier.parameters(), 'lr': args.lr}], lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)


    loss_history = list()
    epoch_losses = list()
    train_losses = list()
    best_map = -np.inf
    for epoch in range(args.num_epochs):
        if args.mining_mode == "online":

            if args.train_dataset == "artdl":
                '''online mining training list'''
                train_origin = generate_train_list(args)
                query_train_o = list(train_origin['anchor_query'])
                p_train_o = list(train_origin['ref_positive'])
                n_train_o = list(train_origin['ref_negative'])

                '''extract features of each triplets'''
                query_o = ImageList(query_train_o, transform=transforms, imsize=args.imsize)
                p_o = ImageList(p_train_o, transform=transforms, imsize=args.imsize)
                n_o = ImageList(n_train_o, transform=transforms, imsize=args.imsize)
                query_dataloader = DataLoader(dataset=query_o, shuffle=False, num_workers=args.num_workers,
                                              batch_size=args.batch_size)
                p_dataloader = DataLoader(dataset=p_o, shuffle=False, num_workers=args.num_workers,
                                          batch_size=args.batch_size)
                n_dataloader = DataLoader(dataset=n_o, shuffle=False, num_workers=args.num_workers,
                                          batch_size=args.batch_size)
                query_f_o = generate_features(args, net, query_dataloader)
                p_f_o = generate_features(args, net, p_dataloader)
                n_f_o = generate_features(args, net, n_dataloader)

                '''calculate distances between each triplets'''
                query_f_o.to(args.device)
                p_f_o.to(args.device)
                n_f_o.to(args.device)
                score_pos = (1 - F.cosine_similarity(query_f_o, p_f_o)).cpu().numpy()
                score_neg = (1 - F.cosine_similarity(query_f_o, n_f_o)).cpu().numpy()

                '''select only semi-hard triplets'''
                true_list = (score_pos < score_neg) * (score_neg < score_pos + args.margin)
                true_list = list(true_list)
                train_origin.insert(train_origin.shape[1], 'label', true_list)
                train_selected = train_origin[train_origin['label']]

                '''generate new training list'''
                query_train = list(train_selected['anchor_query'])
                p_train = list(train_selected['ref_positive'])
                n_train = list(train_selected['ref_negative'])
                num_triplets = len(query_train)
                train_list = []
                for i in range(len(query_train)):
                    train_list.append((query_train[i], p_train[i], n_train[i]))

                '''eraly stop if not able to find semi-hard triplets'''
                if num_triplets == 0:
                    break

        elif args.mining_mode == "offline":
            # train_origin = train
            # query_train_o = query_train
            # p_train_o = p_train
            # n_train_o = n_train

            '''extract features of each triplets'''
            query_o = ImageList(query_train_o, transform=transforms, imsize=args.imsize)
            p_o = ImageList(p_train_o, transform=transforms, imsize=args.imsize)
            n_o = ImageList(n_train_o, transform=transforms, imsize=args.imsize)
            query_dataloader = DataLoader(dataset=query_o, shuffle=False, num_workers=args.num_workers,
                                          batch_size=args.batch_size)
            p_dataloader = DataLoader(dataset=p_o, shuffle=False, num_workers=args.num_workers,
                                      batch_size=args.batch_size)
            n_dataloader = DataLoader(dataset=n_o, shuffle=False, num_workers=args.num_workers,
                                      batch_size=args.batch_size)
            query_f_o = generate_features(args, net, query_dataloader)
            p_f_o = generate_features(args, net, p_dataloader)
            n_f_o = generate_features(args, net, n_dataloader)

            '''calculate distances between each triplets'''
            query_f_o.to(args.device)
            p_f_o.to(args.device)
            n_f_o.to(args.device)
            score_pos = (1 - F.cosine_similarity(query_f_o, p_f_o)).cpu().numpy()
            score_neg = (1 - F.cosine_similarity(query_f_o, n_f_o)).cpu().numpy()

            '''select only semi-hard triplets'''
            true_list = (score_pos < score_neg) * (score_neg < score_pos + args.margin)
            true_list = list(true_list)
            train_new = train_origin.copy()
            train_new.insert(train_new.shape[1], 'label', true_list)
            train_selected = train_new[train_new['label']]

            '''generate new training list'''
            query_train = list(train_selected['anchor_query'])
            p_train = list(train_selected['ref_positive'])
            n_train = list(train_selected['ref_negative'])
            num_triplets = len(query_train)
            train_list = []
            for i in range(len(query_train)):
                train_list.append((query_train[i], p_train[i], n_train[i]))

            '''eraly stop if not able to find semi-hard triplets'''
            if num_triplets == 0:
                break

        image_pairs = TripletValList(train_list, transform=transforms, imsize=args.imsize, argumentation=augmentations_list)
        train_dataloader = DataLoader(dataset=image_pairs, shuffle=True, num_workers=args.num_workers,
                                      batch_size=args.batch_size)
        net.train()
        # Training over batches
        for i, batch in enumerate(train_dataloader, 0):
            query_img, rp_img, rn_img = batch
            query_img = query_img.to(args.device)
            rp_img = rp_img.to(args.device)
            rn_img = rn_img.to(args.device)
            if args.model == 'resnet50':
                q1, q2, q3, q4, p1, p2, p3, p4, n1, n2, n3, n4 = net(query_img, rp_img, rn_img)
                optimizer.zero_grad()
                loss = criterion(q1, q2, q3, q4, p1, p2, p3, p4, n1, n2, n3, n4, args.margin, lam, cos=True)
            elif args.model == 'vgg' or args.model == 'vgg_fc7':
                q1, q2, q3, q4, q5, q6, p1, p2, p3, p4, p5, p6, n1, n2, n3, n4, n5, n6 = net(query_img, rp_img, rn_img)
                optimizer.zero_grad()
                loss = criterion(q1, q2, q3, q4, q5, q6, p1, p2, p3, p4, p5, p6, n6, args.margin, lam,
                                 cos=True)
            loss.backward()
            optimizer.step()
            loss_history.append(loss)

        mean_loss = torch.mean(torch.Tensor(loss_history))
        loss_history.clear()

        print("Epoch:{},  Current training loss {}, num_triplets={}\n".format(epoch, mean_loss, num_triplets))
        train_losses.append(mean_loss)  # Q: Does this only store the mean loss of the last 10 iterations?

        # Validating over batches
        net.eval()
        val_features = generate_features(args, net, val_dataloader)
        val_features = val_features.cuda()
        map_10 = ranked_mean_precision(gt_array, val_features, 10)

        print("Epoch:{},  Current map {}\n".format(epoch, map_10))

        # This re-write the model if validation loss is lower
        if map_10 >= best_map:
            best_map = map_10
            best_model_name = 'Triplet_best.pth'
            model_full_path = args.net + best_model_name
            torch.save(net.state_dict(), model_full_path)
            print('best model updated\n')

    print("Training finished")
    return best_map



if __name__ == "__main__":

    siamese_args = siamese_args()

    # defining augmentations
    augmentations_list = [
        VerticalFlip(probability=0.25),
        HorizontalFlip(probability=0.25),
        AuglyRotate(0.25),
    ]

    def objective(trial):
        lam = trial.suggest_float('lambda', 1.0, 5.0)
        best_map = train(siamese_args, augmentations_list, lam)
        return best_map

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print(study.best_params)
    # print('best map is:{}'.format(best_map))