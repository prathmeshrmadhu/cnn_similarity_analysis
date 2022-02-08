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

from src.lib.loss import TripletLoss, CustomLoss, CustomLoss_vgg, SimCLR_Loss
from src.lib.siamese.args import siamese_args
from src.lib.siamese.dataset import generate_train_dataset, get_transforms, add_file_list
from src.lib.augmentations import *
from src.data.siamese_dataloader import TripletTrainList, TripletValList, ImageList
from src.lib.siamese.model import TripletSiameseNetwork, TripletSiameseNetwork_custom


def generate_features(args, net, data_loader):
    if args.loss == "custom":
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
                    _, _, _, _, feats5 = net.forward_once(images)
                    features_list.append(feats5.cpu().numpy())
        features = torch.Tensor(np.vstack(features_list))

    else:
        features_list = list()
        with torch.no_grad():
            for no, data in enumerate(data_loader):
                images = data
                images = images.to(args.device)
                feats = net.forward_once(images)
                features_list.append(feats.cpu().numpy())
        features = torch.Tensor(np.vstack(features_list))

    return features


def train(args, augmentations_list):
    if args.device == "gpu":
        print("hardware_image_description:", torch.cuda.get_device_name(0))
        # defining the transforms
    transforms = get_transforms(args)

    if args.train_dataset == "image_collation":
        print("Used dataset: Image Collation")
        d1_images = [args.d1 + 'illustration/' + l.strip() for l in open(args.d1 + 'files.txt', "r")]
        d2_images = [args.d2 + 'illustration/' + l.strip() for l in open(args.d2 + 'files.txt', "r")]
        d3_images = [args.d3 + 'illustration/' + l.strip() for l in open(args.d3 + 'files.txt', "r")]

        gt_d1d2 = read_config(args.gt_list + 'D1-D2.json')
        gt_d2d3 = read_config(args.gt_list + 'D2-D3.json')
        gt_d1d3 = read_config(args.gt_list + 'D1-D3.json')

        query_val = []
        p_val = []
        n_val = []
        query_val, p_val, n_val = add_file_list(query_val, p_val, n_val, gt_d1d3, d1_images, d3_images)

        if args.mining_mode == "offline":
            query_train = []
            p_train = []
            n_train = []
            query_train, p_train, n_train = add_file_list(query_train, p_train, n_train, gt_d1d2, d1_images, d2_images)
            query_train, p_train, n_train = add_file_list(query_train, p_train, n_train, gt_d2d3, d2_images, d3_images)
            train_list = []
            for i in range(len(query_train)):
                train_list.append((query_train[i], p_train[i], n_train[i]))

    if args.train_dataset == "artdl":

        if args.mining_mode == "offline":
            print("Used dataset: Image Collation")
            val_list = args.data_path + args.val_list
            val = pd.read_csv(val_list)
            query_val = list(val['anchor_query'])
            p_val = list(val['ref_positive'])
            n_val = list(val['ref_negative'])

            train_list = args.data_path + args.train_list
            train = pd.read_csv(train_list)
            query_train = list(train['anchor_query'])
            p_train = list(train['ref_positive'])
            n_train = list(train['ref_negative'])
            train_list = []
            for i in range(len(query_train)):
                train_list.append((query_train[i], p_train[i], n_train[i]))

        elif args.mining_mode == "online":
            print("Used dataset: Image Collation")
            val = generate_val_list(args)
            query_val = list(val['anchor_query'])
            p_val = list(val['ref_positive'])
            n_val = list(val['ref_negative'])

    if args.train_dataset == "the_MET":
        if args.mining_model == 'offline':
            print("Used dataset: The MET")
            train_file = args.data_path + args.train_list
            train_frame = pd.read_csv(train_file)
            train_frame = train_frame[:10000]
            val_file = args.data_path + args.val_list
            val_frame = pd.read_csv(val_file)

    val_list = []
    if args.train_dataset == 'the_MET':
        val_pairs = TripletTrainList(args.data_path, val_frame, transform=transforms, imsize=args.imsize,
                                     argumentation=augmentations_list)
    else:
        for j in range(len(query_val)):
            val_list.append((query_val[j], p_val[j], n_val[j]))

        val_pairs = TripletValList(val_list, transform=transforms, imsize=args.imsize, argumentation=augmentations_list)
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
    net.to(args.device)

    if args.optimizer == "adam":
        if args.loss == "normal" or args.loss == "simclr":
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.loss == "custom":
            if args.model == 'resnet50':
                optimizer = torch.optim.Adam([{'params': net.head.conv1.parameters(), 'lr': args.lr * 0.25},
                                             {'params': net.head.layer1.parameters(), 'lr': args.lr * 0.25},
                                             {'params': net.head.layer2.parameters(), 'lr': args.lr * 0.5},
                                             {'params': net.head.layer3.parameters(), 'lr': args.lr * 0.75},
                                             {'params': net.head.layer4.parameters(), 'lr': args.lr}], lr=args.lr,
                                             weight_decay=args.weight_decay)
            elif args.model == 'vgg' or args.model == 'vgg_fc7':
                optimizer = torch.optim.Adam([{'params': net.head.features[:3].parameters(), 'lr': args.lr * 0.25},
                                              {'params': net.head.features[4:9].parameters(), 'lr': args.lr * 0.5},
                                              {'params': net.head.features[9:16].parameters(), 'lr': args.lr * 0.5},
                                              {'params': net.head.features[16:23].parameters(), 'lr': args.lr * 0.75},
                                              {'params': net.head.features[23:].parameters(), 'lr': args.lr},
                                              {'params': net.head.avgpool.parameters(), 'lr': args.lr},
                                              {'params': net.head.classifier.parameters(), 'lr': args.lr}], lr=args.lr,
                                              weight_decay=args.weight_decay)

    elif args.optimizer == "sgd":
        if args.loss == "normal" or args.loss == "simclr":
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        elif args.loss == "custom":
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
    # epoch_size = int(len(train_list) / args.epoch)
    best_val_loss = np.inf
    for epoch in range(args.num_epochs):
        if args.mining_mode == "online":
            if args.train_dataset == "image_collation":
                query_train = []
                p_train = []
                n_train = []
                query_train, p_train, n_train = add_file_list(query_train, p_train, n_train, gt_d1d2, d1_images, d2_images)
                query_train, p_train, n_train = add_file_list(query_train, p_train, n_train, gt_d2d3, d2_images, d3_images)
                train_list = []
                for i in range(len(query_train)):
                    train_list.append((query_train[i], p_train[i], n_train[i]))

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

        if args.train_dataset == 'the_MET':
            image_pairs = TripletTrainList(args.data_path, train_frame, transform=transforms, imsize=args.imsize,
                                           argumentation=augmentations_list)
        else:
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
            if args.loss == 'normal':
                p_score, n_score = net(query_img, rp_img, rn_img)
                optimizer.zero_grad()
                loss = criterion(p_score, n_score, args.margin)
            elif args.loss == 'simclr':
                q_emb = net.forward_once(query_img)
                p_emb = net.forward_once(rp_img)
                loss = criterion(q_emb, p_emb)
                print(loss)
            elif args.loss == 'custom':
                if args.model == 'resnet50':
                    q1, q2, q3, q4, p1, p2, p3, p4, n1, n2, n3, n4 = net(query_img, rp_img, rn_img)
                    optimizer.zero_grad()
                    loss = criterion(q1, q2, q3, q4, p1, p2, p3, p4, n1, n2, n3, n4, args.margin, args.regular, cos=True)
                elif args.model == 'vgg' or args.model == 'vgg_fc7':
                    q1, q2, q3, q4, q5, p1, p2, p3, p4, p5, n1, n2, n3, n4, n5 = net(query_img, rp_img, rn_img)
                    optimizer.zero_grad()
                    loss = criterion(q1, q2, q3, q4, q5, p1, p2, p3, p4, p5, n5, args.margin, args.regular,
                                     cos=True)
            loss.backward()
            optimizer.step()
            loss_history.append(loss)

        mean_loss = torch.mean(torch.Tensor(loss_history))
        loss_history.clear()

        print("Epoch:{},  Current training loss {}, num_triplets={}\n".format(epoch, mean_loss, num_triplets))
        train_losses.append(mean_loss)  # Q: Does this only store the mean loss of the last 10 iterations?

        # Validating over batches
        val_loss = []
        p_score_list = []
        n_score_list = []
        net.eval()
        with torch.no_grad():
            for j, batch in enumerate(val_dataloader, 0):
                query_img, rp_img, rn_img = batch
                query_img = query_img.to(args.device)
                rp_img = rp_img.to(args.device)
                rn_img = rn_img.to(args.device)
                if args.loss == 'normal':
                    p_score, n_score = net(query_img, rp_img, rn_img)
                    val_loss.append(criterion(p_score, n_score, args.margin))
                    p_score_list.append(torch.mean(p_score))
                    n_score_list.append(torch.mean(n_score))
                elif args.loss == 'simclr':
                    p_score, n_score = net(query_img, rp_img, rn_img)
                    q_emb = net.forward_once(query_img)
                    p_emb = net.forward_once(rp_img)
                    val_loss.append(criterion(q_emb, p_emb))
                    p_score_list.append(torch.mean(p_score))
                    n_score_list.append(torch.mean(n_score))
                elif args.loss == 'custom':
                    if args.model == 'resnet50':
                        q1, q2, q3, q4, p1, p2, p3, p4, n1, n2, n3, n4 = net(query_img, rp_img, rn_img)
                        loss = criterion(q1, q2, q3, q4, p1, p2, p3, p4, n1, n2, n3, n4, args.margin, args.regular, cos=True)
                        p_score = F.cosine_similarity(q3, p3)
                        n_score = F.cosine_similarity(q3, n3)
                    elif args.model == 'vgg' or args.model == 'vgg_fc7':
                        q1, q2, q3, q4, q5, p1, p2, p3, p4, p5, n1, n2, n3, n4, n5 = net(query_img, rp_img, rn_img)
                        loss = criterion(q1, q2, q3, q4, q5, p1, p2, p3, p4, p5, n5, args.margin, args.regular,
                                         cos=True)
                        p_score = F.cosine_similarity(q5, p5)
                        n_score = F.cosine_similarity(q5, n5)
                    val_loss.append(loss)
                    p_score_list.append(torch.mean(p_score))
                    n_score_list.append(torch.mean(n_score))
            val_loss = torch.mean(torch.Tensor(val_loss))
        print("Epoch:{},  Current validation loss {}\n".format(epoch, val_loss))
        epoch_losses.append(val_loss.cpu())

        # This re-write the model if validation loss is lower
        if val_loss.cpu() <= best_val_loss:
            best_val_loss = val_loss.cpu()
            avg_p_score = torch.mean(torch.Tensor(p_score_list)).cpu()
            avg_n_score = torch.mean(torch.Tensor(n_score_list)).cpu()
            if args.loss == "simclr":
                best_model_name = 'SimCLR.pth'
            else:
                best_model_name = 'Triplet_normalize.pth'
            model_full_path = args.net + best_model_name
            torch.save(net.state_dict(), model_full_path)
            print('best model updated\n')

    print("Training finished")
    print("Average Positive Score: {}\n".format(avg_p_score))
    print("Average Negative Score: {}\n".format(avg_n_score))
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


if __name__ == "__main__":

    siamese_args = siamese_args()

    # defining augmentations
    augmentations_list = [
        VerticalFlip(probability=0.25),
        HorizontalFlip(probability=0.25),
        AuglyRotate(0.25),
    ]

    train(siamese_args, augmentations_list)
