import sys
sys.path.append('/cluster/yinan/cnn_similarity_analysis/')
import numpy as np
import torch
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader
from src.lib.siamese.args import siamese_args
from src.lib.siamese.model import load_siamese_checkpoint, TripletSiameseNetwork
from src.data.siamese_dataloader import ImageList
from src.lib.siamese.dataset import get_transforms
from lib.io import read_config
from lib.metrics import calculate_distance
import faiss
import random


def generate_features(args, net, image_names, data_loader):
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
    print(f"image_description_time: {(t1 - t0) / len(image_names):.5f} s per image")
    return features


def train(args):
    if args.device == "gpu":
        print("hardware_image_description:", torch.cuda.get_device_name(0))

    if args.train_dataset == "image_collation":
        d1_images = [args.d1 + 'illustration/' + l.strip() for l in open(args.d1 + 'files.txt', "r")]
        d2_images = [args.d2 + 'illustration/' + l.strip() for l in open(args.d2 + 'files.txt', "r")]
        d3_images = [args.d3 + 'illustration/' + l.strip() for l in open(args.d3 + 'files.txt', "r")]

        train_images = d1_images + d2_images + d3_images

    if args.train_dataset == 'isc2021':
        TRAIN = '/cluster/shared_dataset/isc2021/training_images/training_images/'
        train_images = [TRAIN + l.strip() for l in open(args.train_list, "r")]

    transforms = get_transforms(args)
    train_dataset = ImageList(train_images, transform=transforms)
    train_loader = DataLoader(dataset=train_dataset, shuffle=False, num_workers=args.num_workers,
                              batch_size=args.batch_size)

    net = TripletSiameseNetwork(args.model)
    net.to(args.device)
    net.eval()

    train_features = generate_features(args, net, train_images, train_loader)

    d = train_features.shape[1]
    pca = faiss.PCAMatrix(d, 256, -0.5)
    print(f"Train PCA {pca.d_in} -> {pca.d_out}")
    pca.train(train_features)
    print(f"Storing PCA to {args.pca_file}")
    faiss.write_VectorTransform(pca, args.pca_file)

    if args.val_dataset == 'image_collation':

        d1_features = pca.apply_py(train_features[:len(d1_images)])
        d2_features = pca.apply_py(train_features[len(d1_images): len(d1_images)+len(d2_images)])
        d3_features = pca.apply_py(train_features[len(d1_images)+len(d2_images): len(train_features)])

        gt_d1d2 = read_config(args.gt_list + 'D1-D2.json')
        gt_d2d3 = read_config(args.gt_list + 'D2-D3.json')
        gt_d1d3 = read_config(args.gt_list + 'D1-D3.json')

        dp_d1d2, dn_d1d2, sp_d1d2, sn_d1d2 = calculate_distance(gt_d1d2, d1_features, d2_features)
        dp_d2d3, dn_d2d3, sp_d2d3, sn_d2d3 = calculate_distance(gt_d2d3, d2_features, d3_features)
        dp_d1d3, dn_d1d3, sp_d1d3, sn_d1d3 = calculate_distance(gt_d1d3, d1_features, d3_features)

        mean_dp = np.mean(np.array(dp_d1d2 + dp_d2d3 + dp_d1d3))
        mean_dn = np.mean(np.array(dn_d1d2 + dn_d2d3 + dn_d1d3))
        mean_sp = np.mean(np.array(sp_d1d2 + sp_d2d3 + sp_d1d3))
        mean_sn = np.mean(np.array(sn_d1d2 + sn_d2d3 + sn_d1d3))

        print('average positive distance: {}'.format(mean_dp))
        print('average negative distance: {}'.format(mean_dn))
        print('\n')
        print('average positive similarity: {}'.format(mean_sp))
        print('average negative similarity: {}'.format(mean_sn))





if __name__ == "__main__":

    pca_args = siamese_args()
    if pca_args.device == "cuda:0":
        print("hardware_image_description:", torch.cuda.get_device_name(0))

    train(pca_args)














