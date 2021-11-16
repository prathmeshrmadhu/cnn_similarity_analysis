import sys
sys.path.append('/cluster/yinan/cnn_similarity_analysis/')
import numpy as np
import torch
import time
import torchvision
from torch.utils.data import DataLoader
from src.lib.siamese.args import siamese_args
from src.lib.siamese.model import load_siamese_checkpoint, TripletSiameseNetwork, ResNet50Conv4
from src.data.siamese_dataloader import ImageList
from src.lib.siamese.dataset import get_transforms
from sklearn.decomposition import PCA
from lib.io import *
import joblib
import faiss


def generate_pca_features(features, image_names, save_path, estimator=None):
    if estimator:
        print(f"Apply PCA {estimator.d_in} -> {estimator.d_out}")
        pca_features = estimator.apply_py(features)
    else:
        pca_features = features
    write_pickle_descriptors(pca_features, image_names, save_path)
    print(f"writing descriptors to {save_path}")


def generate_features(args, net, image_names, data_loader):
    features_list = list()
    # images_list = list()
    t0 = time.time()
    with torch.no_grad():
        for no, data in enumerate(data_loader):
            images = data
            images = images.to(args.device)
            feats = net.forward_once(images)
            features_list.append(feats.cpu().numpy())
            # images_list.append(images.cpu().numpy())
    t1 = time.time()
    features = np.vstack(features_list)
    print(f"image_description_time: {(t1 - t0) / len(image_names):.5f} s per image")
    return features


def embedding_features(args):
    # defining the transforms
    transforms = get_transforms(args)

    # resnet_50 = torchvision.models.resnet50(pretrained=True)
    # net = ResNet50Conv4(resnet_50)
    net = TripletSiameseNetwork(args.model)
    if args.net:
        state_dict = torch.load(args.net + args.checkpoint)
        net.load_state_dict(state_dict)
    net.to(args.device)
    net.eval()

    if args.pca:
        print("Load PCA matrix", args.pca_file)
        pca = faiss.read_VectorTransform(args.pca_file)


    if args.val_dataset == "image_collation":
        p1_images = [args.p1 + 'illustration/' + l.strip() for l in open(args.p1 + 'files.txt', "r")]
        p2_images = [args.p2 + 'illustration/' + l.strip() for l in open(args.p2 + 'files.txt', "r")]
        p3_images = [args.p3 + 'illustration/' + l.strip() for l in open(args.p3 + 'files.txt', "r")]
        d1_images = [args.d1 + 'illustration/' + l.strip() for l in open(args.d1 + 'files.txt', "r")]
        d2_images = [args.d2 + 'illustration/' + l.strip() for l in open(args.d2 + 'files.txt', "r")]
        d3_images = [args.d3 + 'illustration/' + l.strip() for l in open(args.d3 + 'files.txt', "r")]


        p1_dataset = ImageList(p1_images, transform=transforms)
        p2_dataset = ImageList(p2_images, transform=transforms)
        p3_dataset = ImageList(p3_images, transform=transforms)
        d1_dataset = ImageList(d1_images, transform=transforms)
        d2_dataset = ImageList(d2_images, transform=transforms)
        d3_dataset = ImageList(d3_images, transform=transforms)

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

        p1_features = generate_features(args, net, p1_images, p1_loader)
        p2_features = generate_features(args, net, p2_images, p2_loader)
        p3_features = generate_features(args, net, p3_images, p3_loader)
        d1_features = generate_features(args, net, p1_images, d1_loader)
        d2_features = generate_features(args, net, p2_images, d2_loader)
        d3_features = generate_features(args, net, p3_images, d3_loader)

        if args.pca:
            generate_pca_features(p1_features, p1_images, args.p1_f, pca)
            generate_pca_features(p2_features, p2_images, args.p2_f, pca)
            generate_pca_features(p3_features, p3_images, args.p3_f, pca)
            generate_pca_features(d1_features, d1_images, args.d1_f, pca)
            generate_pca_features(d2_features, d2_images, args.d2_f, pca)
            generate_pca_features(d3_features, d3_images, args.d3_f, pca)
        else:
            generate_pca_features(p1_features, p1_images, args.p1_f)
            generate_pca_features(p2_features, p2_images, args.p2_f)
            generate_pca_features(p3_features, p3_images, args.p3_f)
            generate_pca_features(d1_features, d1_images, args.d1_f)
            generate_pca_features(d2_features, d2_images, args.d2_f)
            generate_pca_features(d3_features, d3_images, args.d3_f)


if __name__ == "__main__":

    siamese_args = siamese_args()
    if siamese_args.device == "cuda:0":
        print("hardware_image_description:", torch.cuda.get_device_name(0))

    embedding_features(siamese_args)