import sys
sys.path.append('/cluster/yinan/cnn_similarity_analysis/')
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from src.lib.siamese.args import siamese_args
from src.lib.siamese.model import load_siamese_checkpoint, TripletSiameseNetwork
from src.data.siamese_dataloader import ImageList
from src.lib.siamese.dataset import get_transforms
from sklearn.decomposition import PCA
import faiss
import joblib


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

    if args.dataset == "image_collation":
        d1_images = [args.d1 + 'illustration/' + l.strip() for l in open(args.d1 + 'files.txt', "r")]
        d2_images = [args.d2 + 'illustration/' + l.strip() for l in open(args.d2 + 'files.txt', "r")]
        d3_images = [args.d3 + 'illustration/' + l.strip() for l in open(args.d3 + 'files.txt', "r")]

        train_images = d1_images + d2_images + d3_images

    transforms = get_transforms(args)
    train_dataset = ImageList(train_images, transform=transforms)
    train_loader = DataLoader(dataset=train_dataset, shuffle=False, num_workers=args.num_workers,
                              batch_size=args.batch_size)

    net = TripletSiameseNetwork(args.model)
    net.to(args.device)
    net.eval()

    train_features = generate_features(args, net, train_images, train_loader)

    # estimator = PCA(n_components=256)
    # estimator.fit(train_features)
    # joblib.dump(estimator, args.pca_file)
    d = train_features.shape[1]
    pca = faiss.PCAMatrix(d, 256, -0.5)
    print(f"Train PCA {pca.d_in} -> {pca.d_out}")
    pca.train(train_features)
    print(f"Storing PCA to {args.pca_file}")
    faiss.write_VectorTransform(pca, args.pca_file)


if __name__ == "__main__":

    pca_args = siamese_args()
    if pca_args.device == "cuda:0":
        print("hardware_image_description:", torch.cuda.get_device_name(0))

    train(pca_args)














