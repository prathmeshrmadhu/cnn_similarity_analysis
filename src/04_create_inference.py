"""
Finding the similarity from the embeddings and retrieval code
filepath: cnn_similarity_analysis/src

@author: Prathmesh R. Madhu
"""

import os
import pdb

from CONFIG import CONFIG, DEFAULT_ARGS
import torch
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as transforms
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from lib.model_setup import load_model
from lib.utils import load_experiment_parameters, create_directory
from lib.arguments import process_experiment_directory_argument
from data.utils import load_data

def process_arguments():
    """
    Processing command line arguments
    """

    # defining command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Path to the experiment directory")
    parser.add_argument("--dataset_name", help="Datasets to fit into the retrieval object Either of ['chrisarch', 'styled_coco'].",
                        default="chrisarch")
    parser.add_argument("--test_image_path", required=True, help="Path to the test image")
    # parser.add_argument("--model_path", required=True, help="path to pre-trained model")
    arguments = parser.parse_args()
    exp_directory = process_experiment_directory_argument(arguments.exp_directory)

    return arguments, exp_directory

def load_cnn_model(exp_directory, device):
    """
    Loading pretrained models for feature extraction
    """
    exp_data = load_experiment_parameters(exp_directory)

    # seting up cnn model
    cnn_model = load_model(exp_data, pretrained=True)
    cnn_model = cnn_model
    cnn_model = cnn_model.to(device)
    cnn_model.eval()

    return cnn_model

def load_image_tensor(image_path, device):
    """
    Loads a given image to device.
    Args:
    image_path: path to image to be loaded.
    device: "cuda" or "cpu"
    """
    image_tensor = transforms.ToTensor()(Image.open(image_path))
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor.to(device)


def compute_similar_images(image_path, num_images, embedding, device):
    """
    Given an image and number of similar images to generate.
    Returns the num_images closest neares images.

    Args:
    image_path: Path to image whose similar images are to be found.
    num_images: Number of similar images to find.
    embedding : A (num_images, embedding_dim) Embedding of images learnt from auto-encoder.
    device : "cuda" or "cpu" device.
    """

    image_tensor = load_image_tensor(image_path, device)

    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()

    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))

    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)

    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()

    return indices_list


def plot_similar_images(args, exp_directory, indices_list, image_filenames):
    """
    Plots images that are similar to indices obtained from computing simliar images.
    Args:
    indices_list : List of List of indexes. E.g. [[1, 2, 3]]
    """

    exp_data = load_experiment_parameters(exp_directory)
    model_name = exp_data['model']['model_name']

    query_filename = args.test_image_path.split('/')[-1].split('.')[0]
    indices = indices_list[0]
    for index in indices:
        if index == 0:
            # index 0 is a dummy embedding.
            pass
        else:
            img_path = image_filenames[index]
            img_name = img_path.split('/')[-1].split('.')[0]

            img = Image.open(img_path).convert("RGB")
            save_dir_path = os.path.join(exp_directory, "results", model_name, f"retrieval_{query_filename}")
            create_directory(save_dir_path)
            img.save(os.path.join(save_dir_path, f"recommended_{img_name}_{index}.jpg"))

    query_img = Image.open(args.test_image_path).convert("RGB")
    query_img.save(os.path.join(save_dir_path, f"query_{query_filename}.jpg"))

def compute_similar_features(image_path, num_images, embedding, nfeatures=30):
    """
    Given a image, it computes features using ORB detector and finds similar images to it
    Args:
    image_path: Path to image whose features and simlar images are required.
    num_images: Number of similar images required.
    embedding: 2 Dimensional Embedding vector.
    nfeatures: (optional) Number of features ORB needs to compute
    """

    image = cv2.imread(image_path)
    orb = cv2.ORB_create(nfeatures=nfeatures)

    # Detect features
    keypoint_features = orb.detect(image)
    # compute the descriptors with ORB
    keypoint_features, des = orb.compute(image, keypoint_features)

    # des contains the description to features

    des = des / 255.0
    des = np.expand_dims(des, axis=0)
    des = np.reshape(des, (des.shape[0], -1))
    # print(des.shape)
    # print(embedding.shape)

    pca = PCA(n_components=des.shape[-1])
    reduced_embedding = pca.fit_transform(
        embedding,
    )
    # print(reduced_embedding.shape)

    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(reduced_embedding)
    _, indices = knn.kneighbors(des)

    indices_list = indices.tolist()
    # print(indices_list)
    return indices_list


if __name__ == "__main__":
    # Loading the arguments
    os.system("clear")
    arguments, exp_directory = process_arguments()
    # pdb.set_trace()

    # setting up the device
    torch.backends.cudnn.fastest = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Loading the require experiment parameters
    exp_directory = process_experiment_directory_argument(arguments.exp_directory)
    exp_data = load_experiment_parameters(exp_directory)
    model_name = exp_data['model']['model_name']
    layer = exp_data['model']['layer']

    # Loads the model
    encoder = load_cnn_model(exp_directory, device)
    encoder.to(device)

    # Loads the embedding
    embeddings, image_filenames = load_data(arguments.dataset_name, model_name, layer)

    num_images = DEFAULT_ARGS["retrieval"]["num_images"]
    indices_list = compute_similar_images(
        arguments.test_image_path, num_images=num_images, embedding=embeddings, device=device
    )
    plot_similar_images(arguments, exp_directory, indices_list, image_filenames)