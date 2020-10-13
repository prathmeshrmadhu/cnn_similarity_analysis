"""
Fitting a k-nearest neighbor tree for image similarity and retrieval purposes
filepath: cnn_similarity_analysis/src

@author: Prathmesh R. Madhu
"""

import os
import pdb
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from lib.utils import create_directory
from data.utils import load_data

from CONFIG import CONFIG

def process_arguments():
    """
    Processing command line arguments
    """

    # defining command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Path to the experiment directory")
    parser.add_argument("--dataset_name", help="Datasets to fit into the retrieval object Either of ['chrisarch', 'styled_coco'].",
                        default="chrisarch")
    parser.add_argument("--metric", help="Metric used for retrieval: ['euclidean_distance'," \
                                         " 'cosine_similarity'].", default="euclidean_distance")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    metric = args.metric

    # ensuring correct values
    assert metric in ["euclidean_distance", "cosine_similarity"]
    assert ("chrisarch" in dataset_name or "styled_coco" in dataset_name)

    return args

def cluster_images(args, embedding, pca_num_components: int, tsne_num_components: int):
    """
    Clusters and plots the images using PCA + T-SNE approach.
    Args:
    embedding: A 2D Vector of image embeddings.
    pca_num_components: Number of componenets PCA should reduce.
    tsne_num_components: Number of componenets T-SNE should reduce to. Suggested: 2
    """

    visualization_path = os.path.join(CONFIG["paths"]["visualization_path"])
    create_directory(visualization_path)
    pca_file_name = os.path.abspath(os.path.join(visualization_path, f"pca_{args.dataset_name}_{pca_num_components}.pkl"))
    tsne_file_name = os.path.abspath(os.path.join(visualization_path, f"tsne_{args.dataset_name}_{tsne_num_components}.pkl"))
    tsne_embeddings_file_name = os.path.abspath(os.path.join(
        visualization_path, f"tsne_embeddings_{args.dataset_name}_{tsne_num_components}.pkl"
    ))

    print("Reducing Dimensions using PCA")

    pca = PCA(n_components=pca_num_components, random_state=42)
    reduced_embedding = pca.fit_transform(embedding)
    # print(reduced_embedding.shape)

    # Cluster them using T-SNE.
    print("Using T-SNE to cluster them")
    tsne_obj = TSNE(
        n_components=tsne_num_components,
        verbose=1,
        random_state=42,
        perplexity=200,
        n_iter=1000
    )

    tsne_embedding = tsne_obj.fit_transform(reduced_embedding)

    # print(tsne_embedding.shape)

    # Dump the TSNE and PCA object.
    pickle.dump(pca, open(pca_file_name, "wb"))
    # pickle.dump(tsne_embedding)
    pickle.dump(tsne_obj, open(tsne_file_name, "wb"))

    # Vizualize the TSNE.
    vizualise_tsne(tsne_embedding)

    # Save the embeddings.
    pickle.dump(tsne_embedding, open(tsne_embeddings_file_name, "wb"))

def vizualise_tsne(tsne_embedding):
    """
    Plots the T-SNE embedding.
    Args:
    tsne_embedding: 2 Dimensional T-SNE embedding.
    """

    x = tsne_embedding[:, 0]
    y = tsne_embedding[:, 1]

    plt.scatter(x, y, c=y)
    plt.show()

if __name__ == '__main__':
    os.system("clear")
    args = process_arguments()
    pdb.set_trace()
    ## Loads the embeddings
    embeddings, image_filenames = load_data(args.dataset_name)

    ## Cluster the embeddings
    cluster_images(args, embeddings, pca_num_components=50, tsne_num_components=2)



