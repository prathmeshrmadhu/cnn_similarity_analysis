import os
import pdb
import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal
import pickle
from PIL import Image

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
    # parser.add_argument("--test_image_path", required=True, help="Path to the test image")

    arguments = parser.parse_args()

    return arguments

def getSimilarityMatrix(vectors, image_filenames):
    v = np.array(list(vectors)).T
    sim = np.inner(v.T, v.T) / ((np.linalg.norm(v, axis=0).reshape(-1,1)) * ((np.linalg.norm(v, axis=0).reshape(-1,1)).T))
    keys = list(image_filenames)
    matrix = pd.DataFrame(sim, columns = keys, index = keys)
    return matrix

def setAxes(ax, image, query=False, **kwargs):
    value = kwargs.get("value", None)
    if query:
        ax.set_xlabel("Query Image\n{0}".format(image), fontsize=12)
    else:
        ax.set_xlabel("Similarity value {1:1.3f}\n{0}".format(image, value), fontsize=12)

    ax.set_xticks([])
    ax.set_yticks([])


def getSimilarImages(image, simNames, simVals):
    if image in set(simNames.index):
        imgs = list(simNames.loc[image, :])
        vals = list(simVals.loc[image, :])
        if image in imgs:
            assert_almost_equal(max(vals), 1, decimal=5)
            imgs.remove(image)
            vals.remove(max(vals))

        return imgs, vals
    else:
        print("'{}' Unknown image".format(image))


def plotSimilarImages(image, simiarNames, similarValues):
    simImages, simValues = getSimilarImages(image, similarNames, similarValues)
    fig = plt.figure(figsize=(10, 20))
    # now plot the most similar images
    for j in range(0, numCol * numRow):
        ax = []
        if j == 0:
            img = Image.open(image)
            ax = fig.add_subplot(numRow, numCol, 1)
            setAxes(ax, image, query=True)
        else:
            img = Image.open(simImages[j - 1])
            ax.append(fig.add_subplot(numRow, numCol, j + 1))
            setAxes(ax[-1], simImages[j - 1], value=simValues[j - 1])
        img = img.convert('RGB')
        plt.imshow(img)
        img.close()
    plt.show()

if __name__ == "__main__":

    arguments = process_arguments()
    k = 10

    # Loads the embedding
    embeddings, image_filenames = load_data(arguments.dataset_name)

    similarityMatrix = getSimilarityMatrix(embeddings, image_filenames)
    similarNames = pd.DataFrame(index=similarityMatrix.index, columns=range(k))
    similarValues = pd.DataFrame(index=similarityMatrix.index, columns=range(k))

    for j in tqdm(range(similarityMatrix.shape[0])):
        kSimilar = similarityMatrix.iloc[j, :].sort_values(ascending=False).head(k)
        similarNames.iloc[j, :] = list(kSimilar.index)
        similarValues.iloc[j, :] = kSimilar.values

    numCol = 4
    numRow = 1

    inputImages = image_filenames[:2]

    for image in inputImages:
        plotSimilarImages(image, similarNames, similarValues)



