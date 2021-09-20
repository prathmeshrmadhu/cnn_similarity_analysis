import sys
import numpy as np
import torch
import torch.nn as nn

import torchvision.transforms as transforms
import torchvision.models as models

from scipy import signal
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from img2vec import Img2Vec

from torch.autograd import Variable

def image_cross_correlation(image1_array, image2_array):
    cross_correlation_list = []
    for i in range(np.shape(image2_array)[2]):
        print (i)
        image1_channel = image1_array[:,:, i]
        image2_channel = image2_array[:, :, i]
        channel_cross_correlation = signal.correlate2d(image1_channel, image2_channel)
        cross_correlation_list.append(channel_cross_correlation)
    avg_cross_correlation = np.average(cross_correlation_list)

    return avg_cross_correlation

def find_distance(vector1, vector2):
    cos_dist = cosine(vector1, vector2)
    return cos_dist

def get_model_distance(image1, image2, model_name):

    img2vec = Img2Vec(model=model_name)
    img1_vec = img2vec.get_vec(image1)
    img2_vec = img2vec.get_vec(image2)

    distance = find_distance(img1_vec, img2_vec)

    return distance








