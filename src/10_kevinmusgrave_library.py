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

from src.lib.loss import TripletLoss, CustomLoss, CustomLoss_vgg
from src.lib.siamese.args import siamese_args
from src.lib.siamese.dataset import generate_train_dataset, get_transforms, add_file_list
from src.lib.augmentations import *
from src.data.siamese_dataloader import TripletTrainList, TripletValList, ImageList
from src.lib.siamese.model import TripletSiameseNetwork, TripletSiameseNetwork_custom