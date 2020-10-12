"""
Extracting the cnn features from the arch data
to compare the similarity between them and visualization

@author: Prathmesh R. Madhu
"""

import os
import pickle
from tqdm import tqdm
import argparse

import numpy as np
import torch
from torch.nn import DataParallel
import torch.nn.functional as F
import torchvision.transforms as transforms

from CONFIG import CONFIG
from lib.logger import Logger, log_function, print_
from lib.model_setup import load_checkpoint, load_model, setup_detector
from lib.utils import create_directory, for_all_methods, load_experiment_parameters
from lib.arguments import process_experiment_directory_argument, process_checkpoint


@log_function
def process_arguments():
    """
    Processing command line arguments
    """

    # defining command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Path to the experiment directory")
    args = parser.parse_args()

    exp_directory = args.exp_directory

    # making sure experiment directory and checkpoint file exist
    exp_directory = process_experiment_directory_argument(exp_directory)

    return exp_directory

@for_all_methods(log_function)
class ArchDataExtractor:
    """
    Class to extract cnn features (2d and 1d) for all images
    from the ArchData dataset
    Args:
    -----
    exp_path: string
        path to the experiment directory
    """

    def __init__(self, exp_path, params=None):
        """
        Initializer of the ArchData extractor object
        """

        self.exp_path = exp_path
        self.params = params if params is not None else {}
        self.exp_data = load_experiment_parameters(exp_path)

        # model and processing parameters
        self.num_classes = len(self.class_ids)


        # defining and creating directories to save the results
        plots_path = os.path.join(self.exp_path, "plots")
        create_directory(plots_path)
        self.results_path = os.path.join(plots_path, "final_result_imgs")
        create_directory(self.results_path)

        return


    def load_dataset(self):
        """
        Loading the ArchData dataset and fitting a data loader
        """

        train_loader, _  = get_classification_dataset(exp_data=self.exp_data, train=True,
                                                 validation=False, shuffle_train=False,
                                                 shuffle_valid=False, valid_size=0,
                                                 class_ids=self.class_ids)
        self.train_loader = train_loader

        return


    def load_models(self):
        """
        Loading pretrained models for person detection as well as keypoint detection
        """

        # setting up the device
        torch.backends.cudnn.fastest = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # seting up person detector model
        det_model_name = self.exp_data["model"]["detector_name"]
        det_model_type = self.exp_data["model"]["detector_type"]
        det_model = setup_detector(model_name=det_model_name,
                                   model_type=det_model_type,
                                   pretrained=True,
                                   num_classes=self.num_classes)
        det_model.eval()
        det_model = DataParallel(det_model).to(self.device)
        # loading pretrained person detector checkpoint if specified
        if(self.person_det_checkpoint is not None):
            print_(f"Loading checkpoint {self.person_det_checkpoint}")
            checkpoint_path = os.path.join(self.exp_path, "models", "detector",
                                          self.person_det_checkpoint)
            det_model = load_checkpoint(checkpoint_path,
                                        model=det_model,
                                        only_model=True)
        self.det_model = det_model

        # setting up pretrained keypoint detector
        pose_model = load_model(self.exp_data, checkpoint=self.keypoint_det_checkpoint)
        self.pose_model = DataParallel(pose_model)
        if(self.keypoint_det_checkpoint is not None):
            print_(f"Loading checkpoint {self.keypoint_det_checkpoint}")
            checkpoint_path = os.path.join(self.exp_path, "models",
                                          self.keypoint_det_checkpoint)
            self.pose_model = load_checkpoint(checkpoint_path,
                                              model=self.pose_model,
                                              only_model=True)
        self.pose_model = self.pose_model.to(self.device)
        self.pose_model.eval()

        return
