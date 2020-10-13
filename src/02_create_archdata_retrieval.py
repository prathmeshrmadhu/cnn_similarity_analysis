"""
Extracting the cnn features from the arch data
to compare the similarity between them and visualization
filepath: cnn_similarity_analysis/src

@author: Prathmesh R. Madhu
"""

import os
import pdb
import pickle
from tqdm import tqdm
import argparse

import numpy as np
import torch
from torch.nn import DataParallel
import torch.nn.functional as F
import torchvision.transforms as transforms

from CONFIG import CONFIG, DEFAULT_ARGS
from lib.logger import Logger, log_function, print_
from lib.model_setup import load_model
from lib.utils import create_directory, for_all_methods, load_experiment_parameters
from lib.arguments import process_experiment_directory_argument, process_checkpoint
from data.dataloader import get_classification_dataset


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

        train_loader, num_classes = get_classification_dataset(exp_data=self.exp_data, train=True,
                                                      shuffle_train=False, get_dataset=False)
        self.train_loader = train_loader
        self.num_classes = num_classes

        return


    def load_models(self):
        """
        Loading pretrained models for feature extraction
        """

        # setting up the device
        torch.backends.cudnn.fastest = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # seting up cnn model
        cnn_model = load_model(self.exp_data, pretrained=True)
        self.cnn_model = cnn_model
        self.cnn_model = self.cnn_model.to(self.device)
        self.cnn_model.eval()

        return

    def create_embedding(self):
        """
        Creates embedding using encoder from dataloader.
        encoder: A convolutional Encoder. E.g. torch_model ConvEncoder
        full_loader: PyTorch dataloader, containing (images, images) over entire dataset.
        embedding_dim: Tuple (c, h, w) Dimension of embedding = output of encoder dimesntions.
        device: "cuda" or "cpu"
        Returns: Embedding of size (num_images_in_loader + 1, c, h, w)
        """
        # Set encoder to eval mode.
        self.cnn_model.eval()

        # Again we do not compute loss here so. No gradients.
        with torch.no_grad():
            for batch_idx, (train_img, target_img, img_path) in enumerate(tqdm(self.train_loader)):
                # We can compute this on GPU. be faster
                train_img = train_img.to(self.device)

                # Get encoder outputs and move outputs to cpu
                enc_output = self.cnn_model(train_img).cpu()
                # Keep adding these outputs to embeddings.
                self.retrieval_db[img_path] = enc_output

        return

    @torch.no_grad()
    def extract_retrieval_dataset(self):
        """
        Iterating over all images from the ArchData dataset, extracting the cnn features.
        The results are saved in a json file to then use for retrieval database purposes
        """

        self.retrieval_db = {}
        self.embedding_dim = 2048

        self.create_embedding()

        return

    def save_retrieval_db(self):
        """
        Saving the retrieval db into a pickle file
        """

        database_root = CONFIG["paths"]["database_path"]
        create_directory(database_root)
        database_path = os.path.join(database_root,
                                     f"database_{DEFAULT_ARGS['dataset']['dataset_name']}.pkl")
        with open(database_path, "wb") as file:
            pickle.dump(self.retrieval_db, file)

        return

if __name__ == "__main__":
    os.system("clear")
    exp_path = process_arguments()

    # initializing logger and logging the beggining of the experiment
    logger = Logger(exp_path)
    message = f"Starting to extract ArchData retrieval dataset."
    logger.log_info(message=message, message_type="new_exp")

    extractor = ArchDataExtractor(exp_path=exp_path)
    extractor.load_dataset()
    extractor.load_models()
    extractor.extract_retrieval_dataset()
    extractor.save_retrieval_db()
    logger.log_info(message=f"Dataset extracted successfully.")
