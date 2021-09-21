"""
Methods for Data loading for either of training, validation or testing

cnn_similarity_analysis/src/data
@author: Prathmesh R. Madhu
"""

import os
import pdb
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import make_dataset, default_loader, IMG_EXTENSIONS

from CONFIG import CONFIG
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class FolderDataset(VisionDataset):
    """
    Creates a PyTorch dataset from folder, returning two tensor images.
    Args:
    main_dir : directory where images are stored.
    transform (optional) : torchvision transforms to be applied while making dataset
    """
    def __init__(self, root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None):
        super(FolderDataset, self).__init__(root, transform=transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.classes = classes
        self.num_classes = len(classes)
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

    def __len__(self):
        return len(self.samples)

def get_dataset_loader(dataset, batch_size=64, shuffle=False):
    """
    Fitting a dataset split into a data loader
    Args:
    -----
    dataset: string
        name of the dataset to load
    batch_size: integer
        number of elements in each batch
    shuffle: boolean
        if True, images are accessed randomly
    Returns:
    --------
    data_loader: DataLoader
        data loader to iterate the dataset split using batches
    """

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=CONFIG["num_workers"])

    return data_loader


def get_classification_dataset(exp_data, train=True, shuffle_train=False, get_dataset=False):
    """
    Loading the detection dataset and fitting data loaders to iterate the different splits
    Args:
    -----
    exp_data: dictionary
        parameters corresponding to the different experiment
    train, validation: boolean
        if True, a data loader is created for the given split
    shuffle_train, shuffle_train: boolean
        if True, images are accessed randomly
    class_ids: list of integers
        list containing the ids of the classes to detect. By defaul [1] (person class)
    Returns:
    --------
    train_loader: DataLoader
        data loader for the training set
    valid_loader: DataLoader
        data loader for the validation set
    """

    data_path = CONFIG["paths"]["data_path"]
    batch_size = 1
    dataset_name = exp_data["dataset"]["dataset_name"]

    train_loader, valid_loader = None, None
    train_set, valid_set = None, None

    #######################################
    # Loading training set if necessary
    #######################################

    if (train):
        if (dataset_name == "chrisarch"):
            dataset = FolderDataset(root=data_path,
                                    loader=default_loader,
                                    extensions=IMG_EXTENSIONS,
                                    transform=transforms.Compose([
                                            transforms.ToTensor(),
                                        ])
                                    )

        train_loader = get_dataset_loader(dataset, batch_size, shuffle_train)
        train_set = dataset


    if (get_dataset):
        return train_loader, train_set, train_set.num_classes
    else:
        return train_loader, train_set.num_classes

#