"""
Auxiliary methods to handle logs, argument files and other
functions that do not belong to any particular class

cnn_similarity_analysis/src/lib
@author: Prathmesh R Madhu
"""

import os
import json
import datetime

import numpy as np
from matplotlib import pyplot as plt

from CONFIG import CONFIG, DEFAULT_ARGS


def create_configuration_file(exp_path, config, args):
    """
    Creating a configuration file for an experiment, including the hyperparemters
    used and other relevant variables
    Args:
    -----
    exp_path: string
        path to the experiment folder
    config: dictionary
        dictionary with the global configuration parameters: paths, ...
    args: Namespace
        dictionary-like object with the data corresponding to the command line arguments
    Returns:
    --------
    exp_data: dictionary
        dictionary containing all the parameters is the created experiments
    """

    exp_data = {}
    exp_data["exp_created"] = timestamp()
    exp_data["last_modified"] = timestamp()
    exp_data["random_seed"] = config["random_seed"]
    exp_data["num_workers"] = config["num_workers"]

    # loading default args
    args_dict = vars(args)

    # dataset parameters
    exp_data["dataset"] = DEFAULT_ARGS["dataset"]
    for key in DEFAULT_ARGS["dataset"]:
        if(args_dict[key] is not None):
            exp_data["dataset"][key] = args_dict[key]

    # model parameters
    exp_data["model"] = DEFAULT_ARGS["model"]
    for key in DEFAULT_ARGS["model"]:
        if(args_dict[key] is not None):
            exp_data["model"][key] = args_dict[key]

    # training parameters
    exp_data["training"] = DEFAULT_ARGS["training"]
    for key in DEFAULT_ARGS["training"]:
        if(args_dict[key] is not None):
            exp_data["training"][key] = args_dict[key]

    # evaluation parameters
    exp_data["evaluation"] = DEFAULT_ARGS["evaluation"]
    for key in DEFAULT_ARGS["evaluation"]:
        if(args_dict[key] is not None):
            exp_data["evaluation"][key] = args_dict[key]

    # creating file and saving it in the experiment folder
    exp_data_file = os.path.join(exp_path, "experiment_parameters.json")
    with open(exp_data_file, "w") as file:
        json.dump(exp_data, file)

    return exp_data

def load_experiment_parameters(exp_path):
    """
    Loading the experiment parameters given the path to the experiment directory
    Args:
    -----
    exp_path: string
        path to the experiment directory
    Returns:
    --------
    exp_data: dictionary
        dictionary containing the current experiment parameters
    """

    exp_data_path = os.path.join(exp_path, "experiment_parameters.json")
    with open(exp_data_path) as file:
        exp_data = json.load(file)

    return exp_data

def timestamp():
    """
    Obtaining the current timestamp in an human-readable way
    Returns:
    --------
    timestamp: string
        current timestamp in format hh-mm-ss
    """

    timestamp = str(datetime.datetime.now()).split('.')[0] \
                                            .replace(' ', '_') \
                                            .replace(':', '-')

    return timestamp

def create_directory(path, name=None):
    """
    Checking if a directory already exists and creating it if necessary
    Args:
    -----
    path: string
        path/name where the directory will be created
    name: string
        name fo the directory to be created
    """

    if(name is not None):
        path = os.path.join(path, name)

    if(not os.path.exists(path)):
        os.makedirs(path)

    return


def for_all_methods(decorator):
    """
    Decorator that applies a decorator to all methods inside a class
    """
    def decorate(cls):
        for attr in cls.__dict__: # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate


#