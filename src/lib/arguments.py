"""
Methods for reading and processing command line arguments

cnn_similarity_analysis/src/lib
@author: Prathmesh R Madhu
Usage:
"""

import os
import argparse
from CONFIG import CONFIG


def process_create_experiment_arguments():
    """
    Processing command line arguments for 01_create_experiment script
    """

    # defining command line arguments
    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument("-d", "--exp_directory", help="Directory where the experiment" +\
                        "folder will be created", required=True, default="test_dir")

    # dataset parameters
    parser.add_argument('--dataset_name', help="Dataset to take the images from " +\
                        "[arthist, classarch, chrisarch, artdl, image_collation]", required=True, default="arthist")
    parser.add_argument('--image_size', help="Size used to standardize the images (size x size)")
    parser.add_argument('--shuffle_train', help="If True, train set is iterated randomly",
                        action='store_true')
    parser.add_argument('--shuffle_test', help="If True, valid/test set is iterated randomly",
                        action='store_true')

    # transform and augmentations parameters
    parser.add_argument('--flip', help="If True, images might be flipped during training." +\
                        " We recommend setting it to true", action='store_true')
    parser.add_argument('--rot_factor', help="Maximum rotation angle for the affine " +\
                        " transform. A suitable value is 45", type=float)
    parser.add_argument('--scale_factor', help="Maximum scaling factor for the affine " +\
                        "transform. A suitable value is 0.35", type=float)

    # model parameters
    parser.add_argument('--model_name', help="Model to use for feature extraction " +\
                        "[resnet18, resnet34, resnet50]", default="resnet18")
    parser.add_argument('--layer', help="Model layers ", default="2")

    # training parameters
    parser.add_argument('--num_epochs', help="Number of epochs to train for", type=int)
    parser.add_argument('--learning_rate', help="Learning rate", type=float)
    parser.add_argument('--learning_rate_factor', help="Factor to drop the learning rate " +\
                        "when metric does not further improve", type=float)
    parser.add_argument('--patience', help="Patience factor of the lr scheduler", type=int)
    parser.add_argument('--batch_size', help="Number of examples in each batch", type=int)
    parser.add_argument('--save_frequency', help="Number of epochs after which we save " +\
                        "a checkpoint", type=int)
    parser.add_argument('--optimizer', help="Method used to update model parameters")
    parser.add_argument('--momentum', help="Weight factor for the momentum", type=float)
    parser.add_argument('--nesterov', help="If True, Nesterovs momentum is applied", action='store_true')
    parser.add_argument('--gamma1', help="Gamma1 variable of the adam optimizer", type=float)
    parser.add_argument('--gamma2', help="Gamma2 variable of the adam optimizer", type=float)

    args = parser.parse_args()

    # enforcing correct values
    assert args.dataset_name in ["arthist", "classarch", "chrisarch", "artdl", "image_collation"],\
        "Wrong dataset given. Only ['arthist', 'classarch', 'chrisarch'] are allowed"
    assert args.model_name in ["resnet18", "resnet34", "resnet50"],\
        "Wrong model name given. Only ['resnet18', 'resnet34', 'resnet50'] are allowed"
    return args


def get_directory_argument(get_checkpoint=False, get_dataset=False):
    """
    Reading the directory passed as an argument
    """

    # defining command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Path to the experiment directory")
    parser.add_argument("--checkpoint", help="Name of the checkpoint file to load")
    parser.add_argument("--dataset_name", help="Name of the dataset to use for training" \
                        " or evaluation purposes ['arthist', 'classarch', 'chrisarch', 'artdl', 'image_collation']",
                        default="")
    args = parser.parse_args()

    exp_directory = args.exp_directory
    checkpoint = args.checkpoint
    dataset_name = args.dataset_name

    # making sure experiment directory and checkpoint file exist
    exp_directory = process_experiment_directory_argument(exp_directory)
    if(get_checkpoint is True and checkpoint is not None):
        checkpoint = process_checkpoint(checkpoint, exp_directory)
    if(get_dataset is True):
        assert dataset_name in ["", "arthist", "classarch", "chrisarch", "artdl", "image_collation"]
        dataset_name = None if dataset_name == "" else dataset_name

    if(get_dataset==True and get_checkpoint==True):
        return exp_directory, checkpoint, dataset_name
    elif(get_dataset):
        return exp_directory, dataset_name
    elif(get_checkpoint):
        return exp_directory, checkpoint
    return exp_directory


def process_experiment_directory_argument(exp_directory):
    """
    Ensuring that the experiment directory argument exists
    and giving the full path if relative was detected
    """

    was_relative = False
    exp_path = CONFIG["paths"]["experiments_path"]
    if(exp_path not in exp_directory):
        was_relative = True
        exp_directory = os.path.join(exp_path, exp_directory)

    # making sure experiment directory exists
    if(not os.path.exists(exp_directory)):
        print(f"ERROR! Experiment directorty {exp_directory} does not exist...")
        print(f"     The given path was: {exp_directory}")
        if(was_relative):
            print(f"     It was a relative path. The absolute would be: {exp_directory}")
        print("\n\n")
        exit()

    return exp_directory


def process_checkpoint(checkpoint, exp_directory):
    """
    Making sure the checkpoint to load exists
    """

    # checkpoint = None corresponds to untrained model
    if(checkpoint is None):
        return checkpoint

    checkpoint_path = os.path.join(exp_directory, "models", checkpoint)
    checkpoint_path_det = os.path.join(exp_directory, "models", "detector", checkpoint)
    if(not os.path.exists(checkpoint_path) and not  os.path.exists(checkpoint_path_det)):
        print(f"ERROR! Checkpoint {checkpoint_path} does not exist...")
        print(f"ERROR! Checkpoint {checkpoint_path_det} does not exist either...")
        print(f"     The given checkpoint was: {args.checkpoint}")
        print("\n\n")
        exit()

    return checkpoint

#