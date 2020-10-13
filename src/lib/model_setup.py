"""
Methods for initializing the model, loading checkpoints, setting up optimizers and loss
functions, and other model-related methods

cnn_similarity_analysis/src/lib
@author: Prathmesh R. Madhu
"""

import torch
import torchvision
import pdb

def load_model(exp_data, pretrained, checkpoint=None):
    """
    Args:
    -----
    exp_data: dictionary
        parameters corresponding to the different experiment
    Returns:
    --------
    model: nn.Module
        Lightweight CNN model
    """
    model_name = exp_data["model"]["model_name"]

    if(model_name == "resnet18"):
        model = torchvision.models.resnet18(pretrained=pretrained)

    elif(model_name == "resnet34"):
        model = torchvision.models.resnet34(pretrained=pretrained)

    elif(model_name == "resnet50"):
        model = torchvision.models.resnet50(pretrained=pretrained)

    else:
        raise NotImplementedError("So far only ['resnet18', 'resnet34', 'resnet50']" +\
                                  "models have been implemented")

    return model
