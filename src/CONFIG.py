"""
Configuration macros and default argument values

EnhancePseEstimation/src
@author: Angel Villar-Corrales
"""
import os

osuname = os.uname().nodename
print("osuname", osuname)
if osuname == "prathmeshmadhu":
    data_path = "/localhome/prathmeshmadhu/work/EFI/Data/Christian_Arch/src/cv_notebooks/data/classification_all"
    database_path = "/localhome/prathmeshmadhu/work/EFI/Data/Christian_Arch/src/cnn_similarity_analysis/databases"
    visualization_path = "/localhome/prathmeshmadhu/work/EFI/Data/Christian_Arch/src/cnn_similarity_analysis/visualizations"
    experiments_path = "/localhome/prathmeshmadhu/work/EFI/Data/Christian_Arch/src/cnn_similarity_analysis/experiments"
    knn_path = "/localhome/prathmeshmadhu/work/EFI/Data/Christian_Arch/src/cnn_similarity_analysis/knn"
    pretrained_path = "/localhome/prathmeshmadhu/work/EFI/Data/Christian_Arch/src/cnn_similarity_analysis/resources"
elif osuname == "lme242":
    data_path = "/cluster/open_datasets/Omniart"
    database_path = "/cluster/yinan/cnn_similarity_analysis/databases"
    visualization_path = "/cluster/yinan/cnn_similarity_analysis/visualizations"
    experiments_path = "/cluster/yinan/cnn_similarity_analysis/experiments"
    knn_path = "/cluster/yinan/cnn_similarity_analysis/knn"
    pretrained_path = "/cluster/yinan/cnn_similarity_analysis/resources"



CONFIG = {

# "paths": {
#         "data_path": "/localhome/prathmeshmadhu/work/EFI/Data/Christian_Arch/src/cv_notebooks/data/classification_all",
#         "database_path": "D:\work\codes\cnn_similarity_analysis\databases",
#         "experiments_path":
#             "D:\work\codes\cnn_similarity_analysis\experiments",
#         "knn_path": "D:\work\codes\cnn_similarity_analysis\knn",
#         "pretrained_path": "D:\work\codes\cnn_similarity_analysis\resources",
#         "submission": "submission_dict.json"
#     },

    "paths": {
        "data_path": data_path,
        "database_path": database_path,
        "visualization_path": visualization_path,
        "experiments_path": experiments_path,
        "knn_path": knn_path,
        "pretrained_path": pretrained_path,
        "submission": "submission_dict.json"
    },
    "num_workers": 0,
    "random_seed": 42
}

DEFAULT_ARGS = {
    "dataset": {
        "dataset_name": "chrisarch",
        "image_size": 400,
        "flip": False,
    },
    "model": {
        "model_name": "resnet18",
        "layer": "2"
    },
    "training": {
        "num_epochs": 100,
        "learning_rate": 0.001,
        "learning_rate_factor": 0.333,
        "patience": 10,
        "batch_size": 32,
        "save_frequency": 5,
        "optimizer": "adam",
        "momentum": 0.9,
        "nesterov": False,
        "gamma1": 0.99,
        "gamma2": 0
    },
    "evaluation": {
    },
    "retrieval":{
        "num_images": 5
    }
}

#