"""
Configuration macros and default argument values

EnhancePseEstimation/src
@author: Angel Villar-Corrales
"""


CONFIG = {

"paths": {
        "data_path": "/localhome/prathmeshmadhu/work/EFI/Data/Christian_Arch/src/cv_notebooks/data/classification_all",
        "database_path": "D:\work\codes\cnn_similarity_analysis\databases",
        "experiments_path":
            "D:\work\codes\cnn_similarity_analysis\experiments",
        "knn_path": "D:\work\codes\cnn_similarity_analysis\knn",
        "pretrained_path": "D:\work\codes\cnn_similarity_analysis\resources",
        "submission": "submission_dict.json"
    },

    # "paths": {
    #     "data_path": "/localhome/prathmeshmadhu/work/EFI/Data/Christian_Arch/src/cv_notebooks/data/classification_all",
    #     "database_path": "/localhome/prathmeshmadhu/work/EFI/Data/Christian_Arch/src/cnn_similarity_analysis/databases",
    #     "experiments_path":
    #         "/localhome/prathmeshmadhu/work/EFI/Data/Christian_Arch/src/cnn_similarity_analysis/experiments",
    #     "knn_path": "/localhome/prathmeshmadhu/work/EFI/Data/Christian_Arch/src/cnn_similarity_analysis/knn",
    #     "pretrained_path": "/localhome/prathmeshmadhu/work/EFI/Data/Christian_Arch/src/cnn_similarity_analysis/resources",
    #     "submission": "submission_dict.json"
    # },
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
    }
}

#