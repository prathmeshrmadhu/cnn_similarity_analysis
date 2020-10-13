import os
import pickle
import numpy as np
from CONFIG import CONFIG


def load_data(dataset_name):
    """
    Loading pickled objects
    Args:
    -----
    dataset_names: list
        list with the datasets to fit into the knn structure
    Returns:
    --------
    data: dictionary
        dictionary containing data and metadata from all desired datasets
    """

    all_dicts = []
    data = {}

    # loading data from all datasets
    pickle_path = os.path.join(CONFIG["paths"]["database_path"], f"database_{dataset_name}.pkl")
    with open(pickle_path, "rb") as file:
        database = pickle.load(file)
    data = [np.array(val[0]) for val in database.values()]
    image_filenames = [val[0] for val in database.keys()]

    return data, image_filenames
