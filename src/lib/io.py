# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

from typing import Iterable, List, Optional

import json
import numpy as np
import h5py
import pickle
import pandas as pd
from pandas.core.frame import DataFrame
import random

from .metrics import GroundTruthMatch, PredictedMatch


def read_config(cfg_path):
    with open(cfg_path, 'r') as f:
        params = json.load(f)

    return params


def read_names(path):
    with open(path, 'r') as f:
        params = json.load(f)
    file_names = [i.split('.jpg')[0] + '.jpg' for i in params['_via_image_id_list']]
    return file_names


def read_ground_truth(filename: str) -> List[GroundTruthMatch]:
    """
    Read groundtruth csv file.
    Must contain query_image_id,db_image_id on each line.
    handles the no header version and DD's version with header
    """
    gt_pairs = []
    with open(filename, "r") as cfile:
        for line in cfile:
            line = line.strip()
            if line == 'query_id,reference_id':
                continue
            q, db = line.split(",")
            if db == '':
                continue
            gt_pairs.append(GroundTruthMatch(q, db))
    return gt_pairs


def read_predictions(filename: str) -> List[PredictedMatch]:
    """
    Read predictions csv file.
    Must contain query_image_id,db_image_id,score on each line.
    Header optional
    """
    predictions = []
    with open(filename, "r") as cfile:
        for line in cfile:
            line = line.strip()
            if line == "query_id,reference_id,score":
                continue
            q, db, score = line.split(",")
            predictions.append(PredictedMatch(q, db, float(score)))
    return predictions


def write_predictions(
    predictions: Iterable[PredictedMatch],
    preds_filepath: str,
):
    with open(preds_filepath, "w") as pfile:
        pfile.write("query_id,reference_id,score\n")
        for p in predictions:
            row = f"{p.query},{p.db},{p.score:.6f}"
            pfile.write(row + "\n")


def write_predictions_from_arrays(
    S: np.ndarray,
    I: np.ndarray,
    dbids: np.ndarray,
    qids: np.ndarray,
    preds_filepath: str,
    nmax: Optional[int] = None,
    score_min: Optional[float] = None,
):
    """
    Write CSV predictions file from arrays returned by FAISS.


    Parameters
    ----------
    S : np.ndarray
        Scores for each query. Shape nq, k.
        For each query, scores must be in decreasing order.
    I : np.ndarray
        Nearest neighbors indices for each query. Shape nq, k
    dbids : np.ndarray
        Image ids for each reference image. Shape [nb, ]
    qids : np.ndarray
        Image ids for each query image. Shape [nq, ]
    preds_filepath : str
        Output file path.
    nmax : Optional[int], optional
        Maximum number of predictions to write. Will pick the ones with highest score.
        If None, no limit on the number of predictions.
    score_min : Optional[float], optional
        Minimum score to write a predictions.
        If None, no limit.
    """
    nq, k = S.shape
    # Find the minimum score to return at most nmax predictions
    # TODO(lowik) might not work if duplicates scores at nmax index...
    if nmax is None or nq * k <= nmax:
        score_min = -1e6
    else:
        S = S.copy()   # we are going to overwrite it
        scores = S.ravel()  # linear view of array
        pivot = len(scores) - nmax
        o = scores.argpartition(pivot)  #
        scores[o[:pivot]] = -1e7
        score_min = -1e6

    with open(preds_filepath, "w") as pfile:
        for qidx in range(nq):
            query_id = qids[qidx]
            for score, dbidx in zip(S[qidx], I[qidx]):
                db_id = dbids[dbidx]
                if score >= score_min:
                    row = f"{query_id},{db_id},{score:.6f}"
                    pfile.write(row + "\n")
                else:
                    # Assume scores are in decreasing order in the array
                    # next scores on this row won't pass the threshold
                    break


def write_predictions_from_range_arrays(
    lims: np.ndarray,
    S: np.ndarray,
    I: np.ndarray,
    dbids: np.ndarray,
    qids: np.ndarray,
    preds_filepath: str,
    nmax: Optional[int] = None,
):
    """
    Write CSV predictions file from range search arrays returned by FAISS.

    Parameters
    ----------
    lims : np.ndarray
        limits between queries, shape (nq + 1)
    S : np.ndarray
        Flat array of scores
    I : np.ndarray
        Flat array of indices
    dbids : np.ndarray
        Image ids for each reference image. Shape [nb, ]
    qids : np.ndarray
        Image ids for each query image. Shape [nq, ]
    preds_filepath : str
        Output file path.
    nmax : Optional[int], optional
        Maximum number of predictions to write. Will pick the ones with highest score.
        If None, no limit on the number of predictions.
    """
    npred, = S.shape
    nq = len(lims) - 1
    assert lims[-1] == npred
    assert I.shape == (npred, )

    if nmax is None or S.size <= nmax:
        score_min = -1e6
    else:
        S = S.copy()   # we are going to overwrite it
        scores = S.ravel()
        pivot = len(scores) - nmax
        o = scores.argpartition(pivot)  #
        scores[o[:pivot]] = -1e7
        score_min = -1e6

    with open(preds_filepath, "w") as pfile:
        for qidx in range(nq):
            l0, l1 = lims[qidx:qidx + 2]
            query_id = qids[qidx]
            for dbidx in range(l0, l1):
                db_id = dbids[I[dbidx]]
                score = S[dbidx]
                if score > score_min:
                    row = f"{query_id},{db_id},{score:.6f}"
                    pfile.write(row + "\n")


def write_hdf5_descriptors(vectors, image_names, fname):
    """
    write image description vectors in HDF5 format.
    """
    # image_names = np.array(image_names)
    vectors = np.ascontiguousarray(vectors, dtype='float32')
    image_names = np.array([
        bytes(name, "ascii")
        for name in image_names
    ])
    with h5py.File(fname, "w") as f:
        f.create_dataset("vectors", data=vectors)
        f.create_dataset("image_names", data=image_names)


def write_pickle_descriptors(vectors, image_names, fname):
    """
    write image description vectors in pickle format.
    """
    vectors = np.ascontiguousarray(vectors, dtype='float32')
    fw = open(fname, 'wb')
    pickle.dump(image_names, fw)
    pickle.dump(vectors, fw)
    fw.close()


def write_pickle_descriptors_mix(vectors1, vectors2, vectors3, vectors4, image_names, fname):
    """
    write image description vectors in pickle format.
    """
    vectors1 = np.ascontiguousarray(vectors1, dtype='float32')
    vectors2 = np.ascontiguousarray(vectors2, dtype='float32')
    vectors3 = np.ascontiguousarray(vectors3, dtype='float32')
    vectors4 = np.ascontiguousarray(vectors4, dtype='float32')
    fw = open(fname, 'wb')
    pickle.dump(image_names, fw)
    pickle.dump(vectors1, fw)
    pickle.dump(vectors2, fw)
    pickle.dump(vectors3, fw)
    pickle.dump(vectors4, fw)
    fw.close()


def read_pickle_descriptors(fname):
    """
    read image description vectors from pickle file.
    """
    fw = open(fname, 'rb')
    image_names = np.asarray(pickle.load(fw))
    vectors = pickle.load(fw)
    fw.close()
    return image_names, vectors


def read_pickle_descriptors_mix(fname):
    """
    read image description vectors from pickle file.
    """
    fw = open(fname, 'rb')
    image_names = np.asarray(pickle.load(fw))
    vectors1 = pickle.load(fw)
    vectors2 = pickle.load(fw)
    vectors3 = pickle.load(fw)
    vectors4 = pickle.load(fw)
    fw.close()
    return image_names, vectors1, vectors2, vectors3, vectors4

def read_descriptors(filenames):
    """ read descriptors from a set of HDF5 files """
    descs = []
    names = []
    for filename in filenames:
        hh = h5py.File(filename, "r")
        descs.append(np.array(hh["vectors"]))
        names += np.array(hh["image_names"][:], dtype=object).astype(str).tolist()
    # strip paths and extensions from the filenames
    names = [
        name.split('/')[-1]
        for name in names
    ]
    names = [
        name[:-4] if name.endswith(".jpg") or name.endswith(".png") else name
        for name in names
    ]
    return names, np.vstack(descs)

def generate_train_list(args):
    """generate random train triplets"""
    train_df = pd.read_csv(args.data_path + args.train_list)
    anchor_query = []
    ref_positive = []
    ref_negative = []
    if args.train_dataset == "artdl":
        for i in range(10):
            sub_df_p = train_df[train_df['label_encoded'] == i]
            sub_df_n = train_df[train_df['label_encoded'] != i]
            for ite in sub_df_p['item']:
                for j in range(10):
                    r_p = random.choice(list(sub_df_p['item']))
                    r_n = random.choice(list(sub_df_n['item']))
                    anchor_query.append(args.database_path + ite + '.jpg')
                    ref_positive.append(args.database_path + r_p + '.jpg')
                    ref_negative.append(args.database_path + r_n + '.jpg')
    elif args.train_dataset == "photoart50":
        for i in range(50):
            sub_df_p = train_df[train_df['label_encoded'] == i]
            sub_df_n = train_df[train_df['label_encoded'] != i]
            for ite in sub_df_p['item']:
                for j in range(10):
                    r_p = random.choice(list(sub_df_p['item']))
                    r_n = random.choice(list(sub_df_n['item']))
                    anchor_query.append(args.database_path + ite)
                    ref_positive.append(args.database_path + r_p)
                    ref_negative.append(args.database_path + r_n)


    train_disc = {'anchor_query': anchor_query,
                  'ref_positive': ref_positive,
                  'ref_negative': ref_negative}
    train_data = DataFrame(train_disc)
    return train_data


def generate_val_list(args):
    """generate random train triplets"""
    val_df = pd.read_csv(args.data_path + args.val_list)
    anchor_query = []
    ref_positive = []
    ref_negative = []
    if args.train_dataset == "artdl" or args.train_dataset == 'iconart':
        for i in range(10):
            sub_df_p = val_df[val_df['label_encoded'] == i]
            sub_df_n = val_df[val_df['label_encoded'] != i]
            for ite in sub_df_p['item']:
                for j in range(10):
                    r_p = random.choice(list(sub_df_p['item']))
                    r_n = random.choice(list(sub_df_n['item']))
                    anchor_query.append(args.database_path + ite + '.jpg')
                    ref_positive.append(args.database_path + r_p + '.jpg')
                    ref_negative.append(args.database_path + r_n + '.jpg')
    elif args.train_dataset == "photoart50":
        for i in range(50):
            sub_df_p = val_df[val_df['label_encoded'] == i]
            sub_df_n = val_df[val_df['label_encoded'] != i]
            for ite in sub_df_p['item']:
                for j in range(10):
                    r_p = random.choice(list(sub_df_p['item']))
                    r_n = random.choice(list(sub_df_n['item']))
                    anchor_query.append(args.database_path + ite)
                    ref_positive.append(args.database_path + r_p)
                    ref_negative.append(args.database_path + r_n)

    val_disc = {'anchor_query': anchor_query,
                  'ref_positive': ref_positive,
                  'ref_negative': ref_negative}
    val_data = DataFrame(val_disc)
    return val_data


def generate_test_list(args):
    """generate random train triplets"""
    test_df = pd.read_csv(args.data_path + args.test_list)
    test_list = []
    test_labels = list(test_df["label_encoded"])
    if args.train_dataset == "artdl" or args.train_dataset == 'iconart':
        for i in range(len(test_labels)):
            test_list.append(args.database_path + list(test_df['item'])[i] + '.jpg')
    elif args.train_dataset == "photoart50":
        for i in range(len(test_labels)):
            test_list.append(args.database_path + list(test_df['item'])[i])
    test_disc = {'test_images': test_list,
                 'label': test_labels}
    test_data = DataFrame(test_disc)
    return test_data


def generate_focal_train_list(args):
    train_df = pd.read_csv(args.data_path + args.train_list)
    query = []
    db = []
    match_label = []
    for i in range(50):
        sub_df_p = train_df[train_df['label_encoded'] == i]
        sub_df_n = train_df[train_df['label_encoded'] != i]
        for ite in sub_df_p['item']:
            for j in range(10):
                label = random.randint(0, 1)
                match_label.append(label)
                query.append(args.database_path + ite)
                if label == 1:
                    r_p = random.choice(list(sub_df_p['item']))
                    db.append(args.database_path + r_p)
                elif label == 0:
                    r_n = random.choice(list(sub_df_n['item']))
                    db.append(args.database_path + r_n)

    train_disc = {'query': query,
                  'reference': db,
                  'label': match_label}

    train_data = DataFrame(train_disc)
    return train_data

def generate_focal_val_list(args):
    val_df = pd.read_csv(args.data_path + args.val_list)
    query = []
    db = []
    match_label = []
    for i in range(50):
        sub_df_p = val_df[val_df['label_encoded'] == i]
        sub_df_n = val_df[val_df['label_encoded'] != i]
        for ite in sub_df_p['item']:
            for j in range(10):
                label = random.randint(0, 1)
                match_label.append(label)
                query.append(args.database_path + ite)
                if label == 1:
                    r_p = random.choice(list(sub_df_p['item']))
                    db.append(args.database_path + r_p)
                elif label == 0:
                    r_n = random.choice(list(sub_df_n['item']))
                    db.append(args.database_path + r_n)

    val_disc = {'query': query,
                'reference': db,
                'label': match_label}

    val_data = DataFrame(val_disc)
    return val_data


def generate_test_focal_list(args):
    test_df = pd.read_csv(args.data_path + args.test_list)
    query = []
    db = []
    match_label = []
    for i in range(50):
        sub_df_p = test_df[test_df['label_encoded'] == i]
        sub_df_n = test_df[test_df['label_encoded'] != i]
        for ite in sub_df_p['item']:
            for j in range(10):
                label = random.randint(0, 1)
                match_label.append(label)
                query.append(args.database_path + ite)
                if label == 1:
                    r_p = random.choice(list(sub_df_p['item']))
                    db.append(args.database_path + r_p)
                elif label == 0:
                    r_n = random.choice(list(sub_df_n['item']))
                    db.append(args.database_path + r_n)

    test_disc = {'query': query,
                'reference': db,
                'label': match_label}

    test_data = DataFrame(test_disc)
    return test_data




