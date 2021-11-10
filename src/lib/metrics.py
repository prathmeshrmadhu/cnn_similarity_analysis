# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

from dataclasses import astuple, dataclass
from typing import List, Optional, Tuple
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import random


@dataclass
class GroundTruthMatch:
    query: str
    db: str


@dataclass
class PredictedMatch:
    query: str
    db: str
    score: float


@dataclass
class Metrics:
    average_precision: float
    precisions: np.ndarray
    recalls: np.ndarray
    thresholds: np.ndarray
    recall_at_p90: float
    threshold_at_p90: float
    recall_at_rank1: float
    recall_at_rank10: float

def argsort(seq):
    # from https://stackoverflow.com/a/3382369/3853462
    return sorted(range(len(seq)), key=seq.__getitem__)


def precision_recall(
    y_true: np.ndarray, probas_pred: np.ndarray, num_positives: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precisions, recalls and thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        Binary label of each prediction (0 or 1). Shape [n, k] or [n*k, ]
    probas_pred : np.ndarray
        Score of each prediction (higher score == images more similar, ie not a distance)
        Shape [n, k] or [n*k, ]
    num_positives : int
        Number of positives in the groundtruth.

    Returns
    -------
    precisions, recalls, thresholds
        ordered by increasing recall.
    """
    probas_pred = probas_pred.flatten()
    y_true = y_true.flatten()
    # to handle duplicates scores, we sort (score, NOT(jugement)) for predictions
    # eg,the final order will be (0.5, False), (0.5, False), (0.5, True), (0.4, False), ...
    # This allows to have the worst possible AP.
    # It prevents participants from putting the same score for all predictions to get a good AP.
    order = argsort(list(zip(probas_pred, ~y_true)))
    order = order[::-1]  # sort by decreasing score
    probas_pred = probas_pred[order]
    y_true = y_true[order]

    ntp = np.cumsum(y_true)  # number of true positives <= threshold
    nres = np.arange(len(y_true)) + 1  # number of results

    precisions = ntp / nres
    recalls = ntp / num_positives
    return precisions, recalls, probas_pred


def average_precision_old(recalls: np.ndarray, precisions: np.ndarray):
    """
    Compute the micro average-precision score (uAP).

    Parameters
    ----------
    recalls : np.ndarray
        Recalls, can be in any order.
    precisions : np.ndarray
        Precisions for each recall value.

    Returns
    -------
    uAP: float
    """

    # Order by increasing recall
    order = np.argsort(recalls)
    recalls = recalls[order]
    precisions = precisions[order]
    return ((recalls[1:] - recalls[:-1]) * precisions[:-1]).sum()

# Jay Qi's version
def average_precision(recalls: np.ndarray, precisions: np.ndarray):
    # Order by increasing recall
    # order = np.argsort(recalls)
    # recalls = recalls[order]
    # precisions = precisions[order]

    # Check that it's ordered by increasing recall
    if not np.all(recalls[:-1] <= recalls[1:]):
        raise Exception("recalls array must be sorted before passing in")

    return ((recalls - np.concatenate([[0], recalls[:-1]])) * precisions).sum()

def find_operating_point(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, required_x: float
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Find the highest y with x at least `required_x`.

    Returns
    -------
    x, y, z
        The best operating point (highest y) with x at least `required_x`.
        If we can't find a point with the required x value, return
        x=required_x, y=None, z=None
    """
    valid_points = x >= required_x
    if not np.any(valid_points):
        return required_x, None, None

    valid_x = x[valid_points]
    valid_y = y[valid_points]
    valid_z = z[valid_points]
    best_idx = np.argmax(valid_y)
    return valid_x[best_idx], valid_y[best_idx], valid_z[best_idx]


def check_duplicates(predictions: List[PredictedMatch]) -> List[PredictedMatch]:
    """
    Raise an exception if predictions contains duplicates
    (ie several predictions for the same (query, db) pair).
    """
    unique_pairs = set((p.query, p.db) for p in predictions)
    if len(unique_pairs) != len(predictions):
        raise ValueError("Predictions contains duplicates.")


def sanitize_predictions(predictions: List[PredictedMatch]) -> List[PredictedMatch]:
    # TODO(lowik) check for other possible loopholes
    check_duplicates(predictions)
    return predictions


def to_arrays(gt_matches: List[GroundTruthMatch], predictions: List[PredictedMatch]):
    """Convert from list of matches to arrays"""
    predictions = sanitize_predictions(predictions)

    gt_set = {astuple(g) for g in gt_matches}
    probas_pred = np.array([p.score for p in predictions])
    y_true = np.array([(p.query, p.db) in gt_set for p in predictions], dtype=bool)
    return y_true, probas_pred

def find_tp_ranks(gt_matches: List[GroundTruthMatch], predictions: List[PredictedMatch]):
    q_to_res = defaultdict(list)
    for p in predictions:
        q_to_res[p.query].append(p)
    ranks = []
    not_found = int(1<<35)
    for m in gt_matches:
        if m.query not in q_to_res:
            ranks.append(not_found)
            continue
        res = q_to_res[m.query]
        res = np.array([
            (p.score, m.db == p.db)
            for p in res
        ])
        i, = np.where(res[:, 1] == 1)
        if i.size == 0:
            ranks.append(not_found)
        else:
            i = i[0]
            rank = (res[:, 0] >= res[i, 0]).sum() - 1
            ranks.append(rank)
    return np.array(ranks)


def evaluate(
    gt_matches: List[GroundTruthMatch], predictions: List[PredictedMatch]
) -> Metrics:
    predictions = sanitize_predictions(predictions)
    y_true, probas_pred = to_arrays(gt_matches, predictions)
    p, r, t = precision_recall(y_true, probas_pred, len(gt_matches))
    ap = average_precision(r, p)
    pp90, rp90, tp90 = find_operating_point(p, r, t, required_x=0.9)  # @Precision=90%
    ranks = find_tp_ranks(gt_matches, predictions)
    recall_at_rank1 = (ranks == 0).sum() / ranks.size
    recall_at_rank10 = (ranks < 10).sum() / ranks.size

    return Metrics(
        average_precision=ap,
        precisions=p,
        recalls=r,
        thresholds=t,
        recall_at_p90=rp90,
        threshold_at_p90=tp90,
        recall_at_rank1=recall_at_rank1,
        recall_at_rank10=recall_at_rank10,
    )

def print_metrics(metrics: Metrics):
    print(f"Average Precision: {metrics.average_precision:.5f}")
    if metrics.recall_at_p90 is None:
        print("Does not reach P90")
    else:
        print(f"Recall at P90    : {metrics.recall_at_p90:.5f}")
        print(f"Threshold at P90 : {metrics.threshold_at_p90:g}")
    print(f"Recall at rank 1:  {metrics.recall_at_rank1:.5f}")
    print(f"Recall at rank 10: {metrics.recall_at_rank10:.5f}")


def generate_5_matched_names(q_vector, db_vectors, db_names):
    diff = db_vectors - q_vector
    l2_distance = np.linalg.norm(diff, axis=1)
    matched_index = np.argsort(l2_distance)[:5]
    matched_names = db_names[list(matched_index)]
    return matched_names


def confusion_matrix(data1, data2, gt, threshold, mode='euclidean'):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    if mode == 'euclidean':
        for item in gt:
            l2_distances = np.squeeze(euclidean_distances(data1[item[0]].reshape(1, -1), data2))
            if l2_distances[item[1]] <= threshold:
                tp += 1
                fn += 0
            else:
                tp += 0
                fn += 1
            l2_distance_n = np.delete(l2_distances, item[1])
            tn += l2_distance_n[l2_distance_n > threshold].shape[0]
            fp += l2_distance_n[l2_distance_n <= threshold].shape[0]
    elif mode == 'cosine':
        for item in gt:
            cos_similarity = np.squeeze(cosine_similarity(data1[item[0]].reshape(1, -1), data2))
            if cos_similarity[item[1]] >= threshold:
                tp += 1
                fn += 0
            else:
                tp += 0
                fn += 1
            cos_similarity_n = np.delete(cos_similarity, item[1])
            tn += cos_similarity_n[cos_similarity_n < threshold].shape[0]
            fp += cos_similarity_n[cos_similarity_n >= threshold].shape[0]
    return tp, tn, fp, fn


def calculate_top_accuracy(gt, query, database):
    hit = 0
    hit_cos= 0
    miss = 0
    miss_cos = 0
    hit_5 = 0
    hit_5_cos = 0
    miss_5 = 0
    miss_5_cos = 0
    for pair in gt:
        q_vector = query[pair[0]].reshape(1, -1)
        l2_distance = np.squeeze(euclidean_distances(q_vector, database))
        cos_similarity = np.squeeze(cosine_similarity(q_vector, database))
        matched_index = np.argsort(l2_distance)[:5]
        matched_cos_index = np.argsort(-cos_similarity)[:5]
        if pair[1] in matched_index:
            hit_5 += 1
        else:
            miss_5 += 1
        if pair[1] == matched_index[0]:
            hit += 1
        else:
            miss += 1
        if pair[1] in matched_cos_index:
            hit_5_cos += 1
        else:
            miss_5_cos += 1
        if pair[1] == matched_cos_index[0]:
            hit_cos += 1
        else:
            miss_cos += 1
    # accuracy = hit / (hit + miss)
    # accuracy_cos = hit_cos / (hit_cos + miss_cos)
    # accuracy_5 = hit_5 / (hit_5 + miss_5)
    # accuracy_5_cos = hit_5_cos / (hit_5_cos + miss_5_cos)
    return hit, hit_5, hit_cos, hit_5_cos


def calculate_distance(ground_truth, data1, data2):
    d_positive = []
    d_negative = []
    s_positive = []
    s_negative = []
    for item in ground_truth:
        d_positive.append(euclidean_distances(data1[item[0]].reshape(1, -1), data2[item[1]].reshape(1, -1)))
        s_positive.append(cosine_similarity(data1[item[0]].reshape(1, -1), data2[item[1]].reshape(1, -1)))
        ind_list = list(range(len(data2)))
        ind_list.remove(item[1])
        n_ind = random.choice(ind_list)
        d_negative.append(euclidean_distances(data1[item[0]].reshape(1, -1), data2[n_ind].reshape(1, -1)))
        s_negative.append(cosine_similarity(data1[item[0]].reshape(1, -1), data2[n_ind].reshape(1, -1)))
    # mean_p_diff = np.mean(np.array(d_positive))
    # mean_n_diff = np.mean(np.array(d_negative))
    # mean_p_similarity = np.mean(np.array(s_positive))
    # mean_n_similarity = np.mean(np.array(s_negative))
    return d_positive, d_negative, s_positive, s_negative

