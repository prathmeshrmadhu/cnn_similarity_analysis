import sys
sys.path.append('/cluster/yinan/cnn_similarity_analysis/')
from lib.io import *
from lib.metrics import generate_5_matched_names
from src.lib.siamese.args import siamese_args


def evaluation(args):
    q_names, q_vectors = read_pickle_descriptors(args.query_f)
    db_names, db_vectors = read_pickle_descriptors(args.db_f)
    matched_list = []
    hit = 0
    miss = 0
    hit_5 = 0
    miss_5 = 0

    for i in range(len(q_names)):
        vec = q_vectors[i]
        # diff = db_vectors - vec
        # l2_distance = np.linalg.norm(diff, axis=1)
        # matched_index = np.argsort(l2_distance)[:5]
        matched_names = generate_5_matched_names(vec, db_vectors, db_names)
        matched_list.append(matched_names)
        if matched_names[0] == db_names[i]:
            hit += 1
        else:
            miss += 1
        if db_names[i] in matched_names:
            hit_5 += 1
        else:
            miss_5 += 1

    accuracy = hit / (hit + miss)
    accuracy_5 = hit_5 / (hit_5 + miss_5)
    print('TOP_1 accuracy:{}'.format(accuracy))
    print('TOP_5 accuracy:{}'.format(accuracy_5))

    fw = open(args.matched_f, 'wb')
    pickle.dump(matched_list, fw)
    fw.close()


if __name__ == '__main__':
    eval_args = siamese_args()
    evaluation(eval_args)