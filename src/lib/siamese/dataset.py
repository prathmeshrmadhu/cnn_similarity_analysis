import os
import random
import torchvision

QUERY = '/cluster/shared_dataset/isc2021/query_images/'
REFERENCE = '/cluster/shared_dataset/isc2021/reference_images/'
TRAIN = '/cluster/shared_dataset/Inria/train/'
TEST = '/cluster/shared_dataset/Inria/test/'


def get_transforms(args):
    # transform
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if args.model == "transformer" or args.model == "visformer":
        transforms = [
            torchvision.transforms.Resize((384, 384)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ]
    else:
        transforms = [
            torchvision.transforms.Resize((args.imsize, args.imsize)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ]

    # if args.transpose != -1:
    #     transforms.insert(TransposeTransform(args.transpose), 0)

    transforms = torchvision.transforms.Compose(transforms)

    return transforms

def generate_siamese_train_dataset(query_list, gt_list, train_list, len_data):
    # TODO: generate training list with length len_data
    # random.seed(1)
    t_list = list()
    gt_list = gt_list[0: int(len(gt_list)*3/4)]
    for i in range(len_data):
        label = random.randint(0, 1)
        if label == 0:
            gt = random.sample(gt_list, 1)[0]
            q = gt.query
            r = gt.db
            q = QUERY + q + ".jpg"
            r = REFERENCE + r + ".jpg"
            t_list.append((q, r, label))
        else:
            q = random.sample(query_list, 1)[0]
            r = random.sample(train_list, 1)[0]
            q = QUERY + q + ".jpg"
            t = TRAIN + r + ".jpg"
            t_list.append((q, t, label))
    return t_list


def generate_validation_dataset(query_list, gt_list, train_list, len_data):
    # TODO: generate training list with length len_data
    # random.seed(3)
    v_list = list()
    gt_list = gt_list[int(len(gt_list)*3/4): -1]
    for i in range(len_data):
        label = random.randint(0, 1)
        if label == 0:
            gt = random.sample(gt_list, 1)[0]
            q = gt.query
            r = gt.db
            q = QUERY + q + ".jpg"
            r = REFERENCE + r + ".jpg"
            v_list.append((q, r, label))
        else:
            q = random.sample(query_list, 1)[0]
            t = random.sample(train_list, 1)[0]
            # q = QUERY + q + ".jpg"
            # t = TRAIN + r + ".jpg"
            v_list.append((q, t, label))
    return v_list


def generate_extraction_dataset(query_list, db_list, train_list):
    query_images = [TEST + q + ".jpg" for q in query_list]
    db_images = [TEST + r + ".jpg" for r in db_list]
    train_images = [TEST + t + ".jpg" for t in train_list]
    return query_images, db_images, train_images


def generate_train_dataset(query, p_list, n_list):
    query_images = [TRAIN + q + ".jpg" for q in query]
    p_images = [TRAIN + p + ".jpg" for p in p_list]
    n_images = [TRAIN + n + ".jpg" for n in n_list]
    return query_images, p_images, n_images


def add_file_list(query, ref_p, ref_n, gt, data1, data2):
    for i in len(gt):
        query.append(data1[gt[i][0]])
        ref_p.append(data2[gt[i][1]])
        b = data2.copy()
        b.remove(data2[gt[i][1]])
        ref_n.append(random.choice(b))

    return query, ref_p, ref_n