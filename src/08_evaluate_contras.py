import sys
import torch
import numpy as np
sys.path.append('/cluster/yinan/yinan_cnn/cnn_similarity_analysis/')
import pandas as pd
from pandas.core.frame import DataFrame
import torch.nn.functional as F
from src.lib.io import *
from src.lib.metrics import *
from src.lib.siamese.args import siamese_args
from lib.siamese.dataset import get_transforms
from lib.siamese.model import ContrastiveSiameseNetwork
from data.siamese_dataloader import ContrastiveValList
from torch.utils.data import DataLoader


def conpute_confidence(args, net, data_loader):
    confidence_list = list()
    with torch.no_grad():
        for j, batch in enumerate(data_loader):
            query_img, reference_img, label = batch
            query_img = query_img.to(args.device)
            reference_img = reference_img.to(args.device)
            p_score = net(query_img, reference_img)
            confidence_list.append(p_score.cpu().numpy())
    confidence = np.vstack(confidence_list)
    confidence = np.squeeze(confidence)
    return confidence.tolist()


def evaluate(args):
    transforms = get_transforms(args)
    if args.test_dataset == "artdl" or args.test_dataset == "photoart50":
        test = pd.read_csv(args.data_path + args.test_list)
        query_test = list(test['query'])
        db_test = list(test['reference'])
        label_test = list(test['label'])
        class_list = list(test['class'])

    test_list = []
    for j in range(len(query_test)):
        test_list.append((query_test[j], db_test[j], label_test[j]))

    test_pairs = ContrastiveValList(test_list, transform=transforms, imsize=args.imsize,
                                     augmentation=None)
    test_dataloader = DataLoader(dataset=test_pairs, shuffle=False, num_workers=args.num_workers,
                                  batch_size=args.batch_size)

    net = ContrastiveSiameseNetwork(args.model)
    state_dict = torch.load(args.net + args.checkpoint)
    net.load_state_dict(state_dict)
    net.eval()
    net.to(args.device)

    conf_list = conpute_confidence(args, net, test_dataloader)

    eval_disc = {'confidence': conf_list,
                 'ground_truth': label_test,
                 'class': class_list}
    eval_data = DataFrame(eval_disc)
    precision_10 = []
    for i in range(50):
        eval_sub = eval_data[eval_data['class'] == i]
        eval_sub_sort = eval_sub.sort_values(by=['confidence'], na_position='first', ascending=False)
        sub_gt = list(eval_sub_sort['label_test'])[:11]
        tp = sub_gt.count(1)
        fp = sub_gt.count(0)
        precision_10.append(tp/(tp+fp))
    precision = np.mean(precision_10)
    print('precision: {}'.format(precision))

    


if __name__ == "__main__":

    siamese_args = siamese_args()
    if siamese_args.device == "cuda:0":
        print("hardware_image_description:", torch.cuda.get_device_name(0))
        
    evaluate(siamese_args)


