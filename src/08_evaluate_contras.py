import sys
import torch
import numpy as np
sys.path.append('/cluster/yinan/yinan_cnn/cnn_similarity_analysis/')
import pandas as pd
import torch.nn.functional as F
from src.lib.io import *
from src.lib.metrics import *
from src.lib.siamese.args import siamese_args
from lib.siamese.dataset import get_transforms
from lib.siamese.model import ContrastiveSiameseNetwork
from data.siamese_dataloader import ContrastiveValList
from torch.utils.data import DataLoader


def evaluate(args):
    transforms = get_transforms(args)
    if args.test_dataset == "artdl" or args.test_dataset == "photoart50":
        test = generate_test_focal_list(args)
        query_test = list(test['query'])
        db_test = list(test['reference'])
        label_test = list(test['label'])

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

    hit = 0
    num_tot = len(query_test)
    print(num_tot)

    with torch.no_grad():
        for j, batch in enumerate(test_dataloader):
            query_img, reference_img, label = batch
            query_img = query_img.to(args.device)
            reference_img = reference_img.to(args.device)
            label = label.to(args.device)
            p_score = net(query_img, reference_img)
            # p_score = p_score.unsqueeze(1)
            label = label.unsqueeze(1)
            predict = p_score.clone()
            predict[predict >= 0.5] = 1
            predict[predict < 0.5] = 0
            match_num = torch.count_nonzero(predict == label)
            hit += match_num.cpu()
    
    acc = hit/num_tot
    print("accuracy is {}".format(acc))
    


if __name__ == "__main__":

    siamese_args = siamese_args()
    if siamese_args.device == "cuda:0":
        print("hardware_image_description:", torch.cuda.get_device_name(0))
        
    evaluate(siamese_args)


