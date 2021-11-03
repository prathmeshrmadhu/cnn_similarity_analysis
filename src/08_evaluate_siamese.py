import sys
sys.path.append('/cluster/yinan/cnn_similarity_analysis/')
from lib.io import *
from lib.metrics import generate_5_matched_names, confusion_matrix
from src.lib.siamese.args import siamese_args


def evaluation(args):
    q_names, q_vectors = read_pickle_descriptors(args.query_f)
    db_names, db_vectors = read_pickle_descriptors(args.db_f)
    matched_list = []
    hit = 0
    miss = 0
    hit_5 = 0
    miss_5 = 0
    precision = []

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
        tp, tn, fp, fn = confusion_matrix(vec, db_vectors, i, args.threshold)
        precision.append(tp/(tp+fp))
    mAP = np.mean(np.asarray(precision))
    accuracy = hit / (hit + miss)
    accuracy_5 = hit_5 / (hit_5 + miss_5)
    print('TOP_1 accuracy:{}'.format(accuracy))
    print('TOP_5 accuracy:{}'.format(accuracy_5))
    print('mAP: {}'.format(mAP))

    fw = open(args.matched_f, 'wb')
    pickle.dump(matched_list, fw)
    fw.close()

    # if visualization:
    #     test_list = generate_validation_dataset(query_images, groundtruth_list, train_images, 50)
    #     test_data = ContrastiveValList(test_list, transform=transforms, imsize=args.imsize)
    #     test_loader = DataLoader(dataset=test_data, shuffle=True, num_workers=args.num_workers,
    #                              batch_size=1)
    #     with torch.no_grad():
    #         distance_p = []
    #         distance_n = []
    #         for i, data in enumerate(test_loader, 0):
    #             img_name = 'test_{}.jpg'.format(i)
    #             img_pth = args.images + img_name
    #             query_img, reference_img, label = data
    #             concatenated = torch.cat((query_img, reference_img), 0)
    #             query_img = query_img.to(args.device)
    #             reference_img = reference_img.to(args.device)
    #             score = net(query_img, reference_img).cpu()
    #
    #             if label == 0:
    #                 label = 'matched'
    #                 distance_p.append(score.item())
    #                 print('matched with distance: {:.4f}\n'.format(score.item()))
    #             if label == 1:
    #                 label = 'not matched'
    #                 distance_n.append(score.item())
    #                 print('not matched with distance: {:.4f}\n'.format(score.item()))
    #
    #             imshow(torchvision.utils.make_grid(concatenated),
    #                    'Dissimilarity: {:.2f} Label: {}'.format(score.item(), label), should_save=True, pth=img_pth)
    #     mean_distance_p = torch.mean(torch.Tensor(distance_p))
    #     mean_distance_n = torch.mean(torch.Tensor(distance_n))
    #     print('-------------------------------------------------------------')
    #     print('not matched mean distance: {:.4f}\n'.format(mean_distance_n))
    #     print('matched mean distance: {:.4f}\n'.format(mean_distance_p))


if __name__ == '__main__':
    eval_args = siamese_args()
    evaluation(eval_args)