import sys
sys.path.append('/cluster/yinan/cnn_similarity_analysis/')
from lib.io import *
from lib.metrics import generate_5_matched_names, confusion_matrix, calculate_top_accuracy
from src.lib.siamese.args import siamese_args


def evaluation(args):
    # q_names, q_vectors = read_pickle_descriptors(args.query_f)
    # db_names, db_vectors = read_pickle_descriptors(args.db_f)

    p1_names, p1_vectors = read_pickle_descriptors(args.p1_f)
    p2_names, p2_vectors = read_pickle_descriptors(args.p2_f)
    p3_names, p3_vectors = read_pickle_descriptors(args.p3_f)
    gt_p1p2 = read_config(args.gt_list + 'P1-P2.json')
    gt_p2p3 = read_config(args.gt_list + 'P2-P3.json')
    gt_p1p3 = read_config(args.gt_list + 'P1-P3.json')

    matched_list = []
    precision = []

    hit_p1p2, hit_5_p1p2, hit_p1p2_cos, hit_5_p1p2_cos = calculate_top_accuracy(gt_p1p2, p1_vectors, p2_vectors)
    hit_p2p3, hit_5_p2p3, hit_p2p3_cos, hit_5_p2p3_cos = calculate_top_accuracy(gt_p2p3, p2_vectors, p3_vectors)
    hit_p1p3, hit_5_p1p3, hit_p1p3_cos, hit_5_p1p3_cos = calculate_top_accuracy(gt_p1p3, p1_vectors, p3_vectors)

    accuracy = (hit_p1p2 + hit_p2p3 + hit_p1p3) / (len(gt_p1p2) + len(gt_p1p2) + len(gt_p1p2))
    accuracy_5 = (hit_5_p1p2 + hit_5_p2p3 + hit_5_p1p3) / (len(gt_p1p2) + len(gt_p1p2) + len(gt_p1p2))
    accuracy_cos = (hit_p1p2_cos + hit_p2p3_cos + hit_p1p3_cos) / (len(gt_p1p2) + len(gt_p1p2) + len(gt_p1p2))
    accuracy_5_cos = (hit_5_p1p2_cos + hit_5_p2p3_cos + hit_5_p1p3_cos) / (len(gt_p1p2) + len(gt_p1p2) + len(gt_p1p2))

    tp_d1d2, tn_d1d2, fp_d1d2, fn_d1d2 = confusion_matrix(p1_vectors, p2_vectors, gt_p1p2, args.threshold_d)
    tp_d2d3, tn_d2d3, fp_d2d3, fn_d2d3 = confusion_matrix(p2_vectors, p3_vectors, gt_p2p3, args.threshold_d)
    tp_d1d3, tn_d1d3, fp_d1d3, fn_d1d3 = confusion_matrix(p1_vectors, p3_vectors, gt_p1p3, args.threshold_d)
    mAP = (tp_d1d2 + tp_d2d3 + tp_d1d3) / (tp_d1d2 + tp_d2d3 + tp_d1d3 + fp_d1d2 + fp_d2d3 + fp_d1d3)

    tp_d1d2, tn_d1d2, fp_d1d2, fn_d1d2 = confusion_matrix(p1_vectors, p2_vectors, gt_p1p2, args.threshold_s, False)
    tp_d2d3, tn_d2d3, fp_d2d3, fn_d2d3 = confusion_matrix(p2_vectors, p3_vectors, gt_p2p3, args.threshold_s, False)
    tp_d1d3, tn_d1d3, fp_d1d3, fn_d1d3 = confusion_matrix(p1_vectors, p3_vectors, gt_p1p3, args.threshold_s, False)
    mAP_cos = (tp_d1d2 + tp_d2d3 + tp_d1d3) / (tp_d1d2 + tp_d2d3 + tp_d1d3 + fp_d1d2 + fp_d2d3 + fp_d1d3)


    # print('mAP: {}'.format(mAP))
    print('TOP_1 accuracy :{}'.format(accuracy))
    print('TOP_5 accuracy :{}'.format(accuracy_5))
    print('\n')
    print('TOP_1 accuracy with cosine similarity:{}'.format(accuracy_cos))
    print('TOP_5 accuracy with cosine similarity:{}'.format(accuracy_5_cos))
    print('\n')
    print('mAP: {}'.format(mAP))
    print('mAP with cosine similarity: {}'.format(mAP_cos))


    # fw = open(args.matched_f, 'wb')
    # pickle.dump(matched_list, fw)
    # fw.close()

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