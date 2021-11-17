import sys
sys.path.append('/cluster/yinan/cnn_similarity_analysis/')
from lib.io import *
from lib.metrics import feature_map_matching, feature_vector_matching, calculate_gap
from src.lib.siamese.args import siamese_args


def reshape_feature_map(input_map):
    output = input_map.transpose((0, 2, 3, 1))
    output_map = output.reshape((output.shape[0], output.shape[1]*output.shape[2], output.shape[3]))
    return output_map


def evaluation(args):
    # q_names, q_vectors = read_pickle_descriptors(args.query_f)
    # db_names, db_vectors = read_pickle_descriptors(args.db_f)

    if args.val_dataset == 'image_collation':
        p1_names, p1_vectors = read_pickle_descriptors(args.p1_f)
        p2_names, p2_vectors = read_pickle_descriptors(args.p2_f)
        p3_names, p3_vectors = read_pickle_descriptors(args.p3_f)
        d1_names, d1_vectors = read_pickle_descriptors(args.d1_f)
        d2_names, d2_vectors = read_pickle_descriptors(args.d2_f)
        d3_names, d3_vectors = read_pickle_descriptors(args.d3_f)
        # p1_vectors = reshape_feature_map(p1_vectors)
        # p2_vectors = reshape_feature_map(p2_vectors)
        # p3_vectors = reshape_feature_map(p3_vectors)
        # d1_vectors = reshape_feature_map(d1_vectors)
        # d2_vectors = reshape_feature_map(d2_vectors)
        # d3_vectors = reshape_feature_map(d3_vectors)
        gt_p1p2 = read_config(args.gt_list + 'P1-P2.json')
        gt_p2p3 = read_config(args.gt_list + 'P2-P3.json')
        gt_p1p3 = read_config(args.gt_list + 'P1-P3.json')
        gt_d1d2 = read_config(args.gt_list + 'D1-D2.json')
        gt_d2d3 = read_config(args.gt_list + 'D2-D3.json')
        gt_d1d3 = read_config(args.gt_list + 'D1-D3.json')

        if args.mpa:
            confidence_p1p2, correct_p1p2, accuracy_p1p2 = feature_map_matching(gt_p1p2, p1_vectors, p2_vectors)
            confidence_p2p3, correct_p2p3, accuracy_p2p3 = feature_map_matching(gt_p2p3, p2_vectors, p3_vectors)
            confidence_p1p3, correct_p1p3, accuracy_p1p3 = feature_map_matching(gt_p1p3, p1_vectors, p3_vectors)
            confidence_d1d2, correct_d1d2, accuracy_d1d2 = feature_map_matching(gt_d1d2, d1_vectors, d2_vectors)
            confidence_d2d3, correct_d2d3, accuracy_d2d3 = feature_map_matching(gt_d2d3, d2_vectors, d3_vectors)
            confidence_d1d3, correct_d1d3, accuracy_d1d3 = feature_map_matching(gt_d1d3, d1_vectors, d3_vectors)
        else:
            confidence_p1p2, correct_p1p2, accuracy_p1p2 = feature_vector_matching(gt_p1p2, p1_vectors, p2_vectors)
            confidence_p2p3, correct_p2p3, accuracy_p2p3 = feature_vector_matching(gt_p2p3, p2_vectors, p3_vectors)
            confidence_p1p3, correct_p1p3, accuracy_p1p3 = feature_vector_matching(gt_p1p3, p1_vectors, p3_vectors)
            confidence_d1d2, correct_d1d2, accuracy_d1d2 = feature_vector_matching(gt_d1d2, d1_vectors, d2_vectors)
            confidence_d2d3, correct_d2d3, accuracy_d2d3 = feature_vector_matching(gt_d2d3, d2_vectors, d3_vectors)
            confidence_d1d3, correct_d1d3, accuracy_d1d3 = feature_vector_matching(gt_d1d3, d1_vectors, d3_vectors)

        gap_p1p2 = calculate_gap(confidence_p1p2, correct_p1p2, gt_p1p2)
        gap_p2p3 = calculate_gap(confidence_p2p3, correct_p2p3, gt_p2p3)
        gap_p1p3 = calculate_gap(confidence_p1p3, correct_p1p3, gt_p1p3)
        gap_d1d2 = calculate_gap(confidence_d1d2, correct_d1d2, gt_d1d2)
        gap_d2d3 = calculate_gap(confidence_d2d3, correct_d2d3, gt_d2d3)
        gap_d1d3 = calculate_gap(confidence_d1d3, correct_d1d3, gt_d1d3)

        print('Evaluation results:\n')
        print('Accuracy p1-p2: {}'.format(accuracy_p1p2))
        print('Accuracy p1-p3: {}'.format(accuracy_p1p3))
        print('Accuracy p2-p3: {}'.format(accuracy_p2p3))
        print('Accuracy d1-d2: {}'.format(accuracy_d1d2))
        print('Accuracy d1-d3: {}'.format(accuracy_d1d3))
        print('Accuracy d2-d3: {}'.format(accuracy_d2d3))
        print("\n")
        print('GAP p1-p2: {}'.format(gap_p1p2))
        print('GAP p1-p3: {}'.format(gap_p1p3))
        print('GAP p2-p3: {}'.format(gap_p2p3))
        print('GAP d1-d2: {}'.format(gap_d1d2))
        print('GAP d1-d3: {}'.format(gap_d1d3))
        print('GAP d2-d3: {}'.format(gap_d2d3))




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