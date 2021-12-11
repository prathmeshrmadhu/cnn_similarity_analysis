# Creat Experiment
```
python3 cnn_similarity_analysis/src/01_create_experiment.py \
--exp_directory test_exp \
--dataset_name image_collation \
--image_size 248 \
--rot_factor 45 \
--scale_factor 0.35 \
--num_epochs 20 \
--learning_rate 0.00001 \
--learning_rate_factor 0.5 \
--patience 20 \
--batch_size 10 \
--save_frequency 1 \
--optimizer sgd \
--momentum 0.9 \
--gamma1 0.99 \
--gamma2 0.9
```

# Image collation dataset
## Network training

You can train a model on Image Collation dataset via:
```
python3 cnn_similarity_analysis/src/06_train_triplet_siamese.py \
--start \
--gt_list /cluster/shared_dataset/ImageCollation/IllustrationMatcher/ground_truth/ \
--d1 /cluster/shared_dataset/ImageCollation/ManuscriptDownloader/download/D1/
--d2 /cluster/shared_dataset/ImageCollation/ManuscriptDownloader/download/D2/
--d3 /cluster/shared_dataset/ImageCollation/ManuscriptDownloader/download/D3/
--p1 /cluster/shared_dataset/ImageCollation/ManuscriptDownloader/download/P1/
--p2 /cluster/shared_dataset/ImageCollation/ManuscriptDownloader/download/P2/
--p3 /cluster/shared_dataset/ImageCollation/ManuscriptDownloader/download/P3/
--net cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/models/ \
--plots cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/plots/ \
--num_epochs 10 \
--model resnet50 \
--margin 0.5 \
--lr 0.001 \
--imsize 256 \
--train_dataset image_collation \
--optimizer sgd \
--loss normal 
```
'-- start': start training or continue from a checkpoint.

'-- margin': margin in loss function.

'-- loss': 'normal' means just use triplet loss, 'custom' means use custom defined regularized loss.

'-- optimizer': choose 'sgd' or 'adam' optimizer.

'-- net': save or load checkpoint from this path

## PCA training
You can train a pca via:
```
python3 cnn_similarity_analysis/src/09_train_pca.py \
--train_dataset image_collation \
--val_dataset image_collation \
--d1 /cluster/shared_dataset/ImageCollation/ManuscriptDownloader/download/D1/
--d2 /cluster/shared_dataset/ImageCollation/ManuscriptDownloader/download/D2/
--d3 /cluster/shared_dataset/ImageCollation/ManuscriptDownloader/download/D3/
--p1 /cluster/shared_dataset/ImageCollation/ManuscriptDownloader/download/P1/
--p2 /cluster/shared_dataset/ImageCollation/ManuscriptDownloader/download/P2/
--p3 /cluster/shared_dataset/ImageCollation/ManuscriptDownloader/download/P3/
--pca_file cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/models/pca.vt \
--gt_list /cluster/shared_dataset/ImageCollation/IllustrationMatcher/ground_truth/  \
--model resnet50 \
--pca \
--imsize 256 \
--net cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/models/ \
--loss normal
```
'--pca_file': save trained pca to this path

## Feature extraction

The feature extraction process can be done via:
```
python3 cnn_similarity_analysis/src/07_extract_features_siamese.py \
--net cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/models/ \
--d1 /cluster/shared_dataset/ImageCollation/ManuscriptDownloader/download/D1/
--d2 /cluster/shared_dataset/ImageCollation/ManuscriptDownloader/download/D2/
--d3 /cluster/shared_dataset/ImageCollation/ManuscriptDownloader/download/D3/
--p1 /cluster/shared_dataset/ImageCollation/ManuscriptDownloader/download/P1/
--p2 /cluster/shared_dataset/ImageCollation/ManuscriptDownloader/download/P2/
--p3 /cluster/shared_dataset/ImageCollation/ManuscriptDownloader/download/P3/
--p1_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/p1.pkl \
--p2_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/p2.pkl \
--p3_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/p3.pkl \
--d1_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/d1.pkl \
--d2_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/d2.pkl \
--d3_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/d3.pkl \
--model resnet50 \
--imsize 256 \
--loss custom \
--test_dataset image_collation \
```
'p1_f' - 'd3_f': save extracted features to these paths

## PCA Feature embedding
Use trained PCA to reduce dimensionality:
```
python3 cnn_similarity_analysis/src/09_embedding_pca_features.py \
--val_dataset image_collation \
--pca_file cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/models/pca.vt \
--p1_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/p1.pkl \
--p2_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/p2.pkl \
--p3_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/p3.pkl \
--d1_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/d1.pkl \
--d2_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/d2.pkl \
--d3_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/d3.pkl \
--imsize 256 \
--model random \
--net cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/models/ \
--pca \
--loss normal
```
'--pca_file': load pca from this path

