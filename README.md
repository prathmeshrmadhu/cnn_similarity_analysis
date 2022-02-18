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
this will creat an experiment path for example: 'cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55'
you can change the EXP_PATH parameter in 'cnn_similarity_analysis/src/lib/siamese/args.py' to load saved parameter,
later you can also change them when you run the experiments

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

‘--d1', '--d2'...'--p3': path to images and generated file lists

'--plots': path to save plot of losses

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
'--pca': apply trained pca on validation data

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
Use trained PCA to reduce dimensionality (you do not need to run feature extraction first):
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

## Evaluate results

```
python3 cnn_similarity_analysis/src/08_evaluate_siamese.py \
--test_dataset image_collation \
--p1_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/p1.pkl \
--p2_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/p2.pkl \
--p3_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/p3.pkl \
--d1_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/d1.pkl \
--d2_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/d2.pkl \
--d3_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/d3.pkl \
--gt_list /cluster/shared_dataset/ImageCollation/IllustrationMatcher/ground_truth/ \
--loss normal
```
'--gt_list': path to ground truth files

# ArtDL dataset
## Network training

You can train a model on Image Collation dataset via:
```
python3 cnn_similarity_analysis/src/06_train_triplet_siamese.py \
--train \
--start \
--net cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/models/ \
--plots cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/plots/ \
--num_epochs 10 \
--model resnet50 \
--margin 0.5 \
--lr 0.001 \
--imsize 256 \
--train_dataset artdl \
--optimizer sgd \
--loss normal \
--train_list /cluster/shared_dataset/DEVKitArtDL/artdl_train_list.csv \
--val_list /cluster/shared_dataset/DEVKitArtDL/artdl_val_list.csv
```
'-- train_list', '--val_list': generated .csv file with strcture ['anchor_query', 'ref_positive', 'ref_negative']

'-- start': start training or continue from a checkpoint.

'-- margin': margin in loss function.

'-- loss': 'normal' means just use triplet loss, 'custom' means use custom defined regularized loss.

'-- optimizer': choose 'sgd' or 'adam' optimizer.

'-- net': save or load checkpoint from this path.

'--plots': path to save plot of losses.

## Network training

You can train a PCA on ArtDL dataset via:
```
python3 yinan_cnn/cnn_similarity_analysis/src/09_train_pca.py \
--train_dataset artdl \
--val_dataset artdl \
--train_list /cluster/shared_dataset/DEVKitArtDL/artdl_train_list.csv \
--val_list /cluster/shared_dataset/DEVKitArtDL/artdl_val_list.csv \
--pca_file cnn_similarity_analysis/experiments/test_exp/experiment_2021-12-12_11-11-01/models/pca.vt \
--model resnet50 \
--pca \
--imsize 256 \
--net cnn_similarity_analysis/experiments/test_exp/experiment_2021-12-12_11-11-01/models/ \
--loss normal
```
'--pca_file': save trained pca to this path.

'--pca': apply trained pca on validation data.

'--loss': 'normal' means just use triplet loss, 'custom' means use custom defined regularized loss.

## Feature Extraction
You can direct use trained model to extract features from test dataset
```
python3 cnn_similarity_analysis/src/07_extract_features_siamese.py \
--net cnn_similarity_analysis/experiments/test_exp/experiment_2021-12-12_11-11-01/models/ \
--test_list /cluster/shared_dataset/DEVKitArtDL/artdl_test_list.csv \
--test_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-12-12_11-11-01/artdl_test.pkl \
--db_list /cluster/shared_dataset/DEVKitArtDL/artdl_sample_list.csv\
--db_f cnn_similarity_analysis/experiments/test_exp/experiment_2021-12-12_11-11-01/db.pkl \
--model resnet50 \
--imsize 256 \
--loss normal \
--test_dataset artdl
```
'--net': path to loaded checkpoint.

'test_list': generated test .csv file list with structure ['test_images', 'label'].

'test_f': path to save generated test features.

'db_list': generated sample .csv file, (one image for one class) for evaluation.

'db_f': path to save sample features.

## PCA Feature embedding
Apply a PCA after feature extraction via (you do not need to run feature extraction first):
```
python3 yinan_cnn/cnn_similarity_analysis/src/09_embedding_pca_features.py \
--test_dataset artdl \
--pca_file yinan_cnn/cnn_similarity_analysis/experiments/test_exp/experiment_2021-12-12_11-11-01/models/pca.vt \
--imsize 256 \
--model resnet50 \
--net yinan_cnn/cnn_similarity_analysis/experiments/test_exp/experiment_2021-12-12_11-11-01/models/ \
--loss normal \
--test_list /cluster/shared_dataset/DEVKitArtDL/artdl_test_list.csv \
--test_f yinan_cnn/cnn_similarity_analysis/experiments/test_exp/experiment_2021-12-12_11-11-01/artdl_test.pkl \
--db_list /cluster/shared_dataset/DEVKitArtDL/artdl_sample_list.csv \
--db_f yinan_cnn/cnn_similarity_analysis/experiments/test_exp/experiment_2021-12-12_11-11-01/db.pkl
```
'--pca_file': pca file to be loaded.

'--net': loaded model check point.

'test_list': generated test .csv file list with structure ['test_images', 'label'].

'test_f': path to save generated test features.

'db_list': generated sample .csv file, (one image represented one class) for evaluation.

'db_f': path to save sample features.

## Evaluation
```
python3 yinan_cnn/cnn_similarity_analysis/src/08_evaluate_siamese.py \
--test_dataset artdl \
--test_list /cluster/shared_dataset/DEVKitArtDL/artdl_test_list.csv \
--test_f yinan_cnn/cnn_similarity_analysis/experiments/test_exp/experiment_2021-12-12_11-11-01/artdl_test.pkl \
--db_f yinan_cnn/cnn_similarity_analysis/experiments/test_exp/experiment_2021-12-12_11-11-01/db.pkl \
```
'--test_list': .csv file with test ground truth labels.

'--test_f': load test features from .pkl file.

'--db_f': load sample features.

## Optuna Experiment
```
python3 yinan_cnn/cnn_similarity_analysis/src/11_optuna_experiment.py \
--train \
--start \
--num_epochs 5 \
--model vgg_fc7 \
--margin 0.5 \
--lr 0.0001 \
--weight_decay 0.001 \
--imsize 256 \
--train_dataset artdl \
--optimizer sgd \
--loss custom \
--regular 0.001 \
--train_list artdl_train_list_full.csv \
--val_list artdl_valid.csv \
--num_workers 8 \
--method feature_map \
--mining_mode offline \
--batch_size 8 \
--upper_bound 0.1 \
--lower_bound 0.00001 \
--num_experiment 20
```
'--upper_bound': upper bound of the to be searched hyperparameter 
'--lower_bound': lower bound of the to be searched hyperparameter 
'--num_experiment': number of search experiment