
# Usage
## Image collation dataset
- Training process
example:
```
python3 cnn_similarity_analysis/src/06_train_triplet_siamese.py --train --start --gt_list /cluster/shared_dataset/ImageCollation/IllustrationMatcher/ground_truth/ --i0 0 --i1 50 --net cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/models/ --images cnn_similarity_analysis/experiments/test_exp/experiment_2021-10-10_12-16-55/plots/ --num_epochs 10 --model resnet50 --margin 0.5 --lr 0.001 --imsize 256 --train_dataset artdl --optimizer sgd --loss normal --train_list /cluster/shared_dataset/DEVKitArtDL/artdl_train_list.csv --val_list /cluster/shared_dataset/DEVKitArtDL/artdl_val_list.csv
```

