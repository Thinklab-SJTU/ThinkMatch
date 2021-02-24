# PCA-GM
## Paddle-Version
To find the original work, click [here](https://github.com/Thinklab-SJTU/PCA-GM/tree/c426792d0fd566807c8fbf9ea056d7291e717263)

This repository contains PyTorch and PaddlePaddle implementation of our ICCV 2019 paper (for oral presentation): [Learning Combinatorial Embedding Networks for Deep Graph Matching.](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Learning_Combinatorial_Embedding_Networks_for_Deep_Graph_Matching_ICCV_2019_paper.pdf)

It contains our implementation of following deep graph matching methods: 

* **GMN** Andrei Zanfir and Cristian Sminchisescu, "Deep Learning of Graph Matching." CVPR 2018.
* **PCA-GM** Runzhong Wang, Junchi Yan and Xiaokang Yang, "Learning Combinatorial Embedding Networks for Deep Graph Matching." ICCV 2019.

This repository also include training/evaluation protocol on Pascal VOC Keypoint and Willow Object Class dataset, inline with the experiment part in our ICCV 2019 paper.

## Problem setting
In this codebase inline with our ICCV 2019 paper, a keypoint matching problem in images is considered. 
Given two images with their labeled keypoint positions, our models are required to predict the correspondence between keypoints in two images, which is solved by deep graph matching.
Especially, the following settings are made:
* The matched two graphs contain equally number of inliers.
* The graph structure is unknown to the model, only keypoint positions are available.
* The predicted correspondence is bijective and one-to-one correspondence of nodes in two graphs. The correspondence can be represented by a permutation matrix.


#### Preprocessing steps on Pascal VOC Keypoint dataset:
Here we describe our preprocessing steps on Pascal VOC Keypoint dataset for fair comparison and to ease future research.
1. Filter out instances with label 'difficult', 'occluded' and 'truncated', together with 'people' after 2008. 
1. Randomly select two instances from the same category.
1. Crop these two instances from the background images using bounding box annotation.
1. Filter out non-overlapping keypoints (i.e. outliers) in two instances respectively and leave only inliers. **If the resulting inlier number is less than 3, omit it** (because the problem is too trivial).
1. Build graph structures from keypoint positions for two graphs independently (in PCA-GM, it is Delaunay triangulation).

## Get started

1. Install and configure pytorch 1.1+ (with GPU support)
1. Install ninja-build: ``apt-get install ninja-build``
1. Install python packages: ``pip install tensorboardX scipy easydict pyyaml``
1. If you want to run experiment on Pascal VOC Keypoint dataset:
    1. Download [VOC2011 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html) and make sure it looks like ``data/PascalVOC/VOC2011``
    1. Download keypoint annotation for VOC2011 from [Berkeley server](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz) or [google drive](https://drive.google.com/open?id=1D5o8rmnY1-DaDrgAXSygnflX5c-JyUWR) and make sure it looks like ``data/PascalVOC/annotations``
    1. The train/test split is available in ``data/PascalVOC/voc2011_pairs.npz``
1. If you want to run experiment on Willow ObjectClass dataset, please refer to [this section](#detailed-instructions-on-willow-object-class-dataset)

## Training

Run training and evaluation

``python train_eval.py --cfg path/to/your/yaml`` 

and replace ``path/to/your/yaml`` by path to your configuration file. Default configuration files are stored in``experiments/``.

## Evaluation

Run evaluation on epoch ``k``

``python eval.py --cfg path/to/your/yaml --epoch k`` 

## Detailed instructions on Willow Object Class dataset

1. Download [Willow ObjectClass dataset](http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip)
1. Unzip the dataset and make sure it looks like ``data/WILLOW-ObjectClass``
1. If you want to initialize model weights on Pascal VOC Keypoint dataset (as reported in the paper), please:
    1. Remove cached VOC index ``rm data/cache/voc_db_*``
    1. Uncomment L156-159 in ``data/pascal_voc.py`` to filter out overlapping images in Pascal VOC
    1. Train model on Pascal VOC Keypoint dataset, e.g. ``python train_eval.py --cfg experiments/vgg16_pca_voc.yaml``
    1. Copy Pascal VOC's cached weight to the corresponding directory of Willow. E.g. copy Pascal VOC's model weight at epoch 10 for willow
    ```bash
    cp output/vgg16_pca_voc/params/*_0010.pt output/vgg16_pca_willow/params/
    ```
    1. Set the ``START_EPOCH`` parameter to load the pretrained weights, e.g. in ``experiments/vgg16_pca_willow.yaml`` set
    ```yaml
    TRAIN:
       START_EPOCH: 10
    ```


## Benchmark

We report performance on Pascal VOC Keypoint and Willow Object Class datasets. These are consistent with the numbers reported in our paper.

**Pascal VOC Keypoint** (mean accuracy is on the last column)

| method | aero | bike | bird | boat | bottle | bus  | car  | cat  | chair | cow  | table | dog  | horse | mbike | person | plant | sheep | sofa | train | tv   | mean     |
| ------ | ---- | ---- | ---- | ---- | ------ | ---- | ---- | ---- | ----- | ---- | ----- | ---- | ----- | ----- | ------ | ----- | ----- | ---- | ----- | ---- | -------- |
| GMN    | 31.9 | 47.2 | 51.9 | 40.8 | 68.7   | 72.2 | 53.6 | 52.8 | 34.6  | 48.6 | 72.3  | 47.7 | 54.8  | 51.0  | 38.6   | 75.1  | 49.5  | 45.0 | 83.0  | 86.3 | 55.3     |
| PCA-GM | 40.9 | 55.0 | 65.8 | 47.9 | 76.9   | 77.9 | 63.5 | 67.4 | 33.7  | 65.5 | 63.6  | 61.3 | 68.9  | 62.8  | 44.9   | 77.5  | 67.4  | 57.5 | 86.7  | 90.9 | **63.8** |

**Willow Object Class**

| method        | face      | m-bike   | car      | duck     | w-bottle |
| ------------- | --------- | -------- | -------- | -------- | -------- |
| HARG-SSVM     | 91.2      | 44.4     | 58.4     | 55.2     | 66.6     |
| GMN-VOC       | 98.1      | 65.0     | 72.9     | 74.3     | 70.5     |
| GMN-Willow    | 99.3      | 71.4     | 74.3     | 82.8     | 76.7     |
| PCA-GM-VOC    | **100.0** | 69.8     | 78.6     | 82.4     | 95.1     |
| PCA-GM-Willow | **100.0** | **76.7** | **84.0** | **93.5** | **96.9** |

Suffix *VOC* means model trained on VOC dataset, and suffix *Willow* means model tuned on Willow dataset.

## Citation

If you find this repository helpful to your research, please consider citing:
```text
@InProceedings{Wang_2019_ICCV,
author = {Wang, Runzhong and Yan, Junchi and Yang, Xiaokang},
title = {Learning Combinatorial Embedding Networks for Deep Graph Matching},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```
