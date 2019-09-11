# PCA-GM

This repository contains PyTorch implementation of our ICCV 2019 paper (for oral presentation): [Learning Combinatorial Embedding Networks for Deep Graph Matching.](https://arxiv.org/abs/1904.00597)

It contains our implementation of following deep graph matching methods: 

* **GMN** Andrei Zanfir and Cristian Sminchisescu, "Deep Learning of Graph Matching." CVPR 2018.
* **PCA-GM** Runzhong Wang, Junchi Yan and Xiaokang Yang, "Learning Combinatorial Embedding Networks for Deep Graph Matching." ICCV 2019.

This repository also include training/evaluation protocol on Pascal VOC Keypoint and Willow Object Class dataset, inline with the experiment part in our ICCV 2019 paper.

## Get started

1. Install and configure pytorch 1.1+ (with GPU support)
1. Install ninja-build: ``apt-get install ninja-build``
1. Install python packages: ``pip install tensorboardX scipy easydict pyyaml``
1. If you want to run experiment on Pascal VOC Keypoint dataset:
    1. Download [VOC2011 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html) and make sure it looks like ``data/PascalVOC/VOC2011``
    1. Download [keypoint annotation for VOC2011](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz) and make sure it looks like ``data/PascalVOC/annotations``
    1. The train/test split is available in ``data/PascalVOC/voc2011_pairs.npz``
1. If you want to run experiment on Willow ObjectClass dataset:
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

## Training

Run training and evaluation

``python train_eval.py --cfg path/to/your/yaml`` 

and replace ``path/to/your/yaml`` by path to your configuration file. Default configuration files are stored in``experiments/``.

## Evaluation

Run evaluation on epoch ``k``

``python eval.py --cfg path/to/your/yaml --epoch k`` 

## Benchmark

We report performance on Pascal VOC Keypoint and Willow Object Class datasets. These are consistent with the numbers reported in our paper.

**Pascal VOC Keypoint** (mean accuracy on last column)

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
@article{wangICCV19pcagm,
author = {Wang, Runzhong and Yan, Junchi and Yang, Xiaokang},
title = {Learning Combinatorial Embedding Networks for Deep Graph Matching},
journal = {arXiv preprint arXiv:1904.00597},
year = {2019}
}
```