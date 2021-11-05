# CIE-H

This folder contains our implementation of the following paper:
 * Tianshu Yu, Runzhong Wang, Junchi Yan, Baoxin Li. "Learning deep graph matching with channel-independent embedding and Hungarian attention." _ICLR 2020_.
    [[paper]](https://openreview.net/forum?id=rJgBd2NYPH)

CIE-H follows the CNN-GNN-metric-Sinkhorn pipeline proposed by PCA-GM, and it improves PCA-GM from two aspects:
1) A channel-independent edge embedding module for better graph feature extraction;
2) A Hungarian Attention module that dynamically constructs a structured and sparsely connected layer,
taking into account the most contributing matching pairs as hard attention during training.

## Benchmark Results
### PascalVOC - 2GM

experiment config: ``experiments/vgg16_cie_voc.yaml``

pretrained model: https://drive.google.com/file/d/1oRwcnw06t1rCbrIN_7p8TJZY-XkBOFEp/view?usp=sharing

| model                 | year | aero   | bike   | bird   | boat   | bottle | bus    | car    | cat    | chair  | cow    | table  | dog    | horse  | mbkie  | person | plant  | sheep  | sofa   | train  | tv     | mean   |
| --------------------- | ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| [CIE-H](/models/CIE)   | 2020 | 0.4994 | 0.6313 | 0.7065 | 0.5298 | 0.8243 | 0.7536 | 0.6766 | 0.7230 | 0.4235 | 0.6688 | 0.6990 | 0.6952 | 0.7074 | 0.6196 | 0.4667 | 0.8504 | 0.7000 | 0.6175 | 0.8023 | 0.9178 | 0.6756 |

## File Organization
```
├── model.py
|   the implementation of training/evaluation procedures of BBGM
└── model_config.py
    the declaration of model hyperparameters
```
some files are borrowed from ``models/PCA``

## Credits and Citation

Please cite the following paper if you use this model in your research:
```
@inproceedings{YuICLR20,
  title={Learning deep graph matching with channel-independent embedding and Hungarian attention},
  author={Yu, Tianshu and Wang, Runzhong and Yan, Junchi and Li, Baoxin},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```
