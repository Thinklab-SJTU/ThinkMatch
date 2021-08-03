# CIE-H

Our implementation of the following paper:
 * Tianshu Yu, Runzhong Wang, Junchi Yan, Baoxin Li. "Learning deep graph matching with channel-independent embedding and Hungarian attention." _ICLR 2020_.
    [[paper]](https://openreview.net/forum?id=rJgBd2NYPH)

CIE-H follows the CNN-GNN-metric-Sinkhorn pipeline proposed by PCA-GM, and it improves PCA-GM from two aspects:
1) A channel-independent edge embedding module for better graph feature extraction;
2) A Hungarian Attention module that dynamically constructs a structured and sparsely connected layer,
taking into account the most contributing matching pairs as hard attention during training.

## Benchmark Results
### PascalVOC - 2GM

experiment config: ``experiments/vgg16_cie_voc.yaml``

pretrained model: [google drive](https://drive.google.com/file/d/1oRwcnw06t1rCbrIN_7p8TJZY-XkBOFEp/view?usp=sharing)

| model                 | year | aero   | bike   | bird   | boat   | bottle | bus    | car    | cat    | chair  | cow    | table  | dog    | horse  | mbkie  | person | plant  | sheep  | sofa   | train  | tv     | mean   |
| --------------------- | ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| [CIE-H](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#cie-h) | 2020 | 0.5250 | 0.6858 | 0.7015 | 0.5706 | 0.8207 | 0.7700 | 0.7073 | 0.7313 | 0.4383 | 0.6994 | 0.6237 | 0.7018 | 0.7031 | 0.6641 | 0.4763 | 0.8525 | 0.7172 | 0.6400 | 0.8385 | 0.9168 | 0.6892 |

### Willow Object Class - 2GM

experiment config: ``experiments/vgg16_cie_willow.yaml``

pretrained model: [google drive](https://drive.google.com/file/d/1aUdNTWlFxk-sf-bj08ADUoo9CSIQjzDb/view?usp=sharing)

| model                    | year | remark          | Car    | Duck   | Face   | Motorbike | Winebottle | mean   |
| ------------------------ | ---- | --------------- | ------ | ------ | ------ | --------- | ---------- | ------ |
| [CIE-H](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#cie-h) | 2020 | -               | 0.8581 | 0.8206 | 0.9994 | 0.8836    | 0.8871     | 0.8898 |


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
