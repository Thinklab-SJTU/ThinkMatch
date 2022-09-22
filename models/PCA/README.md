# PCA-GM

Our implementation of the following papers:
* Runzhong Wang, Junchi Yan and Xiaokang Yang. "Combinatorial Learning of Robust Deep Graph Matching: an Embedding based Approach." _TPAMI 2020_.
    [[paper]](https://ieeexplore.ieee.org/abstract/document/9128045/), [[project page]](https://thinklab.sjtu.edu.cn/IPCA_GM.html)
* Runzhong Wang, Junchi Yan and Xiaokang Yang. "Learning Combinatorial Embedding Networks for Deep Graph Matching." _ICCV 2019_. 
  [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Learning_Combinatorial_Embedding_Networks_for_Deep_Graph_Matching_ICCV_2019_paper.pdf)


**PCA-GM** proposes the first deep graph matching network based on graph embedding, and it is composed of the following components:
* VGG16 CNN to extract image features
* Delaunay triangulation to build graphs
* Graph Convolutional Network (GCN) with cross-graph convolution to embed graph structure features
* Solving the resulting Linear Assignment Problem by Sinkhorn network
* Supervised learning based on cross entropy loss (known as "permutation loss" in this paper)

Such a CNN-GNN-Sinkhorn-CrossEntropy framework has be adopted by many following papers.

**PCA-GM** is proposed in the conference version. In the journal version, we propose **IPCA-GM**, whereby the cross-graph convolution step is implemented in an iterative manner.
The motivation of the iterative update scheme is that: better embedding features will lead to better cross-graph weight matrix and vice versa.

## Benchmark Results
### PascalVOC - 2GM
* PCA-GM
  
  experiment config: ``experiments/vgg16_pca_voc.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/1JnX3cSPvRYBSrDKVwByzp7CADgVCJCO_/view?usp=sharing)
  
* IPCA-GM

  experiment config: ``experiments/vgg16_ipca_voc.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/1TGrbSQRmUkClH3Alz2OCwqjl8r8gf5yI/view?usp=sharing)

| model                  | year | aero   | bike   | bird   | boat   | bottle | bus    | car    | cat    | chair  | cow    | table  | dog    | horse  | mbkie  | person | plant  | sheep  | sofa   | train  | tv     | mean   |
| ---------------------- | ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| [PCA-GM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#pca-gm) | 2019 | 0.4979 | 0.6193 | 0.6531 | 0.5715 | 0.7882 | 0.7556 | 0.6466 | 0.6969 | 0.4164 | 0.6339 | 0.5073 | 0.6705 | 0.6671 | 0.6164 | 0.4447 | 0.8116 | 0.6782 | 0.5922 | 0.7845 | 0.9042 | 0.6478 |
| [IPCA-GM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#pca-gm) | 2020 | 0.5378 | 0.6622 | 0.6714 | 0.6120 | 0.8039 | 0.7527 | 0.7255 | 0.7252 | 0.4455 | 0.6524 | 0.5430 | 0.6724 | 0.6790 | 0.6421 | 0.4793 | 0.8435 | 0.7079 | 0.6398 | 0.8380 | 0.9083 | 0.6770 |

### Willow Object Class - 2GM
* PCA-GM
  
  experiment config: ``experiments/vgg16_pca_willow.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/1BYFevb7C1mUW9vK-L9wOo0Omtp4V15Ub/view?usp=sharing)
  
* IPCA-GM

  experiment config: ``experiments/vgg16_ipca_willow.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/1-OcLEwlKiudxs3KoKbFW56kzspqFsoWH/view?usp=sharing)

| model                    | year | remark          | Car    | Duck   | Face   | Motorbike | Winebottle | mean   |
| ------------------------ | ---- | --------------- | ------ | ------ | ------ | --------- | ---------- | ------ |
| [PCA-GM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#pca-gm) | 2019 | -               | 0.8760 | 0.8360 | 1.0000 | 0.7760    | 0.8840     | 0.8744 |
| [IPCA-GM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#pca) | 2020 | -               | 0.9040 | 0.8860 | 1.0000 | 0.8300    | 0.8830     | 0.9006 |

## File Organization
```
├── affinity_layer.py
|   the implementation of affinity layer to compute the affinity matrix for PCA-GM and IPCA-GM
├── model.py
|   the implementation of training/evaluation procedures of PCA-GM and IPCA-GM
└── model_config.py
    the declaration of model hyperparameters
```

## Credits and Citation

Please cite the following paper if you use this model in your research:
```
@inproceedings{WangICCV19,
  author = {Wang, Runzhong and Yan, Junchi and Yang, Xiaokang},
  title = {Learning Combinatorial Embedding Networks for Deep Graph Matching},
  booktitle = {IEEE International Conference on Computer Vision},
  pages={3056--3065},
  year = {2019}
}
```
