# BBGM

Our implementation of the following paper:
* Michal Rolínek, Paul Swoboda, Dominik Zietlow, Anselm Paulus, Vít Musil, Georg Martius. "Deep Graph Matching via Blackbox Differentiation of Combinatorial Solvers." _ECCV 2020_. 
    [[paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730409.pdf)

BBGM proposes a new feature extractor by using the global feature of VGG16 and a Spline Convolution module, and such an improved backbone is found effective for image matching problems. 
NOTE: Spline convolution is officially named as [SplineCNN](https://arxiv.org/abs/1711.08920), however, since the term "CNN" is conventionally used for image feature extractors, and SplineCNN works on graphs, here we name it as "Spline Convolution" for disambiguation.

The resulting quadratic assignment problem is solved by a discrete [LPMP solver](https://github.com/LPMP/LPMP), and the gradient is approximated by the [black-box combinatorial solver technique](https://arxiv.org/abs/1912.02175).

## Benchmark Results
### PascalVOC - 2GM

experiment config: ``experiments/vgg16_bbgm_voc.yaml``

pretrained model: [google drive](https://drive.google.com/file/d/1RxC7daviZf3kz2Nvr76DldR_oMHfNB4h/view?usp=sharing)

| model                  | year | aero   | bike   | bird   | boat   | bottle | bus    | car    | cat    | chair  | cow    | table  | dog    | horse  | mbkie  | person | plant  | sheep  | sofa   | train  | tv     | mean   |
| ---------------------- | ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| [BBGM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#bbgm) | 2020 | 0.6187 | 0.7106 | 0.7969 | 0.7896 | 0.8740 | 0.9401 | 0.8947 | 0.8022 | 0.5676 | 0.7914 | 0.6458 | 0.7892 | 0.7615 | 0.7512 | 0.6519 | 0.9818 | 0.7729 | 0.7701 | 0.9494 | 0.9393 | 0.7899 |

### Willow Object Class - 2GM & MGM

experiment config: ``experiments/vgg16_bbgm_willow.yaml``

pretrained model: [google drive](https://drive.google.com/file/d/1bt8wBeimM0ofm3QWEVOWWKxoIVRfFwi-/view?usp=sharing)

| model                    | year | remark          | Car    | Duck   | Face   | Motorbike | Winebottle | mean   |
| ------------------------ | ---- | --------------- | ------ | ------ | ------ | --------- | ---------- | ------ |
| [BBGM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#bbgm) | 2020 | -               | 0.9680 | 0.8990 | 1.0000 | 0.9980    | 0.9940     | 0.9718 |

## File Organization
```
├── affinity_layer.py
|   the implementation of the inner-product with weight affinity layer proposed by BBGM
├── model.py
|   the implementation of training/evaluation procedures of BBGM
├── model_config.py
|   the declaration of model hyperparameters
└── sconv_archs.py
    the implementation of spline convolution (SpilneCNN) operations
```

## Remarks
It is worth noting that our reproduced result of BBGM is different from the result from their paper. By looking into the code released by BBGM, we find that the authors of BBGM filter out keypoints which are out of the bounding box but we do not. Therefore, the number of nodes of graphs in BBGM is smaller than ours, and the graph matching problem is less challenging than ours. In this repository, we modify BBGM to fit into our setting, and we report our reproduced result for fair comparison.

## Credits and Citation
This code is developed based on the [official implementation of BBGM](https://github.com/martius-lab/blackbox-deep-graph-matching). The code is modified to fit our general framework.

Please cite the following paper if you use this model in your research:
```
@inproceedings{RolinekECCV20,
  title={Deep Graph Matching via Blackbox Differentiation of Combinatorial Solvers},
  author={Rol{\'\i}nek, Michal and Swoboda, Paul and Zietlow, Dominik and Paulus, Anselm and Musil, V{\'\i}t and Martius, Georg},
  booktitle={European Conference on Computer Vision},
  pages={407--424},
  year={2020},
  organization={Springer}
}
```
