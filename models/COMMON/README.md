# COMMON

Official implementation of the following paper:

* Yijie Lin, Mouxing Yang, Jun Yu, Peng Hu, Changqing Zhang, Xi Peng. "Graph Matching with Bi-level Noisy
  Correspondence". ICCV, 2023.  [[paper]](https://arxiv.org/pdf/2212.04085.pdf)

COMMON proposes a deep graph matching pipeline to solve **Noisy Correspondence** in GM problem, which refers to node-level noisy correspondence (NNC) and edge-level noisy correspondence (ENC).

<img src="https://github.com/Lin-Yijie/Graph-Matching-Networks/blob/main/COMMON/docs/images/nc_example.png" alt="COMMON, ICCV 2023" width="100%">

As shown above, due to the poor recognizability and viewpoint differences between images, it is inevitable to inaccurately annotate some
keypoints with offset and confusion, leading to the mismatch between two associated nodes (NNC). The noisy node-to-node
correspondence will further contaminate the edge-to-edge correspondence (ENC).

It proposes to combine the following components to formulate the GM pipeline:

* VGG16 CNN to extract image features
* [SplineCNN](https://arxiv.org/abs/1711.08920) to embed the graph information
* GM customized quadratic contrastive learning
* Momentum teacher to adaptively penalizing the noisy assignments

## Benchmark Results

### PascalVOC - 2GM

experiment config: ``experiments/vgg16_common_voc.yaml``

pretrained model: [google drive](https://drive.google.com/file/d/1tS5h4_EjsbZP4rbaRVK4g7MU9075u4eV/view?usp=share_link)

| model                                          | year | aero   | bike   | bird   | boat   | bottle | bus    | car    | cat    | chair  | cow    | table  | dog    | horse  | mbkie  | person | plant  | sheep  | sofa   | train  | tv     | mean   |
|------------------------------------------------|------| ------ |--------| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| [COMMON](https://github.com/Lin-Yijie/Graph-Matching-Networks/tree/main/COMMON) | 2023 | 0.6560 | 0.7520 | 0.8080 | 0.7950    |0.8930 | 0.9230 | 0.9010 | 0.8180 | 0.6160 | 0.8070| 0.9500 | 0.8200 |    0.8160    | 0.7950 | 0.6660 |    0.9890 | 0.7890 | 0.8090 | 0.9930 |    0.9380 | 0.8270 |  

### Willow Object Class - 2GM

experiment config: ``experiments/vgg16_common_willow.yaml``

pretrained model: [google drive](https://drive.google.com/file/d/1Ym_P8JdHPE4MTvc8ievYmJzAezELWkP7/view?usp=share_link)

| model                    | year | remark        | Car    | Duck   | Face   | Motorbike | Winebottle | mean   |
| ------------------------ |------| ------------- | ------ | ------ | ------ | ------ |------------| ------ |
| [COMMON]([https://arxiv.org/pdf/2212.04085.pdf](https://github.com/Lin-Yijie/Graph-Matching-Networks/tree/main/COMMON)) | 2023 | -             | 0.9760 | 0.9820 | 1.0000 | 1.0000 | 0.9960     | 0.9910 |

### SPair-71k - 2GM

experiment config: ``experiments/vgg16_common_spair71k.yaml``

pretrained model: [google drive](https://drive.google.com/file/d/1gSOQciMh1dUy9C07H3ZV3PIrLry-APYy/view?usp=share_link)

| model   | year | aero   | bike   | bird   | boat   | bottle | bus    | car    | cat    | chair  | cow    | dog    | horse  | mtbike | person | plant  | sheep  | train  | tv     | mean |
| ------- |------|--------| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| [COMMON]([https://arxiv.org/pdf/2212.04085.pdf](https://github.com/Lin-Yijie/Graph-Matching-Networks/tree/main/COMMON))    | 2023 | 0.7730 | 0.6820 | 0.9200 | 0.7950 | 0.7040 | 0.9750 | 0.9160 | 0.8250 | 0.7220 | 0.8800 | 0.8000| 0.7410 | 0.8340 | 0.8280 | 0.9990 | 0.8440 | 0.9820 | 0.9980| 0.8450 |


## File Organization

```
├── model.py
|   the implementation of training/evaluation procedures of COMMON
├── model_config.py
|   the declaration of model hyperparameters
└── sconv_archs.py
    the implementation of spline convolution (SpilneCNN) operations, the same with BBGM
```

Note that the network structure of projection layer is defined in ``model.py``, which is a two-layer MLP with ReLU activation.

## Credits and Citation

Please cite the following paper if you use this model in your research:

```
@article{lin2023graph,
  title={Graph Matching with Bi-level Noisy Correspondence},
  author={Lin, Yijie and Yang, Mouxing and Yu, Jun and Hu, Peng and Zhang, Changqing and Peng, Xi},
  journal={IEEE International Conference on Computer Vision},
  year={2023}
}
```
