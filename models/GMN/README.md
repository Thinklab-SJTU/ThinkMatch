# GMN

Our implementation of the following paper:
* Andrei Zanfir and Cristian Sminchisescu. "Deep Learning of Graph Matching." _CVPR 2018_.
    [[paper]](http://openaccess.thecvf.com/content_cvpr_2018/html/Zanfir_Deep_Learning_of_CVPR_2018_paper.html)

GMN proposes the first deep graph matching pipeline which is end-to-end trainable via supervised learning and gradient descent. It proposes to combine the following components to formulate the graph matching pipeline:
* VGG16 CNN to extract image features
* Delaunay triangulation to build graphs
* Building affinity matrix efficiently via Factorized Graph Matching (FGM)
* Solving the resulting Quadratic Assignment Problem (QAP) by Spectral Matching (SM) and Sinkhorn algorithm which are differentiable
* Supervised learning based on pixel offset loss (known as "Robust loss" in the paper)

## Benchmark Results
### PascalVOC - 2GM
experiment config: ``experiments/vgg16_gmn_voc.yaml``

pretrained model: [google drive](https://drive.google.com/file/d/1X8p4XjzqGDniYirwSNqsQhBLWB5VcqN2/view?usp=sharing)

| model                  | year | aero   | bike   | bird   | boat   | bottle | bus    | car    | cat    | chair  | cow    | table  | dog    | horse  | mbkie  | person | plant  | sheep  | sofa   | train  | tv     | mean   |
| ---------------------- | ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| [GMN](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#gmn) | 2018 | 0.4163 | 0.5964 | 0.6027 | 0.4795 | 0.7918 | 0.7020 | 0.6735 | 0.6488 | 0.3924 | 0.6128 | 0.6693 | 0.5976 | 0.6106 | 0.5975 | 0.3721 | 0.7818 | 0.6800 | 0.4993 | 0.8421 | 0.9141 | 0.6240 |

### Willow Object Class - 2GM
experiment config: ``experiments/vgg16_gmn_willow.yaml``

pretrained model: [google drive](https://drive.google.com/file/d/1PWM1i0oywH3hrwPdYerPazRmhApC0B4U/view?usp=sharing)

| model                    | year | remark          | Car    | Duck   | Face   | Motorbike | Winebottle | mean   |
| ------------------------ | ---- | --------------- | ------ | ------ | ------ | --------- | ---------- | ------ |
| [GMN](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#gmn) | 2018 | -               | 0.6790 | 0.7670 | 0.9980 | 0.6920    | 0.8310     | 0.7934 |

## File Organization
```
├── affinity_layer.py
|   the implementation of affinity layer to compute the affinity matrix for GMN 
├── displacement_layer.py
|   the implementation of the displacement layer to compute the pixel offset loss
├── model.py
|   the implementation of training/evaluation procedures of GMN
├── model_config.py
|   the declaration of model hyperparameters
└── voting_layer.py
    the implementation of voting layer to compute the row-stotatic matrix with softmax
```

## Credits and Citation

Please cite the following paper if you use this model in your research:
```
@inproceedings{ZanfirCVPR18,
  author = {A. Zanfir and C. Sminchisescu},
  title = {Deep Learning of Graph Matching},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2684--2693},
  year={2018}
}
```
