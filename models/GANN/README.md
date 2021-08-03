# GANN

Our implementation of the following paper:
  * Runzhong Wang, Junchi Yan and Xiaokang Yang. "Graduated Assignment for Joint Multi-Graph Matching and Clustering with Application to Unsupervised Graph Matching Network Learning." _NeurIPS 2020_.
    [[paper]](https://papers.nips.cc/paper/2020/hash/e6384711491713d29bc63fc5eeb5ba4f-Abstract.html)
  * Runzhong Wang, Shaofei Jiang, Junchi Yan and Xiaokang Yang. "Robust Self-supervised Learning of Deep Graph Matching with Mixture of Modes." _Submitted to TPAMI_. 
    [[project page]](https://thinklab.sjtu.edu.cn/project/GANN-GM/index.html)

GANN proposes a self-supervised learning framework by leveraging graph matching solvers to provide pseudo labels to train the neural network module in deep graph matching pipeline. We propose a general graph matching solver for various graph matching settings based on the classic Graduated Assignment (GA) algorithm.

The variants on three different graph matching settings are denoted by different suffixes:
* **GANN-2GM**: self-supervised learning graduated assignment neural network for **two-grpah matching**
* **GANN-MGM**: self-supervised learning graduated assignment neural network for **multi-grpah matching**
* **GANN-MGM3**: self-supervised learning graduated assignment neural network for **multi-graph matching with a mixture of modes** (this setting is also known as multi-graph matching and clustering in the NeurIPS paper)

GANN-MGM notably surpass supervised learning methods on the relatively small dataset Willow Object Class.

## Benchmark Results
### Willow Object Class - MGM

experiment config: ``experiments/vgg16_gann-mgm_willow.yaml``

pretrained model: [google drive](https://drive.google.com/file/d/15Sg6mi9nrpsD4MAjp8b138-t-17VbYsw/view?usp=sharing)

| model                    | year | remark          | Car    | Duck   | Face   | Motorbike | Winebottle | mean   |
| ------------------------ | ---- | --------------- | ------ | ------ | ------ | --------- | ---------- | ------ |
| [GANN-MGM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#gann) | 2020 | self-supervised | 0.9600 | 0.9642 | 1.0000 | 1.0000    | 0.9879     | 0.9906 |

## File Organization
```
├── graduated_assignment.py
|   the implementation of the graduated assignment algorithm covering all scenarios
├── model.py
|   the implementation of training/evaluation procedures of GANN-GM/MGM/MGM3
└── model_config.py
    the declaration of model hyperparameters
```

## Credits and Citation

Please cite the following papers if you use this model in your research:
```
@inproceedings{WangNeurIPS20,
  author = {Runzhong Wang and Junchi Yan and Xiaokang Yang},
  title = {Graduated Assignment for Joint Multi-Graph Matching and Clustering with Application to Unsupervised Graph Matching Network Learning},
  booktitle = {Neural Information Processing Systems},
  year = {2020}
}

@unpublished{WangPAMIsub21,
  title={Robust Self-supervised Learning of Deep Graph Matching with Mixture of Modes},
  author={Wang, Runzhong and Jiang, Shaofei and Yan, Junchi and Yang, Xiaokang},
  note={submitted to IEEE Transactions of Pattern Analysis and Machine Intelligence},
  year={2021}
}
```
