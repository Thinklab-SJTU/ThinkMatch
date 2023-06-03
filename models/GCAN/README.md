# GCAN

Our implementation of the following paper:
* Zheheng Jiang, Hossein Rahmani, Plamen Angelov, Sue Black, Bryan M. Williams. "Graph-Context Attention Networks for Size-Varied Deep Graph Matching." _CVPR 2022_.
    [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_Graph-Context_Attention_Networks_for_Size-Varied_Deep_Graph_Matching_CVPR_2022_paper.html)

GCAN proposes a deep graph matching pipeline to solve size-varied GM problem. It learns both node-level and graph-level matching. It proposes to combine the following components to formulate the GM pipeline:
* VGG16 CNN to extract image features
* Positional encoding and Graph-context attention (GCA) module to extract node features
* Solving node correspondence via Integer Linear Programming (ILP) attention loss
* Learning graph-level matching via margin-based pairwise loss

## Benchmark Results
### IMC-PT-SparseGM-50 - 2GM (unfiltered)
experiment config: ``experiments/vgg16_gcan_imcpt_50.yaml``

pretrained model: [google drive](https://drive.google.com/file/d/1XwHJ-nBvDf4rXQ4Zzn8y7N9jSEZQYJ0e/view?usp=share_link)

| model                  | year | reichstag | sacre coeur | st peters square | mean   |
| ---------------------- | ---- | ------ | ------ | ------ | ------ |
| [GCAN](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#gcan) | 2022 | 0.872 | 0.551 | 0.630 | 0.684 |

### IMC-PT-SparseGM-100 - 2GM (unfiltered)
experiment config: ``experiments/vgg16_gcan_imcpt_100.yaml``

pretrained model: [google drive](https://drive.google.com/file/d/13GSNuxgLomzFAFA-mzyDtGU0DN3Rz0Lm/view?usp=share_link)

| model                  | year | reichstag | sacre coeur | st peters square | mean   |
| ---------------------- | ---- | ------ | ------ | ------ | ------ |
| [GCAN](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#gcan) | 2022 | 0.804 | 0.557 | 0.728 | 0.696|

## File Organization
```
├── cross_attention_layer.py
|   the implementation of cross-attention layer used in GCA module
├── GCA_module.py
|   the implementation of GCA module
├── GCAN_model.py
|   the implementation of training/evaluation procedures of GCAN 
├── GCAN_model_topk.py
|   the implementation of training/evaluation procedures of GCAN combined with topk-GM algorithm and partial matching handling AFA modules
├── model_config.py
|   the declaration of model hyperparameters
├── positional_encoding_layer.py
|   the implementation of positional encoding
└── self_attention_layer.py
    the implementation of self-attention layer used in GCA module
```

## Credits and Citation

Please cite the following paper if you use this model in your research:
```
@InProceedings{JiangCVPR22,
    author={Jiang, Zheheng and Rahmani, Hossein and Angelov, Plamen and Black, Sue and Williams, Bryan M.},
    title={Graph-Context Attention Networks for Size-Varied Deep Graph Matching},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022},
}
```
