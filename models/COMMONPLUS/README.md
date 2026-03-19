# COMMON+

Official implementation of the following paper:

* Yijie Lin, Mouxing Yang, Peng Hu, Jiancheng Lv, Hao Chen, Xi Peng. "Learning with Partial and Noisy Correspondence in Graph Matching". TPAMI, 2026.  [[paper]](https://xlearning-lab.com/assets/2026-TPAMI-Learning-With-Partial-and-Noisy-Correspondence-in-Graph-Matching.pdf)


COMMON+ proposes a unified deep graph matching framework for learning under **partial and noisy correspondence**, where graph matching models must simultaneously handle:

* **Partial correspondence**: some nodes have no valid matches due to outliers, occlusion, truncation, or structural mismatch;
* **Noisy correspondence**: the supervision itself is imperfect, including both inaccurate node-level matches and corrupted edge-level relations.

<img src="https://github.com/Lin-Yijie/Graph-Matching-Networks/blob/main/COMMON/docs/images/common%2B.jpg" alt="COMMON+, TPAMI 2026" width="100%">

As illustrated above, in real-world graph matching scenarios, images often exhibit large appearance variations, viewpoint changes, partial overlaps, and ambiguous structures. These factors make it difficult to establish clean one-to-one correspondences: some keypoints may be incorrectly annotated or spatially shifted, resulting in noisy node-to-node matches, while some nodes may have no valid counterparts at all. Such imperfect node correspondence further contaminates higher-order structural relations, leading to unreliable edge-level correspondence.

To address these challenges, COMMON+ introduces an **Align–Fuse–Refine** pipeline that combines complementary graph matching experts:

* Supporting **VGG16 CNN** or **VIT** backbone to extract visual features;
* A **quadratic contrastive learning objective** to enhance correspondence-aware structural alignment;
* A **dual-expert collaborative framework** that integrates **KB-QAP-based alignment** and **L-QAP-based fusion/matching**;
* A **momentum cooperation and refinement mechanism** to progressively identify outliers and correct noisy supervision during training.

This design enables COMMON+ to robustly handle both missing correspondences and corrupted annotations within a unified graph matching framework.


## Benchmark Results

COMMON+ supports multiple graph matching benchmarks under different settings, including standard 2GM, partial matching, and noisy-correspondence evaluation.

### VGG16-based configs

#### PascalVOC
- **Task**: 2GM
- **Config**: `experiments/vgg16_common_plus_voc.yaml`

#### PascalVOC (with outliers)
- **Task**: 2GM
- **Config**: `experiments/vgg16_common_plus_voc-all.yaml`

#### Willow ObjectClass
- **Task**: 2GM
- **Config**: `experiments/vgg16_common_plus_willow.yaml`

#### Willow ObjectClass (noisy correspondence)
- **Task**: 2GM with noisy correspondence
- **Config**: `experiments/vgg16_common_plus_willow_nc.yaml`

#### Willow ObjectClass (partial matching with outliers)
- **Task**: partial 2GM setting
- **Config**: `experiments/vgg16_common_plus_willow_outlier.yaml`

#### SPair-71k
- **Task**: 2GM
- **Config**: `experiments/vgg16_common_plus_spair71k.yaml`

#### IMC-PT-50
- **Task**: partial graph matching
- **Config**: `experiments/vgg16_common_plus_imcpt_50.yaml`

#### IMC-PT-100
- **Task**: partial graph matching
- **Config**: `experiments/vgg16_common_plus_imcpt_100.yaml`

---

### ViT-based configs

#### PascalVOC
- **Task**: 2GM
- **Config**: `experiments/vit_common_plus_voc.yaml`

#### Willow ObjectClass
- **Task**: 2GM
- **Config**: `experiments/vit_common_plus_willow.yaml`

#### SPair-71k
- **Task**: 2GM
- **Config**: `experiments/vit_common_plus_spair71k.yaml`



## File Organization


```
├── README.md
│   project overview and usage instructions
├── model.py
│   implementation of the training and evaluation procedures of COMMON+
├── model_config.py
│   declaration of model hyperparameters and configurations
├── sconv_archs.py
│   implementation of spline convolution (SplineCNN) operations
└── transformconv.py
    implementation of transformer-based graph fusion modules
```

## Credits and Citation

Please cite the following paper if you use this model in your research:

```
@article{lin2022graph,
  title={Graph Matching with Bi-level Noisy Correspondence},
  author={Lin, Yijie and Yang, Mouxing and Yu, Jun and Hu, Peng and Zhang, Changqing and Peng, Xi},
  journal={IEEE International Conference on Computer Vision},
  year={2022}
}
@article{lin2026learning,
  title={Learning With Partial and Noisy Correspondence in Graph Matching},
  author={Lin, Yijie and Yang, Mouxing and Hu, Peng and Lv, Jiancheng and Chen, Hao and Peng, Xi},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2026}
}
```
