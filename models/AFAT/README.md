# AFAT

Our implementation of the following paper:
* Runzhong Wang, Ziao Guo, Shaofei Jiang, Xiaokang Yang, Junchi Yan. "Deep Learning of Partial Graph Matching via Differentiable Top-K." _CVPR 2023_.
  [[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Deep_Learning_of_Partial_Graph_Matching_via_Differentiable_Top-K_CVPR_2023_paper.html)

AFAT is a partial matching handling approach, used to resolve partial graph matching problem, i.e., graph matching with the existence of outliers.
It includes a topk-GM algorithm to suppress matchings with low confidence, and two AFA modules to predict the number of inliers k.
The implementation of topk-GM algorithm and AFA modules is shown in ``models.AFAT.sinkhorn_topk.py`` and ``models.AFAT.k_pred_net.py`` respectively.


AFAT has the following components:
* An Optimal Transport (OT) layer utilizing Sinkhorn algorithm with marginal distributions to resolve topk selection problem
* AFA-U (unified bipartite graph modeling) module to predict the number of inliers k
* AFA-I (individual graph modeling) module to predict the number of inliers k
* Greedy-topk algorithm to select matches with topk confidences

We also show that such a general scheme can be readily plugged into SOTA deep graph matching pipelines.
We adopt quadratic matching network NGMv2 as the backend graph matching network, and the file ``models.NGM.model_v2_topk.py`` can be referred to for details.
We also choose linear matching network GCAN as the backend graph matching network, and the file ``models.GCAN.GCAN_model_topk.py`` can be referred to for details.


## Benchmark Results
### PascalVOC - 2GM (unfiltered)
* NGMv2-AFAT-U

  experiment config: ``experiments/ngmv2-afat/vgg16_ngmv2-afat-u_voc_stage1.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-u_voc_stage2.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-u_voc_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/1hF43g6D5IO8EGrknCeiuWALT0JcqdIEf/view?usp=share_link)

* NGMv2-AFAT-I
  
  experiment config: ``experiments/ngmv2-afat/vgg16_ngmv2-afat-i_voc_stage1.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-i_voc_stage2.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-i_voc_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/1bv3z4WuxuoOt_MnmfNVzSekubIXg9p1o/view?usp=share_link)

* GCAN-AFAT-U

  experiment config: ``experiments/gcan-afat/vgg16_gcan-afat-u_voc_stage1.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-u_voc_stage2.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-u_voc_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/1ZdKBTpw3uoPORQbA5mqEm-5KJBIIuPvT/view?usp=share_link)

* GCAN-AFAT-I

  experiment config: ``experiments/gcan-afat/vgg16_gcan-afat-i_voc_stage1.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-i_voc_stage2.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-i_voc_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/1o0upD_z1cgtnLNnMMjxL3AVqYa2iKiOg/view?usp=share_link)

| model                  | year | aero   | bike   | bird   | boat   | bottle | bus    | car    | cat    | chair  | cow    | table  | dog    | horse  | mbkie  | person | plant  | sheep  | sofa   | train  | tv     | mean   |
| ---------------------- | ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| [NGMv2-AFAT-U](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat) | 2023 | 0.457 | 0.677 | 0.573 | 0.449 | 0.901 | 0.655 | 0.499 | 0.593 | 0.440 | 0.620 | 0.549 | 0.584 | 0.586 | 0.638 | 0.459 | 0.948 | 0.509 | 0.373 | 0.742 | 0.828 | 0.602 |
| [NGMv2-AFAT-I](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat) | 2023 | 0.450 | 0.673 | 0.559 | 0.456 | 0.903 | 0.646 | 0.487 | 0.580 | 0.447 | 0.602 | 0.548 | 0.572 | 0.575 | 0.634 | 0.452 | 0.953 | 0.493 | 0.416 | 0.736 | 0.824 | 0.599 |
| [GCAN-AFAT-U](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat) | 2023 | 0.471 | 0.708 | 0.581 | 0.458 | 0.908 | 0.665 | 0.496 | 0.588 | 0.506 | 0.646 | 0.472 | 0.605 | 0.623 | 0.657 | 0.463 | 0.954 | 0.527 | 0.474 | 0.742 | 0.838 | 0.620 |
| [GCAN-AFAT-I](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat) | 2023 | 0.461 | 0.699 | 0.561 | 0.466 | 0.907 | 0.661 | 0.481 | 0.579 | 0.499 | 0.639 | 0.504 | 0.590 | 0.616 | 0.650 | 0.447 | 0.955 | 0.509 | 0.492 | 0.740 | 0.838 | 0.616 |

### Willow Object Class - 2GM & MGM
* NGMv2-AFAT-U

  experiment config: ``experiments/ngmv2-afat/vgg16_ngmv2-afat-u_willow_stage1.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-u_willow_stage2.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-u_willow_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/1tUsYNlnZh_aW6h4eBce0DZRspbR6bw6w/view?usp=share_link)

* NGMv2-AFAT-I
  
  experiment config: ``experiments/ngmv2-afat/vgg16_ngmv2-afat-i_willow_stage1.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-i_willow_stage2.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-i_willow_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/173THcpubDa0r6F7GPpyD1dUNvVu74H9t/view?usp=share_link)

* GCAN-AFAT-U

  experiment config: ``experiments/gcan-afat/vgg16_gcan-afat-u_willow_stage1.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-u_willow_stage2.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-u_willow_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/1YoajZilGAK4HLh9ot_evLJcnihvJGEos/view?usp=share_link)

* GCAN-AFAT-I

  experiment config: ``experiments/gcan-afat/vgg16_gcan-afat-i_willow_stage1.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-i_willow_stage2.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-i_willow_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/1IUrH61uUdF2ftfpD1Lcs_YL9zAW8YdIq/view?usp=share_link)

| model                    | year  | Car    | Duck   | Face   | Motorbike | Winebottle | mean   |
| ------------------------ | ----  | ------ | ------ | ------ | --------- | ---------- | ------ |
| [NGMv2-AFAT-U](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat) | 2023  | 0.826 | 0.745 | 0.906 | 0.739    | 0.870     | 0.817 |
| [NGMv2-AFAT-I](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat) | 2023  | 0.846 | 0.757 | 0.920 | 0.745    | 0.886     | 0.831 |
| [GCAN-AFAT-U](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat) | 2023 | 0.801 | 0.780 | 0.906 | 0.760    | 0.870     | 0.823 |
| [GCAN-AFAT-I](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat) | 2023 | 0.822 | 0.777 | 0.927 | 0.772    | 0.886     | 0.837 |

### SPair-71k - 2GM (unfiltered)
* NGMv2-AFAT-U

  experiment config: ``experiments/ngmv2-afat/vgg16_ngmv2-afat-u_spair71k_stage1.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-u_spair71k_stage2.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-u_spair71k_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/12vwkPXv_0hmHJeR78h29RPINlSjudNIq/view?usp=share_link)

* NGMv2-AFAT-I
  
  experiment config: ``experiments/ngmv2-afat/vgg16_ngmv2-afat-i_spair71k_stage1.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-i_spair71k_stage2.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-i_spair71k_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/1kMbEk4kE0IqJDQSE4eWI6LqIGiFHCqSO/view?usp=share_link)

* GCAN-AFAT-U

  experiment config: ``experiments/gcan-afat/vgg16_gcan-afat-u_spair71k_stage1.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-u_spair71k_stage2.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-u_spair71k_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/196fMBID1rLbtWHceWHOnZ1EJAU9ISiBv/view?usp=share_link)

* GCAN-AFAT-I

  experiment config: ``experiments/gcan-afat/vgg16_gcan-afat-i_spair71k_stage1.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-i_spair71k_stage2.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-i_spair71k_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/1UmAZAA7SUcWAPugXPmjMgo7h9KAzKzQT/view?usp=share_link)

| model   | year | aero   | bike   | bird   | boat   | bottle | bus    | car    | cat    | chair  | cow    | dog    | horse  | mtbike | person | plant  | sheep  | train  | tv     | mean |
| ------- | ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| [NGMv2-AFAT-U](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat)     | 2023 | 0.503 | 0.435 | 0.638 | 0.324 | 0.590 | 0.601 | 0.397 | 0.686 | 0.361 | 0.636 | 0.565 | 0.463 | 0.514 | 0.433 | 0.770 | 0.512 | 0.811 | 0.894 | 0.563 |
| [NGMv2-AFAT-I](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat)  | 2023 | 0.504 | 0.436 | 0.639 | 0.321 | 0.612 | 0.585 | 0.380 | 0.684 | 0.357 | 0.627 | 0.564 | 0.477 | 0.519 | 0.443 | 0.785 | 0.507 | 0.792 | 0.912 | 0.564 |
| [GCAN-AFAT-U](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat) | 2023 | 0.467 | 0.433 | 0.658 | 0.333 | 0.615 | 0.549 | 0.352 | 0.684 | 0.377 | 0.599 | 0.560 | 0.476 | 0.472 | 0.435 | 0.803 | 0.477 | 0.838 | 0.890 | 0.557 |
| [GCAN-AFAT-I](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat) | 2023 | 0.468 | 0.443 | 0.659 | 0.324 | 0.615 | 0.538 | 0.337 | 0.684 | 0.381 | 0.601 | 0.563 | 0.479 | 0.483 | 0.438 | 0.812 | 0.484 | 0.829 | 0.880 | 0.557 |

### IMC-PT-SparseGM-50 - 2GM (unfiltered)
* NGMv2-AFAT-U

  experiment config: ``experiments/ngmv2-afat/vgg16_ngmv2-afat-u_imcpt_50_stage1.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-u_imcpt_50_stage2.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-u_imcpt_50_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/1VWCC7g2R5A8m1jnoOWh52G0dnJiR5DAE/view?usp=share_link)

* NGMv2-AFAT-I
  
  experiment config: ``experiments/ngmv2-afat/vgg16_ngmv2-afat-i_imcpt_50_stage1.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-i_imcpt_50_stage2.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-i_imcpt_50_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/12OfbTsIafJJGLHDGHG_zeHDqREQCILzU/view?usp=share_link)

* GCAN-AFAT-U

  experiment config: ``experiments/gcan-afat/vgg16_gcan-afat-u_imcpt_50_stage1.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-u_imcpt_50_stage2.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-u_imcpt_50_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/1altuEr7otvsaSmIJ2EfeQZc7ZGx08TQ3/view?usp=share_link)

* GCAN-AFAT-I

  experiment config: ``experiments/gcan-afat/vgg16_gcan-afat-i_imcpt_50_stage1.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-i_imcpt_50_stage2.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-i_imcpt_50_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/1EBisjv_RZv0u_hmOeO-Uvocfxu3ZSoEZ/view?usp=share_link)

| model   | year | reichstag   | sacre coeur   | st peters square | mean |
| ------- | ---- | ------ | ------ | ------ | ------ |
| [NGMv2-AFAT-U](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat)     | 2023 | 0.905 | 0.587 | 0.669 | 0.720 |
| [NGMv2-AFAT-I](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat)  | 2023 | 0.923 | 0.587 | 0.667 | 0.728 |
| [GCAN-AFAT-U](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat) | 2023 | 0.869 | 0.594 | 0.671 | 0.711  | 
| [GCAN-AFAT-I](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat) | 2023 | 0.910 | 0.603 | 0.673 | 0.729 |

### IMC-PT-SparseGM-100 - 2GM (unfiltered)
* NGMv2-AFAT-U

  experiment config: ``experiments/ngmv2-afat/vgg16_ngmv2-afat-u_imcpt_100_stage1.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-u_imcpt_100_stage2.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-u_imcpt_100_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/181_q0fQRI0qpUUvAiYuoscLk8-3bqBvl/view?usp=share_link)

* NGMv2-AFAT-I
  
  experiment config: ``experiments/ngmv2-afat/vgg16_ngmv2-afat-i_imcpt_100_stage1.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-i_imcpt_100_stage2.yaml``, ``experiments/ngmv2-afat/vgg16_ngmv2-afat-i_imcpt_100_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/1N8dPd8YEFqrsg9eMG19t4FGyKvfF8Nsr/view?usp=share_link)

* GCAN-AFAT-U

  experiment config: ``experiments/gcan-afat/vgg16_gcan-afat-u_imcpt_100_stage1.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-u_imcpt_100_stage2.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-u_imcpt_100_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/1C4bGyxOMIS4DAimv8gt9ZVStZL5H9hoz/view?usp=share_link)

* GCAN-AFAT-I

  experiment config: ``experiments/gcan-afat/vgg16_gcan-afat-i_imcpt_100_stage1.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-i_imcpt_100_stage2.yaml``, ``experiments/gcan-afat/vgg16_gcan-afat-i_imcpt_100_stage3.yaml``

  pretrained model: [google drive](https://drive.google.com/file/d/1Zcm2EhFp2ZkIy0gl6CbRHlAPY2I9Q8_0/view?usp=share_link)

| model   | year | reichstag   | sacre coeur   | st peters square | mean |
| ------- | ---- | ------ | ------ | ------ | ------ |
| [NGMv2-AFAT-U](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat)     | 2023 | 0.817 | 0.570 | 0.722 | 0.703 |
| [NGMv2-AFAT-I](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat)  | 2023 | 0.820 | 0.570 | 0.714 | 0.701 |
| [GCAN-AFAT-U](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat) | 2023 | 0.826 | 0.582 | 0.738 | 0.715  | 
| [GCAN-AFAT-I](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#afat) | 2023 | 0.827 | 0.578 | 0.724 | 0.709 |

## File Organization
```
├── k_pred_net.py
|   the implementation of AFA-I and AFA-U modules 
└── sinkhorn_topk.py
    the implementation of topk-GM algorithm and greedy-topk algorithm
```

## Credits and Citation

Please cite the following paper if you use this model in your research:
```
@InProceedings{WangCVPR23,
    author={Wang, Runzhong and Guo, Ziao and Jiang, Shaofei and Yang, Xiaokang and Yan, Junchi},
    title={Deep Learning of Partial Graph Matching via Differentiable Top-K},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023},
}
```
