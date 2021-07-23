# PCA-GM
To find the original work( powered by pytorch), click [here](https://github.com/Thinklab-SJTU/PCA-GM/tree/c426792d0fd566807c8fbf9ea056d7291e717263)

## Paddle-Version ( UNFINISHED )

- [x] Parameter convertion tools (see `utils_pdl/convert_params.py`)
- [x] Inference with pretrained parameters
- [ ] Training from the scratch

This repository contains PaddlePaddle implementation of our ICCV 2019 paper (for oral presentation): [Learning Combinatorial Embedding Networks for Deep Graph Matching.](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Learning_Combinatorial_Embedding_Networks_for_Deep_Graph_Matching_ICCV_2019_paper.pdf)

It contains our implementation of following deep graph matching methods:

**PCA-GM** Runzhong Wang, Junchi Yan and Xiaokang Yang, "Learning Combinatorial Embedding Networks for Deep Graph Matching." ICCV 2019.

This repository also include training/evaluation protocol on Pascal VOC Keypoint and Willow Object Class dataset, inline with the experiment part in our ICCV 2019 paper.

## Extra requirements
paddlepaddle-gpu==2.1.0
visualdl
