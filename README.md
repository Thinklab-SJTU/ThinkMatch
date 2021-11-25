# ThinkMatch--Paddle version
To find the original work( powered by pytorch), click [here](https://github.com/Thinklab-SJTU/PCA-GM/tree/c426792d0fd566807c8fbf9ea056d7291e717263)

## Paddle-Version ( UNFINISHED )

- [x] Parameter convertion tools (see `src/convert_params.py`)
- [x] Inference with pretrained parameters (get the params from the function `pca_convert()` in `src/convert_params.py`)
- [ ] Training from the scratch

This repository contains PaddlePaddle implementation of our ICCV 2019 paper (for oral presentation): [Learning Combinatorial Embedding Networks for Deep Graph Matching.](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Learning_Combinatorial_Embedding_Networks_for_Deep_Graph_Matching_ICCV_2019_paper.pdf)

It contains our implementation of following deep graph matching methods:

**PCA-GM** Runzhong Wang, Junchi Yan and Xiaokang Yang, "Learning Combinatorial Embedding Networks for Deep Graph Matching." ICCV 2019.

This repository also include training/evaluation protocol on Pascal VOC Keypoint and Willow Object Class dataset, inline with the experiment part in our ICCV 2019 paper.

To run this, you can
```
python eval_pdl.py --cfg experiments/vggpdl_pca_voc.yaml
```

## Parameter\_convertion
`convert_param.py` includes a function to convert the parameters in Pytorch (as `.pt` form) into those in Paddle (as `.pdparam` form).

### Example
```
def vgg_convert():
    with fluid.dygraph.guard():
        # define the Pytorch model and load the paramters
        model_th = models.vgg16_bn(pretrained=True)
        # define the coresponding Paddle model
        model_pd = vision.models.vgg16(pretrained=False, batch_norm=True)
        # define the output path
        model_path = "./vgg16_bn"
        # convert it 
        convert_params(model_th, model_pd, model_path)
```
---

## Extra requirements
paddlepaddle-gpu==2.1.0 [Warning: Do not use version >=2.1.1 or version >=2.2.0. The former one has bugs that will reduce the accuracy of our model while the latter one has an unknown problem that possibly explodes your GPU memory]
visualdl

## bugs still not fixed
https://github.com/PaddlePaddle/Paddle/issues/34633 That's the reason we cannot train the model well now : )

