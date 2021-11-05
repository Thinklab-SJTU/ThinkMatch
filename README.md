# ThinkMatch--Mindspore version
To find the original work (powered by pytorch), click [here](https://github.com/Thinklab-SJTU/PCA-GM/tree/c426792d0fd566807c8fbf9ea056d7291e717263)

## Mindspore-Version ( UNFINISHED )

- [x] Parameter convertion tools (see `api/convert_ckpt.py`)
- [x] Inference with pretrained parameters
- [ ] Training from the scratch

This repository contains Mindspore implementation of our ICCV 2019 paper (for oral presentation): [Learning Combinatorial Embedding Networks for Deep Graph Matching.](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Learning_Combinatorial_Embedding_Networks_for_Deep_Graph_Matching_ICCV_2019_paper.pdf)

It contains our implementation of following deep graph matching methods:

**PCA-GM** Runzhong Wang, Junchi Yan and Xiaokang Yang, "Learning Combinatorial Embedding Networks for Deep Graph Matching." ICCV 2019.

Only 2GM problem is supported now.

## Parameter conversion
`api/convert_ckpt.py` includes a function to convert the parameters in Pytorch (as `.pt` form) into those in Mindspore (as `.ckpt` form).

Set the path of original parameters in its main function. After running it, the converted parameters will be saved as 'torch2ms.ckpt'

Note: if you want to use this converter, you should install both mindspore and pytorch.

## Requirements

Besides the requirements (except pytorch) in the master branch, the mindspore version also requires:

```bash
mindinsight==1.2.0
mindspore-gpu==1.2.1
mindspore-hub==1.5.0
pygmtools==0.1.13
```

Since the installation of mindspore-gpu is hard, we recommend using [this docker](https://drive.google.com/file/d/1PBQIPSUCIFR5s4_4JDieJYpVAeL2xRMx/view?usp=sharing) to build the environment.
It can be used by docker or other container runtimes that support docker images e.g. [singularity](https://sylabs.io/singularity/).

## Available datasets
1. PascalVOC-Keypoint
    1. Download [VOC2011 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html) and make sure it looks like ``data/PascalVOC/VOC2011``
    1. Download keypoint annotation for VOC2011 from [Berkeley server](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz) or [google drive](https://drive.google.com/open?id=1D5o8rmnY1-DaDrgAXSygnflX5c-JyUWR) and make sure it looks like ``data/PascalVOC/annotations``
    1. The train/test split is available in ``data/PascalVOC/voc2011_pairs.npz``
   
You must download the dataset before running the experiment.
 
## Run the Experiment

Run evaluation
```bash
python eval.py --cfg path/to/your/yaml
```

and replace ``path/to/your/yaml`` by path to your configuration file, e.g.
```bash
python eval.py --cfg experiments/test.yaml
```

Default configuration files are stored in``experiments/``.

## Pretrained model

You can download a pretrained model of PCA from below. Both pytorch and mindspore versions of the same checkpoint are provided.

[pytorch](https://drive.google.com/file/d/1f6mXjW1cIDIm2OxYrlX3Lh6-NpSmwCdw/view?usp=sharing)

[mindspore](https://drive.google.com/file/d/1_p78Mwot0el4olZmvTUJQ6OVxCf2h_cd/view?usp=sharing)

Please change the 'PRETRAINED_PATH' in the yaml file to the path of mindspore checkpoint before running.

