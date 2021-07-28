"""Graph matching config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
from easydict import EasyDict as edict
import numpy as np
import importlib
import src.dataset

__C = edict()
# Consumers can get config by:
#   from config import cfg
cfg = __C

# Minibatch size
__C.BATCH_SIZE = 4

#
# Problem settings. Set these parameters the same for fair comparison.
#
__C.PROBLEM = edict()

# Problem type.
# Candidates can be '2GM' (two graph matching), 'MGM' (multi-graph matching), 'MGMC' (multi-graph matching and clustering)
__C.PROBLEM.TYPE = '2GM'

# If UNSUPERVISED==True, ground truth permutations will not be provided during training.
__C.PROBLEM.UNSUPERVISED = False

# Rescaled image size
__C.PROBLEM.RESCALE = (256, 256)

# Do not include the problem if n_1 x n_2 > MAX_PROB_SIZE. -1 for no filtering
__C.PROBLEM.MAX_PROB_SIZE = -1

# Allow outlier in source graph. Useful for 2GM
__C.PROBLEM.SRC_OUTLIER = False

# Allow outlier in target graph. Useful for 2GM
__C.PROBLEM.TGT_OUTLIER = False

# Number of graphs in a MGM/MGMC problem. Useful for MGM & MGMC
# No effect if TEST_ALL_GRAPHS/TRAIN_ALL_GRAPHS=True
__C.PROBLEM.NUM_GRAPHS = 3

# Number of clusters in MGMC problem. Useful for MGMC
__C.PROBLEM.NUM_CLUSTERS = 1

# During testing, jointly match all graphs from the same class. Useful for MGM & MGMC
__C.PROBLEM.TEST_ALL_GRAPHS = False

# During training, jointly match all graphs from the same class. Useful for MGM & MGMC
__C.PROBLEM.TRAIN_ALL_GRAPHS = False

# Shape of candidates, useful for the setting in Zanfir et al CVPR2018
#__C.PROBLEM.CANDIDATE_SHAPE = (16, 16)
#__C.PROBLEM.CANDIDATE_LENGTH = np.cumprod(__C.PAIR.CANDIDATE_SHAPE)[-1]

#
# Graph construction settings.
#
__C.GRAPH = edict()

# The ways of constructing source graph/target graph.
# Candidates can be 'tri' (Delaunay triangulation), 'fc' (Fully-connected)
__C.GRAPH.SRC_GRAPH_CONSTRUCT = 'tri'
__C.GRAPH.TGT_GRAPH_CONSTRUCT = 'fc'

# Build a symmetric adjacency matrix, else only the upper right triangle of adjacency matrix will be filled
__C.GRAPH.SYM_ADJACENCY = True

# Padding length on number of keypoints for batched operation
__C.GRAPH.PADDING = 23

#
# Training options
#

__C.TRAIN = edict()

# Iterations per epochs
__C.TRAIN.EPOCH_ITERS = 7000

# Training start epoch. If not 0, will be resumed from checkpoint.
__C.TRAIN.START_EPOCH = 0

# Total epochs
__C.TRAIN.NUM_EPOCHS = 30

# Optimizer type
__C.TRAIN.OPTIMIZER = 'SGD'

# Start learning rate
__C.TRAIN.LR = 0.01

# Use separate learning rate for the CNN backbone
__C.TRAIN.SEPARATE_BACKBONE_LR = False

# Start learning rate for backbone
__C.TRAIN.BACKBONE_LR = __C.TRAIN.LR

# Learning rate decay
__C.TRAIN.LR_DECAY = 0.1

# Learning rate decay step (in epochs)
__C.TRAIN.LR_STEP = [10, 20]

# SGD momentum
__C.TRAIN.MOMENTUM = 0.9

# RobustLoss normalization
__C.TRAIN.RLOSS_NORM = max(__C.PROBLEM.RESCALE)

# Specify a class for training
__C.TRAIN.CLASS = 'none'

# Loss function. Should be 'offset' or 'perm'
__C.TRAIN.LOSS_FUNC = 'perm'


#
# Evaluation options
#

__C.EVAL = edict()

# Evaluation epoch number
__C.EVAL.EPOCH = 30

# PCK metric (deprecated)
#__C.EVAL.PCK_ALPHAS = []
#__C.EVAL.PCK_L = float(max(__C.PROBLEM.RESCALE))  # PCK reference.

# Number of samples for testing. Stands for number of image pairs in each classes (VOC)
__C.EVAL.SAMPLES = 1000

# Evaluated classes
__C.EVAL.CLASS = 'all'


#
# MISC
#

# name of backbone net
__C.BACKBONE = 'VGG16_bn'

# Parallel GPU indices ([0] for single GPU)
__C.GPUS = [0]

# num of dataloader processes
__C.DATALOADER_NUM = __C.BATCH_SIZE

# path to load pretrained model weights
__C.PRETRAINED_PATH = ''

# Mean and std to normalize images
__C.NORM_MEANS = [0.485, 0.456, 0.406]
__C.NORM_STD = [0.229, 0.224, 0.225]

# Data cache path
__C.CACHE_PATH = 'data/cache'

# Model name and dataset name
__C.MODEL_NAME = ''
__C.DATASET_NAME = ''
__C.DATASET_FULL_NAME = ''

# Module path of module
__C.MODULE = ''

# Output path (for checkpoints, running logs)
__C.OUTPUT_PATH = ''

# The step of iteration to print running statistics.
# The real step value will be the least common multiple of this value and batch_size
__C.STATISTIC_STEP = 100

# random seed used for data loading
__C.RANDOM_SEED = 123

# enable fp16 instead of fp32 in the model (via nvidia/apex)
__C.FP16 = False

def lcm(x, y):
    """
    Compute the least common multiple of x and y. This function is used for running statistics.
    """
    greater = max(x, y)
    while True:
        if (greater % x == 0) and (greater % y == 0):
            lcm = greater
            break
        greater += 1
    return lcm


def get_output_dir(model, dataset):
    """
    Return the directory where experimental artifacts are placed.
    :param model: model name
    :param dataset: dataset name
    :return: output path (checkpoint and log)
    """
    outp_path = os.path.join('output', '{}_{}'.format(model, dataset))
    return outp_path


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            if type(b[k]) is float and type(v) is int:
                v = float(v)
            else:
                if not k in ['CLASS']:
                    raise ValueError('Type mismatch ({} vs. {}) for config key: {}'.format(type(b[k]), type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.full_load(f))

    if 'MODULE' in yaml_cfg and yaml_cfg.MODULE not in __C:
        model_cfg_module = '.'.join(yaml_cfg.MODULE.split('.')[:-1] + ['model_config'])
        mod = importlib.import_module(model_cfg_module)
        __C.update(mod.model_cfg)

    if 'DATASET_FULL_NAME' in yaml_cfg and yaml_cfg.DATASET_FULL_NAME not in __C:
        __C[yaml_cfg.DATASET_FULL_NAME] = src.dataset.dataset_cfg[yaml_cfg.DATASET_FULL_NAME]

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d.keys()
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d.keys()
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
