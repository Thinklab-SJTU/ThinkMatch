from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

# COMMON model options.
__C.COMMON = edict()
__C.COMMON.FEATURE_CHANNEL = 512
__C.COMMON.ALPHA = 0.4
__C.COMMON.DISTILL = True
__C.COMMON.WARMUP_STEP = 0
__C.COMMON.MOMENTUM = 0.995
__C.COMMON.SOFTMAXTEMP = 0.07
