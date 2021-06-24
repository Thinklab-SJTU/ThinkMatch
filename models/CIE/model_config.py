from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

# CIE model options
__C.CIE = edict()
__C.CIE.FEATURE_CHANNEL = 512
__C.CIE.SK_ITER_NUM = 20
__C.CIE.SK_EPSILON = 1.0e-10
__C.CIE.SK_TAU = 0.005
__C.CIE.GNN_LAYER = 5
__C.CIE.GNN_FEAT = 1024
__C.CIE.CROSS_ITER = False
