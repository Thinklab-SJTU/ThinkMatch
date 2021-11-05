from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

# GANN model options
__C.GANN = edict()
__C.GANN.FEATURE_CHANNEL = 1024
__C.GANN.SK_ITER_NUM = 20
__C.GANN.SK_TAU = 0.05
__C.GANN.SK_EPSILON = 1e-10
__C.GANN.UNIV_SIZE = 10
__C.GANN.CLUSTER_ITER = 10
__C.GANN.MGM_ITER = [200, 200]
__C.GANN.INIT_TAU = [0.5, 0.5]
__C.GANN.GAMMA = 0.5
__C.GANN.BETA = [1., 0.]
__C.GANN.CONVERGE_TOL = 1e-5
__C.GANN.MIN_TAU = [1e-2, 1e-2]
__C.GANN.SCALE_FACTOR = 1.
__C.GANN.QUAD_WEIGHT = 1.
__C.GANN.CLUSTER_QUAD_WEIGHT = 1.
__C.GANN.PROJECTOR = ['sinkhorn', 'sinkhorn']
__C.GANN.NORM_QUAD_TERM = False
