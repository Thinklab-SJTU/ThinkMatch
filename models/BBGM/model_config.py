from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

# BBGM model options
__C.BBGM = edict()
__C.BBGM.SOLVER_NAME = 'LPMP'
__C.BBGM.LAMBDA_VAL = 80.0
__C.BBGM.SOLVER_PARAMS = edict()
__C.BBGM.SOLVER_PARAMS.timeout = 1000
__C.BBGM.SOLVER_PARAMS.primalComputationInterval = 10
__C.BBGM.SOLVER_PARAMS.maxIter = 100
__C.BBGM.FEATURE_CHANNEL = 1024