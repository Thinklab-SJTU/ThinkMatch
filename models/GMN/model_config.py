from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

# GMN model options
__C.GMN = edict()
__C.GMN.FEATURE_CHANNEL = 512
__C.GMN.PI_ITER_NUM = 50
__C.GMN.PI_STOP_THRESH = 2e-7
__C.GMN.BS_ITER_NUM = 10
__C.GMN.BS_EPSILON = 1e-10
__C.GMN.VOTING_ALPHA = 2e8
__C.GMN.GM_SOLVER = 'SM'