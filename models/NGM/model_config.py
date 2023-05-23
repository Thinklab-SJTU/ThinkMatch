from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

# NGM model options
__C.NGM = edict()
__C.NGM.FEATURE_CHANNEL = 512
__C.NGM.SK_ITER_NUM = 10
__C.NGM.SK_EPSILON = 1e-10
__C.NGM.SK_TAU = 0.005
__C.NGM.MGM_SK_TAU = 0.005
__C.NGM.GNN_FEAT = [16, 16, 16]
__C.NGM.GNN_LAYER = 3
__C.NGM.GAUSSIAN_SIGMA = 1.
__C.NGM.SIGMA3 = 1.
__C.NGM.WEIGHT2 = 1.
__C.NGM.WEIGHT3 = 1.
__C.NGM.EDGE_FEATURE = 'cat' # 'cat' or 'geo'
__C.NGM.ORDER3_FEATURE = 'none' # 'cat' or 'geo' or 'none'
__C.NGM.FIRST_ORDER = True
__C.NGM.EDGE_EMB = False
__C.NGM.SK_EMB = 1
__C.NGM.GUMBEL_SK = 0 # 0 for no gumbel, other wise for number of gumbel samples
__C.NGM.UNIV_SIZE = -1
__C.NGM.POSITIVE_EDGES = True
__C.NGM.THRESHOLDING = 0
__C.NGM.GT_K = False
__C.NGM.SPARSE_MODEL = False

__C.AFA = edict()
__C.AFA.UNIV_SIZE = -1
__C.AFA.K_FACTOR = 50.
__C.AFA.REG_HIDDEN_FEAT = 8
__C.AFA.REGRESSION = True
__C.AFA.HEAD_NUM = 16
__C.AFA.KQV_DIM = 16
__C.AFA.FF_HIDDEN_DIM = 256
__C.AFA.MS_HIDDEN_DIM = 16
__C.AFA.MS_LAYER1_INIT = 10
__C.AFA.MS_LAYER2_INIT = 10
__C.AFA.MEAN_K = False
__C.AFA.K_GNN_LAYER = 2
__C.AFA.TN_NEURONS = 16
__C.AFA.AFAU = False
