from easydict import EasyDict as edict

__C = edict()

dataset_cfg = __C
# Pascal VOC 2011 dataset with keypoint annotations
__C.PascalVOC = edict()
__C.PascalVOC.KPT_ANNO_DIR = 'data/PascalVOC/annotations/'  # keypoint annotation
__C.PascalVOC.ROOT_DIR = 'data/PascalVOC/VOC2011/'  # original VOC2011 dataset
__C.PascalVOC.SET_SPLIT = 'data/PascalVOC/voc2011_pairs.npz'  # set split path
__C.PascalVOC.CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                         'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                         'tvmonitor']

# Willow-Object Class dataset
__C.WillowObject = edict()
__C.WillowObject.ROOT_DIR = 'data/WILLOW-ObjectClass'
__C.WillowObject.CLASSES = ['Car', 'Duck', 'Face', 'Motorbike', 'Winebottle']
__C.WillowObject.KPT_LEN = 10
__C.WillowObject.TRAIN_NUM = 20
__C.WillowObject.SPLIT_OFFSET = 0
__C.WillowObject.TRAIN_SAME_AS_TEST = False
__C.WillowObject.RAND_OUTLIER = 0

# Synthetic dataset
__C.SYNTHETIC = edict()
__C.SYNTHETIC.DIM = 1024
__C.SYNTHETIC.TRAIN_NUM = 100  # training graphs
__C.SYNTHETIC.TEST_NUM = 100  # testing graphs
__C.SYNTHETIC.MIXED_DATA_NUM = 10  # num of samples in mixed synthetic test
__C.SYNTHETIC.RANDOM_EXP_ID = 0  # id of random experiment
__C.SYNTHETIC.EDGE_DENSITY = 0.3  # edge_num = X * node_num^2 / 4
__C.SYNTHETIC.KPT_NUM = 10  # number of nodes (inliers)
__C.SYNTHETIC.OUT_NUM = 0 # number of outliers
__C.SYNTHETIC.FEAT_GT_UNIFORM = 1.  # reference node features in uniform(-X, X) for each dimension
__C.SYNTHETIC.FEAT_NOISE_STD = 0.1  # corresponding node features add a random noise ~ N(0, X^2)
__C.SYNTHETIC.POS_GT_UNIFORM = 256.  # reference keypoint position in image: uniform(0, X)
__C.SYNTHETIC.POS_AFFINE_DXY = 50.  # corresponding position after affine transform: t_x, t_y ~ uniform(-X, X)
__C.SYNTHETIC.POS_AFFINE_S_LOW = 0.8  # corresponding position after affine transform: s ~ uniform(S_LOW, S_HIGH)
__C.SYNTHETIC.POS_AFFINE_S_HIGH = 1.2
__C.SYNTHETIC.POS_AFFINE_DTHETA = 60.  # corresponding position after affine transform: theta ~ uniform(-X, X)
__C.SYNTHETIC.POS_NOISE_STD = 10.  # corresponding position add a random noise ~ N(0, X^2) after affine transform

# QAPLIB dataset
__C.QAPLIB = edict()
__C.QAPLIB.DIR = 'data/qapdata'
__C.QAPLIB.FEED_TYPE = 'affmat' # 'affmat' (affinity matrix) or 'adj' (adjacency matrix)
__C.QAPLIB.ONLINE_REPO = 'http://anjos.mgi.polymtl.ca/qaplib/'
__C.QAPLIB.MAX_TRAIN_SIZE = 200
__C.QAPLIB.MAX_TEST_SIZE = 100

# CUB2011 dataset
__C.CUB2011 = edict()
__C.CUB2011.ROOT_PATH = 'data/CUB_200_2011'
__C.CUB2011.CLASS_SPLIT = 'ori' # choose from 'ori' (original split), 'sup' (super class) or 'all' (all birds as one class)

# IMC_PT_SparseGM dataset
__C.IMC_PT_SparseGM = edict()
__C.IMC_PT_SparseGM.CLASSES = {'train': ['brandenburg_gate', 'buckingham_palace', 'colosseum_exterior',
                                      'grand_place_brussels', 'hagia_sophia_interior', 'notre_dame_front_facade',
                                      'palace_of_westminster', 'pantheon_exterior', 'prague_old_town_square',
                                      'taj_mahal', 'temple_nara_japan', 'trevi_fountain', 'westminster_abbey'],
                            'test': ['reichstag', 'sacre_coeur', 'st_peters_square']}
__C.IMC_PT_SparseGM.ROOT_DIR_NPZ = 'data/IMC_PT_SparseGM/annotation'
__C.IMC_PT_SparseGM.ROOT_DIR_IMG = 'data/IMC_PT_SparseGM/Image_Matching_Challange_Data'
__C.IMC_PT_SparseGM.TOTAL_KPT_NUM = 50
