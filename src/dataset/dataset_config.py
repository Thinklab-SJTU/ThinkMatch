from easydict import EasyDict as edict

__C = edict()

dataset_cfg = __C

# QAPLIB dataset
__C.QAPLIB = edict()
__C.QAPLIB.DIR = 'data/qapdata'
__C.QAPLIB.FEED_TYPE = 'affmat' # 'affmat' (affinity matrix) or 'adj' (adjacency matrix)
__C.QAPLIB.ONLINE_REPO = 'http://anjos.mgi.polymtl.ca/qaplib/'
__C.QAPLIB.MAX_TRAIN_SIZE = 200
__C.QAPLIB.MAX_TEST_SIZE = 100
