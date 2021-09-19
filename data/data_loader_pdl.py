# line 138 139 has sparse_tensor !! do not know whether it will be used?
# unfixed random seed is NOT available now
import paddle
import paddle.nn.functional as F
from paddle.io import Dataset
from paddle.vision import transforms
import numpy as np
import random
from data.pascal_voc import PascalVOC
from data.willow_obj import WillowObject
from src.utils_pdl.build_graphs import build_graphs
# Now only implement PCA
# so following files are useless, in theory
from src.utils_pdl.fgm import kronecker_sparse
from src.sparse_torch import CSRMatrix3d

from src.utils.config import cfg


class GMDataset(Dataset):
    def __init__(self, name, length, cls=None, **args):
        self.name = name
        self.ds = eval(self.name)(**args)
        self.length = length  # NOTE images pairs are sampled randomly, so there is no exact definition of dataset size
                              # length here represents the iterations between two checkpoints
        self.obj_size = self.ds.obj_resize
        self.classes = self.ds.classes
        self.cls = None if cls == 'none' else cls

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        anno_pair, perm_mat = self.ds.get_pair(self.cls)
        if perm_mat.size <= 2 * 2:
            return self.__getitem__(idx)

        P1_gt = [(kp['x'], kp['y']) for kp in anno_pair[0]['keypoints']]
        P2_gt = [(kp['x'], kp['y']) for kp in anno_pair[1]['keypoints']]

        n1_gt, n2_gt = len(P1_gt), len(P2_gt)

        P1_gt = np.array(P1_gt)
        P2_gt = np.array(P2_gt)

        G1_gt, H1_gt, e1_gt = build_graphs(P1_gt, n1_gt, stg=cfg.PAIR.GT_GRAPH_CONSTRUCT)
        if cfg.PAIR.REF_GRAPH_CONSTRUCT == 'same':
            G2_gt = perm_mat.transpose().dot(G1_gt)
            H2_gt = perm_mat.transpose().dot(H1_gt)
            e2_gt= e1_gt
        else:
            G2_gt, H2_gt, e2_gt = build_graphs(P2_gt, n2_gt, stg=cfg.PAIR.REF_GRAPH_CONSTRUCT)

        ret_dict = {'Ps': [paddle.to_tensor(x) for x in [P1_gt, P2_gt]],
                    'ns': [paddle.to_tensor(x) for x in [n1_gt, n2_gt]],
                    'es': [paddle.to_tensor(x) for x in [e1_gt, e2_gt]],
                    'gt_perm_mat': perm_mat,
                    'Gs': [paddle.to_tensor(x) for x in [G1_gt, G2_gt]],
                    'Hs': [paddle.to_tensor(x) for x in [H1_gt, H2_gt]]}

        imgs = [anno['image'] for anno in anno_pair]
        if imgs[0] is not None:
            trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)
                    ])
            imgs = [trans(img) for img in imgs]
            ret_dict['images'] = imgs
        elif 'feat' in anno_pair[0]['keypoints'][0]:
            feat1 = np.stack([kp['feat'] for kp in anno_pair[0]['keypoints']], axis=-1)
            feat2 = np.stack([kp['feat'] for kp in anno_pair[1]['keypoints']], axis=-1)
            ret_dict['features'] = [paddle.to_tensor(x) for x in [feat1, feat2]]

        return ret_dict


def collate_fn(data: list):
    """
    Create mini-batch data for training.
    :param data: data dict
    :return: mini-batch
    """
    def pad_tensor(inp):
        assert type(inp[0]) == paddle.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            pad_pattern = pad_pattern.tolist()
            #print('pad_pattern is', pad_pattern)
            if (len(pad_pattern) == 2):
                tt = t.reshape((1,1,t.shape[0]))
                padded_ts.append(F.pad(tt, pad_pattern, 'constant', 0, 'NCL').squeeze())
            elif len(pad_pattern) == 4:
                tt = t.reshape((1,1,t.shape[0],-1))
                tt = F.pad(tt, pad_pattern, 'constant', 0, 'NCHW')
                padded_ts.append(tt.reshape((1,tt.shape[2],-1)).squeeze())
            elif len(pad_pattern) == 6:
                tt = t.reshape((1,1,t.shape[0],t.shape[1],-1))
                padded_ts.append(F.pad(tt, pad_pattern, 'constant', 0, data_format='NCDHW').squeeze())

        return padded_ts

    def stack(inp):
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Key value mismatch."
                #if k == 'ns' : print('original ', vs)
                ret[k] = stack(vs)
                #if k == 'ns' : print('After stack, ', ret[k])
        elif type(inp[0]) == paddle.Tensor:
            new_t = pad_tensor(inp)
            if len(new_t) == 0:
                ret = paddle.to_tensor(new_t)
            else:
                ret = paddle.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([paddle.to_tensor(x) for x in inp])
            ret = paddle.stack(new_t, 0)
        elif type(inp[0]) == str:
            ret = inp
        else:
            raise ValueError('Cannot handle type {}'.format(type(inp[0])))
        return ret

    ret = stack(data)

    # compute CPU-intensive matrix K1, K2 here to leverage multi-processing nature of dataloader
    if 'Gs' in ret and 'Hs' in ret:
        try:
            G1_gt, G2_gt = ret['Gs']
            H1_gt, H2_gt = ret['Hs']
            sparse_dtype = np.float32
            K1G = [kronecker_sparse(x.squeeze(), y.squeeze()).astype(sparse_dtype) for x, y in zip(G2_gt, G1_gt)]  # 1 as source graph, 2 as target graph
            K1H = [kronecker_sparse(x.squeeze(), y.squeeze()).astype(sparse_dtype) for x, y in zip(H2_gt, H1_gt)]
            K1G = CSRMatrix3d(K1G)
            K1H = CSRMatrix3d(K1H).transpose()

            ret['Ks'] = K1G, K1H #, K1G.transpose(keep_type=True), K1H.transpose(keep_type=True)
        except ValueError:
            pass

    return ret


def worker_init_fix(worker_id):
    """
    Init dataloader workers with fixed seed.
    """
    random.seed(cfg.RANDOM_SEED + worker_id)
    np.random.seed(cfg.RANDOM_SEED + worker_id)


def worker_init_rand(worker_id):
    """
    Init dataloader workers with torch.initial_seed().
    torch.initial_seed() returns different seeds when called from different dataloader threads.
    """
    random.seed(paddle.initial_seed())
    np.random.seed(paddle.initial_seed() % 2 ** 32)


def get_dataloader(dataset, fix_seed=True, shuffle=False):
    fix_seed = True #"Paddle version now do NOT support unfixed seed"
    return paddle.io.DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.DATALOADER_NUM, collate_fn=collate_fn,
         worker_init_fn=worker_init_fix if fix_seed else worker_init_rand
    )
