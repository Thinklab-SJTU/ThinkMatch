import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
from data.pascal_voc import PascalVOC
from data.willow_obj import WillowObject
from src.utils.build_graphs import build_graphs
from src.utils.fgm import kronecker_sparse
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

        ret_dict = {'Ps': [torch.Tensor(x) for x in [P1_gt, P2_gt]],
                    'ns': [torch.tensor(x) for x in [n1_gt, n2_gt]],
                    'es': [torch.tensor(x) for x in [e1_gt, e2_gt]],
                    'gt_perm_mat': perm_mat,
                    'Gs': [torch.Tensor(x) for x in [G1_gt, G2_gt]],
                    'Hs': [torch.Tensor(x) for x in [H1_gt, H2_gt]]}

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
            ret_dict['features'] = [torch.Tensor(x) for x in [feat1, feat2]]

        return ret_dict


def collate_fn(data: list):
    """
    Create mini-batch data for training.
    :param data: data dict
    :return: mini-batch
    """
    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor
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
            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, 'constant', 0))

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
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
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
            K1G = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(G2_gt, G1_gt)]  # 1 as source graph, 2 as target graph
            K1H = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(H2_gt, H1_gt)]
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
    random.seed(torch.initial_seed())
    np.random.seed(torch.initial_seed() % 2 ** 32)


def get_dataloader(dataset, fix_seed=True, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.DATALOADER_NUM, collate_fn=collate_fn,
        pin_memory=False, worker_init_fn=worker_init_fix if fix_seed else worker_init_rand
    )
