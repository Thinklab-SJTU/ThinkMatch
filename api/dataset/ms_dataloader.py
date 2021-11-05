import sys
sys.path.insert(0,"/mnt/nas/home/lixinyang/mindspore3/ThinkMatch")

import mindspore.dataset as msdataset
import mindspore
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
import mindspore.context as context

context.set_context(device_target="GPU")

import hashlib
import numpy as np
import random
from api.build_graphs import build_graphs
from api.dataset import *
from pygmtools.benchmark import Benchmark
import time

from api.utils.config import cfg

from itertools import combinations, product


class GMDataset:
    def __init__(self, bm, length, using_all_graphs=False, cls=None, problem='2GM'):
        self.totensor = msdataset.vision.py_transforms.ToTensor()
        self.norm = msdataset.vision.py_transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)
        self.hash2id = dict()

        self.bm = bm
        self.using_all_graphs = using_all_graphs
        self.obj_size = self.bm.obj_resize
        self.test = True if self.bm.sets == 'test' else False
        self.cls = None if cls in ['none', 'all'] else cls

        if self.cls is None:
            if problem == 'MGMC':
                self.classes = list(combinations(self.bm.classes, cfg.PROBLEM.NUM_CLUSTERS))
            else:
                self.classes = self.bm.classes
        else:
            self.classes = [self.cls]

        self.problem_type = problem
        self.img_num_list = self.bm.compute_img_num(self.classes)

        if self.problem_type == '2GM':
            self.id_combination, self.length = self.bm.get_id_combination(self.cls)
            self.length_list = []
            for cls in self.classes:
                cls_length = self.bm.compute_length(cls)
                self.length_list.append(cls_length)
        else:
            self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.problem_type == '2GM':
            return self.get_pair(idx, self.cls)
        elif self.problem_type == 'MGM':
            raise NotImplementedError
        elif self.problem_type == 'MGMC':
            raise NotImplementedError
        else:
            raise NameError("Unknown problem type: {}".format(self.problem_type))

    def get_pair(self, idx, cls):

        cls_num = random.randrange(0, len(self.classes))
        ids = list(self.id_combination[cls_num][idx % self.length_list[cls_num]])
        anno_pair, perm_mat_, id_list = self.bm.get_data(ids)
        perm_mat = perm_mat_[(0, 1)].toarray()
        while min(perm_mat.shape[0], perm_mat.shape[1]) <= 2 or perm_mat.size >= cfg.PROBLEM.MAX_PROB_SIZE > 0:
            anno_pair, perm_mat_, id_list = self.bm.rand_get_data(cls)
            perm_mat = perm_mat_[(0, 1)].toarray()

        cls = [anno['cls'] for anno in anno_pair]
        P1 = [(kp['x'], kp['y']) for kp in anno_pair[0]['kpts']]
        P2 = [(kp['x'], kp['y']) for kp in anno_pair[1]['kpts']]

        n1, n2 = len(P1), len(P2)
        univ_size = [anno['univ_size'] for anno in anno_pair]

        P1 = np.array(P1)
        P2 = np.array(P2)

        A1, G1, H1, e1 = build_graphs(P1, n1, stg=cfg.GRAPH.SRC_GRAPH_CONSTRUCT, sym=cfg.GRAPH.SYM_ADJACENCY)
        if cfg.GRAPH.TGT_GRAPH_CONSTRUCT == 'same':
            G2 = perm_mat.transpose().dot(G1)
            H2 = perm_mat.transpose().dot(H1)
            A2 = G2.dot(H2.transpose())
            e2 = e1
        else:
            A2, G2, H2, e2 = build_graphs(P2, n2, stg=cfg.GRAPH.TGT_GRAPH_CONSTRUCT, sym=cfg.GRAPH.SYM_ADJACENCY)

        ret_dict = {'Ps': [np.array(x) for x in [P1, P2]],
                    'ns': [np.array(x) for x in [n1, n2]],
                    'es': [np.array(x) for x in [e1, e2]],
                    'Gs': [np.array(x) for x in [G1, G2]],
                    'Hs': [np.array(x) for x in [H1, H2]],
                    'As': [np.array(x) for x in [A1, A2]],
                    'cls': [str(x) for x in cls],
                    'univ_size': [np.array(int(x)) for x in univ_size],
                    }
        imgs = [anno['img'] for anno in anno_pair]
        if imgs[0] is not None:
            templist = []
            for img in imgs:
                img = self.totensor(img)
                img = self.norm(img)
                templist.append(img)
            imgs = templist

        elif 'feat' in anno_pair[0]['kpts'][0]:

            feat1 = np.stack([kp['feat'] for kp in anno_pair[0]['kpts']], axis=-1)
            feat2 = np.stack([kp['feat'] for kp in anno_pair[1]['kpts']], axis=-1)
            ret_dict['features'] = [Tensor(x) for x in [feat1, feat2]]

        # mindspore的数据集加载器只支持numpy数组，为了传递字符串只能将其存在数据集类里
        self.hash2id[hashlib.md5(imgs[0][0,0]).hexdigest()] = id_list[0]
        self.hash2id[hashlib.md5(imgs[1][0, 0]).hexdigest()] = id_list[1]

        return ret_dict['Ps'] + ret_dict['ns'] + ret_dict['es'] + ret_dict['Gs'] + ret_dict[
            'Hs'] + ret_dict['As'] + ret_dict['univ_size'] + imgs

def pad_tensor(inp):
    assert type(inp[0]) == np.ndarray
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
    max_shape = np.array(max_shape, dtype=np.int64)

    padded_ts = []
    for t in inp:
        pad_pattern = []
        for axis in range(len(max_shape)):
            pad_pattern.append((0, max_shape[axis] - t.shape[axis]))
        if len(pad_pattern) > 0:
            padded_ts.append(np.pad(t, pad_pattern, 'constant', constant_values=0))
        else:
            padded_ts.append(t)

    return padded_ts

def stack(inp):

    if type(inp[0]) == np.ndarray:
        ret = pad_tensor(inp)
        ret = np.stack(ret,0)
    else:
        raise ValueError('Cannot handle type {}'.format(type(inp[0])))
    return ret

def collate_fn(P1, P2, n1, n2, e1, e2, G1, G2, H1, H2, A1, A2, univ_size0, univ_size1, img0, img1, batchInfo):
    """
    Create mini-batch data for training.
    :param data: data dict
    :return: mini-batch
    """
    P1 = stack(P1)
    P2 = stack(P2)
    n1 = stack(n1)
    n2 = stack(n2)
    e1 = stack(e1)
    e2 = stack(e2)
    G1 = stack(G1)
    G2 = stack(G2)
    H1 = stack(H1)
    H2 = stack(H2)
    A1 = stack(A1)
    A2 = stack(A2)
    univ_size0 = stack(univ_size0)
    univ_size1 = stack(univ_size1)
    img0 = stack(img0)
    img1 = stack(img1)

    return P1, P2, n1, n2, e1, e2, G1, G2, H1, H2, A1, A2, univ_size0, univ_size1, img0, img1

def get_dataloader(dataset, columns_names, fix_seed=True, shuffle=False):

    out =  msdataset.GeneratorDataset(dataset, columns_names, num_parallel_workers=1, shuffle=shuffle, python_multiprocessing=False)
    out = out.batch(batch_size=cfg.BATCH_SIZE, input_columns=columns_names, per_batch_map=collate_fn)
    return out

if __name__ == '__main__':
    import pdb
    image_dataset = GMDataset(
        bm=Benchmark('PascalVOC', 'test', (256, 256), problem='2GM'),
        length=None,
        cls='all',
        problem='2GM')
    columns_names = ['P1', 'P2', 'n1', 'n2', 'e1', 'e2',
                                                'G1', 'G2', 'H1', 'H2', 'A1', 'A2', 'univ_size0', 'univ_size1', 'img0', 'img1']

    dataloader = get_dataloader(image_dataset, columns_names)
    print(image_dataset.classes)
    #print(sum(1 for _ in dataloader.create_dict_iterator()))
    print(image_dataset.length)
    for i, data in enumerate(dataloader.create_dict_iterator()):
        print(i)
        for key in data.keys():
            print(key,data[key].shape)
        print('hash',data['img0'].asnumpy()[0].shape,hashlib.md5(data['img0'].asnumpy()[0,0,0]).hexdigest())
        print('shape',data['img0'].asnumpy()[0,0,0].shape)
        print('hashdict',image_dataset.hash2id[hashlib.md5(data['img0'].asnumpy()[0,0,0]).hexdigest()])
        print('batchsize',data['P1'].shape[0])
