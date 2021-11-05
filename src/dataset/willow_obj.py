from pathlib import Path
import scipy.io as sio
from PIL import Image
import numpy as np
from src.utils.config import cfg
from src.dataset.base_dataset import BaseDataset
import random


'''
Important Notice: Face image 160 contains only 8 labeled keypoints (should be 10)
'''

class WillowObject(BaseDataset):
    def __init__(self, sets, obj_resize):
        """
        :param sets: 'train' or 'test'
        :param obj_resize: resized object size
        """
        super(WillowObject, self).__init__()
        self.classes = cfg.WillowObject.CLASSES
        self.kpt_len = [cfg.WillowObject.KPT_LEN for _ in cfg.WillowObject.CLASSES]

        self.root_path = Path(cfg.WillowObject.ROOT_DIR)
        self.obj_resize = obj_resize

        assert sets in ('train', 'test'), 'No match found for dataset {}'.format(sets)
        self.sets = sets
        self.split_offset = cfg.WillowObject.SPLIT_OFFSET
        self.train_len = cfg.WillowObject.TRAIN_NUM
        self.rand_outlier = cfg.WillowObject.RAND_OUTLIER

        self.mat_list = []
        for cls_name in self.classes:
            assert type(cls_name) is str
            cls_mat_list = [p for p in (self.root_path / cls_name).glob('*.mat')]
            if cls_name == 'Face':
                cls_mat_list.remove(self.root_path / cls_name / 'image_0160.mat')
                assert not self.root_path / cls_name / 'image_0160.mat' in cls_mat_list
            ori_len = len(cls_mat_list)
            if self.split_offset % ori_len + self.train_len <= ori_len:
                if sets == 'train' and not cfg.WillowObject.TRAIN_SAME_AS_TEST:
                    self.mat_list.append(
                        cls_mat_list[self.split_offset % ori_len: (self.split_offset + self.train_len) % ori_len]
                    )
                else:
                    self.mat_list.append(
                        cls_mat_list[:self.split_offset % ori_len] +
                        cls_mat_list[(self.split_offset + self.train_len) % ori_len:]
                    )
            else:
                if sets == 'train' and not cfg.WillowObject.TRAIN_SAME_AS_TEST:
                    self.mat_list.append(
                        cls_mat_list[:(self.split_offset + self.train_len) % ori_len - ori_len] +
                        cls_mat_list[self.split_offset % ori_len:]
                    )
                else:
                    self.mat_list.append(
                        cls_mat_list[(self.split_offset + self.train_len) % ori_len - ori_len: self.split_offset % ori_len]
                    )

    def get_pair(self, cls=None, shuffle=True):
        """
        Randomly get a pair of objects from WILLOW-object dataset
        :param cls: None for random class, or specify for a certain set
        :param shuffle: random shuffle the keypoints
        :return: (pair of data, groundtruth permutation matrix)
        """
        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)
        assert type(cls) == int and 0 <= cls < len(self.classes)

        anno_pair = []
        for mat_name in random.sample(self.mat_list[cls], 2):
            anno_dict = self.__get_anno_dict(mat_name, cls)
            if shuffle:
                random.shuffle(anno_dict['keypoints'])
            anno_pair.append(anno_dict)

        perm_mat = np.zeros([len(_['keypoints']) for _ in anno_pair], dtype=np.float32)
        row_list = []
        col_list = []
        for i, keypoint in enumerate(anno_pair[0]['keypoints']):
            for j, _keypoint in enumerate(anno_pair[1]['keypoints']):
                if keypoint['name'] == _keypoint['name']:
                    if keypoint['name'] != 'outlier':
                        perm_mat[i, j] = 1
                    row_list.append(i)
                    col_list.append(j)
                    break
        row_list.sort()
        col_list.sort()
        perm_mat = perm_mat[row_list, :]
        perm_mat = perm_mat[:, col_list]
        anno_pair[0]['keypoints'] = [anno_pair[0]['keypoints'][i] for i in row_list]
        anno_pair[1]['keypoints'] = [anno_pair[1]['keypoints'][j] for j in col_list]

        return anno_pair, perm_mat

    def get_multi(self, cls=None, num=2, shuffle=True):
        """
        Randomly get multiple objects from Willow Object Class dataset for multi-matching.
        :param cls: None for random class, or specify for a certain set
        :param num: number of objects to be fetched
        :param shuffle: random shuffle the keypoints
        :return: (list of data, list of permutation matrices)
        """
        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)
        assert type(cls) == int and 0 <= cls < len(self.classes)

        anno_list = []
        for mat_name in random.sample(self.mat_list[cls], num):
            anno_dict = self.__get_anno_dict(mat_name, cls)
            if shuffle:
                random.shuffle(anno_dict['keypoints'])
            anno_list.append(anno_dict)

        perm_mat = [np.zeros([len(anno_list[0]['keypoints']), len(x['keypoints'])], dtype=np.float32) for x in
                    anno_list]
        row_list = []
        col_lists = []
        for i in range(num):
            col_lists.append([])
        for i, keypoint in enumerate(anno_list[0]['keypoints']):
            kpt_idx = []
            for anno_dict in anno_list:
                kpt_name_list = [x['name'] for x in anno_dict['keypoints']]
                if keypoint['name'] in kpt_name_list:
                    kpt_idx.append(kpt_name_list.index(keypoint['name']))
                else:
                    kpt_idx.append(-1)
            row_list.append(i)
            for k in range(num):
                j = kpt_idx[k]
                if j != -1:
                    col_lists[k].append(j)
                    if keypoint['name'] != 'outlier':
                        perm_mat[k][i, j] = 1

        row_list.sort()
        for col_list in col_lists:
            col_list.sort()

        for k in range(num):
            perm_mat[k] = perm_mat[k][row_list, :]
            perm_mat[k] = perm_mat[k][:, col_lists[k]]
            anno_list[k]['keypoints'] = [anno_list[k]['keypoints'][j] for j in col_lists[k]]
            perm_mat[k] = perm_mat[k].transpose()

        return anno_list, perm_mat
    
    def __get_anno_dict(self, mat_file, cls):
        """
        Get an annotation dict from .mat annotation
        """
        assert mat_file.exists(), '{} does not exist.'.format(mat_file)

        img_name = mat_file.stem + '.png'
        img_file = mat_file.parent / img_name

        struct = sio.loadmat(mat_file.open('rb'))
        kpts = struct['pts_coord']

        with Image.open(str(img_file)) as img:
            ori_sizes = img.size
            obj = img.resize(self.obj_resize, resample=Image.BICUBIC)
            xmin = 0
            ymin = 0
            w = ori_sizes[0]
            h = ori_sizes[1]

        keypoint_list = []
        for idx, keypoint in enumerate(np.split(kpts, kpts.shape[1], axis=1)):
            attr = {
                'name': idx,
                'x': float(keypoint[0]) * self.obj_resize[0] / w,
                'y': float(keypoint[1]) * self.obj_resize[1] / h
            }
            keypoint_list.append(attr)

        for idx in range(self.rand_outlier):
            attr = {
                'name': 'outlier',
                'x': random.uniform(0, self.obj_resize[0]),
                'y': random.uniform(0, self.obj_resize[1])
            }
            keypoint_list.append(attr)

        anno_dict = dict()
        anno_dict['image'] = obj
        anno_dict['keypoints'] = keypoint_list
        anno_dict['bounds'] = xmin, ymin, w, h
        anno_dict['ori_sizes'] = ori_sizes
        anno_dict['cls'] = cls
        anno_dict['univ_size'] = 10

        return anno_dict

    def len(self, cls):
        if type(cls) == int:
            cls = self.classes[cls]
        assert cls in self.classes
        return len(self.mat_list[self.classes.index(cls)])


if __name__ == '__main__':
    cfg.WillowObject.ROOT_DIR = 'WILLOW-ObjectClass'
    cfg.WillowObject.SPLIT_OFFSET = 0
    train = WillowObject('train', (256, 256))
    test = WillowObject('test', (256, 256))
    for train_cls_list, test_cls_list in zip(train.mat_list, test.mat_list):
        for t in train_cls_list:
            assert t not in test_cls_list
    pass
