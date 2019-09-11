from pathlib import Path
import scipy.io as sio
from PIL import Image
import numpy as np
from utils.config import cfg
from data.base_dataset import BaseDataset
import random


class WillowObject(BaseDataset):
    def __init__(self, sets, obj_resize):
        """
        :param sets: 'train' or 'test'
        :param obj_resize: resized object size
        """
        super(WillowObject, self).__init__()
        self.classes = cfg.WILLOW.CLASSES
        self.kpt_len = [cfg.WILLOW.KPT_LEN for _ in cfg.WILLOW.CLASSES]

        self.root_path = Path(cfg.WILLOW.ROOT_DIR)
        self.obj_resize = obj_resize

        assert sets == 'train' or 'test', 'No match found for dataset {}'.format(sets)
        self.split_offset = cfg.WILLOW.TRAIN_OFFSET
        self.train_len = cfg.WILLOW.TRAIN_NUM

        self.mat_list = []
        for cls_name in self.classes:
            assert type(cls_name) is str
            cls_mat_list = [p for p in (self.root_path / cls_name).glob('*.mat')]
            ori_len = len(cls_mat_list)
            assert ori_len > 0, 'No data found for WILLOW Object Class. Is the dataset installed correctly?'
            if self.split_offset % ori_len + self.train_len <= ori_len:
                if sets == 'train':
                    self.mat_list.append(
                        cls_mat_list[self.split_offset % ori_len: (self.split_offset + self.train_len) % ori_len]
                    )
                else:
                    self.mat_list.append(
                        cls_mat_list[:self.split_offset % ori_len] +
                        cls_mat_list[(self.split_offset + self.train_len) % ori_len:]
                    )
            else:
                if sets == 'train':
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
            attr = {'name': idx}
            attr['x'] = float(keypoint[0]) * self.obj_resize[0] / w
            attr['y'] = float(keypoint[1]) * self.obj_resize[1] / h
            keypoint_list.append(attr)

        anno_dict = dict()
        anno_dict['image'] = obj
        anno_dict['keypoints'] = keypoint_list
        anno_dict['bounds'] = xmin, ymin, w, h
        anno_dict['ori_sizes'] = ori_sizes
        anno_dict['cls'] = cls

        return anno_dict
