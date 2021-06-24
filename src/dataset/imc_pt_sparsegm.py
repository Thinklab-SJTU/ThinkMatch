from pathlib import Path
from PIL import Image
import numpy as np
from src.utils.config import cfg
from src.dataset.base_dataset import BaseDataset
import random


class IMC_PT_SparseGM(BaseDataset):
    def __init__(self, sets, obj_resize):
        """
        :param sets: 'train' or 'test'
        :param obj_resize: resized object size
        """
        super(IMC_PT_SparseGM, self).__init__()
        assert sets in ('train', 'test'), 'No match found for dataset {}'.format(sets)
        self.sets = sets
        self.classes = cfg.IMC_PT_SparseGM.CLASSES[sets]
        self.total_kpt_num = cfg.IMC_PT_SparseGM.TOTAL_KPT_NUM

        self.root_path_npz = Path(cfg.IMC_PT_SparseGM.ROOT_DIR_NPZ)
        self.root_path_img = Path(cfg.IMC_PT_SparseGM.ROOT_DIR_IMG)
        self.obj_resize = obj_resize

        self.img_lists = [np.load(self.root_path_npz / cls / 'img_info.npz')['img_name'].tolist()
                          for cls in self.classes]

    def get_pair(self, cls=None, shuffle=True, tgt_outlier=False, src_outlier=False):
        """
        Randomly get a pair of objects from Photo Tourism dataset
        :param cls: None for random class, or specify for a certain set
        :param shuffle: random shuffle the keypoints
        :param src_outlier: allow outlier in the source graph (first graph)
        :param tgt_outlier: allow outlier in the target graph (second graph)
        :return: (pair of data, groundtruth permutation matrix)
        """
        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)
        assert type(cls) == int and 0 <= cls < len(self.classes)

        anno_pair = []
        for img_name in random.sample(self.img_lists[cls], 2):
            anno_dict = self.__get_anno_dict(img_name, cls)
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
        if not src_outlier:
            perm_mat = perm_mat[row_list, :]
            anno_pair[0]['keypoints'] = [anno_pair[0]['keypoints'][i] for i in row_list]
        if not tgt_outlier:
            perm_mat = perm_mat[:, col_list]
            anno_pair[1]['keypoints'] = [anno_pair[1]['keypoints'][j] for j in col_list]

        return anno_pair, perm_mat

    def get_multi(self, cls=None, num=2, shuffle=True, filter_outlier=False):
        """
        Randomly get multiple objects from Willow Object Class dataset for multi-matching.
        :param cls: None for random class, or specify for a certain set
        :param num: number of objects to be fetched
        :param shuffle: random shuffle the keypoints
        :param filter_outlier: filter out outlier keypoints among images
        :return: (list of data, list of permutation matrices)
        """
        assert not filter_outlier, 'Multi-matching on IMC_PT_SparseGM dataset with filtered outliers is not supported'

        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)
        assert type(cls) == int and 0 <= cls < len(self.classes)

        anno_list = []
        for img_name in random.sample(self.img_lists[cls], num):
            anno_dict = self.__get_anno_dict(img_name, cls)
            if shuffle:
                random.shuffle(anno_dict['keypoints'])
            anno_list.append(anno_dict)

        perm_mat = [np.zeros([self.total_kpt_num, len(x['keypoints'])], dtype=np.float32) for x in anno_list]
        for k, anno_dict in enumerate(anno_list):
            kpt_name_list = [x['name'] for x in anno_dict['keypoints']]
            for j, name in enumerate(kpt_name_list):
                perm_mat[k][name, j] = 1
        for k in range(num):
            perm_mat[k] = perm_mat[k].transpose()

        return anno_list, perm_mat

    def __get_anno_dict(self, img_name, cls):
        """
        Get an annotation dict from .npz annotation
        """
        img_file = self.root_path_img / self.classes[cls] / 'dense' / 'images' / img_name
        npz_file = self.root_path_npz / self.classes[cls] / (img_name + '.npz')

        assert img_file.exists(), '{} does not exist.'.format(img_file)
        assert npz_file.exists(), '{} does not exist.'.format(npz_file)

        with Image.open(str(img_file)) as img:
            ori_sizes = img.size
            obj = img.resize(self.obj_resize, resample=Image.BICUBIC)
            xmin = 0
            ymin = 0
            w = ori_sizes[0]
            h = ori_sizes[1]

        with np.load(str(npz_file)) as npz_anno:
            kpts = npz_anno['points']
            if len(kpts.shape) != 2:
                ValueError('{} contains no keypoints.'.format(img_file))

        keypoint_list = []
        for i in range(kpts.shape[1]):
            kpt_index = int(kpts[0, i])
            assert kpt_index < self.total_kpt_num
            attr = {
                'name': kpt_index,
                'x': kpts[1, i] * self.obj_resize[0] / w,
                'y': kpts[2, i] * self.obj_resize[1] / h
            }
            keypoint_list.append(attr)

        anno_dict = dict()
        anno_dict['image'] = obj
        anno_dict['keypoints'] = keypoint_list
        anno_dict['bounds'] = xmin, ymin, w, h
        anno_dict['ori_sizes'] = ori_sizes
        anno_dict['cls'] = self.classes[cls]
        anno_dict['univ_size'] = self.total_kpt_num

        return anno_dict

    def len(self, cls):
        if type(cls) == int:
            cls = self.classes[cls]
        assert cls in self.classes
        return len(self.img_lists[self.classes.index(cls)])
