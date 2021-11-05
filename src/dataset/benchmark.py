import sys
sys.path.insert(0,"/data/lixinyang/mindspore3/ThinkMatch")
#sys.path.insert(0,"/data/lixinyang/mindspore3/ThinkMatch/src/dataset")

import src.dataset.dataset as data
#import dataset as data
import requests
import os
import zipfile
import tarfile
from pathlib import Path
from PIL import Image
import numpy as np
import random
import json
import itertools
from scipy.sparse import coo_matrix
import time
from tqdm import tqdm


class Benchmark:
    def __init__(self, name, sets, obj_resize, length, cls, problem='2GM', filter='intersection'):
        
        assert name == 'PascalVOC' or name == 'SPair-71k' or name == 'WillowObject', 'No match found for dataset {}'.format(name)
        assert problem == '2GM' or problem == 'MGM', 'No match found for problem {}'.format(problem)
        assert filter == 'intersection' or filter == 'inclusion' or filter == 'unfiltered', 'No match found for filter {}'.format(filter)
        assert not(problem == 'MGM' and filter == 'inclusion'), 'The filter inclusion only matches 2GM'
        
        self.name = name
        self.problem = problem
        self.filter = filter
        self.sets = sets
        self.obj_resize = obj_resize
        self.data_path = os.path.join('data', name, 'data.json')
        self.data_list_path = os.path.join('data', name, sets + '.json')
        self.gt = dict()
        self.length = length
        self.cls = None if cls in ['none', 'all'] else cls
        
        if name == 'PascalVOC':
            data_set = data.PascalVOC(self.sets, self.obj_resize, self.problem, self.filter)
        if name == 'WillowObject':
            data_set = data.WillowObject(self.sets, self.obj_resize, self.problem, self.filter)
        if name == 'SPair-71k':
            data_set = data.SPair71k(self.sets, self.obj_resize, self.problem, self.filter)
        
        self.classes = data_set.classes

    #def __getitem__(self, index):
    #    return rand_get_data(self, cls=None, num=2, test=False, shuffle=True):

    def __len__(self):
        return self.length

    def get_data(self, ids, test=False, shuffle=True):
        assert (self.problem == '2GM' and len(ids) == 2) or (self.problem == 'MGM' and len(ids) > 2), '{} problem cannot get {} data'.format(self.problem, len(ids))

        with open(self.data_path) as f:
            data_dict = json.load(f)

        data_list = []
        for keys in ids:
            obj_dict = dict()
            boundbox = data_dict[keys]['bounds']
            img_file = data_dict[keys]['path']
            with Image.open(str(img_file)) as img:
                obj = img.resize(self.obj_resize, resample=Image.BICUBIC, box=(boundbox[0], boundbox[1], boundbox[2], boundbox[3]))
            obj_dict['img'] = np.array(obj)
            obj_dict['kpts'] = data_dict[keys]['kpts']
            obj_dict['cls'] = data_dict[keys]['cls']
            if shuffle:
                random.shuffle(obj_dict['kpts'])
            data_list.append(obj_dict)

        perm_mat_dict = dict()
        id_combination = list(itertools.combinations(list(range(len(ids))), 2))
        for id_tuple in id_combination:
            perm_mat = np.zeros([len(data_list[_]['kpts']) for _ in id_tuple], dtype=np.float32)
            row_list = []
            col_list = []
             
            for i, keypoint in enumerate(data_list[id_tuple[0]]['kpts']):
                for j, _keypoint in enumerate(data_list[id_tuple[1]]['kpts']):
                    if keypoint['labels'] == _keypoint['labels']:
                        if keypoint['labels'] != 'outlier':
                            perm_mat[i, j] = 1
                        row_list.append(i)
                        col_list.append(j)
                        break
            row_list.sort()
            col_list.sort()
            if self.filter == 'intersection':
                perm_mat = perm_mat[row_list, :]
                perm_mat = perm_mat[:, col_list]
                data_list[id_tuple[0]]['kpts'] = [data_list[id_tuple[0]]['kpts'][i] for i in row_list]
                data_list[id_tuple[1]]['kpts'] = [data_list[id_tuple[1]]['kpts'][i] for i in col_list]
            elif self.filter == 'inclusion':
                perm_mat = perm_mat[row_list, :]
                data_list[id_tuple[0]]['kpts'] = [data_list[id_tuple[0]]['kpts'][i] for i in row_list]
            if not(len(ids) > 2 and self.filter == 'intersection'):
                sparse_perm_mat = coo_matrix(perm_mat)
                perm_mat_dict[id_tuple] = sparse_perm_mat

        if len(ids) > 2 and self.filter == 'intersection':
            for p in range(len(ids) - 1):
                perm_mat_list = [np.zeros([len(data_list[p]['kpts']), len(x['kpts'])], dtype=np.float32) for x in
                                 data_list[p + 1: len(ids)]]
                row_list = []
                col_lists = []
                for i in range(len(ids) - p - 1):
                    col_lists.append([])

                for i, keypoint in enumerate(data_list[p]['kpts']):
                    kpt_idx = []
                    for anno_dict in data_list[p + 1: len(ids)]:
                        kpt_name_list = [x['labels'] for x in anno_dict['kpts']]
                        if keypoint['labels'] in kpt_name_list:
                            kpt_idx.append(kpt_name_list.index(keypoint['labels']))
                        else:
                            kpt_idx.append(-1)
                    row_list.append(i)
                    for k in range(len(ids) - p - 1):
                        j = kpt_idx[k]
                        if j != -1:
                            col_lists[k].append(j)
                            if keypoint['labels'] != 'outlier':
                                perm_mat_list[k][i, j] = 1

                row_list.sort()
                for col_list in col_lists:
                    col_list.sort()

                for k in range(len(ids) - p - 1):
                    perm_mat_list[k] = perm_mat_list[k][row_list, :]
                    perm_mat_list[k] = perm_mat_list[k][:, col_lists[k]]
                    id_tuple = (p, k + p + 1)
                    perm_mat_dict[id_tuple] = coo_matrix(perm_mat_list[k])

        for pair in id_combination:
            id_pair = (ids[pair[0]], ids[pair[1]])
            if not (id_pair in self.gt.keys()):
                self.gt[id_pair] = perm_mat_dict[pair]

        if not test:
            return data_list, perm_mat_dict, id_combination
        else:
            return data_list, id_combination

    def rand_get_data(self, cls=None, num=2, test=False, shuffle=True):
        if cls == None:
            cls = random.randrange(0, len(self.classes))
            clss = self.classes[cls]
        elif type(cls) == str:
            clss = cls

        with open(self.data_path) as f:
            data_dict = json.load(f)

        with open(self.data_list_path) as f1:
            data_id = json.load(f1)

        data_list = []
        for id in data_id:
            if data_dict[id]['cls'] == clss:
                data_list.append(id)

        ids = []
        for objID in random.sample(data_list, num):
            ids.append(objID)
        
        return self.get_data(ids, test, shuffle)

    def get_id_combination(self, cls, num=2):
        if cls == None:
            clss = None
        elif type(cls) == str:
            clss = cls

        with open(self.data_path) as f:
            data_dict = json.load(f)

        with open(self.data_list_path) as f1:
            data_id = json.load(f1)

        if clss != None:
            data_list = []
            for id in data_id:
                if data_dict[id]['cls'] == clss:
                    data_list.append(id)

            id_combination = list(itertools.combinations(data_list, num))
            return id_combination
        else:
            id_combination = []
            for clss in self.classes:
                data_list = []
                for id in data_id:
                    if data_dict[id]['cls'] == clss:
                        data_list.append(id)
                id_combination = id_combination + list(itertools.combinations(data_list, num))
            return id_combination

    def compute_length(self, cls, num=2):
        if cls == None:
            clss = None
        elif type(cls) == str:
            clss = cls

        with open(self.data_path) as f:
            data_dict = json.load(f)

        with open(self.data_list_path) as f1:
            data_id = json.load(f1)

        length = 0

        if clss != None:
            data_list = []
            for id in data_id:
                if data_dict[id]['cls'] == clss:
                    data_list.append(id)

            id_combination = list(itertools.combinations(data_list, num))
            length = length + len(id_combination)

        else:
            for clss in self.classes:
                data_list = []
                for id in data_id:
                    if data_dict[id]['cls'] == clss:
                        data_list.append(id)
                id_combination = list(itertools.combinations(data_list, num))
                length = length + len(id_combination)
        return length

    def eval(self, prediction, verbose=False):
        with open(self.data_path) as f:
            data_dict = json.load(f)
        cls_dict = dict()
        pred_cls_dict = dict()
        result = dict()

        for cls in self.classes:
            cls_dict[cls] = 0
            pred_cls_dict[cls] = 0
            result[cls] = dict()
            result[cls]['precision'] = 0
            result[cls]['recall'] = 0
            result[cls]['f1'] = 0

        for obj in data_dict:
            cls_dict[obj['cls']] = cls_dict[obj['cls']] + 1

        for pair_dict in prediction:
            tmp = [pair_dict['ids'][0], pair_dict['ids'][1]]
            tmp.sort()
            ids = (tmp[0], tmp[1])
            pred_cls_dict[pair_dict['cls']] = pred_cls_dict[pair_dict['cls']] + 1

            perm_mat = pair_dict['permmat']
            gt = self.gt[ids]
            gt_array = gt.toarray()
            if type(perm_mat) != type(gt_array):
                perm_mat = perm_mat.toarray()

            precesion = (perm_mat * gt_array).sum() / perm_mat.sum()
            recall = (perm_mat * gt_array).sum() / gt_array.sum()
            f1_score = (2 * precesion * recall) / (precesion + recall)

            result[pair_dict['cls']]['precision'] = result[pair_dict['cls']]['precision'] + precesion
            result[pair_dict['cls']]['recall'] = result[pair_dict['cls']]['recall'] + recall
            result[pair_dict['cls']]['f1'] = result[pair_dict['cls']]['f1'] + f1_score

        p_sum = 0
        r_sum = 0
        f1_sum = 0
        pred_sum = 0
        total = 0
        for cls in self.classes:
            result[cls]['precision'] = result[cls]['precision'] / pred_cls_dict[cls]
            result[cls]['recall'] = result[cls]['recall'] / pred_cls_dict[cls]
            result[cls]['f1'] = result[cls]['f1'] / pred_cls_dict[cls]
            result[cls]['coverage'] = 2 * pred_cls_dict[cls] / (cls_dict[cls] * (cls_dict[cls] - 1))
            p_sum = p_sum + result[cls]['precision']
            r_sum = r_sum + result[cls]['recall']
            f1_sum = f1_sum + result[cls]['f1']
            pred_sum = pred_sum + pred_cls_dict[cls]
            total = total + (cls_dict[cls] * (cls_dict[cls] - 1)) / 2

        result['mean'] = dict()
        result['mean']['presicion'] = p_sum / len(self.classes)
        result['mean']['recall'] = r_sum / len(self.classes)
        result['mean']['f1'] = f1_sum / len(self.classes)
        result['mean']['coverage'] = pred_sum / total

        if verbose:
            print(result)
        return result


if __name__ == '__main__':
    #data = Benchmark('PascalVOC', 'test', (256, 256), problem='2GM')
    data = Benchmark(name='PascalVOC', sets='test', obj_resize=(256, 256),
              length=1000, cls='all', problem='2GM')
    b = data.rand_get_data(num=2, test=False, shuffle=True)
    print(len(b[0]))
    pass
