import sys
sys.path.insert(0,"/mnt/nas/home/lixinyang/mindspore3/ThinkMatch")
import tempfile
import requests
import os
import shutil
import zipfile
import tarfile
from pathlib import Path
from PIL import Image
import numpy as np
import random
import json
import itertools
from scipy.sparse import coo_matrix
from api.dataset.dataset import *


# from dataset import *


class Benchmark:
    def __init__(self, name, sets, obj_resize, problem='2GM', filter='intersection'):

        # assert name == 'PascalVOC' or name == 'SPair-71k' or name == 'WillowObject', 'No match found for dataset {}'.format(name)
        assert problem == '2GM' or problem == 'MGM', 'No match found for problem {}'.format(problem)
        assert filter == 'intersection' or filter == 'inclusion' or filter == 'unfiltered', 'No match found for filter {}'.format(
            filter)
        assert not (problem == 'MGM' and filter == 'inclusion'), 'The filter inclusion only matches 2GM'

        self.name = name
        self.problem = problem
        self.filter = filter
        self.sets = sets
        self.obj_resize = obj_resize

        data_set = eval(self.name)(self.sets, self.obj_resize)
        self.data_path = os.path.join(data_set.dataset_dir, 'data.json')
        self.data_list_path = os.path.join(data_set.dataset_dir, sets + '.json')
        self.classes = data_set.classes

        with open(self.data_path) as f:
            self.data_dict = json.load(f)

        if self.sets == 'test':
            tmpfile = tempfile.gettempdir()
            pid_num = os.getpid()
            cache_dir = str(pid_num) + '_gt_cache'
            self.gt_cache_path = os.path.join(tmpfile, cache_dir)

            if not os.path.exists(self.gt_cache_path):
                os.mkdir(self.gt_cache_path)
                print('gt perm mat cache built')

    def get_data(self, ids, test=False, shuffle=True):
        assert (self.problem == '2GM' and len(ids) == 2) or (
                    self.problem == 'MGM' and len(ids) > 2), '{} problem cannot get {} data'.format(self.problem,
                                                                                                    len(ids))

        ids.sort()
        data_list = []
        for keys in ids:
            obj_dict = dict()
            boundbox = self.data_dict[keys]['bounds']
            img_file = self.data_dict[keys]['path']
            with Image.open(str(img_file)) as img:
                obj = img.resize(self.obj_resize, resample=Image.BICUBIC,
                                 box=(boundbox[0], boundbox[1], boundbox[2], boundbox[3]))
                if self.name == 'CUB2011':
                    if not obj.mode == 'RGB':
                        obj = obj.convert('RGB')
            obj_dict['img'] = np.array(obj)
            obj_dict['kpts'] = self.data_dict[keys]['kpts']
            obj_dict['cls'] = self.data_dict[keys]['cls']
            obj_dict['univ_size'] = self.data_dict[keys]['univ_size']
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
            if not (len(ids) > 2 and self.filter == 'intersection'):
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

        if self.sets == 'test':
            for pair in id_combination:
                id_pair = (ids[pair[0]], ids[pair[1]])
                gt_path = os.path.join(self.gt_cache_path, str(id_pair) + '.npy')
                if not os.path.exists(gt_path):
                    np.save(gt_path, perm_mat_dict[pair])

        id_combination = list(itertools.combinations(ids, 2))
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

        with open(self.data_list_path) as f1:
            data_id = json.load(f1)

        data_list = []
        for id in data_id:
            if self.data_dict[id]['cls'] == clss:
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

        with open(self.data_list_path) as f1:
            data_id = json.load(f1)

        length = 0
        id_combination_list = []
        if clss != None:
            data_list = []
            for id in data_id:
                if self.data_dict[id]['cls'] == clss:
                    data_list.append(id)
            id_combination = list(itertools.combinations(data_list, num))
            length = length + len(id_combination)
            id_combination_list.append(id_combination)
        else:
            for clss in self.classes:
                data_list = []
                for id in data_id:
                    if self.data_dict[id]['cls'] == clss:
                        data_list.append(id)
                id_combination = list(itertools.combinations(data_list, num))
                length = length + len(id_combination)
                id_combination_list.append(id_combination)
        return id_combination_list, length

    def compute_length(self, cls, num=2):
        if cls == None:
            clss = None
        elif type(cls) == str:
            clss = cls

        with open(self.data_list_path) as f1:
            data_id = json.load(f1)

        length = 0

        if clss != None:
            data_list = []
            for id in data_id:
                if self.data_dict[id]['cls'] == clss:
                    data_list.append(id)
            id_combination = list(itertools.combinations(data_list, num))
            length = length + len(id_combination)

        else:
            for clss in self.classes:
                data_list = []
                for id in data_id:
                    if self.data_dict[id]['cls'] == clss:
                        data_list.append(id)
                id_combination = list(itertools.combinations(data_list, num))
                length = length + len(id_combination)
        return length

    def compute_img_num(self, classes):
        with open(self.data_list_path) as f1:
            data_id = json.load(f1)
        for clss in classes:
            cls_img_num = 0
            num_list = []
            for id in data_id:
                if self.data_dict[id]['cls'] == clss:
                    cls_img_num += 1
            num_list.append(cls_img_num)

        return num_list

    def eval(self, prediction, classes, verbose=False):

        with open(self.data_list_path) as f1:
            data_id = json.load(f1)

        cls_dict = dict()
        pred_cls_dict = dict()
        result = dict()
        id_cache = []

        for cls in classes:
            cls_dict[cls] = 0
            pred_cls_dict[cls] = 0
            result[cls] = dict()
            result[cls]['precision'] = 0
            result[cls]['recall'] = 0
            result[cls]['f1'] = 0

        for key, obj in self.data_dict.items():
            if key in data_id:
                cls_dict[obj['cls']] += 1

        for pair_dict in prediction:
            ids = (pair_dict['ids'][0], pair_dict['ids'][1])
            if ids not in id_cache:
                id_cache.append(ids)
                pred_cls_dict[pair_dict['cls']] += 1
                perm_mat = pair_dict['perm_mat']
                gt_path = os.path.join(self.gt_cache_path, str(ids) + '.npy')
                gt = np.load(gt_path, allow_pickle=True).item()
                gt_array = gt.toarray()
                assert type(perm_mat) == type(gt_array)

                if perm_mat.sum() == 0 or gt_array.sum() == 0:
                    precision = 1
                    recall = 1
                else:
                    precision = (perm_mat * gt_array).sum() / perm_mat.sum()
                    recall = (perm_mat * gt_array).sum() / gt_array.sum()
                if precision == 0 or recall == 0:
                    f1_score = 0
                else:
                    f1_score = (2 * precision * recall) / (precision + recall)

                result[pair_dict['cls']]['precision'] += precision
                result[pair_dict['cls']]['recall'] += recall
                result[pair_dict['cls']]['f1'] += f1_score

        p_sum = 0
        r_sum = 0
        f1_sum = 0
        pred_sum = 0
        total = 0
        for cls in classes:
            # print(pred_cls_dict[cls])
            # print(cls_dict[cls])
            result[cls]['precision'] /= pred_cls_dict[cls]
            result[cls]['recall'] /= pred_cls_dict[cls]
            result[cls]['f1'] /= pred_cls_dict[cls]
            result[cls]['coverage'] = 2 * pred_cls_dict[cls] / (cls_dict[cls] * (cls_dict[cls] - 1))
            p_sum += result[cls]['precision']
            r_sum += result[cls]['recall']
            f1_sum += result[cls]['f1']
            pred_sum += pred_cls_dict[cls]
            total += (cls_dict[cls] * (cls_dict[cls] - 1)) / 2

        result['mean'] = dict()
        result['mean']['precision'] = p_sum / len(classes)
        result['mean']['recall'] = r_sum / len(classes)
        result['mean']['f1'] = f1_sum / len(classes)
        result['mean']['coverage'] = pred_sum / total

        if verbose:
            print('Matching accuracy')
            for cls in classes:
                print('{}: {}'.format(cls, 'p = {:.4f}, r = {:.4f}, f1 = {:.4f}, cvg = {:.4f}' \
                                      .format(result[cls]['precision'], result[cls]['recall'], result[cls]['f1'],
                                              result[cls]['coverage']
                                              )))
            print('average accuracy: {}'.format('p = {:.4f}, r = {:.4f}, f1 = {:.4f}, cvg = {:.4f}' \
                                                .format(result['mean']['precision'], result['mean']['recall'],
                                                        result['mean']['f1'], result['mean']['coverage']
                                                        )))
        return result

    def rm_gt_cache(self, last_epoch=False):
        if os.path.exists(self.gt_cache_path):
            shutil.rmtree(self.gt_cache_path)
            print('gt perm mat cache deleted')

            if not last_epoch:
                os.mkdir(self.gt_cache_path)


if __name__ == '__main__':
    data = Benchmark('PascalVOC', 'train', (256, 256), problem='2GM')
    data_list, perm_mat_dict, id_combination = data.get_data(0)
    print(type(data_list[0]),type(data_list[1]),type(perm_mat_dict),type(id_combination[0]))
    for i in data_list[0].keys():
        print(i)
    print('aa')
    for i in data_list[1].keys():
        print(i)
    print('bb')
    print(len(perm_mat_dict))
    for i in perm_mat_dict.values():
        print(i.toarray())
    print(len(data_list), len(perm_mat_dict), len(id_combination))
    # data = Benchmark('IMC_PT_SparseGM', 'test', (256, 256), problem='MGM')
    # data = Benchmark('CUB2011', 'train', (256, 256), problem='2GM')
    # b = data.rand_get_data(num=4, test=False, shuffle=True)
    pass