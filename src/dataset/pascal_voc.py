from pathlib import Path
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import random
import pickle

from src.utils.config import cfg

KPT_NAMES = {
    'cat': ['L_B_Elbow', 'L_B_Paw', 'L_EarBase', 'L_Eye', 'L_F_Elbow',
            'L_F_Paw', 'Nose', 'R_B_Elbow', 'R_B_Paw', 'R_EarBase', 'R_Eye',
            'R_F_Elbow', 'R_F_Paw', 'TailBase', 'Throat', 'Withers'],
    'bottle': ['L_Base', 'L_Neck', 'L_Shoulder', 'L_Top', 'R_Base', 'R_Neck',
               'R_Shoulder', 'R_Top'],
    'horse': ['L_B_Elbow', 'L_B_Paw', 'L_EarBase', 'L_Eye', 'L_F_Elbow',
              'L_F_Paw', 'Nose', 'R_B_Elbow', 'R_B_Paw', 'R_EarBase', 'R_Eye',
              'R_F_Elbow', 'R_F_Paw', 'TailBase', 'Throat', 'Withers'],
    'motorbike': ['B_WheelCenter', 'B_WheelEnd', 'ExhaustPipeEnd',
                  'F_WheelCenter', 'F_WheelEnd', 'HandleCenter', 'L_HandleTip',
                  'R_HandleTip', 'SeatBase', 'TailLight'],
    'boat': ['Hull_Back_Bot', 'Hull_Back_Top', 'Hull_Front_Bot',
             'Hull_Front_Top', 'Hull_Mid_Left_Bot', 'Hull_Mid_Left_Top',
             'Hull_Mid_Right_Bot', 'Hull_Mid_Right_Top', 'Mast_Top', 'Sail_Left',
             'Sail_Right'],
    'tvmonitor': ['B_Bottom_Left', 'B_Bottom_Right', 'B_Top_Left',
                  'B_Top_Right', 'F_Bottom_Left', 'F_Bottom_Right', 'F_Top_Left',
                  'F_Top_Right'],
    'cow': ['L_B_Elbow', 'L_B_Paw', 'L_EarBase', 'L_Eye', 'L_F_Elbow',
            'L_F_Paw', 'Nose', 'R_B_Elbow', 'R_B_Paw', 'R_EarBase', 'R_Eye',
            'R_F_Elbow', 'R_F_Paw', 'TailBase', 'Throat', 'Withers'],
    'chair': ['BackRest_Top_Left', 'BackRest_Top_Right', 'Leg_Left_Back',
              'Leg_Left_Front', 'Leg_Right_Back', 'Leg_Right_Front',
              'Seat_Left_Back', 'Seat_Left_Front', 'Seat_Right_Back',
              'Seat_Right_Front'],
    'car': ['L_B_RoofTop', 'L_B_WheelCenter', 'L_F_RoofTop', 'L_F_WheelCenter',
            'L_HeadLight', 'L_SideviewMirror', 'L_TailLight', 'R_B_RoofTop',
            'R_B_WheelCenter', 'R_F_RoofTop', 'R_F_WheelCenter', 'R_HeadLight',
            'R_SideviewMirror', 'R_TailLight'],
    'person': ['B_Head', 'HeadBack', 'L_Ankle', 'L_Ear', 'L_Elbow', 'L_Eye',
               'L_Foot', 'L_Hip', 'L_Knee', 'L_Shoulder', 'L_Toes', 'L_Wrist', 'Nose',
               'R_Ankle', 'R_Ear', 'R_Elbow', 'R_Eye', 'R_Foot', 'R_Hip', 'R_Knee',
               'R_Shoulder', 'R_Toes', 'R_Wrist'],
    'diningtable': ['Bot_Left_Back', 'Bot_Left_Front', 'Bot_Right_Back',
                    'Bot_Right_Front', 'Top_Left_Back', 'Top_Left_Front', 'Top_Right_Back',
                    'Top_Right_Front'],
    'dog': ['L_B_Elbow', 'L_B_Paw', 'L_EarBase', 'L_Eye', 'L_F_Elbow',
            'L_F_Paw', 'Nose', 'R_B_Elbow', 'R_B_Paw', 'R_EarBase', 'R_Eye',
            'R_F_Elbow', 'R_F_Paw', 'TailBase', 'Throat', 'Withers'],
    'bird': ['Beak_Base', 'Beak_Tip', 'Left_Eye', 'Left_Wing_Base',
             'Left_Wing_Tip', 'Leg_Center', 'Lower_Neck_Base', 'Right_Eye',
             'Right_Wing_Base', 'Right_Wing_Tip', 'Tail_Tip', 'Upper_Neck_Base'],
    'bicycle': ['B_WheelCenter', 'B_WheelEnd', 'B_WheelIntersection',
                'CranksetCenter', 'F_WheelCenter', 'F_WheelEnd', 'F_WheelIntersection',
                'HandleCenter', 'L_HandleTip', 'R_HandleTip', 'SeatBase'],
    'train': ['Base_Back_Left', 'Base_Back_Right', 'Base_Front_Left',
              'Base_Front_Right', 'Roof_Back_Left', 'Roof_Back_Right',
              'Roof_Front_Middle'],
    'sheep': ['L_B_Elbow', 'L_B_Paw', 'L_EarBase', 'L_Eye', 'L_F_Elbow',
              'L_F_Paw', 'Nose', 'R_B_Elbow', 'R_B_Paw', 'R_EarBase', 'R_Eye',
              'R_F_Elbow', 'R_F_Paw', 'TailBase', 'Throat', 'Withers'],
    'aeroplane': ['Bot_Rudder', 'Bot_Rudder_Front', 'L_Stabilizer',
                  'L_WingTip', 'Left_Engine_Back', 'Left_Engine_Front',
                  'Left_Wing_Base', 'NoseTip', 'Nose_Bottom', 'Nose_Top',
                  'R_Stabilizer', 'R_WingTip', 'Right_Engine_Back',
                  'Right_Engine_Front', 'Right_Wing_Base', 'Top_Rudder'],
    'sofa': ['Back_Base_Left', 'Back_Base_Right', 'Back_Top_Left',
             'Back_Top_Right', 'Front_Base_Left', 'Front_Base_Right',
             'Handle_Front_Left', 'Handle_Front_Right', 'Handle_Left_Junction',
             'Handle_Right_Junction', 'Left_Junction', 'Right_Junction'],
    'pottedplant': ['Bottom_Left', 'Bottom_Right', 'Top_Back_Middle',
                    'Top_Front_Middle', 'Top_Left', 'Top_Right'],
    'bus': ['L_B_Base', 'L_B_RoofTop', 'L_F_Base', 'L_F_RoofTop', 'R_B_Base',
            'R_B_RoofTop', 'R_F_Base', 'R_F_RoofTop']
}


class PascalVOC:
    def __init__(self, sets, obj_resize):
        """
        :param sets: 'train' or 'test'
        :param obj_resize: resized object size
        """
        self.classes = cfg.PascalVOC.CLASSES
        self.kpt_len = [len(KPT_NAMES[_]) for _ in cfg.PascalVOC.CLASSES]

        anno_path = cfg.PascalVOC.KPT_ANNO_DIR
        img_path = cfg.PascalVOC.ROOT_DIR + 'JPEGImages'
        ori_anno_path = cfg.PascalVOC.ROOT_DIR + 'Annotations'
        set_path = cfg.PascalVOC.SET_SPLIT
        cache_path = cfg.CACHE_PATH

        self.classes_kpts = {cls: len(KPT_NAMES[cls]) for cls in self.classes}

        self.anno_path = Path(anno_path)
        self.img_path = Path(img_path)
        self.ori_anno_path = Path(ori_anno_path)
        self.obj_resize = obj_resize
        self.sets = sets

        assert sets in ('train', 'test'), 'No match found for dataset {}'.format(sets)
        cache_name = 'voc_db_' + sets + '.pkl'
        self.cache_path = Path(cache_path)
        self.cache_file = self.cache_path / cache_name
        if self.cache_file.exists():
            with self.cache_file.open(mode='rb') as f:
                self.xml_list = pickle.load(f)
            print('xml list loaded from {}'.format(self.cache_file))
        else:
            print('Caching xml list to {}...'.format(self.cache_file))
            self.cache_path.mkdir(exist_ok=True, parents=True)
            with np.load(set_path, allow_pickle=True) as f:
                self.xml_list = f[sets]
            before_filter = sum([len(k) for k in self.xml_list])
            self.filter_list()
            after_filter = sum([len(k) for k in self.xml_list])
            with self.cache_file.open(mode='wb') as f:
                pickle.dump(self.xml_list, f)
            print('Filtered {} images to {}. Annotation saved.'.format(before_filter, after_filter))

    def filter_list(self):
        """
        Filter out 'truncated', 'occluded' and 'difficult' images following the practice of previous works.
        In addition, this dataset has uncleaned label (in person category). They are omitted as suggested by README.
        """
        for cls_id in range(len(self.classes)):
            to_del = []
            for xml_name in self.xml_list[cls_id]:
                xml_comps = xml_name.split('/')[-1].strip('.xml').split('_')
                ori_xml_name = '_'.join(xml_comps[:-1]) + '.xml'
                voc_idx = int(xml_comps[-1])
                xml_file = self.ori_anno_path / ori_xml_name
                assert xml_file.exists(), '{} does not exist.'.format(xml_file)
                tree = ET.parse(xml_file.open())
                root = tree.getroot()
                obj = root.findall('object')[voc_idx - 1]

                difficult = obj.find('difficult')
                if difficult is not None: difficult = int(difficult.text)
                occluded = obj.find('occluded')
                if occluded is not None: occluded = int(occluded.text)
                truncated = obj.find('truncated')
                if truncated is not None: truncated = int(truncated.text)
                if difficult or occluded or truncated:
                    to_del.append(xml_name)
                    continue

                # Exclude uncleaned images
                if self.classes[cls_id] == 'person' and int(xml_comps[0]) > 2008:
                    to_del.append(xml_name)
                    continue

                # Exclude overlapping images in Willow
                #if self.sets == 'train' and (self.classes[cls_id] == 'motorbike' or self.classes[cls_id] == 'car') \
                #        and int(xml_comps[0]) == 2007:
                #    to_del.append(xml_name)
                #    continue

            for x in to_del:
                self.xml_list[cls_id].remove(x)

    def get_pair(self, cls=None, shuffle=True, tgt_outlier=False, src_outlier=False):
        """
        Randomly get a pair of objects from VOC-Berkeley keypoints dataset
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
        for xml_name in random.sample(self.xml_list[cls], 2):
            anno_dict = self.__get_anno_dict(xml_name, cls)
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
        if not src_outlier:
            perm_mat = perm_mat[row_list, :]
            anno_pair[0]['keypoints'] = [anno_pair[0]['keypoints'][i] for i in row_list]
        if not tgt_outlier:
            perm_mat = perm_mat[:, col_list]
            anno_pair[1]['keypoints'] = [anno_pair[1]['keypoints'][j] for j in col_list]

        return anno_pair, perm_mat

    def get_multi(self, cls=None, num=2, shuffle=True, filter_outlier=True):
        """
        Randomly get multiple objects from VOC-Berkeley keypoints dataset for multi-matching.
        The first image is fetched with all appeared keypoints, and the rest images are fetched with only inliers.
        :param cls: None for random class, or specify for a certain set
        :param num: number of objects to be fetched
        :param shuffle: random shuffle the keypoints
        :param filter_outlier: filter out outlier keypoints among images
        :return: (list of data, list of permutation matrices)
        """
        assert filter_outlier == True, 'Multi-matching on PascalVOC dataset with unfiltered outliers is not supported'

        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)
        assert type(cls) == int and 0 <= cls < len(self.classes)

        anno_list = []
        for xml_name in random.sample(self.xml_list[cls], num):
            anno_dict = self.__get_anno_dict(xml_name, cls)
            if shuffle:
                random.shuffle(anno_dict['keypoints'])
            anno_list.append(anno_dict)

        perm_mat = [np.zeros([len(anno_list[0]['keypoints']), len(x['keypoints'])], dtype=np.float32) for x in anno_list]
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

    def get_single_to_ref(self, idx, cls, shuffle=True):
        """
        Get a single image, against a reference model containing all ground truth keypoints.
        :param idx: index in this class
        :param cls: specify for a certain class
        :param shuffle: random shuffle the keypoints
        :return: (data, groundtruth permutation matrix)
        """
        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)
        else:
            cls = cls
        assert type(cls) == int and 0 <= cls < len(self.classes)

        xml_name = self.xml_list[cls][idx]
        anno_dict = self.__get_anno_dict(xml_name, cls)
        if shuffle:
            random.shuffle(anno_dict['keypoints'])

        ref = self.__get_ref_model(cls)

        perm_mat = np.zeros((len(anno_dict['keypoints']), len(ref['keypoints'])), dtype=np.float32)
        for i, keypoint in enumerate(anno_dict['keypoints']):
            for j, _keypoint in enumerate(ref['keypoints']):
                if keypoint['name'] == _keypoint['name']:
                    perm_mat[i, j] = 1

        return anno_dict, perm_mat

    def __get_ref_model(self, cls):
        """
        Get a reference model for a certain class. The reference model contains all ground truth keypoints
        :param cls: specify a certain class (by integer ID)
        :return: annotation dict
        """
        anno_dict = dict()
        anno_dict['keypoints'] = [{'name': x} for x in KPT_NAMES[self.classes[cls]]]
        anno_dict['cls'] = self.classes[cls]
        return anno_dict

    def __get_anno_dict(self, xml_name, cls):
        """
        Get an annotation dict from xml file
        """
        xml_file = self.anno_path / xml_name
        assert xml_file.exists(), '{} does not exist.'.format(xml_file)

        tree = ET.parse(xml_file.open())
        root = tree.getroot()

        img_name = root.find('./image').text + '.jpg'
        img_file = self.img_path / img_name
        bounds = root.find('./visible_bounds').attrib
        h = float(bounds['height'])
        w = float(bounds['width'])
        xmin = float(bounds['xmin'])
        ymin = float(bounds['ymin'])
        with Image.open(str(img_file)) as img:
            ori_sizes = img.size
            obj = img.resize(self.obj_resize, resample=Image.BICUBIC, box=(xmin, ymin, xmin + w, ymin + h))

        keypoint_list = []
        for keypoint in root.findall('./keypoints/keypoint'):
            attr = keypoint.attrib
            attr['x'] = (float(attr['x']) - xmin) * self.obj_resize[0] / w
            attr['y'] = (float(attr['y']) - ymin) * self.obj_resize[1] / h
            keypoint_list.append(attr)

        anno_dict = dict()
        anno_dict['image'] = obj
        anno_dict['keypoints'] = keypoint_list
        anno_dict['bounds'] = xmin, ymin, w, h
        anno_dict['ori_sizes'] = ori_sizes
        anno_dict['cls'] = self.classes[cls]
        anno_dict['univ_size'] = len(KPT_NAMES[anno_dict['cls']])

        return anno_dict

    @property
    def length(self):
        l = 0
        for cls in self.classes:
            l += len(self.xml_list[self.classes.index(cls)])
        return l

    def length_of(self, cls):
        return len(self.xml_list[self.classes.index(cls)])


if __name__ == '__main__':
    dataset = PascalVOC('train', (256, 256))
    a = dataset.get_pair()
    pass
