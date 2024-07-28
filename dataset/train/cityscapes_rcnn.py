# TODO: change utils to our settings
# TODO: keypoints extracting, combine with base.py

import math
import numpy as np
import random
import torch.utils.data as data
from pycocotools.coco import COCO
from .douglas import Douglas
import cv2
import glob
import json
import os

from .utils import transform_polys, filter_tiny_polys, get_cw_polys, gaussian_radius, draw_umich_gaussian,\
uniformsample, four_idx, get_img_gt, img_poly_to_can_poly, augment, get_dist_center, transform_bbox, \
    get_mask_from_poly, get_boundary_map, Mosaic, get_poly_area, get_extreme_points, get_quadrangle, \
    get_octagon, uniformsample_rcnn, get_ellipse
## Here are the utils from polysnake, check how to replace them
# from lib.utils.snake import snake_cityscapes_utils, visualize_utils
# from lib.utils.snake import snake_voc_utils
# from lib.utils import data_utils
import dataset.rcnn_snake_config as snake_config

################# from cityscapes.py #############################
#Globals ----------------------------------------------------------------------
COCO_LABELS = {24: 1,
               26: 2,
               27: 3,
               25: 4,
               33: 5,
               32: 6,
               28: 7,
               31: 8}

# Label number to name and color
INSTANCE_LABELS = {26: {'name': 'car', 'color': [0, 0, 142]},
                   24: {'name': 'person', 'color': [220, 20, 60]},
                   25: {'name': 'rider', 'color': [255, 0, 0]},
                   32: {'name': 'motorcycle', 'color': [0, 0, 230]},
                   33: {'name': 'bicycle', 'color': [119, 11, 32]},
                   27: {'name': 'truck', 'color': [0, 0, 70]},
                   28: {'name': 'bus', 'color': [0, 60, 100]},
                   31: {'name': 'train', 'color': [0, 80, 100]}}

# Label name to number
LABEL_DICT = {'car': 26, 'person': 24, 'rider': 25, 'motorcycle': 32,
              'bicycle': 33, 'truck': 27, 'bus': 28, 'train': 31}
# LABEL_DICT = {'bicycle': 33}

# Label name to contiguous number
JSON_DICT = dict(car=0, person=1, rider=2, motorcycle=3, bicycle=4, truck=5, bus=6, train=7)
# JSON_DICT = dict(bicycle=0)
# Contiguous number to name
NUMBER_DICT = {0: 'car', 1: 'person', 2: 'rider', 3: 'motorcycle',
               4: 'bicycle', 5: 'truck', 6: 'bus', 7: 'train'}
# NUMBER_DICT = {0:'bicycle'}
# Array of keys
KEYS = np.array([[26000, 26999], [24000, 24999], [25000, 25999],
                 [32000, 32999], [33000, 33999], [27000, 27999],
                 [28000, 28999], [31000, 31999]])

NUM_CLASS = {'person': 17914, 'rider': 1755, 'car': 26944, 'truck': 482,
             'bus': 379, 'train': 168, 'motorcycle': 735, 'bicycle': 3658}

# ------------------------------------------------------------------------------

def read_dataset(ann_files):
    if not isinstance(ann_files, tuple):
        ann_files = (ann_files,)

    ann_file = []
    for ann_file_dir in ann_files:
        ann_file += glob.glob(os.path.join(ann_file_dir, '*/*.json'))

    ann_filter = []
    for fname in ann_file:
        with open(fname, 'r') as f:
            ann = json.load(f)
            examples = []
            for instance in ann:
                instance_label = instance['label']
                if instance_label not in LABEL_DICT:
                    continue
                examples.append(instance)
            if len(examples) > 0:
                ann_filter.append(fname)
    return ann_filter
############################## end ####################################

def dict_value_to_np(d):
    for k,v in d.items():
        d[k] = np.array(v)
    return d

class Dataset(data.Dataset):
    def __init__(self, anno_file, data_root, split, cfg):
        super(Dataset, self).__init__()
        print("cityscapes two detectors dataset")
        ##############################################
        self.cfg = cfg
        self.mda_kpt = cfg.train.mda_kpt
        self.down_ratio = cfg.commen.down_ratio
        self.polygon_origin_size = cfg.train.polygon_origin_size
        self.data_root = data_root
        self.train_boundary_head = cfg.model.train_boundary_head
        self.split = split
        self.istrain = split == "train"
        self.d = Douglas(ratio=self.down_ratio) if self.polygon_origin_size else Douglas()
        self.object_in_center = True
        #############################################

        self.anns = np.array(read_dataset(anno_file)[:])
        self.anns = self.anns[:500] if split == 'mini' else self.anns
        self.json_category_id_to_continuous_id = JSON_DICT
        
    def process_info(self, fname):  # checked
        data_root = self.data_root
        with open(fname, 'r') as f:
            ann = json.load(f)
        examples = []
        for instance in ann:
            instance_label = instance['label']
            if instance_label not in LABEL_DICT:
                continue
            examples.append(instance)
        img_path = os.path.join(data_root, '/'.join(ann[0]['img_path'].split('/')[-3:]))
        img_id = ann[0]['image_id']
        return examples, img_path, img_id
    
    def read_original_data(self, anno, path):   # checked
        img = cv2.imread(path)

        ## object <- here is what we want
        instance_polys = [np.array(obj['components']) for obj in anno]
        ## component
        # instance_polys = [[np.array(comp['poly']) for comp in obj['components']] for obj in anno]
        cls_ids = [self.json_category_id_to_continuous_id[obj['label']] for obj in anno]
        return img, instance_polys, cls_ids

    def transform_original_data(self, instance_polys, flipped, width, trans_output, hw):
        output_h, output_w = hw
        instance_polys_ = []
        for instance in instance_polys:
            polys = [np.array(poly['poly']).reshape(-1, 2) for poly in instance]

            if flipped:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_

            polys = transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def get_valid_polys(self, instance_polys, hw):
        output_h, output_w = hw
        instance_polys_ = []
        ratio = self.down_ratio if self.polygon_origin_size else 1
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            for poly in instance:
                poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
            polys = filter_tiny_polys(instance, ratio)
            polys = get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            instance_polys_.append(polys)
        return instance_polys_

    def get_amodal_boxes(self, extreme_points):
        boxes = []
        for instance_points in extreme_points:
            if len(instance_points) == 0:
                box = []
            else:
                instance = np.concatenate(instance_points)
                box = np.concatenate([np.min(instance, axis=0), np.max(instance, axis=0)])
            boxes.append(box)
        return boxes

    def get_extreme_points(self, instance_polys):
        extreme_points = []
        for instance in instance_polys:
            points = [get_extreme_points(poly) for poly in instance]
            extreme_points.append(points)
        return extreme_points

    def prepare_adet(self, box, ct_hm, cls_id, wh, ct_ind):
        if len(box) == 0:
            return
        box = box / self.down_ratio if self.polygon_origin_size else box
        ct_hm = ct_hm[cls_id]

        x_min, y_min, x_max, y_max = box
        ct = np.round([(x_min + x_max) / 2, (y_min + y_max) / 2]).astype(np.int32)

        h, w = y_max - y_min, x_max - x_min
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct = np.clip(ct, a_min=0, a_max=199)
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])

    ## ROI from original bounding box
    def prepare_rcnn(self, abox, instance, cp_hm, cp_wh, cp_ind):
        ## abox: abs coord; instance: abs coord
        if len(abox) == 0:
            return
        abox = abox / self.down_ratio if self.polygon_origin_size else abox
        instance = [(ins / self.down_ratio) for ins in instance]  if self.polygon_origin_size else instance

        ## NOTE: assert the abox is not in order from min to max
        x_min, y_min, x_max, y_max = abox
        ct = np.round([(x_min + x_max) / 2, (y_min + y_max) / 2]).astype(np.int32)
        h, w = y_max - y_min, x_max - x_min
        abox = np.array([ct[0] - w/2, ct[1] - h/2, ct[0] + w/2, ct[1] + h/2])

        ## NOTE: snake config -> write in dataset/rcnn_snake_config.py
        hm = np.zeros([1, snake_config.cp_h, snake_config.cp_w], dtype=np.float32)
        abox_w, abox_h = abox[2] - abox[0], abox[3] - abox[1]
        cp_wh_ = []
        cp_ind_ = []
        ratio = [snake_config.cp_w, snake_config.cp_h] / np.array([abox_w, abox_h])

        decode_boxes = []

        for ex in instance:
            box = np.concatenate([np.min(ex, axis=0), np.max(ex, axis=0)])
            box_w, box_h = box[2] - box[0], box[3] - box[1]
            ## Weird, maybe change coordinate is better
            cp_wh_.append([box_w, box_h])
            ## NOTE: relative coordinate, phase2
            # cp_wh_.append([box_w * ratio, box_h * ratio])

            center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            shift = center - abox[:2]
            ro_center = shift / [abox_w, abox_h] * [snake_config.cp_w, snake_config.cp_h]
            ro_center = np.floor(ro_center).astype(np.int32)
            ro_center[0] = np.clip(ro_center[0], a_min=0, a_max=snake_config.cp_w-1)
            ro_center[1] = np.clip(ro_center[1], a_min=0, a_max=snake_config.cp_h-1)
            cp_ind_.append(ro_center[1] * hm.shape[2] + ro_center[0])

            ro_box_w, ro_box_h = [box_w, box_h] * ratio
            radius = gaussian_radius((math.ceil(ro_box_h), math.ceil(ro_box_w)))
            radius = max(0, int(radius))
            draw_umich_gaussian(hm[0], ro_center, radius)

            center = ro_center / [snake_config.cp_w, snake_config.cp_h] * [abox_w, abox_h] + abox[:2]
            x_min, y_min = center[0] - box_w / 2, center[1] - box_h / 2
            x_max, y_max = center[0] + box_w / 2, center[1] + box_h / 2
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            decode_boxes.append([x_min, y_min, x_max, y_max])

        cp_hm.append(hm)
        cp_wh.append(cp_wh_)
        cp_ind.append(cp_ind_)

        return decode_boxes

    def get_keypoints_mask(self, img_gt_poly, reserve_points=[]):
        key_mask = self.d.sample(img_gt_poly)
        if len(reserve_points) > 0 :
            for i in reserve_points:
                key_mask[i] = 1
        return key_mask

    def prepare_evolution(self, bbox, poly, extreme_point, img_init_polys, can_init_polys, img_gt_polys, can_gt_polys, keyPointsMask):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])
        
        if self.istrain:
            if random.random() > 0.33:
                aug_para = (np.random.rand(4)) * 3 - 1.5
                bbox_ = list(np.array(bbox) + aug_para)
                bbox_ = np.clip(bbox_, 0, 199)
                if bbox_[0] < bbox_[2] and bbox_[1] < bbox_[3]:
                    bbox = bbox_
                assert bbox[0] < bbox[2] and bbox[1] < bbox[3], f'bbox: {bbox}'
        ## NOTE: make octagon initial contour, like Deep Snake
        ### get quadrangle
        extreme_point = get_quadrangle(bbox)
        octagon = get_octagon(extreme_point)
        # octagon = get_cw_polys(octagon)
        img_init_poly = uniformsample_rcnn(octagon, self.cfg.data.points_per_poly)
        can_init_poly = img_poly_to_can_poly(img_init_poly)

        ## NOTE: ellipse initial contour, this could do multiple alignment
        # img_init_poly = get_ellipse(bbox, self.cfg.data.points_per_poly)
        # can_init_poly = img_poly_to_can_poly(img_init_poly)

        ## NOTE: ground truth poly, same to E2EC
        img_gt_poly = uniformsample(poly, len(poly) * self.cfg.data.points_per_poly)
        idx = four_idx(img_gt_poly)
        img_gt_poly = get_img_gt(img_gt_poly, idx, t=self.cfg.data.points_per_poly)
        # tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0]*self.down_ratio, 2).sum(axis=1))
        # img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)
        can_gt_poly = img_poly_to_can_poly(img_gt_poly)
        ## roll
        gt_align_idx = np.arange(4) * (self.cfg.data.points_per_poly // len(idx))
        key_mask = self.get_keypoints_mask(img_gt_poly, gt_align_idx if self.mda_kpt else [])

        # gt_align_idx = np.arange(4) * (self.cfg.data.points_per_poly // len(idx)) -tt_idx
        # gt_align_idx[gt_align_idx < 0] = gt_align_idx[gt_align_idx < 0] + self.cfg.data.points_per_poly
        # key_mask = self.get_keypoints_mask(img_gt_poly, gt_align_idx if self.mda_kpt else [])

        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)
        keyPointsMask.append(key_mask)

    def __getitem__(self, index):
        ann = self.anns[index]

        anno, path, img_id = self.process_info(ann)
        img, instance_polys, cls_ids = self.read_original_data(anno, path)
        height, width = img.shape[0], img.shape[1]

        _, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            augment(
                img, self.split,
                self.cfg.data.data_rng, self.cfg.data.eig_val, self.cfg.data.eig_vec,
                self.cfg.data.mean, self.cfg.data.std, self.cfg.commen.down_ratio,
                self.cfg.data.input_h, self.cfg.data.input_w, self.cfg.data.scale_range,
                self.cfg.data.scale, self.cfg.test.test_rescale, self.cfg.data.test_scale,
                poly_instance = instance_polys if self.object_in_center else None
            )

        if self.polygon_origin_size:
            instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_input, inp_out_hw[:2])
            instance_polys = self.get_valid_polys(instance_polys, inp_out_hw[:2])
            extreme_points = self.get_extreme_points(instance_polys)    # abs coord 
            boxes = self.get_amodal_boxes(extreme_points)
        else:
            instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_output, inp_out_hw[2:])
            instance_polys = self.get_valid_polys(instance_polys, inp_out_hw[2:])
            extreme_points = self.get_extreme_points(instance_polys)
            boxes = self.get_amodal_boxes(extreme_points)

        # detection
        output_h, output_w = inp_out_hw[2:]

        ## NOTE: first detector
        act_hm = np.zeros([8, output_h, output_w], dtype=np.float32)
        awh = []
        act_ind = []

        # component
        cp_hm = []
        cp_wh = []
        cp_ind = []

        # evolution
        i_it_pys = []
        c_it_pys = []
        i_gt_pys = []
        c_gt_pys = []
        keyPointsMask = []

        for i in range(len(anno)):
            cls_id = cls_ids[i]
            instance_poly = instance_polys[i]
            instance_points = extreme_points[i]
            ## prepare detection: objects
            self.prepare_adet(boxes[i], act_hm, cls_id, awh, act_ind)
            ## prepare detection: components
            decode_boxes = self.prepare_rcnn(boxes[i], instance_points, cp_hm, cp_wh, cp_ind)

            for j in range(len(instance_poly)):
                poly = instance_poly[j] # in origin size
                extreme_point = instance_points[j]
                
                poly_ = poly.copy()/self.down_ratio
                x_min, y_min = np.min(poly_[:, 0]), np.min(poly_[:, 1])
                x_max, y_max = np.max(poly_[:, 0]), np.max(poly_[:, 1])
                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue

                self.prepare_evolution(bbox, poly, extreme_point, i_it_pys, c_it_pys, i_gt_pys, c_gt_pys, keyPointsMask)

        if self.train_boundary_head:
            if self.polygon_origin_size:
                polys_draw = i_gt_pys.copy()
                down_ratio = self.down_ratio
            else:
                polys_draw = i_gt_pys.copy()
                down_ratio = 1

            bd_map = get_boundary_map(polys_draw, inp_out_hw[2:], down_ratio)
        # the meaning of the returned data
        # inp: image
        # act_hm: 'ct_hm' means the heatmap of the object center; 'a' means 'amodal', which includes the complete object
        # awh: 'wh' means the width and height of the object bounding box
        # act_ind: the index in an image, row * width + col
        # cp_hm: component heatmap
        # cp_ind: the index in an RoI
        # i_it_4py: initial 4-vertex polygon for extreme point prediction, 'i' means 'image', 'it' means 'initial'
        # c_it_4py: normalized initial 4-vertex polygon. 'c' means 'canonical', which indicates that the polygon coordinates are normalized.
        # i_gt_4py: ground-truth 4-vertex polygon.
        # i_it_py: initial n-vertex polygon for contour deformation.

        ret = {'inp': inp}
        adet = {'act_hm': act_hm, 'awh': awh, 'act_ind': act_ind}
        cp = {'cp_hm': cp_hm, 'cp_wh': cp_wh, 'cp_ind': cp_ind}
        evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys, 'keypoints_mask': keyPointsMask}
        
        if self.train_boundary_head:
            evolution.update({'bd_map': bd_map})
        # # to numpy
        # adet = dict_value_to_np(adet)
        # cp = dict_value_to_np(cp)
        # evolution = dict_value_to_np(evolution)

        ret.update(adet)
        ret.update(cp)
        ret.update(evolution)
        act_num = len(act_ind)
        ct_num = len(i_gt_pys)
        meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ann': ann, 'act_num': act_num, 'ct_num': ct_num}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.anns)

