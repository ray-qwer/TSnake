import math
import numpy as np
import copy
import torch.utils.data as data
from pycocotools.coco import COCO
from .douglas import Douglas
from .utils import transform_polys, filter_tiny_polys, get_cw_polys, gaussian_radius, draw_umich_gaussian,\
uniformsample, four_idx, get_img_gt, img_poly_to_can_poly, augment, get_dist_center, transform_bbox, \
get_mask_from_poly

class Dataset(data.Dataset):
    def __init__(self, anno_file, data_root, split, cfg):
        super(Dataset, self).__init__()
        self.cfg = cfg
        
        ## added
        self.from_dist = cfg.train.from_dist
        self.mda_kpt = cfg.train.mda_kpt
        self.fcos = cfg.model.detect_type == "FCOS"
        self.get_origin_size = cfg.train.get_origin_size
        self.get_instance_mask = cfg.train.get_instance_mask
        ##########
        
        self.data_root = data_root
        self.split = split
        dataset_name = cfg.train.dataset
        if dataset_name.split('_')[0] != 'cityscapes':
            self.coco = COCO(anno_file)
            self.anns = np.array(sorted(self.coco.getImgIds()))
            self.anns = self.anns[:500] if split == 'mini' else self.anns
            self.json_category_id_to_continuous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}
        self.d = Douglas()

    def transform_original_data(self, instance_polys, flipped, width, trans_output, hw):
        output_h, output_w = hw
        instance_polys_ = []
        for instance in instance_polys:
            polys = [poly.reshape(-1, 2) for poly in instance]
            if flipped:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_
            polys = transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def transform_original_bbox(self, bboxes, flipped, width, trans_input, hw):
        # bbox: x0, y0, w, h
        input_h, input_w = hw
        new_bboxes = []
        for instance in bboxes:
            bbox = [b.reshape(-1, 2) for b in instance]
            for i in range(len(bbox)):
                bbox[i][1] = bbox[i][0] + bbox[i][1]
            if flipped:
                bbox_ = []
                for b in bbox:
                    b[:, 0] = width - np.array(b[:, 0]) - 1
                    b[0,0], b[1,0] = b[1,0], b[0,0]
                    bbox_.append(b.copy())
                bbox = bbox_
            bbox = transform_bbox(bbox, trans_input, input_h, input_w)
            new_bboxes.append(bbox)
        return new_bboxes

    def get_valid_polys(self, instance_polys, hw):
        output_h, output_w = hw
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            for poly in instance:
                poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
            polys = filter_tiny_polys(instance)
            polys = get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            instance_polys_.append(polys)
        return instance_polys_

    def prepare_detection(self, box, poly, ct_hm, cls_id, wh, ct_cls, ct_ind):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        ct_box_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        ct_box_center = np.round(ct_box_center).astype(np.int32)
        
        # TODO: other centers like mass center or skelenton center
        if self.from_dist:
            ct = np.round(get_dist_center(box, poly)).astype(np.int32)
        else:
            ct = ct_box_center.copy()
        ##########################################################
            
        h, w = y_max - y_min, x_max - x_min
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])

        x_min, y_min = ct_box_center[0] - w / 2, ct_box_center[1] - h / 2
        x_max, y_max = ct_box_center[0] + w / 2, ct_box_center[1] + h / 2
        decode_box = [x_min, y_min, x_max, y_max]

        return decode_box

    def prepare_evolution(self, poly, img_gt_polys, can_gt_polys, keyPointsMask):
        # TODO: add keypoints, add mda first
        img_gt_poly = uniformsample(poly, len(poly) * self.cfg.data.points_per_poly)
        
        ### multiple alignment ###
        # the provided ground truth doesnt reserve the alignment points
        idx = four_idx(img_gt_poly)
        ##########################
        
        ## here will get t keypoints
        img_gt_poly = get_img_gt(img_gt_poly, idx, t=self.cfg.data.points_per_poly)
        can_gt_poly = img_poly_to_can_poly(img_gt_poly)
        ## reserve the keypoints
        ## MultiAlignment points ids
        gt_align_idx = np.arange(4)* (self.cfg.data.points_per_poly // len(idx))
        key_mask = self.get_keypoints_mask(img_gt_poly, gt_align_idx if self.mda_kpt else [])
        
        keyPointsMask.append(key_mask)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)

    def get_keypoints_mask(self, img_gt_poly, reserve_points=[]):
        key_mask = self.d.sample(img_gt_poly)
        if len(reserve_points) > 0 :
            for i in reserve_points:
                key_mask[i] = 1
        return key_mask
    

    def __getitem__(self, index):
        data_input = {}

        ann = self.anns[index]
        anno, image_path, image_id = self.process_info(ann)
        img, instance_polys, cls_ids, bboxes = self.read_original_data(anno, image_path)
        width, height = img.shape[1], img.shape[0]
        # TODO: what is these parameters mean for?
        _, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            augment(
                img, self.split,
                self.cfg.data.data_rng, self.cfg.data.eig_val, self.cfg.data.eig_vec,
                self.cfg.data.mean, self.cfg.data.std, self.cfg.commen.down_ratio,
                self.cfg.data.input_h, self.cfg.data.input_w, self.cfg.data.scale_range,
                self.cfg.data.scale, self.cfg.test.test_rescale, self.cfg.data.test_scale
            )
        origin_size_instance_polys = copy.deepcopy(instance_polys)

        if self.get_origin_size:
            bboxes = self.transform_original_bbox(bboxes, flipped, width, trans_input, inp_out_hw[:2])
            origin_size_instance_polys = self.transform_original_data(origin_size_instance_polys, flipped, width, trans_input, inp_out_hw[:2])
            origin_size_instance_polys = self.get_valid_polys(origin_size_instance_polys, inp_out_hw[:2])
        
        instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_output, inp_out_hw[2:])
        instance_polys = self.get_valid_polys(instance_polys, inp_out_hw[2:])
        

        #detection
        output_h, output_w = inp_out_hw[2:]
        ct_hm = np.zeros([len(self.json_category_id_to_continuous_id), output_h, output_w], dtype=np.float32)
        ct_cls = []
        wh = []
        ct_ind = []

        #segmentation
        img_gt_polys = []
        keyPointsMask = []
        can_gt_polys = []
        bboxes_ = []
        if self.get_instance_mask:
            instance_masks = []

        for i in range(len(anno)):
            cls_id = cls_ids[i]
            instance_poly = instance_polys[i]
            for j in range(len(instance_poly)):
                poly = instance_poly[j]
                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])

                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue
                self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind)
                self.prepare_evolution(poly, img_gt_polys, can_gt_polys, keyPointsMask)
                
                if self.get_origin_size:
                    b = bboxes[i][j]
                    bboxes_.append(b.flatten())

                    if self.get_instance_mask:
                        poly = origin_size_instance_polys[i][j]
                        # poly = poly - b[0]
                        poly = np.round(poly)
                        # h, w = np.round(b[1] - b[0] + 1).astype(np.int)
                        h, w = inp_out_hw[:2]
                        mask = get_mask_from_poly(poly, w=w, h=h)
                        instance_masks.append(mask)
                else:
                    bboxes_.append(np.array(bbox))
                    if self.get_instance_mask:
                        poly = poly - np.array([x_min, y_min])
                        poly = np.round(poly)
                        h, w = round(h), round(w)
                        mask = get_mask_from_poly(poly, w=w, h=h)
                        instance_masks.append(mask)
                    
        
        data_input.update({'inp': inp})
        
        if self.fcos:
            detection = {'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind, "bbox":bboxes_}    # FCOS
        else:
            detection = {'ct_wh': ct_hm.shape[-2:], 'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind, "bbox":bboxes_}    # centernet

        evolution = {'img_gt_polys': img_gt_polys, 'can_gt_polys': can_gt_polys}
        if self.get_instance_mask:
            evolution.update({'mask': instance_masks})
        
        data_input.update(detection)
        data_input.update(evolution)
        data_input.update({'keypoints_mask': keyPointsMask})
        ct_num = len(ct_ind)
        if self.fcos:
            meta = {'ct_wh': ct_hm.shape[-2:], 'center': center, 'scale': scale, 'img_id': image_id, 'ann': ann, 'ct_num': ct_num}
        else:
            meta = {'center': center, 'scale': scale, 'img_id': image_id, 'ann': ann, 'ct_num': ct_num}

        data_input.update({'meta': meta})

        return data_input

    def __len__(self):
        return len(self.anns)

    