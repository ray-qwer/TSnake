## modified: prepare_detection-> the center is the farthest point
import math
import numpy as np
import random
import torch.utils.data as data
from pycocotools.coco import COCO
from .douglas import Douglas
from .utils import transform_polys, filter_tiny_polys, get_cw_polys, gaussian_radius, draw_umich_gaussian,\
uniformsample, four_idx, get_img_gt, img_poly_to_can_poly, augment, get_dist_center, transform_bbox, \
    get_mask_from_poly, get_boundary_map, Mosaic, get_poly_area

class Dataset(data.Dataset):
    def __init__(self, anno_file, data_root, split, cfg):
        super(Dataset, self).__init__()
        self.cfg = cfg
        ## added
        self.from_dist = cfg.train.from_dist
        self.mda_kpt = cfg.train.mda_kpt
        self.fcos = "FCOS" in cfg.model.detect_type
        self.down_ratio = cfg.commen.down_ratio
        self.polygon_origin_size = cfg.train.polygon_origin_size
        self.get_instance_mask = cfg.train.get_instance_mask
        self.train_boundary_head = cfg.train.train_boundary_head
        ##########
        self.data_root = data_root
        self.split = split
        self.mosaic_aug = cfg.train.mosaic_aug
        self.mosaic = Mosaic(cfg.data.scale, crop_img=cfg.train.dataset.split('_')[0] == 'cityscapes')
        self.max_mosaic_ratio = 0.8
        self.min_mosaic_ratio = 0.4
        self.mosaic_ratio = self.max_mosaic_ratio
        self.mosaic_end_epoch = 100.
        self.epoch = 0
        dataset_name = cfg.train.dataset
        self.object_in_center = True
        if dataset_name.split('_')[0] != 'cityscapes':
            self.coco = COCO(anno_file)
            self.anns = np.array(sorted(self.coco.getImgIds()))
            print("before", len(self.anns))
            self.anns = np.array([ann for ann in self.anns if len(self.coco.getAnnIds(imgIds=ann, iscrowd=0))])
            print("after", len(self.anns))
            self.anns = self.anns[:500] if split == 'mini' else self.anns
            self.json_category_id_to_continuous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}
            self.sample_list = range(len(self.anns))
            self.object_in_center = False
        self.d = Douglas(ratio=self.down_ratio) if self.polygon_origin_size else Douglas()
        self.max_training_instance = -1 # negative number for all accept, positive to set the max polygons to train, random choosing
    def update_epoch(self, epoch):
        if epoch > self.mosaic_end_epoch:
            self.mosaic_ratio = self.min_mosaic_ratio
        elif epoch > 175:
            self.mosaic_ratio = 1   # directly set to 0
        else:
            self.mosaic_ratio = (1 - (float(epoch) / self.mosaic_end_epoch)) * (self.max_mosaic_ratio - self.min_mosaic_ratio) + self.min_mosaic_ratio
    def transform_original_data(self, instance_polys, flipped, width, trans_output, hw):
        # output_h, output_w = inp_out_hw[2:]
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
        # bbox is seperated, so does instance polys
        img, instance_polys, cls_ids = self.read_original_data(anno, image_path)
        width, height = img.shape[1], img.shape[0]
        is_mosaic = False
        if self.mosaic_aug and random.random() > self.mosaic_ratio: # what i want is "<", but ">" would get better performance?
            is_mosaic = True
            ################ mosaic ########################### 
            ## choose other 3 images
            indice = random.sample(self.sample_list, 3)
            imgs = [img]
            poly_list = [instance_polys]
            for ind in indice:
                _ann = self.anns[ind]
                _anno, _img_path, _ = self.process_info(_ann)
                _img, _ins_poly, _cls_ids = self.read_original_data(_anno, _img_path)
                imgs.append(_img)
                poly_list.append(_ins_poly)
                cls_ids += _cls_ids
            
            img, instance_polys = self.mosaic(imgs, poly_list)
            width, height = img.shape[1], img.shape[0]
            #################################################

        # TODO: what is these parameters mean for?
        _, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            augment(
                img, "test",
                self.cfg.data.data_rng, self.cfg.data.eig_val, self.cfg.data.eig_vec,
                self.cfg.data.mean, self.cfg.data.std, self.cfg.commen.down_ratio,
                self.cfg.data.input_h, self.cfg.data.input_w, self.cfg.data.scale_range,
                self.cfg.data.scale, self.cfg.test.test_rescale, self.cfg.data.test_scale,
                is_mosaic=is_mosaic, poly_instance = instance_polys if self.object_in_center else None
            )

        # bboxes = self.transform_original_bbox(bboxes, flipped, width, trans_input, inp_out_hw[:2])
        if self.polygon_origin_size:
            instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_input, inp_out_hw[:2])
            instance_polys = self.get_valid_polys(instance_polys, inp_out_hw[:2])
        else:
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
        # bboxes_ = []
        if self.get_instance_mask:
            instance_masks = []

        multi_comp = []
        components = 0
        for i in range(len(instance_polys)):
            cls_id = cls_ids[i]
            instance_poly = instance_polys[i]
            # bbox_multicomp = [1e6,1e6,0,0]
            for j in range(len(instance_poly)):
                poly = instance_poly[j] / self.down_ratio if self.polygon_origin_size else instance_poly[j]
                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue
                self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind)
                # prepare_evolution will append poly's coordiates into list, thus use the polygon that would be used
                self.prepare_evolution(instance_poly[j], img_gt_polys, can_gt_polys, keyPointsMask)
                # for combine
                # b = bboxes[i][j].flatten()
                # bbox_multicomp[0] = min(bbox_multicomp[0], b[0])
                # bbox_multicomp[1] = min(bbox_multicomp[1], b[1])
                # bbox_multicomp[2] = max(bbox_multicomp[2], b[2])
                # bbox_multicomp[3] = max(bbox_multicomp[3], b[3])

                if self.get_instance_mask:
                    poly = instance_polys[i][j]
                    poly = np.round(poly)
                    h, w = inp_out_hw[:2]
                    mask = get_mask_from_poly(poly, w=w, h=h)
                    instance_masks.append(mask)
                
                # multi-component combination
                multi_comp.append(components)
            if len(instance_poly) > 0:
                components += 1
                # bboxes_ += ([bbox_multicomp] * len(instance_poly))

        # boundary gt
        if self.train_boundary_head:
            if self.polygon_origin_size:
                polys_draw = img_gt_polys.copy()
                down_ratio = self.down_ratio
            else:
                polys_draw = img_gt_polys.copy()
                down_ratio = 1

            bd_map = get_boundary_map(polys_draw, inp_out_hw[2:], down_ratio)
        data_input.update({'inp': inp})
        
        wh = np.array(wh)
        ct_cls = np.array(ct_cls)
        ct_ind = np.array(ct_ind)
        img_gt_polys = np.array(img_gt_polys)
        can_gt_polys = np.array(can_gt_polys)
        keyPointsMask = np.array(keyPointsMask)
        ###### not to train that much polygons at ones #########
        if self.max_training_instance > 0 and len(ct_ind) > self.max_training_instance:
            # sort from large to little
            # p_area = get_poly_area(img_gt_polys)
            # args_poly_area = np.argsort(p_area)
            shuffle_list = np.arange(len(wh))
            np.random.shuffle(shuffle_list)
            wh = wh[shuffle_list[:self.max_training_instance]]
            ct_cls = ct_cls[shuffle_list[:self.max_training_instance]]
            ct_ind = ct_ind[shuffle_list[:self.max_training_instance]]
            img_gt_polys = img_gt_polys[shuffle_list[:self.max_training_instance]]
            can_gt_polys = can_gt_polys[shuffle_list[:self.max_training_instance]]
            keyPointsMask = keyPointsMask[shuffle_list[:self.max_training_instance]]
        ########################################################

        if self.fcos:
            # detection = {'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind, 'bbox':bboxes_}    # FCOS
            detection = {'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind}    # FCOS
        else:
            # detection = {'ct_wh': ct_hm.shape[-2:], 'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind, 'bbox': bboxes_}    # centernet
            detection = {'ct_wh': ct_hm.shape[-2:], 'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind}    # centernet
    
        evolution = {'img_gt_polys': img_gt_polys, 'can_gt_polys': can_gt_polys}
        if self.get_instance_mask:
            evolution.update({'mask': instance_masks})
        if self.train_boundary_head:
            evolution.update({'bd_map': bd_map})

        # multi_comp_comb = {'multicomp': multi_comp}
        
        data_input.update(detection)
        data_input.update(evolution)
        # data_input.update(multi_comp_comb)
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
