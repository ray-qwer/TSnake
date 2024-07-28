import os
import json
from ..sbd import utils
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import torch
import numpy as np

class Evaluator:
    def __init__(self, result_dir, anno_dir, combine_threshold=1):
        self.results = []
        self.img_ids = []
        self.aps = []
        self.combine_threshold = combine_threshold
        self.result_dir = result_dir    # save the result at data/result/
        os.system('mkdir -p {}'.format(self.result_dir))

        ann_file = anno_dir
        self.coco = coco.COCO(ann_file) # ground truth

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def evaluate(self, output, batch):
        detection = output['detection']
        score = detection[:, 2].detach().cpu().numpy()
        label = detection[:, 3].detach().cpu().numpy().astype(int)
        py = output['py'][-1].detach().cpu().numpy()
        combine = None
        if 'combine' in output:
            """
                convert combine to binary mask
            """
            # combine = torch.where(output['combine'].squeeze(0) > 0.6, True, False).detach().cpu().numpy()
            combine_pred = torch.where(output['combine'].squeeze(0) > self.combine_threshold, True, False)
            """
                check if symmetric
            """
            triu_combine = torch.triu(combine_pred)
            tril_T_combine = torch.tril(combine_pred).T
            combine = (triu_combine & tril_T_combine)
            # combine = np.where(combine > 0.6, True, False)
            # class-wise choosing
            # NUMBER_DICT = {0: 'car', 1: 'person', 2: 'rider', 3: 'motorcycle',4: 'bicycle', 5: 'truck', 6: 'bus', 7: 'train'}
            # dont combine scores under 0.15, class person, 
            combine_mask = torch.ones_like(combine_pred).to(bool)
            score_under_th = score < 0.10
            combine_mask [score_under_th, :] = 0
            combine_mask [:, score_under_th] = 0

            # class_mask = torch.zeros_like(combine_pred).to(bool)
            # for i in range(8):
            #     label_mask = label == i
            #     tmp1 = torch.zeros_like(combine_pred).to(bool)
            #     tmp2 = torch.zeros_like(combine_pred).to(bool)
            #     tmp1[label_mask, :] = True
            #     tmp2[:, label_mask] = True
            #     class_mask[tmp1 & tmp2] = True
            # #     # class_mask[label_mask][label_mask] = True
            # # # exception: car and truck, bus and train, person and rider
            # label_mask = (label == 0) | (label == 5)
            # tmp1 = torch.zeros_like(combine_pred).to(bool)
            # tmp2 = torch.zeros_like(combine_pred).to(bool)
            # tmp1[label_mask, :] = True
            # tmp2[:, label_mask] = True
            # class_mask[tmp1 & tmp2] = True

            # label_mask = (label == 6) | (label == 7)
            # tmp1 = torch.zeros_like(combine_pred).to(bool)
            # tmp2 = torch.zeros_like(combine_pred).to(bool)
            # tmp1[label_mask, :] = True
            # tmp2[:, label_mask] = True
            # class_mask[tmp1 & tmp2] = True


            # label_mask = (label == 1) | (label == 2)
            # tmp1 = torch.zeros_like(combine_pred).to(bool)
            # tmp2 = torch.zeros_like(combine_pred).to(bool)
            # tmp1[label_mask, :] = True
            # tmp2[:, label_mask] = True
            # class_mask[tmp1 & tmp2] = True
            combine = (combine & combine_mask)
            combine = combine.detach().cpu().numpy()

        if len(py) == 0:
            return

        img_id = int(batch['meta']['img_id'][0])
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy()

        h, w = batch['inp'].size(2), batch['inp'].size(3)
        trans_output_inv = utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img['height'], img['width']
        py = np.array([utils.affine_transform(py_, trans_output_inv) for py_ in py])
        rles = utils.coco_poly_to_rle(py, ori_h, ori_w, combine)

        coco_dets = []
        for i in range(len(rles)):
            detection = {
                'image_id': img_id,
                'category_id': self.contiguous_category_id_to_json_id[label[i]],
                'segmentation': rles[i],
                'score': float('{:.2f}'.format(score[i]))
            }
            coco_dets.append(detection)

        self.results.extend(coco_dets)
        self.img_ids.append(img_id)

    def summarize(self):
        json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
        coco_eval = COCOeval(self.coco, coco_dets, 'segm')
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.results = []
        self.img_ids = []
        self.aps.append(coco_eval.stats[0])
        return {'ap': coco_eval.stats[0]}


class DetectionEvaluator:
    def __init__(self, result_dir, anno_dir):
        self.results = []
        self.img_ids = []
        self.aps = []

        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

        ann_file = anno_dir
        self.coco = coco.COCO(ann_file)

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def evaluate(self, output, batch):
        detection = output['detection']
        detection = detection[0] if detection.dim() == 3 else detection
        # box = detection[:, :4].detach().cpu().numpy() * snake_config.down_ratio
        score = detection[:, 2].detach().cpu().numpy()
        label = detection[:, 3].detach().cpu().numpy().astype(int)

        # NOTE: original py is a list with 3 elements
        if isinstance(output['py'], list):
            py = output['py'][-1].detach()
        else:
            py = output['py'].detach()
            
        if len(py) == 0:
            return 
        box = torch.cat([torch.min(py, dim=1, keepdim=True)[0], torch.max(py, dim=1, keepdim=True)[0]], dim=1)
        box = box.cpu().numpy()

        img_id = int(batch['meta']['img_id'][0])
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy()

        if len(box) == 0:
            return
        # print("len of box", len(box))
        # print("box.shape", box.shape)

        h, w = batch['inp'].size(2), batch['inp'].size(3)
        trans_output_inv = utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img['height'], img['width']

        coco_dets = []
        for i in range(len(label)):
            box_ = utils.affine_transform(box[i].reshape(-1, 2), trans_output_inv).ravel()
            # box for coco: xmin, ymin, w, h
            box_[2] -= box_[0]
            box_[3] -= box_[1]
            box_ = list(map(lambda x: float('{:.2f}'.format(x)), box_))
            detection = {
                'image_id': img_id,
                'category_id': self.contiguous_category_id_to_json_id[label[i]],
                'bbox': box_,
                'score': float('{:.2f}'.format(score[i]))
            }
            coco_dets.append(detection)

        self.results.extend(coco_dets)
        self.img_ids.append(img_id)

    def summarize(self):
        json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
        coco_eval = COCOeval(self.coco, coco_dets, 'bbox')
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.results = []
        self.img_ids = []
        self.aps.append(coco_eval.stats[0])
        return {'ap': coco_eval.stats[0]}

