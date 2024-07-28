import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss
from  torchvision.ops import sigmoid_focal_loss, generalized_box_iou_loss as giou_loss

import random
import numpy as np

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # for same result, set True and False; for faster training, set False and True
    torch.backends.cudnn.deterministic= False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

def sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)

class DMLoss(nn.Module):
    # TODO: modify this loss
    def __init__(self, type='l1'):
        type_list = {'l1': torch.nn.functional.l1_loss, 'smooth_l1': torch.nn.functional.smooth_l1_loss}
        self.crit = type_list[type]
        super(DMLoss, self).__init__()

    def interpolation(self, poly, time=10):
        ori_points_num = poly.size(1)
        poly_roll =torch.roll(poly, shifts=1, dims=1)
        poly_ = poly.unsqueeze(3).repeat(1, 1, 1, time)
        poly_roll = poly_roll.unsqueeze(3).repeat(1, 1, 1, time)
        step = torch.arange(0, time, dtype=torch.float32).cuda() / time
        poly_interpolation = poly_ * step + poly_roll * (1. - step)
        poly_interpolation = poly_interpolation.permute(0, 1, 3, 2).reshape(poly_interpolation.size(0), ori_points_num * time, 2)
        return poly_interpolation

    def compute_distance(self, pred_poly, gt_poly):
        pred_poly_expand = pred_poly.unsqueeze(1)
        gt_poly_expand = gt_poly.unsqueeze(2)
        gt_poly_expand = gt_poly_expand.expand(gt_poly_expand.size(0), gt_poly_expand.size(1),
                                               pred_poly_expand.size(2), gt_poly_expand.size(3))
        pred_poly_expand = pred_poly_expand.expand(pred_poly_expand.size(0), gt_poly_expand.size(1),
                                                   pred_poly_expand.size(2), pred_poly_expand.size(3))
        distance = torch.sum((pred_poly_expand - gt_poly_expand) ** 2, dim=3)
        return distance
    
    def lossPred2NearestGt(self, ini_pred_poly, pred_poly, gt_poly):
        gt_poly_interpolation = self.interpolation(gt_poly)
        distance_pred_gtInterpolation = self.compute_distance(ini_pred_poly, gt_poly_interpolation)
        index_gt = torch.min(distance_pred_gtInterpolation, dim=1)[1]
        index_0 = torch.arange(index_gt.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_gt.size(0), index_gt.size(1))
        loss_predto_nearestgt = self.crit(pred_poly,gt_poly_interpolation[index_0, index_gt, :])
        return loss_predto_nearestgt

    def lossGt2NearestPred(self, ini_pred_poly, pred_poly, gt_poly):
        # NOTE: why here no need to do interpolate?
        distance_pred_gt = self.compute_distance(ini_pred_poly, gt_poly)
        index_pred = torch.min(distance_pred_gt, dim=2)[1]
        index_0 = torch.arange(index_pred.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_pred.size(0), index_pred.size(1))
        loss_gtto_nearestpred = self.crit(pred_poly[index_0, index_pred, :], gt_poly,reduction='none')
        return loss_gtto_nearestpred

    def setloss(self, ini_pred_poly, pred_poly, gt_poly, keyPointsMask):
        keyPointsMask = keyPointsMask.unsqueeze(2).expand(keyPointsMask.size(0), keyPointsMask.size(1), 2)
        lossPred2NearestGt = self.lossPred2NearestGt(ini_pred_poly, pred_poly, gt_poly)
        lossGt2NearestPred = self.lossGt2NearestPred(ini_pred_poly, pred_poly, gt_poly)

        loss_set2set = torch.sum(lossGt2NearestPred * keyPointsMask) / (torch.sum(keyPointsMask) + 1) + lossPred2NearestGt
        return loss_set2set / 2.

    def forward(self, ini_pred_poly, pred_polys_, gt_polys, keyPointsMask):
        return self.setloss(ini_pred_poly, pred_polys_, gt_polys, keyPointsMask)

class DetLoss(nn.Module):
    def __init__(self, ltype='l1', ptype=2):
        """
            ltype: type of loss
            ptype: type of det points
                1 -> 128*2 pnts
                2 -> 2*2 pnts, only the top-left and bottom-right 
                3 -> 8*2 pnts
        """
        super(DetLoss, self).__init__()
        if ltype == 'l1':
            self.loss = torch.nn.functional.l1_loss
        elif ltype == 'smooth_l1':
            self.loss = torch.nn.functional.smooth_l1_loss
        else:
            raise ValueError('No consistent type of Loss!')
        
        if ptype in [1,2,3]:
            self.ptype = ptype
        else:
            raise ValueError('No consistent type of Points!')

    def extract_pnts(self, polys, ct):
        """
        input
            polys: img_gt_polys from ground truth
            ct: the mass center, from ground truth
        output
            polys: polygons
            box: xmin, ymin, xmax, ymax
            align: top, bottom, left, right
        """
        if self.ptype == 1:
            return polys
        else:
            box = torch.cat([torch.min(polys, dim=1, keepdim=True)[0], torch.max(polys, dim=1, keepdim=True)[0]], dim=1)
            if self.ptype == 2:
                return box
            else:
                # ptype = 3
                # use ct to find the alignment points
                return box
        
    def forward(self, pred_polys, gt_polys, ct_ind):
        """
            input:
                pred_polys: Num_polys*4*w*h: the offsets from center to top-left and bottom-right
                gt_polys: Num_polys*N_pnts*2: the offsets for all pnts
                ct_ind: Num_polys: the index of origin image size
        """
        # print("pred_polys:", pred_polys.shape)
        # print("gt_polys:", gt_polys.shape)
        # print("ct_ind:", ct_ind.shape)

        gt_polys = self.extract_pnts(gt_polys, ct_ind)
        
        return self.loss(pred_polys, gt_polys)

class FCOSLoss(nn.Module):
    def __init__(self, stride=10.):
        # classification loss: sigmoid focal loss
        # regression: 
        #   original: GIoU loss
        #   mine: DMLLoss (and GIoU) for polygon
        # ctrness loss:
        #   BCELoss
        super().__init__()
        self.cls_loss = sigmoid_focal_loss  # take 3 arguments: pred, gt, reduction='sum'
        self.stride = stride
    
    def convert_poly_to_box(self, poly):
        x1, y1 = poly.max(dim=-2).values.unbind(dim=-1)
        x0, y0 = poly.min(dim=-2).values.unbind(dim=-1)
        
        x1, y1 = x1.unsqueeze(-1), y1.unsqueeze(-1)
        x0, y0 = x0.unsqueeze(-1), y0.unsqueeze(-1)

        box = torch.cat([x0, y0, x1, y1], dim=-1).to(torch.float32)
        return box

    def box_encode(self, ref_box, proposals): # ref: anchors, proposals: gt_boxes
        # proposals: absolute position 
        # this method is to get relative distance from predicted center to ground truth bbox
        reference_boxes_ctr_x = 0.5 * (ref_box[..., 0] + ref_box[..., 2])
        reference_boxes_ctr_y = 0.5 * (ref_box[..., 1] + ref_box[..., 3])

        # get box regression transformation deltas
        target_l = reference_boxes_ctr_x - proposals[..., 0]
        target_t = reference_boxes_ctr_y - proposals[..., 1]
        target_r = proposals[..., 2] - reference_boxes_ctr_x
        target_b = proposals[..., 3] - reference_boxes_ctr_y

        targets = torch.stack((target_l, target_t, target_r, target_b), dim=-1)
        return targets

    def box_decode(self, poly, boxes):  # poly: poly, boxes: anchor
        rel_box = self.convert_poly_to_box(poly)
        rel_box = rel_box * self.stride
        boxes = boxes.to(poly.dtype)    # center points
        ctr_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
        ctr_y = 0.5 * (boxes[:, 1] + boxes[:, 3])
        
        pred_boxes1 = ctr_x + rel_box[:, 0]
        pred_boxes2 = ctr_y + rel_box[:, 1]
        pred_boxes3 = ctr_x + rel_box[:, 2]
        pred_boxes4 = ctr_y + rel_box[:, 3]
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=1)
        return pred_boxes

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        cls_logits = head_outputs["cls_logits"] # (N,HWA,1)
        poly_regression = head_outputs["poly_regression"]  # [N, HWA, 128, 2]
        poly_ctrness = head_outputs["poly_ctrness"]  # [N, HWA, 1]
        
        all_gt_classes_targets = []
        all_gt_boxes_targets = []
        # _, _, height, width = targets['ct_hm'].shape
        width = targets['meta']['ct_wh'][1][0]
        # TODO: decode ct_ind and ct_cls from targets
        for i in range(targets["ct_01"].shape[0]):
            ct_01 = targets["ct_01"][i]
            ct_cls = targets["ct_cls"][i, ct_01]
            ct_ind = targets["ct_ind"][i, ct_01]    # ind -> real index of heatmap
            img_gt_poly_per_image = targets['img_gt_polys'][i, ct_01]   # absolute position
            img_gt_box_per_image = self.convert_poly_to_box(img_gt_poly_per_image)
            matched_idxs_per_image = matched_idxs[i]
            if len(ct_cls) == 0:
                gt_classes_targets = ct_cls.new_zeros(len(matched_idxs_per_image),)
                # change this
                gt_box_targets = ct_cls.new_zeros(len(matched_idxs_per_image),4)
            else:
                gt_classes_targets = ct_cls[matched_idxs_per_image.clip(min=0)]
                gt_box_targets = img_gt_box_per_image[matched_idxs_per_image.clip(min=0)]
            
            # print(gt_box_targets.shape)
            gt_classes_targets[matched_idxs_per_image < 0] = -1 # background
            all_gt_classes_targets.append(gt_classes_targets)
            all_gt_boxes_targets.append(gt_box_targets)
        
        all_gt_boxes_targets, all_gt_classes_targets, anchors = (
            torch.stack(all_gt_boxes_targets),
            torch.stack(all_gt_classes_targets),
            torch.stack(anchors),
        )

        if anchors.shape[0] == 1:
            anchors = anchors.squeeze(0)

        # compute foregroud
        foregroud_mask = all_gt_classes_targets >= 0
        num_foreground = foregroud_mask.sum().item()

        # classification loss
        gt_classes_targets = torch.zeros_like(cls_logits)
        gt_classes_targets[foregroud_mask, all_gt_classes_targets[foregroud_mask]] = 1.0

        loss_cls = sigmoid_focal_loss(cls_logits, gt_classes_targets, reduction="sum")

        # TODO: regression loss..
        ## decode_single -> add wh with center
        batch, nums, _ = poly_regression.shape
        poly_regression = poly_regression.reshape(batch, nums, -1, 2)
        pred_boxes = [ self.box_decode(poly_reg, anchors) for poly_reg in poly_regression ] 

        # pred_boxes = [
        #     self.box_coder.decode_single(bbox_regression_per_image, anchors_per_image)
        #     for anchors_per_image, bbox_regression_per_image in zip(anchors, bbox_regression)
        # ]
        # amp issue: pred_boxes need to convert float


        loss_bbox_reg = giou_loss(
            torch.stack(pred_boxes)[foregroud_mask].float(),
            all_gt_boxes_targets[foregroud_mask],
            reduction="sum",
        )

        # ctrness loss, make sure if need to regularize
        bbox_reg_targets = self.box_encode(anchors, all_gt_boxes_targets)

        if len(bbox_reg_targets) == 0:
            gt_ctrness_targets = bbox_reg_targets.new_zeros(bbox_reg_targets.size()[:-1])
        else:
            left_right = bbox_reg_targets[:, :, [0, 2]]
            top_bottom = bbox_reg_targets[:, :, [1, 3]]
            gt_ctrness_targets = torch.sqrt(
                (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
                * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
            )

        pred_centerness = poly_ctrness.squeeze(dim=2)
        loss_bbox_ctrness = nn.functional.binary_cross_entropy_with_logits(
            pred_centerness[foregroud_mask], gt_ctrness_targets[foregroud_mask], reduction="sum"
        )

        return {
            "cls_loss": loss_cls / max(1, num_foreground),
            "bbox_regression": loss_bbox_reg / max(1, num_foreground),
            "ctrness": loss_bbox_ctrness / max(1, num_foreground),
        }
    
    def forward(self, targets, head_outputs, anchors, matched_idxs):
        loss = self.compute_loss(targets, head_outputs, anchors, matched_idxs)
        
        return loss