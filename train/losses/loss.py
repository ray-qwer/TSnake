import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss
from  torchvision.ops import sigmoid_focal_loss, generalized_box_iou_loss as giou_loss
from torch.fft import fft, ifft
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

    pos_loss = torch.log(pred + 1e-6) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred + 1e-6) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class EqualLengthLoss(nn.Module):
    # to ensure length of each segment is almost the same
    # 
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.functional.smooth_l1_loss
    
    def forward(self, pred):
        pre_dis = torch.cat((pred[:,1:], pred[:,:1]), dim=1)
        pred_shape = pre_dis - pred
        pred_len = torch.sqrt(torch.sum(pred_shape ** 2, dim=2))
        pred_mean_len = torch.mean(pred_len, dim=-1, keepdim=True).detach().repeat(1, pred_len.size(1))
        return self.loss(pred_len, pred_mean_len)

class ShapeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.functional.smooth_l1_loss
    def forward(self, pred, target=None, target_shape=None):
        if target_shape is None:
            target_dis = torch.cat((target[:,1:], target[:,:1]), dim=1)
            target_shape = target_dis - target
        pre_dis = torch.cat((pred[:,1:], pred[:,:1]), dim=1)
        pred_shape = pre_dis - pred
        return self.loss(pred_shape, target_shape)

class FourierLoss(nn.Module):
    def __init__(self, num_points, num_points_fft, device='cuda'):
        super().__init__()
        self.loss = nn.L1Loss()
        self.mask = torch.zeros((1,num_points), dtype=bool, device=device)
        self.weight = torch.zeros((num_points_fft), dtype=torch.float, device=device)
        self.weight_coef = torch.zeros_like(self.weight)
        self.weight_const = 1.3
        self.total_epoch = 200
        if num_points_fft % 2 == 0:
            self.mask[:,:num_points_fft // 2] = 1
            self.mask[:,-(num_points_fft // 2):] = 1
            self.weight_coef = torch.cat((torch.arange(num_points_fft//2), torch.arange(num_points_fft//2-1, -1, -1)))/ (num_points_fft//2-1)
        else:
            self.mask[:,:num_points_fft//2 + 1] = 1
            self.mask[:,-(num_points_fft // 2):] = 1
            self.weight_coef = torch.cat((torch.arange(num_points_fft//2+1), torch.arange(num_points_fft//2, 0, -1)))/ (num_points_fft//2-1)
        self.weight_coef = self.weight_coef.to(device)
        self.total_weight = torch.sum(self.mask)
        self.epoch = -1
        self.updateWeight(0)

    def updateWeight(self, epoch):
        if self.epoch == epoch:
            return
        self.epoch = epoch
        weight_epoch = self.weight_const - (self.weight_const - 1) * epoch / self.total_epoch
        self.weight = self.weight_const - weight_epoch ** self.weight_coef
        self.weight = self.weight * self.total_weight / torch.sum(self.weight)
        self.weight = self.weight.reshape(1, -1, 1)
    def forward(self, pred_fft, gt_contour):
        # pred_fft: (N, pred_point)
        # gt_contour: (N, n_points, 2)
        N = gt_contour.shape[0]
        gt_contour = gt_contour[:,:,0] + gt_contour[:,:,1] * 1j
        gt_fft = fft(gt_contour)
        mask = self.mask.repeat(N, 1)
        gt_fft = gt_fft[mask].reshape(N, -1)
        gt_fft = torch.cat((gt_fft.real[:,:,None], gt_fft.imag[:,:,None]), dim=-1)
        return self.loss(pred_fft * self.weight, gt_fft*self.weight)

class AngularLoss(nn.Module):
    """
        to get the angle between two vectors or slopes of one vector?
    """
    def __init__(self, mode="angle"):
        super().__init__()
        assert mode in ["angle", "slope"]
        self.mode = mode
        if mode == "angle":
            self.loss = torch.nn.MSELoss(reduction='none')
        else:
            self.loss = torch.nn.MSELoss(reduction='none')
    def getLength(self, vec):
        return torch.sqrt(torch.sum(vec ** 2, dim=2))
    def getVec(self, poly):
        poly_dis = torch.cat((poly[:, 1:], poly[:, :1]), dim=1)
        return poly_dis - poly
    def cosVec(self, vec, vec_, vec_len, vec_len_):
        return torch.sum(vec * vec_, dim=2) / ((vec_len + 1e-6) * (vec_len_ + 1e-6))
    def sinVec(self, vec, vec_, vec_len, vec_len_):
        return (vec[...,0]*vec_[...,1] - vec[...,1]*vec_[...,0]) / ((vec_len + 1e-6) * (vec_len_ + 1e-6))
    def getSlope(self, vec):
        return torch.remainder(torch.atan2(vec[...,1], vec[...,0]) + torch.pi / 2, torch.pi)
    def forward_slope(self, pred, gt_vec):
        pred_vec = self.getVec(pred)
        pred_slope = self.getSlope(pred_vec)
        gt_slope = self.getSlope(gt_vec)
        # mse and take the minimun
        loss_a = self.loss(pred_slope, gt_slope)
        pi_ = torch.ones_like(pred_slope) * torch.pi
        loss_b = self.loss(torch.abs(pred_slope - gt_slope), pi_)
        loss_ = torch.min(loss_a, loss_b)
        # average all
        return torch.mean(loss_)

    def forward_angle(self, pred, gt_vec):
        pred_vec = self.getVec(pred)
        pred_len = self.getLength(pred_vec)
        gt_len = self.getLength(gt_vec)

        # calculate sin and cos
        pred_vec_ = torch.cat((pred_vec[:, 1:], pred_vec[:, :1]), dim=1)
        pred_len_ = torch.cat((pred_len[:, 1:], pred_len[:, :1]), dim=1)
        gt_vec_ = torch.cat((gt_vec[:, 1:], gt_vec[:, :1]), dim=1)
        gt_len_ = torch.cat((gt_len[:, 1:], gt_len[:, :1]), dim=1)
        
        # diff and max
        pred_cos = self.cosVec(pred_vec, pred_vec_, pred_len, pred_len_)
        gt_cos = self.cosVec(gt_vec, gt_vec_, gt_len, gt_len_)
        cos_loss = self.loss(pred_cos, gt_cos)
        pred_sin = self.sinVec(pred_vec, pred_vec_, pred_len, pred_len_)
        gt_sin = self.sinVec(gt_vec, gt_vec_, gt_len, gt_len_)
        sin_loss = self.loss(pred_sin, gt_sin)
        loss_ = torch.max(cos_loss, sin_loss)
        return torch.mean(loss_)
    def forward(self, pred, gt_vec):
        if self.mode == "angle":
            return self.forward_angle(pred, gt_vec)
        else:
            return self.forward_slope(pred, gt_vec)

class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, kernel_size=1, sigma=0.8):
        super().__init__()
        self.loss = torch.nn.functional.smooth_l1_loss
        # kernel
        assert isinstance(kernel_size, int) and kernel_size > 0
        self.width = kernel_size // 2
        dist = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float, device="cuda")
        self.kernel = torch.exp(-(dist[:] ** 2) / (2 * sigma ** 2) ).reshape(1,1,-1)
    def getWeight(self, kpt):
        kpt = torch.cat([kpt[..., -self.width:], kpt, kpt[..., :self.width]], dim=-1)
        kpt = kpt.unsqueeze(1)
        return F.conv1d(kpt, self.kernel).squeeze(1) + 1
    def forward(self, pred, target, kpt):
        weight = self.getWeight(kpt)
        assert weight.shape == kpt.shape, f"weight shape: {weight.shape}; weight shape: {kpt.shape}"
        weight = weight.unsqueeze(-1)
        pred = pred * weight
        target = target * weight
        return self.loss(pred, target)

class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self, do_sigmoid=False):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss
        self.do_sigmoid = do_sigmoid
    def forward(self, out, target):
        if self.do_sigmoid:
            out = torch.sigmoid(out)
        return self.neg_loss(out, target)

class DMLoss(nn.Module):
    # TODO: modify this loss
    def __init__(self, type='l1', kernel_size=1, sigma=0.8):
        type_list = {'l1': torch.nn.functional.l1_loss, 'smooth_l1': torch.nn.functional.smooth_l1_loss}
        self.crit = type_list[type]
        super(DMLoss, self).__init__()
        # weighted for Mask
        assert isinstance(kernel_size, int) and kernel_size > 0
        self.width = kernel_size // 2
        dist = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float, device="cuda")
        self.kernel = torch.exp(-(dist[:] ** 2) / (2 * sigma ** 2) ).reshape(1,1,-1)
        
    def getWeight(self, kpt):
        if self.width == 0:
            kpt = kpt
        else:
            kpt = torch.cat([kpt[..., -self.width:], kpt, kpt[..., :self.width]], dim=-1)
        kpt = kpt.unsqueeze(1)
        return F.conv1d(kpt, self.kernel).squeeze(1)
    
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
        distance_pred_gt = self.compute_distance(ini_pred_poly, gt_poly)
        index_pred = torch.min(distance_pred_gt, dim=2)[1]
        index_0 = torch.arange(index_pred.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_pred.size(0), index_pred.size(1))
        loss_gtto_nearestpred = self.crit(pred_poly[index_0, index_pred, :], gt_poly,reduction='none')
        return loss_gtto_nearestpred

    def setloss(self, ini_pred_poly, pred_poly, gt_poly, keyPointsMask):
        keyPointsMask = self.getWeight(keyPointsMask).clip(0., 1.)
        keyPointsMask = keyPointsMask.unsqueeze(2).expand(keyPointsMask.size(0), keyPointsMask.size(1), 2)
        lossPred2NearestGt = self.lossPred2NearestGt(ini_pred_poly, pred_poly, gt_poly)
        lossGt2NearestPred = self.lossGt2NearestPred(ini_pred_poly, pred_poly, gt_poly)
        loss_set2set = torch.sum(lossGt2NearestPred * keyPointsMask) / (torch.sum(keyPointsMask) + 1) + lossPred2NearestGt
        return loss_set2set / 2.

    def forward(self, ini_pred_poly, pred_polys_, gt_polys, keyPointsMask):
        return self.setloss(ini_pred_poly, pred_polys_, gt_polys, keyPointsMask)

class BMLoss(DMLoss):
    """
        function 'interpolation', 'getWeight', 'compute_distance' could reuse
    """
    def __init__(self, type='l1', dist_threshold=8, kernel_size=1, sigma=0.8):
        super().__init__(type=type, kernel_size=kernel_size, sigma=sigma)
        self.dist_threshold = dist_threshold
    
    def lossPred2NearestGt_Out_of_bd(self, pred_poly, gt_poly):
        # pred to nearest gt
        gt_poly_interpolation = self.interpolation(gt_poly)
        dist_pred_gtIntpoltation = self.compute_distance(pred_poly, gt_poly_interpolation)
        # filter out of boundary
        value_gt, index_gt = torch.min(dist_pred_gtIntpoltation, dim=1)
        out_of_bd_filter = value_gt > self.dist_threshold
        index_0 = torch.arange(index_gt.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_gt.size(0), index_gt.size(1))
        loss_predto_nearsetgt = self.crit(pred_poly, gt_poly_interpolation[index_0, index_gt, :], reduction='none')
        out_of_bd_filter = out_of_bd_filter.unsqueeze(2).expand(out_of_bd_filter.size(0), out_of_bd_filter.size(1), 2)
        loss_predto_nearsetgt_out_of_bd = torch.sum(loss_predto_nearsetgt * out_of_bd_filter) / (torch.sum(out_of_bd_filter) + 1)
        return loss_predto_nearsetgt_out_of_bd

    def lossGt2NearestPred(self, pred_poly, gt_poly, keyPointsMask):
        keyPointsMask = keyPointsMask.unsqueeze(2).expand(keyPointsMask.size(0), keyPointsMask.size(1), 2)
        dist_pred_gt = self.compute_distance(pred_poly, gt_poly)
        index_pred = torch.min(dist_pred_gt, dim=2)[1]
        index_0 = torch.arange(index_pred.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_pred.size(0), index_pred.size(1))
        loss_gtto_nearestpred = self.crit(pred_poly[index_0, index_pred, :], gt_poly,reduction='none')
        loss_gtto_nearestpred = torch.sum(loss_gtto_nearestpred * keyPointsMask) / (torch.sum(keyPointsMask) + 1)
        return loss_gtto_nearestpred

    def forward(self, pred_polys, gt_polys, keyPointsMask):
        # if the points are out of bound, add penalty?
        loss_p2ng_out_of_bd = self.lossPred2NearestGt_Out_of_bd(pred_polys, gt_polys)
        loss_gtnp = self.lossGt2NearestPred(pred_polys, gt_polys, keyPointsMask)
        return (loss_p2ng_out_of_bd + loss_gtnp) / 2
        
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
    def __init__(self, stride=10., is_pred_bbox=True, down_ratio=4., anchor_origin_size=False):
        # classification loss: sigmoid focal loss
        # regression: 
        #   original: GIoU loss
        #   mine: DMLLoss (and GIoU) for polygon
        # ctrness loss:
        #   BCELoss
        super().__init__()
        self.cls_loss = sigmoid_focal_loss  # take 3 arguments: pred, gt, reduction='sum'
        self.stride = stride
        self.is_pred_bbox = is_pred_bbox
        self.anchor_origin_size = anchor_origin_size
        self.down_ratio = 1 if anchor_origin_size else down_ratio
    
    def convert_poly_to_box(self, poly):
        x1, y1 = poly.max(dim=-2).values.unbind(dim=-1)
        x0, y0 = poly.min(dim=-2).values.unbind(dim=-1)
        
        x1, y1 = x1.unsqueeze(-1), y1.unsqueeze(-1)
        x0, y0 = x0.unsqueeze(-1), y0.unsqueeze(-1)

        box = torch.cat([x0, y0, x1, y1], dim=-1).to(torch.float32)
        return box

    def box_encode(self, ref_box, proposals): # ref: anchors, proposals: gt_boxes
        proposals = proposals / self.down_ratio
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
        is_bbox = False
        if poly.size(-2) == 2:
            rel_box = poly.reshape(-1, 4)
            is_bbox = True
        else:
            rel_box = self.convert_poly_to_box(poly)
        rel_box = rel_box * self.stride
        boxes = boxes.to(poly.dtype)    # center points
        ctr_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
        ctr_y = 0.5 * (boxes[:, 1] + boxes[:, 3])
        
        pred_boxes1 = ctr_x - rel_box[:, 0] if is_bbox else ctr_x + rel_box[:, 0]
        pred_boxes2 = ctr_y - rel_box[:, 1] if is_bbox else ctr_y + rel_box[:, 1]
        pred_boxes3 = ctr_x + rel_box[:, 2]
        pred_boxes4 = ctr_y + rel_box[:, 3]
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=1)
        pred_boxes = pred_boxes * self.down_ratio
        return pred_boxes

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # targets: img_gt_polys -> absolute position
        cls_logits = head_outputs["cls_logits"] # (N,HWA,1)
        poly_ctrness = head_outputs["poly_ctrness"]  # [N, HWA, 1]
        poly_regression = head_outputs["poly_regression"]  # [N, HWA, 128, 2]
        
        all_gt_classes_targets = []
        all_gt_boxes_targets = []
        # _, _, height, width = targets['ct_hm'].shape
        width = targets['meta']['ct_wh'][1][0]
        # TODO: decode ct_ind and ct_cls from targets
        for i in range(targets["ct_01"].shape[0]):
            ct_01 = targets["ct_01"][i]
            ct_cls = targets["ct_cls"][i, ct_01]
            ct_ind = targets["ct_ind"][i, ct_01]    # ind -> real index of heatmap, not used here
            img_gt_poly_per_image = targets['img_gt_polys'][i, ct_01]   # absolute position
            img_gt_box_per_image = self.convert_poly_to_box(img_gt_poly_per_image)  # absolute position, (800*800)
            matched_idxs_per_image = matched_idxs[i]
            if len(ct_cls) == 0:
                gt_classes_targets = ct_cls.new_zeros(len(matched_idxs_per_image),)
                # change this
                gt_box_targets = ct_cls.new_zeros(len(matched_idxs_per_image),4)
            else:
                gt_classes_targets = ct_cls[matched_idxs_per_image.clip(min=0)]
                gt_box_targets = img_gt_box_per_image[matched_idxs_per_image.clip(min=0)]
            
            # print(gt_box_targets.shape)
            gt_classes_targets[matched_idxs_per_image < 0] = -1 # background # (200,200) for my case
            all_gt_classes_targets.append(gt_classes_targets)
            all_gt_boxes_targets.append(gt_box_targets)
        
        all_gt_boxes_targets, all_gt_classes_targets, anchors = (
            torch.stack(all_gt_boxes_targets),      # 800*800
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
        pred_boxes = [ self.box_decode(poly_reg, anchors) for poly_reg in poly_regression ]     # multiply down ratio when decode

        # pred_boxes = [
        #     self.box_coder.decode_single(bbox_regression_per_image, anchors_per_image)
        #     for anchors_per_image, bbox_regression_per_image in zip(anchors, bbox_regression)
        # ]
        # amp issue: pred_boxes need to convert float
        pred_boxes = torch.stack(pred_boxes)[foregroud_mask].float()
        loss_bbox_reg = giou_loss(
            pred_boxes,
            all_gt_boxes_targets[foregroud_mask],
            reduction="sum",
        )
        # ctrness loss, make sure if need to regularize
        bbox_reg_targets = self.box_encode(anchors, all_gt_boxes_targets)   # 800*800

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

class CBLoss(nn.Module):
    def __init__(self, loss_type="focal"):
        super().__init__()
        if loss_type == "focal":
            self.cb_loss = FocalLoss()
        elif loss_type == "bce":
            self.cb_loss = torch.nn.BCELoss()
    
    def prepare_target(self, comp_target, instance_num):
        """
            the same components are in the same index
            -1 means there is no components, like 0 in ct_01
            input:
                comp_target:
                    shape: (b, max_len)
            output:
                shape: (instance, max_len)
        """
        batch_size, max_len = comp_target.shape
        comp_matrix = torch.zeros(instance_num, max_len, dtype=torch.float32, device=comp_target.device)
        idx, row = 0, 0
        while idx < batch_size:
            target = comp_target[idx]
            for tgt in target:
                if tgt == -1:
                    break
                comp_matrix[row] = (target == tgt)
                row += 1
            idx += 1
        return comp_matrix
    
    def forward(self, combine_pred, multicomp_target, ct_01):
        instance_num = torch.sum(ct_01)

        multicomp_matrix = self.prepare_target(multicomp_target, instance_num)
        if instance_num == 0:
            combine_loss = torch.sum(combine_pred) * 0.0
        else:
            combine_loss = self.cb_loss(combine_pred, multicomp_matrix)
        
        return combine_loss

# ref: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice