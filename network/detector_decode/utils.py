import torch.nn as nn
import torch
from torch.fft import fft, ifft
import torchvision.models.detection._utils as det_utils
from torchvision.ops import boxes as box_ops, nms as tv_nms
from dataset import rcnn_snake_config

# TODO: rcnn_config 

def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat

def transpose_and_gather_feat_from_coord(feat, coord, down_sample=1.):
    # coord need to down size to feat size
    coord = (coord / down_sample).long()
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat[:, coord[:,1], coord[:,0]]
    return feat

def topk(scores, K=100):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def decode_ct_hm(ct_hm, wh, reg=None, K=100, stride=10., get_ifft=False, num_point=128):
    batch, cat, height, width = ct_hm.size()
    ct_hm = nms(ct_hm)
    scores, inds, clses, ys, xs = topk(ct_hm, K=K)
    wh = transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch * K, -1, 2)
    if get_ifft:
        wh = pred_ifft(wh, num_point)
    wh = wh.view(batch, K, -1, 2)

    if reg is not None:
        reg = transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1)
        ys = ys.view(batch, K, 1)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    ct = torch.cat([xs, ys], dim=2)
    poly = ct.unsqueeze(2).expand(batch, K, wh.size(2), 2) + wh * stride
    detection = torch.cat([ct, scores, clses], dim=2)
    return poly, detection

def decode_rcnn_ct_hm(ct_hm, wh, reg=None, K=100):
    batch, cat, height, width = ct_hm.size()
    ct_hm = nms(ct_hm)

    scores, inds, clses, ys, xs = topk(ct_hm, K=K)
    wh = transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)

    if reg is not None:
        reg = transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1)
        ys = ys.view(batch, K, 1)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    ct = torch.cat([xs, ys], dim=2)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detection = torch.cat([bboxes, scores, clses], dim=2)

    return ct, detection

def decode_FCOS_ctr(output, anchors, img_num, min_ct_score=0.2, K=100, stride=10.):
    ## threshold
    # min_ct_score = 0.1
    nms_thresh = 0.6
    cls_pred, ctrness_pred, wh_pred = output["cls_logits"], output["poly_ctrness"], output["wh"]
    
    detections = []

    for i in range(img_num):
        wh_per_image = [wh[i] for wh in wh_pred]
        logits_per_image = [cp[i] for cp in cls_pred]
        ctrness_per_image = [ctp[i] for ctp in ctrness_pred]
        anchors_per_image = anchors[0][0]

        image_poly = []
        image_scores = []
        image_labels = []
        image_centers = []
        # per level
        for wh_per_level, logits_per_level, box_ctrness_per_level in zip(
            wh_per_image, logits_per_image, ctrness_per_image
        ):

            num_classes = logits_per_level.shape[-1]

            # remove low scoring boxes
            scores_per_level = torch.sqrt(
                torch.sigmoid(logits_per_level) * torch.sigmoid(box_ctrness_per_level)
            ).flatten()
            keep_idxs = scores_per_level > min_ct_score
            scores_per_level = scores_per_level[keep_idxs]
            topk_idxs = torch.where(keep_idxs)[0]
        
            # keep only topk scoring predictions
            num_topk = min(500, topk_idxs.size(0))
            scores_per_level, idxs = scores_per_level.topk(num_topk)
            topk_idxs = topk_idxs[idxs]

            anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
            labels_per_level = topk_idxs % num_classes

            wh_per_level = wh_per_level[anchor_idxs]
            anchor_per_level = anchors_per_image[anchor_idxs]
            ctr_x = (anchor_per_level[:, 0] + anchor_per_level[:, 2]) * 0.5
            ctr_y = (anchor_per_level[:, 1] + anchor_per_level[:, 3]) * 0.5
            center_per_level = torch.stack((ctr_x, ctr_y), dim=1)

            # wh shape: (nums, 256) -> (nums, 128, 2)
            # print(wh_per_level.shape)
            nums, _ = wh_per_level.shape
            wh_per_level = wh_per_level.reshape(nums, -1, 2)
            poly_per_level = center_per_level.unsqueeze(1).expand(nums, wh_per_level.size(1), 2) + wh_per_level * stride

            image_poly.append(poly_per_level)
            image_scores.append(scores_per_level)
            image_labels.append(labels_per_level)
            image_centers.append(center_per_level)

 

        image_poly = torch.cat(image_poly, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)
        image_centers = torch.cat(image_centers, dim=0)
        
        wh_xy = image_poly.reshape(image_poly.size(0), -1, 2)
        xymin = torch.min(wh_xy, dim=-2).values
        xymax = torch.max(wh_xy, dim=-2).values
        # print(xymin.shape)
        image_boxes = torch.cat((xymin, xymax), dim=-1)

        # non-maximum suppression
        keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, nms_thresh)
        keep = keep[: min(K, len(keep))]        

        detections.append(
            {
                "poly": image_poly[keep],
                "scores": image_scores[keep],
                "labels": image_labels[keep],
                "ct": image_centers[keep],
            }
        )

    return detections

def decode_FCOS_Det_ctr(output, anchors, img_num, min_ct_score=0.2, K=100, stride=10., down_sample=4):
    ## threshold
    # min_ct_score = 0.1
    nms_thresh = 0.7
    cls_pred, ctrness_pred, bbox_pred = output["cls_logits"], output["poly_ctrness"], output["bbox"]
    predict_out = False
    if 'wh' in output:
        wh_pred = output["wh"]
        predict_out = True
    detections = []

    for i in range(img_num):
        bbox_per_image = [b[i] for b in bbox_pred]
        logits_per_image = [cp[i] for cp in cls_pred]
        ctrness_per_image = [ctp[i] for ctp in ctrness_pred]
        # deal with anchors
        anchors_no_iter_flag = False        # if this flag is True, then there must no FPN, and only one layer, so no need to iterative
        if len(anchors) == img_num:
            anchors_per_image = anchors[i]
        else:
            anchors_per_image = anchors[0]
            if len(anchors_per_image) != len(bbox_per_image[0]):
                anchors_per_image = anchors[0][0]
                anchors_no_iter_flag = True
        image_bbox = []
        image_scores = []
        image_labels = []
        image_centers = []
        # per level
        for lvl_idx, (bbox_per_level, logits_per_level, box_ctrness_per_level) in enumerate(zip(
            bbox_per_image, logits_per_image, ctrness_per_image
        )):
            if anchors_no_iter_flag:
                anchors_per_level = anchors_per_image
            else:
                anchors_per_level = anchors_per_image[lvl_idx] 
            num_classes = logits_per_level.shape[-1]
            # remove low scoring boxes
            scores_per_level = torch.sqrt(
                torch.sigmoid(logits_per_level) * torch.sigmoid(box_ctrness_per_level)
            ).flatten()
            keep_idxs = scores_per_level > min_ct_score
            scores_per_level = scores_per_level[keep_idxs]
            topk_idxs = torch.where(keep_idxs)[0]
        
            # keep only topk scoring predictions
            num_topk = min(2000, topk_idxs.size(0))
            # num_topk = 0
            scores_per_level, idxs = scores_per_level.topk(num_topk)
            # idxs = torch.tensor([])
            topk_idxs = topk_idxs[idxs]
            anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
            labels_per_level = (topk_idxs % num_classes)

            bbox_per_level = bbox_per_level[anchor_idxs]
            anchor_per_level = anchors_per_level[anchor_idxs]
            ctr_x = (anchor_per_level[:, 0] + anchor_per_level[:, 2]) * 0.5
            ctr_y = (anchor_per_level[:, 1] + anchor_per_level[:, 3]) * 0.5
            center_per_level = torch.stack((ctr_x, ctr_y), dim=1)

            # wh shape: (nums, 256) -> (nums, 128, 2)
            # print(wh_per_level.shape)
            
            nums, _ = bbox_per_level.shape
            # TODO: four coordiantes are all positive nums
            bbox_per_level = torch.cat([center_per_level[:, 0, None] - bbox_per_level[:,0, None], center_per_level[:, 1, None] - bbox_per_level[:,1, None], 
                                        center_per_level[:, 0, None] + bbox_per_level[:,2, None], center_per_level[:, 1, None] + bbox_per_level[:,3, None]], dim=1)
            bbox_per_level = bbox_per_level.reshape(nums, 2, 2)
            
            # bbox_per_level = bbox_per_level.reshape(nums, 2, 2)
            # bbox_per_level = center_per_level.unsqueeze(1).expand(nums, bbox_per_level.size(1), 2) + bbox_per_level * stride # here the stride is 1
            # bbox_per_level = bbox_per_level[:, [1, 0]]

            image_bbox.append(bbox_per_level)
            image_scores.append(scores_per_level)
            image_labels.append(labels_per_level)
            image_centers.append(center_per_level)
 
        image_bbox = torch.cat(image_bbox, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)
        image_centers = torch.cat(image_centers, dim=0)
        
        # image_boxes = torch.cat((xymin, xymax), dim=-1)
        image_boxes = image_bbox.reshape(image_bbox.size(0), 4) # (nums, 4)
        # non-maximum suppression with classes
        keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, nms_thresh)
        # keep = box_ops.nms(image_boxes, image_scores, nms_thresh)
        keep = keep[: min(K, len(keep))]

        image_centers = image_centers[keep]
        image_bbox = image_bbox[keep]   # this is in shape (N, 2, 2)
        image_boxes = image_boxes[keep] # this is in shape (N, 4)
        image_scores = image_scores[keep]
        image_labels = image_labels[keep]
        ## below changes make it worse
        # non-maximum suppression with different class
        # keep_diff_cls = box_ops.nms(image_boxes, image_scores, 0.9) # discard the highly overlapping bbox, even different classes
        # image_bbox = image_bbox[keep_diff_cls]
        # image_boxes = image_boxes[keep_diff_cls]
        # image_scores = image_scores[keep_diff_cls]
        # image_labels = image_labels[keep_diff_cls]
        # image_centers = image_centers[keep_diff_cls]

        # center map is down-sample smaller than input image, which is the same size with wh
        if predict_out:
            poly_init = transpose_and_gather_feat_from_coord(wh_pred, image_centers, down_sample).squeeze(0) # (N, 256)
            # poly
            # print(poly_init.shape)
            if len(poly_init.shape) == 3:
                batch, nums, _ = poly_init.shape
                poly_init = poly_init.reshape(batch*nums, -1, 2)
            else:
                nums, feat = poly_init.shape
                assert feat % 2 == 0
                poly_init = poly_init.reshape(nums, feat//2, 2)
            poly_init = image_centers.unsqueeze(1).expand(poly_init.size(0), poly_init.size(1), 2)/down_sample + poly_init * stride
        
            detections.append(
                {
                    "poly": image_bbox,
                    "scores": image_scores,
                    "labels": image_labels,
                    "ct": image_centers,
                    "poly_init": poly_init
                }
            )
        else:
            detections.append(
                {
                    "poly": image_bbox,
                    "scores": image_scores,
                    "labels": image_labels,
                    "ct": image_centers,
                }
            )

    return detections

def clip_to_image(poly, h, w):
    if poly.shape[-1] == 4:
        # detection box
        poly[..., :2] = torch.clamp(poly[..., :2], min=0)
        poly[..., 2] = torch.clamp(poly[..., 2], max=w-1)
        poly[..., 3] = torch.clamp(poly[..., 3], max=h-1)
        return poly
    else:
        poly[..., :2] = torch.clamp(poly[..., :2], min=0)
        poly[..., 0] = torch.clamp(poly[..., 0], max=w-1)
        poly[..., 1] = torch.clamp(poly[..., 1], max=h-1)
        return poly

def get_gcn_feature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1
    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly)[0].permute(1, 0, 2)
        gcn_feature[ind == i] = feature
    return gcn_feature

# ifft from prediction
def pred_ifft(pred, n_points):
    # input:
    #   pred: (N, K), N is num of polys, K is the number of prediction
    # output
    #   ipred: (N, P, 2)
    assert pred.shape[-1] % 2 == 0
    assert n_points > pred.shape[-1]
    N = pred.shape[0]
    c_fft = torch.zeros(pred.shape[0], n_points, dtype=torch.complex64, device=pred.device)
    pred = pred.reshape(N, -1, 2)
    P = pred.shape[1]
    if P % 2 == 0:
        c_fft[:, :P//2] = pred[:, :P//2, 0] + pred[:, :P//2, 1] * 1j
        c_fft[:, -(P//2):] = pred[:, (P//2):, 0] + pred[:, (P//2):, 1] * 1j
    else:
        c_fft[:, :P//2 + 1] = pred[:, :P//2 + 1, 0] + pred[:, :P//2 + 1, 1] * 1j
        c_fft[:, -(P//2):] = pred[:, (P//2 + 1):, 0] + pred[:, (P//2 + 1):, 1] * 1j
    ipred = ifft(c_fft)
    ipred = torch.cat((ipred.real[:,:,None], ipred.imag[:,:,None]), dim=-1)
    return ipred

def box_to_roi(box, box_01):
    """ box: [b, n, 4] """
    box = box[box_01]
    ind = torch.cat([torch.full([box_01[i].sum()], i) for i in range(len(box_01))], dim=0)
    ind = ind.to(box.device).float()
    roi = torch.cat([ind[:, None], box], dim=1)
    return roi

def decode_cp_detection(cp_hm, cp_wh, abox, adet):
    batch, cat, height, width = cp_hm.size()    # batch here is the RoIs
    if rcnn_snake_config.cp_hm_nms:
        cp_hm = nms(cp_hm)
    is_poly = cp_wh.shape[1] != 2
    if batch == 0:
        device = cp_hm.device
        boxes = torch.zeros(0,4,device=device)
        cp_ind = torch.zeros(0, device=device)
        poly = torch.zeros(0,128,2,device=device)
        return (poly, boxes, cp_ind) if is_poly else (boxes, cp_ind)
    
    ## NOTE: uncomment it if the result is weird
    # abox = abox * 4
    # coord/4
    abox_w, abox_h = abox[..., 2] - abox[..., 0], abox[..., 3] - abox[..., 1]

    scores, inds, clses, ys, xs = topk(cp_hm, rcnn_snake_config.max_cp_det)
    cp_wh = transpose_and_gather_feat(cp_wh, inds)

    if is_poly:
        cp_wh = cp_wh.view(batch, rcnn_snake_config.max_cp_det, -1, 2) # (RoIs, cp_det, 128, 2)
    else:
        cp_wh = cp_wh.view(batch, rcnn_snake_config.max_cp_det, -1) # (RoIs, cp_det, 2)

    cp_hm_h, cp_hm_w = cp_hm.size(2), cp_hm.size(3) # (RoIs, max_det)
    ## NOTE: weird, why cp_wh is in the same coord with abox
    xs = xs / cp_hm_w * abox_w[..., None] + abox[:, 0:1]
    ys = ys / cp_hm_h * abox_h[..., None] + abox[:, 1:2]
    ## NOTE: To relative coordinate, phase 2
    # cp_wh[..., 0] = cp_wh[..., 0] * abox_w[..., None]
    # cp_wh[..., 1] = cp_wh[..., 1] * abox_h[..., None]

    if is_poly:
        cp = torch.cat([xs.unsqueeze(2).to(torch.float32), ys.unsqueeze(2).to(torch.float32)], dim=2)
        init_poly = cp_wh + cp.unsqueeze(2).expand(cp_wh.size(0), cp_wh.size(1),cp_wh.size(2),cp_wh.size(3))

        boxes = torch.stack([xs + torch.min(cp_wh[..., 0], dim=2)[0],
                            ys + torch.min(cp_wh[..., 1], dim=2)[0],
                            xs + torch.max(cp_wh[..., 0], dim=2)[0],
                            ys + torch.max(cp_wh[..., 1], dim=2)[0]], dim=2)

    else:
        boxes = torch.stack([xs - cp_wh[..., 0] / 2,
                            ys - cp_wh[..., 1] / 2,
                            xs + cp_wh[..., 0] / 2,
                            ys + cp_wh[..., 1] / 2], dim=2)

    ascore = adet[..., 4]
    acls = adet[..., 5]
    excluded_clses = [1, 2]
    for cls_ in excluded_clses:
        boxes[acls == cls_, 0] = abox[acls == cls_]
        scores[acls == cls_, 0] = 1
        scores[acls == cls_, 1:] = 0

    ct_num = len(abox)
    boxes_ = []
    if is_poly:
        init_poly_ = []
    abox_list = []
    for i in range(ct_num):
        cp_ind = tv_nms(boxes[i], scores[i], rcnn_snake_config.max_cp_overlap)
        cp_01 = scores[i][cp_ind] > rcnn_snake_config.cp_score
        if torch.any(cp_01) > 0:
            abox_list.append(i)

        boxes_.append(boxes[i][cp_ind][cp_01])
        if is_poly:
            init_poly_.append(init_poly[i][cp_ind][cp_01])

    cp_ind = torch.cat([torch.full([len(boxes_[i])], i) for i in range(len(boxes_))], dim=0)
    cp_ind = cp_ind.to(boxes.device)
    boxes = torch.cat(boxes_, dim=0)
    if is_poly:
        init_poly = torch.cat(init_poly_, dim=0)
    return (init_poly, boxes, cp_ind) if is_poly else (boxes, cp_ind)

