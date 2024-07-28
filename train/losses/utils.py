import numpy as np
import torch
import cv2
import supervision as sv
from torch.nn import functional as F
from torchvision.transforms.functional import InterpolationMode as IPM
from torchvision import transforms

def get_resize_transforms(resolution):
    return transforms.Resize(resolution, interpolation=IPM.BILINEAR)

def poly2mask(poly, img_w, img_h):
    poly = poly.astype(int)
    mask = sv.polygon_to_mask(poly, (img_w, img_h))
    # mask = maskUtils.decode()
    return mask

def cropnResize(poly, union_bbox, resolution=(64,64)):
    poly = poly - union_bbox[:2]
    
    if isinstance(poly, torch.Tensor):
        poly = poly.detach().cpu().numpy()
    
    mask = poly2mask(poly, int(union_bbox[2]-union_bbox[0]+1), int(union_bbox[3]-union_bbox[1]+1))
    mask = cv2.resize(mask, resolution, interpolation=cv2.INTER_LINEAR)
    _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
    return mask

def padnResize(mask, bbox, union_bbox, transforms):
    padding = (bbox[0]-union_bbox[0], union_bbox[2]-bbox[2], bbox[1]-union_bbox[1], union_bbox[3]-bbox[3])
    mask_ = F.pad(mask, padding, 'constant', 0).unsqueeze(0)
    mask_ = transforms(mask_)
    mask_ = (mask_ > 0.5).float().squeeze()
    return mask_

def MaskCropnResize(mask, union_bbox, transforms):
    assert torch.all(union_bbox >= 0.)
    m = mask[int(union_bbox[1]):int(union_bbox[3])+1, int(union_bbox[0]):int(union_bbox[2])+1].unsqueeze(0)
    m = transforms(m)
    m = (m > 0.5).float().squeeze()
    return m