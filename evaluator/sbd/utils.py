import pycocotools.mask as mask_utils
import numpy as np
import cv2

def coco_poly_to_rle(poly, h, w, combine=None):
    rle_ = []
    if combine is None:
        for i in range(len(poly)):
            rles = mask_utils.frPyObjects([poly[i].reshape(-1)], h, w)
            rle = mask_utils.merge(rles)
            rle['counts'] = rle['counts'].decode('utf-8')
            rle_.append(rle)
    else:
        # here only one image per iter
        l = poly.shape[0]
        poly = poly.reshape(l, -1)
        # used = np.zeros(l, dtype=bool)
        for idx, comb in enumerate(combine):
            comb[idx] = True
            comb[:idx] = False
            # comb[used] = False
            multi_polys = poly[comb,:]
            rles = mask_utils.frPyObjects([p for p in multi_polys], h, w)
            rle = mask_utils.merge(rles)
            rle['counts'] = rle['counts'].decode('utf-8')
            rle_.append(rle)
            # used[comb] = True
    return rle_

def rcnn_coco_poly_to_rle(poly, ind_group, h, w):
    rle_ = []
    for i in range(len(ind_group)):
        poly_ = [poly[ind].reshape(-1) for ind in ind_group[i]]
        rles = mask_utils.frPyObjects(poly_, h, w)
        rle = mask_utils.merge(rles)
        rle['counts'] = rle['counts'].decode('utf-8')
        rle_.append(rle)
    return rle_

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def affine_transform(pt, t):
    """pt: [n, 2]"""
    new_pt = np.dot(np.array(pt), t[:, :2].T) + t[:, 2]
    return new_pt


