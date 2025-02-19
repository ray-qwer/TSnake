import numpy as np
import cv2
import random
from shapely.geometry import Polygon
import supervision as sv

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


def get_border(border, size):
    i = 1
    while np.any(size - border // i <= border // i):
        i *= 2
    return border // i

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha

def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)

def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    # functions = [brightness_]
    random.shuffle(functions)
    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)

def augment(img, split, _data_rng, _eig_val, _eig_vec, mean, std,
            down_ratio, input_h, input_w, scale_range, scale=None, test_rescale=None, test_scale=None, 
            is_mosaic=False, poly_instance=None):
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if scale is None:
        scale = max(img.shape[0], img.shape[1]) * 1.0
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    if isinstance(scale, np.ndarray):
        scale = scale.astype(np.float32)
    flipped = False
    if split == 'train':
        if not is_mosaic:
            scale = scale * np.random.uniform(scale_range[0], scale_range[1])
            # scale = scale * (random.random() * (scale_range[1] - scale_range[0]) + scale_range[0])
            if poly_instance is not None:
                seed = np.random.randint(0, len(poly_instance))
                index = np.random.randint(0, len(poly_instance[seed]))
                poly = poly_instance[seed][index]
                x, y = np.array(poly[np.random.randint(0, len(poly))]).astype(np.float32)
            else:
                x, y = center
            w_border = get_border(width/4, scale[0]) + 1
            h_border = get_border(height/4, scale[0]) + 1
            center[0] = np.random.randint(low=max(x - w_border, 0), high=min(x + w_border, width - 1))
            center[1] = np.random.randint(low=max(y - h_border, 0), high=min(y + h_border, height - 1))

        if random.random() < 0.5:
            flipped = True
            img = img[:, ::-1, :]
            center[0] = width - center[0] - 1

    if split != 'train':
        scale = np.array([width, height])
        x = 32
        if test_rescale is not None:
            input_w, input_h = int((width / test_rescale + x - 1) // x * x),\
                               int((height / test_rescale + x - 1) // x * x)
        else:
            if test_scale is None:
                input_w = (int(width / 1.) | (x - 1)) + 1
                input_h = (int(height / 1.) | (x - 1)) + 1
            else:
                scale = max(width, height) * 1.0
                scale = np.array([scale, scale])
                input_w, input_h = test_scale
        center = np.array([width // 2, height // 2])

    trans_input = get_affine_transform(center, scale, 0, [input_w, input_h])    # center, scale, rot, output_size
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    orig_img = inp.copy()
    inp = (inp.astype(np.float32) / 255.)
    if split == 'train':
        color_aug(_data_rng, inp, _eig_val, _eig_vec)

    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)

    output_h, output_w = input_h // down_ratio, input_w // down_ratio
    trans_output = get_affine_transform(center, scale, 0, [output_w, output_h])
    inp_out_hw = (input_h, input_w, output_h, output_w)

    return orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw

def handle_break_point(poly, axis, number, outside_border):
    if len(poly) == 0:
        return []

    if len(poly[outside_border(poly[:, axis], number)]) == len(poly):
        return []

    break_points = np.argwhere(
        outside_border(poly[:-1, axis], number) != outside_border(poly[1:, axis], number)).ravel()
    if len(break_points) == 0:
        return poly

    new_poly = []
    if not outside_border(poly[break_points[0], axis], number):
        new_poly.append(poly[:break_points[0]])

    for i in range(len(break_points)):
        current_poly = poly[break_points[i]]
        next_poly = poly[break_points[i] + 1]
        mid_poly = current_poly + (next_poly - current_poly) * (number - current_poly[axis]) / (next_poly[axis] - current_poly[axis])

        if outside_border(poly[break_points[i], axis], number):
            if mid_poly[axis] != next_poly[axis]:
                new_poly.append([mid_poly])
            next_point = len(poly) if i == (len(break_points) - 1) else break_points[i + 1]
            new_poly.append(poly[break_points[i] + 1:next_point])
        else:
            new_poly.append([poly[break_points[i]]])
            if mid_poly[axis] != current_poly[axis]:
                new_poly.append([mid_poly])

    if outside_border(poly[-1, axis], number) != outside_border(poly[0, axis], number):
        current_poly = poly[-1]
        next_poly = poly[0]
        mid_poly = current_poly + (next_poly - current_poly) * (number - current_poly[axis]) / (next_poly[axis] - current_poly[axis])
        new_poly.append([mid_poly])

    return np.concatenate(new_poly)

def transform_bbox(bboxes, trans_input, input_h, input_w):
    new_bbox = []
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        bbox = affine_transform(bbox, trans_input)
        bbox[0,:] = bbox[0,:].clip(0, input_w-1)
        bbox[1,:] = bbox[1,:].clip(0, input_h-1)
        
        if bbox[0,0] >= bbox[1,0]:
            continue
        if bbox[0,1] >= bbox[1,1]:
            continue
        
        new_bbox.append(bbox)
    return new_bbox        


def transform_polys(polys, trans_output, output_h, output_w):
    new_polys = []
    for i in range(len(polys)):
        poly = polys[i]
        poly = affine_transform(poly, trans_output)
        poly = handle_break_point(poly, 0, 0, lambda x, y: x < y)
        poly = handle_break_point(poly, 0, output_w, lambda x, y: x >= y)
        poly = handle_break_point(poly, 1, 0, lambda x, y: x < y)
        poly = handle_break_point(poly, 1, output_h, lambda x, y: x >= y)
        if len(poly) == 0:
            continue
        if len(np.unique(poly, axis=0)) <= 2:
            continue    
        new_polys.append(poly)
    return new_polys

def filter_tiny_polys(polys, ratio=1):
    return [poly for poly in polys if Polygon(poly).area > (5 * ratio * ratio) ]

def get_poly_area(polys):
    return np.array([Polygon(poly).area for poly in polys])

def get_cw_polys(polys):    # clockwise polygon
    return [poly[::-1] if Polygon(poly).exterior.is_ccw else poly for poly in polys]

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    if b3 ** 2 - 4 * a3 * c3 < 0:
        r3 = min(r1, r2)
    else:
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=(1, 1), rho=0):
    if not isinstance(sigma, tuple):
        sigma = (sigma, sigma)
    sigma_x, sigma_y = sigma

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    energy = (x * x) / (sigma_x * sigma_x) - 2 * rho * x * y / (sigma_x * sigma_y) + (y * y) / (sigma_y * sigma_y)
    h = np.exp(-energy / (2 * (1 - rho * rho)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def uniformsample(pgtnp_px2, newpnum):
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2 # coordinates, (len, 2)

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum    # from 1 to pnum+1(0, start point)
    pgtnext_px2 = pgtnp_px2[idxnext_p]      # rearange
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    ## NOTE: this is groundtruth, could not do this@@
    if pnum > newpnum: # never enter this condition, since newpnum is pnum*points_per_poly
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        # to calculate how many points in one segment
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                # it is short segment
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        
        if edgenumsum != newpnum:
            if edgenumsum > newpnum:
                id = -1
                passnum = edgenumsum - newpnum  # redundant point
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:   # the longest length - extra points
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1  # just reserve one point for the longest points
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1     # add to the longest length
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

            # interpolate
            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp

def four_idx(img_gt_poly):
    x_min, y_min = np.min(img_gt_poly, axis=0)
    x_max, y_max = np.max(img_gt_poly, axis=0)
    center = [(x_min + x_max) / 2., (y_min + y_max) / 2.]
    can_gt_polys = img_gt_poly.copy()
    can_gt_polys[:, 0] -= center[0]
    can_gt_polys[:, 1] -= center[1]
    # distance: the distance to the center
    distance = np.sum(can_gt_polys ** 2, axis=1, keepdims=True) ** 0.5 + 1e-6
    can_gt_polys /= np.repeat(distance, axis=1, repeats=2)
    idx_bottom = np.argmax(can_gt_polys[:, 1])
    idx_top = np.argmin(can_gt_polys[:, 1])
    idx_right = np.argmax(can_gt_polys[:, 0])
    idx_left = np.argmin(can_gt_polys[:, 0])
    return [idx_bottom, idx_right, idx_top, idx_left]

def get_img_gt(img_gt_poly, idx, t=128):
    # there are some points duplicate
    align = len(idx)
    pointsNum = img_gt_poly.shape[0]
    r = []
    k = np.arange(0, t / align, dtype=float) / (t / align) # it seperate the polygon into 4 parts to balance the number for each direction
    for i in range(align):
        begin = idx[i]
        end = idx[(i + 1) % align]
        if begin > end:
            end += pointsNum
        r.append((np.round(((end - begin) * k).astype(int)) + begin) % pointsNum)
        
    r = np.concatenate(r, axis=0)
    return img_gt_poly[r, :]

def img_poly_to_can_poly(img_poly):
    x_min, y_min = np.min(img_poly, axis=0)
    can_poly = img_poly - np.array([x_min, y_min])
    return can_poly

def get_dist_center(box, poly):
    """
    This function is to find out the farthest point regarding the border
        box: bbox, with form x_min, y_min, x_max, y_max
        poly: polygon that has been modified according to resized image
    """
    x_min, y_min, x_max, y_max = box
    poly_map = np.zeros((int(np.ceil(y_max-y_min+1)), int(np.ceil(x_max-x_min+1))), dtype=np.uint8)
    poly_translate = poly - np.array([x_min, y_min])
    cv2.fillPoly(poly_map, [np.round(poly_translate).astype(int)], 255)
    dist = cv2.distanceTransform(poly_map, cv2.DIST_L2, 3)
    farthest = np.max(dist)
    mask = np.where(dist >= farthest, 1, 0).astype(np.uint8)
    nums, labels = cv2.connectedComponents(mask)

    max_length = 0
    for i in range(1, nums):
        label = np.where(labels == i)
        if max_length < len(label[0]):
            max_length = len(label[0])
            ct = np.mean(label, axis=1)
    ct = np.array([ct[1],ct[0]])
    ct += (np.array([x_min, y_min]))
    return np.round(ct)

def get_mask_from_poly(poly, w, h):
    poly = poly.astype(np.int)
    mask = sv.polygon_to_mask(poly, (w, h))
    return mask
    
def get_boundary_map(polys, hw, down_ratio=1):
    bd_map = np.zeros(hw, dtype=np.uint8)
    width = 5  # may add dilation
    # with distance
    # bd_map = cv2.drawContours(bd_map, [np.int32(p / down_ratio) for p in polys], -1, 1, 1)
    # kernel = np.ones((width, width))
    # bd_map = cv2.dilate(bd_map, kernel)
    # dist_map = cv2.distanceTransform(bd_map, cv2.DIST_L2, 3).astype(np.float32)
    # dist_map = cv2.normalize(dist_map, None, 0, 1, cv2.NORM_MINMAX)
    # return dist_map
    return cv2.drawContours(bd_map, [np.int32(p / down_ratio) for p in polys], -1, 1, 1)

class Mosaic:
    """
    Args: 
        img_scale: image size after mosaic pipeline of single image.
        center_ratio_range: center ratio range of mosaic(???
        min_area: the minimum pixel for filtering invalid annotations
        pad_value: default is 128
    """
    def __init__(self, img_scale=[640, 640], 
                 center_ratio_range=[0.5, 1.5],
                 input_hw=[640, 640],
                 scale_range=[0.6, 1.4],
                 crop_img=False,
                 min_area=80,
                 ):
        if img_scale is not None:
            self.img_scale = np.array(img_scale)
        else:
            self.img_scale = np.array([640, 640])
        self.center_ratio_range = center_ratio_range
        self.min_area = min_area
        self.scale_range = scale_range
        self.crop_img = crop_img
    def __call__(self, imgs, polys):
        if self.crop_img:
            return self.crop_img_mosaic(imgs, polys)
        else:
            return self.resize_mosaic(imgs, polys)
    
    def resize_mosaic(self, imgs, polys):
        # YOLOX mosaic
        """
        what i need:
            1. the transform of each pictures
            2. the combination figures
        """
        assert len(imgs) == 4
        H, W = self.img_scale
        large_mosaic_img = np.full((H * 2, W * 2, 3), 114, dtype=np.uint8)
        c_x = random.randint(int(W * self.center_ratio_range[0]), int(W * self.center_ratio_range[1]))
        c_y = random.randint(int(H * self.center_ratio_range[0]), int(H * self.center_ratio_range[1]))

        
        poly_list = []
        for i, (img, poly) in enumerate(zip(imgs, polys)):
            ih, iw = img.shape[:2]
            crop_img = img.copy()
            resized_crop = cv2.resize(crop_img, (W, H))

            if i == 0:      # top-left
                large_mosaic_img[:H, :W] = resized_crop
                x_min, y_min = 0, 0
            elif i == 1:    # top-right
                large_mosaic_img[:H, -W:] = resized_crop
                x_min, y_min = W, 0
            elif i == 2:    # bot-left
                large_mosaic_img[-H:, :W] = resized_crop
                x_min, y_min = 0, H
            elif i == 3:    # bot-right
                large_mosaic_img[-H:, -W:] = resized_crop
                x_min, y_min = W, H


            for _poly in poly:
                pys = [p.reshape(-1,2) for p in _poly]
                _pys = []
                for py in pys:
                    py[:, 0] = py[:, 0] * (W / iw) + x_min
                    py[:, 1] = py[:, 1] * (H / ih) + y_min
                    _pys.append(py)
                poly_list.append(_pys)
            
        # crop images from center point
        new_x_min, new_y_min = c_x - W//2, c_y - H//2
        new_x_max, new_y_max = c_x + W//2, c_y + H//2

        mosaic_img = large_mosaic_img[new_y_min: new_y_max, new_x_min: new_x_max]

        for i in range(len(poly_list)):
            pys = poly_list[i]
            for j in range(len(pys)):
                pys[j][:, 0] = pys[j][:, 0] - new_x_min
                pys[j][:, 1] = pys[j][:, 1] - new_y_min
        return mosaic_img, poly_list
            
    def crop_img_mosaic(self, imgs, polys):
        #### mosaic image property
        assert len(imgs) == 4
        H, W = self.img_scale
        large_mosaic_img = np.full((H * 2, W * 2, 3), 114, dtype=np.uint8)
        c_x = random.randint(int(W * self.center_ratio_range[0]), int(W * self.center_ratio_range[1]))
        c_y = random.randint(int(H * self.center_ratio_range[0]), int(H * self.center_ratio_range[1]))

        poly_list = []
        for i, (img, poly) in enumerate(zip(imgs, polys)):
            height, width = img.shape[0], img.shape[1]
            scale_h, scale_w = (self.img_scale * np.random.uniform(self.scale_range[0], self.scale_range[1])).astype(int)
            ih, iw = min(scale_h, height), min(scale_w, width)
            
            if len(poly) > 0:
                seed = np.random.randint(0, len(poly))
                index = np.random.randint(0, len(poly[seed]))
                p = poly[seed][index]
                x,y = p[np.random.randint(0, len(p))]
                tl_pt = [0, 0]
                if x < scale_w//2:
                    x = scale_w // 2
                elif x > (width - scale_w):
                    x = width - scale_w 
                if y < scale_h//2:
                    y = scale_h // 2
                elif y > (height - scale_h):
                    y = height - scale_h
                tl_pt = max(0, int(x) - scale_w//2), max(0, int(y) - scale_h//2)
                br_pt = tl_pt[0] + iw, tl_pt[1] + ih
            else:
                tl_pt = (random.randint(0, width - iw), random.randint(0, height - ih))
                br_pt = tl_pt[0] + iw, tl_pt[1] + ih
            
            crop_img = img[tl_pt[1]:br_pt[1], tl_pt[0]:br_pt[0]]
            resized_crop = cv2.resize(crop_img, (W, H))

            if i == 0:      # top-left
                large_mosaic_img[:H, :W] = resized_crop
                x_min, y_min = 0, 0
                x_max, y_max = W-1, H-1
            elif i == 1:    # top-right
                large_mosaic_img[:H, -W:] = resized_crop
                x_min, y_min = W, 0
                x_max, y_max = 2*W-1, H-1
            elif i == 2:    # bot-left
                large_mosaic_img[-H:, :W] = resized_crop
                x_min, y_min = 0, H
                x_max, y_max = W-1, 2*H-1
            elif i == 3:    # bot-right
                large_mosaic_img[-H:, -W:] = resized_crop
                x_min, y_min = W, H
                x_max, y_max = 2*W-1, 2*H-1
            
            for _poly in poly:
                pys = [p.reshape(-1,2) for p in _poly]
                _pys = []
                for py in pys:
                    # filter out of range
                    py[:, 0] = (py[:, 0] - tl_pt[0]) * (W / iw) + x_min
                    py[:, 1] = (py[:, 1] - tl_pt[1]) * (H / ih) + y_min
                    py[:, 0] = np.clip(py[:, 0], x_min, x_max)
                    py[:, 1] = np.clip(py[:, 1], y_min, y_max)
                    _pys.append(py)
                poly_list.append(_pys)
        
        new_x_min, new_y_min = c_x - W//2, c_y - H//2
        new_x_max, new_y_max = c_x + W//2, c_y + H//2

        mosaic_img = large_mosaic_img[new_y_min: new_y_max, new_x_min: new_x_max]

        for i in range(len(poly_list)):
            pys = poly_list[i]
            for j in range(len(pys)):
                pys[j][:, 0] = pys[j][:, 0] - new_x_min
                pys[j][:, 1] = pys[j][:, 1] - new_y_min
        return mosaic_img, poly_list