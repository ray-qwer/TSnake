from torch.utils.data.dataloader import default_collate
import torch
import numpy as np


def collate_batch(batch):
    data_input = {}
    inp = {'inp': default_collate([b['inp'] for b in batch])}
    meta = default_collate([b['meta'] for b in batch])
    data_input.update(inp)
    data_input.update({'meta': meta})

    if 'test' in meta:
        return data_input

    #collate detection
    if 'ct_hm' in batch[0]:
        ct_hm = default_collate([b['ct_hm'] for b in batch])
    if 'bd_map' in batch[0]:
        bd_map = default_collate([b['bd_map'] for b in batch])

    max_len = torch.max(meta['ct_num'])
    batch_size = len(batch)
    wh = torch.zeros([batch_size, max_len, 2], dtype=torch.float)   # to max len
    ct_cls = torch.zeros([batch_size, max_len], dtype=torch.int64)
    ct_ind = torch.zeros([batch_size, max_len], dtype=torch.int64)
    ct_01 = torch.zeros([batch_size, max_len], dtype=torch.bool)
    ct_img_idx = torch.zeros([batch_size, max_len], dtype=torch.int64)
    for i in range(batch_size):
        ct_01[i, :meta['ct_num'][i]] = 1    # ct_01 is just like mask, to record where has gt
        ct_img_idx[i, :meta['ct_num'][i]] = i

    if max_len != 0:
        # reg[ct_01] = torch.Tensor(sum([b['reg'] for b in batch], []))
        wh[ct_01] = torch.Tensor(np.concatenate([np.concatenate([b['wh']]) for b in batch if len(b["wh"]) > 0]))
        ct_cls[ct_01] = torch.LongTensor(np.concatenate([np.concatenate([b['ct_cls']]) for b in batch if len(b["ct_cls"]) > 0]))
        ct_ind[ct_01] = torch.LongTensor(np.concatenate([np.concatenate([b['ct_ind']]) for b in batch if len(b["ct_ind"]) > 0]))
        # wh[ct_01] = torch.Tensor(sum([b['wh'] for b in batch], []))
        # ct_cls[ct_01] = torch.LongTensor(sum([b['ct_cls'] for b in batch], []))
        # ct_ind[ct_01] = torch.LongTensor(sum([b['ct_ind'] for b in batch], []))
    
    if 'ct_hm' in batch[0]:
        detection = {'ct_hm': ct_hm, 'ct_cls': ct_cls, 'ct_ind': ct_ind, 'ct_01': ct_01, 'ct_img_idx': ct_img_idx}
    else:
        detection = {'ct_cls': ct_cls, 'ct_ind': ct_ind, 'ct_01': ct_01, 'ct_img_idx': ct_img_idx}

    data_input.update(detection)

    #collate sementation
    num_points_per_poly = 128
    img_gt_polys = torch.zeros([batch_size, max_len, num_points_per_poly, 2], dtype=torch.float)
    can_gt_polys = torch.zeros([batch_size, max_len, num_points_per_poly, 2], dtype=torch.float)
    keyPointsMask = torch.zeros([batch_size, max_len, num_points_per_poly], dtype=torch.float)
    bbox = torch.zeros([max_len, 4], dtype=torch.float)
    multicomp = torch.ones([batch_size, max_len], dtype=torch.int8)*(-1)
    mask_size = inp['inp'].shape[-2:]
    mask = torch.zeros([max_len, mask_size[0], mask_size[1]], dtype=torch.float)

    if max_len != 0:
        # TODO: could optimize the speed here
        img_gt_polys[ct_01] = torch.Tensor(np.concatenate([np.concatenate([b['img_gt_polys']]) for b in batch if len(b["img_gt_polys"]) > 0]))
        can_gt_polys[ct_01] = torch.Tensor(np.concatenate([np.concatenate([b['can_gt_polys']]) for b in batch if len(b["can_gt_polys"]) > 0]))
        keyPointsMask[ct_01] = torch.Tensor(np.concatenate([np.concatenate([b['keypoints_mask']]) for b in batch if len(b["keypoints_mask"]) > 0]))
        if 'multicomp' in batch[0]:
            multicomp[ct_01] = torch.Tensor(np.concatenate([np.concatenate([b['multicomp']]) for b in batch if len(b['multicomp']) > 0])).to(torch.int8)
        if 'bbox' in batch[0]:
            bbox = torch.Tensor(np.concatenate([np.concatenate([b["bbox"]]) for b in batch if len(b["bbox"]) > 0]))
        if 'mask' in batch[0]:    
            mask = torch.Tensor(np.concatenate([np.concatenate([b['mask']]) for b in batch if len(b['mask']) > 0]))
        # img_gt_polys[ct_01] = torch.Tensor(sum([b['img_gt_polys'] for b in batch], []))
        # can_gt_polys[ct_01] = torch.Tensor(sum([b['can_gt_polys'] for b in batch], []))
        # keyPointsMask[ct_01] = torch.Tensor(sum([b['keypoints_mask'] for b in batch], []))
    
    data_input.update({'img_gt_polys': img_gt_polys, 'can_gt_polys': can_gt_polys,
                       'keypoints_mask': keyPointsMask, 'bbox':bbox})
    if 'multicomp' in batch[0]:
        data_input.update({'multicomp': multicomp})
    if 'mask' in batch[0]:
        data_input.update({'mask': mask})
    if 'bd_map' in batch[0]:
        data_input.update({'bd_map': bd_map})
    return data_input
