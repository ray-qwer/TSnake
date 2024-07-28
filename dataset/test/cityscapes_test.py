import cv2
import os
import numpy as np
import glob
import torch.utils.data as data
import json
from ..train.cityscapes import JSON_DICT
from ..train.utils import augment

class Dataset(data.Dataset):
    def __init__(self, anno_file, data_root, split, cfg):
        super(Dataset, self).__init__()
        self.cfg = cfg
        self.data_root = data_root
        self.imgs = glob.glob(os.path.join(data_root, '*/*.png'))
        self.split = split
        # self.anns = np.array(self.read_dataset(anno_file)[:])
        # self.anns = self.anns[:500] if split == 'mini' else self.anns
        self.json_category_id_to_contiguous_id = JSON_DICT
    
    def __getitem__(self, index):
        data_input = {}
        img_name = self.imgs[index]
        
        img_id = os.path.basename(img_name).replace('_leftImg8bit.png', '')
        img = cv2.imread(img_name)
        
        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            augment(
                img, self.split,
                self.cfg.data.data_rng, self.cfg.data.eig_val, self.cfg.data.eig_vec,
                self.cfg.data.mean, self.cfg.data.std, self.cfg.commen.down_ratio,
                self.cfg.data.input_h, self.cfg.data.input_w, self.cfg.data.scale_range,
                self.cfg.data.scale, self.cfg.test.test_rescale, self.cfg.data.test_scale
            )
        data_input.update({'inp': inp})
        meta = {'center': center, 'img_id': img_id, 'scale': scale, 'test': '', 'img_name': img_name}
        data_input.update({'meta': meta})
        return data_input

    def __len__(self):
        return len(self.imgs)