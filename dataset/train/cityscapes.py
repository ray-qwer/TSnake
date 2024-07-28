from .base import Dataset
import os
import cv2
import glob
import json
import numpy as np

#Globals ----------------------------------------------------------------------
COCO_LABELS = {24: 1,
               26: 2,
               27: 3,
               25: 4,
               33: 5,
               32: 6,
               28: 7,
               31: 8}

# Label number to name and color
INSTANCE_LABELS = {26: {'name': 'car', 'color': [0, 0, 142]},
                   24: {'name': 'person', 'color': [220, 20, 60]},
                   25: {'name': 'rider', 'color': [255, 0, 0]},
                   32: {'name': 'motorcycle', 'color': [0, 0, 230]},
                   33: {'name': 'bicycle', 'color': [119, 11, 32]},
                   27: {'name': 'truck', 'color': [0, 0, 70]},
                   28: {'name': 'bus', 'color': [0, 60, 100]},
                   31: {'name': 'train', 'color': [0, 80, 100]}}

# Label name to number
LABEL_DICT = {'car': 26, 'person': 24, 'rider': 25, 'motorcycle': 32,
              'bicycle': 33, 'truck': 27, 'bus': 28, 'train': 31}
# LABEL_DICT = {'bicycle': 33}

# Label name to contiguous number
JSON_DICT = dict(car=0, person=1, rider=2, motorcycle=3, bicycle=4, truck=5, bus=6, train=7)
# JSON_DICT = dict(bicycle=0)
# Contiguous number to name
NUMBER_DICT = {0: 'car', 1: 'person', 2: 'rider', 3: 'motorcycle',
               4: 'bicycle', 5: 'truck', 6: 'bus', 7: 'train'}
# NUMBER_DICT = {0:'bicycle'}
# Array of keys
KEYS = np.array([[26000, 26999], [24000, 24999], [25000, 25999],
                 [32000, 32999], [33000, 33999], [27000, 27999],
                 [28000, 28999], [31000, 31999]])

NUM_CLASS = {'person': 17914, 'rider': 1755, 'car': 26944, 'truck': 482,
             'bus': 379, 'train': 168, 'motorcycle': 735, 'bicycle': 3658}

# ------------------------------------------------------------------------------

def read_dataset(ann_files):
    if not isinstance(ann_files, tuple):
        ann_files = (ann_files,)

    ann_file = []
    for ann_file_dir in ann_files:
        ann_file += glob.glob(os.path.join(ann_file_dir, '*/*.json'))

    ann_filter = []
    for fname in ann_file:
        with open(fname, 'r') as f:
            ann = json.load(f)
            examples = []
            for instance in ann:
                instance_label = instance['label']
                if instance_label not in LABEL_DICT:
                    continue
                examples.append(instance)
            if len(examples) > 0:
                ann_filter.append(fname)
    return ann_filter

class CityscapesDataset(Dataset):
    def __init__(self, anno_file, data_root, split, cfg):
        super(CityscapesDataset, self).__init__(anno_file, data_root, split, cfg)
        self.anns = np.array(read_dataset(anno_file)[:])
        self.anns = self.anns[:500] if split == 'mini' else self.anns
        self.json_category_id_to_continuous_id = JSON_DICT
        self.sample_list = range(len(self.anns))
        
        print("initialize")
        
    def read_original_data(self, anno, path):
        img = cv2.imread(path)
        instance_polys = [[np.array(comp['poly']) for comp in obj['components']] for obj in anno]
        # instance_bbox = [[np.array(comp['bbox']) for comp in obj['components']] for obj in anno]
        cls_ids = [self.json_category_id_to_continuous_id[obj['label']] for obj in anno]
        return img, instance_polys, cls_ids

    def process_info(self, fname):
        data_root = self.data_root
        with open(fname, 'r') as f:
            ann = json.load(f)
        examples = []
        for instance in ann:
            instance_label = instance['label']
            if instance_label not in LABEL_DICT:
                continue
            examples.append(instance)
        img_path = os.path.join(data_root, '/'.join(ann[0]['img_path'].split('/')[-3:]))
        img_id = ann[0]['image_id']
        return examples, img_path, img_id
    