from .base_localmoving import commen, data, model, train, test
import numpy as np

data.scale = np.array([800, 800])
data.input_w, data.input_h = (800, 800)
data.scale_range = [0.5, 1.2]

model.rcnn = True
model.heads = ({'act_hm': 8, 'awh': 2})
model.use_GN = False
model.snake = "shared"
model.evolve_name = ""
model.evolve_iters = 8
model.cp_head = {'cp_hm':1, 'cp_wh':2}

train.dataset = 'cityscapes_rcnn_train'
# polysnake: milestones:[80, 120, 150]
train.optimizer = {'name': 'adam', 'lr': 1e-4,
                'weight_decay':5e-4,
                'milestones': (60, 100, 130, 155, 180), # -> this is from polysnake
                'gamma': 0.5}
train.batch_size = 12   # try
train.num_workers = 8
train.epoch = 200
train.from_dist = False
train.mda_kpt = True
train.with_dml = False
train.start_epoch = 10
train.with_wsll = True
train.polygon_origin_size = True
train.weight_dict = {'ct_loss': 1., 'init': 0.1, 'coarse': 0.1, 'evolve': 1., 'evolve_mask':1.}
train.use_Mask_Loss = False
train.get_instance_mask = False
train.train_boundary_head = True
train.mosaic_aug = False
train.start_module = 3 + 1

test.dataset = 'cityscapesCoco_val'
test.with_nms = True
test.ct_score = 0.05
test.eval_ep = 1
test.baseline_val = 0.32
test.test_scale = 0.85

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test
