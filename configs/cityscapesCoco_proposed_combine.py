from .base_localmoving import commen, data, model, train, test
import numpy as np

data.scale = np.array([800, 800])
data.input_w, data.input_h = (800, 800)

model.heads.update({'bd': 1})
model.heads['ct_hm'] = 8
model.use_GN = False
model.snake = "shared"
model.evolve_name = ""
model.evolve_iters = 8
# combine net
model.combineNet = True
model.combine_layer = 2
model.combine_heads = 8
model.combine_pe_method = "rel"

train.dataset = 'cityscapes_train'
# polysnake: milestones:[80, 120, 150]
train.optimizer = {'name': 'adam', 'lr': 1e-4,
                'weight_decay':5e-4,
                'milestones': [20, 40, 80, 120], # -> this is from polysnake
                'gamma': 0.5}
train.batch_size = 12   # try
train.num_workers = 8
train.epoch = 150
train.from_dist = False
train.mda_kpt = False
train.with_dml = False
train.start_epoch = 0
train.with_wsll = True
train.polygon_origin_size = True
train.weight_dict = {'ct_loss': 1., 'init': 0.1, 'coarse': 0.1, 'evolve': 1., 'evolve_mask':1.}
train.use_Mask_Loss = False
train.get_instance_mask = False
train.train_boundary_head = True
train.mosaic_aug = False
train.start_module = 3
# combineNet
train.combine_from_dataset = True

test.dataset = 'cityscapesCoco_val'
test.with_nms = False
test.ct_score = 0.05
test.eval_ep = 1
test.baseline_val = 0.32
test.test_scale = 0.85
test.combine_threshold = 0.7

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test
