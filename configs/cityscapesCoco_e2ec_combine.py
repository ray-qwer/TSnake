from .base_localmoving import commen, data, model, train, test
import numpy as np

data.scale = np.array([800, 800])
data.input_w, data.input_h = (800, 800)

model.heads['ct_hm'] = 8
model.use_GN = False
model.combineNet = True
model.combine_layer = 2
# model.combine_n_embd = 64 + 2
model.combine_heads = 8
model.combine_pe_method = "rel"

train.dataset = 'cityscapes_train'
train.batch_size = 4
train.num_workers = 8
train.epoch = 150
## added
train.grad_acc = 4
train.optimizer = {'name': 'adam', 'lr': 1e-4,
                'weight_decay': 5e-4,
                'milestones': [20, 40, 80, 120],
                'gamma': 0.5, 'momentum':0.9}
train.weight_dict = {'ct_loss':1, 'init': 0.1, 'coarse': 0.1, 'evolve': 1, 'combine': 1.2}
train.from_dist = False
train.mda_kpt = True
train.polygon_origin_size = True
train.start_epoch = 0
# combine net
train.combine_from_dataset = True
train.eval_ep = 1
train.save_ep = 5

test.test_rescale = 0.85
test.dataset = 'cityscapesCoco_val'
test.with_nms = False
test.combine_threshold = 0.67
# test.ct_score = 0.2

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test
