from .base_localmoving import commen, data, model, train, test

data.scale = None
data.test_scale = (512, 512)

model.heads.update({'bd': 1})
model.heads['ct_hm'] = 20
model.use_GN = False
model.snake = "shared"
model.evolve_name = ""
model.evolve_iters = 8

train.dataset = 'sbd_train'
train.optimizer = {'name': 'adam', 'lr': 1e-4,
                'weight_decay': 5e-4,
                'milestones': [60, 100, 130, 155, 180], # -> this is from polysnake
                'gamma': 0.5}
train.batch_size = 20
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
train.start_module = 3 + 1

test.dataset = 'sbd_mini'
test.with_nms = False
test.ct_score = 0.05
test.eval_ep = 5
test.baseline_val = 0.595

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test
