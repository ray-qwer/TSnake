import numpy as np

class commen(object):
    task = 'e2ec'
    points_per_poly = 128
    down_ratio = 4
    result_dir = 'data/result'
    record_dir = 'data/record'
    model_dir = 'data/model'


class data(object):
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
    data_rng = np.random.RandomState(123)
    eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                       dtype=np.float32)
    eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    down_ratio = commen.down_ratio
    scale = np.array([512, 512])
    input_w, input_h = (512, 512)
    test_scale = None
    scale_range = [0.6, 1.4]
    points_per_poly = commen.points_per_poly

class model(object):
    dla_layer = 34
    head_conv = 256
    use_dcn = True
    points_per_poly = commen.points_per_poly
    ####### NEW ADDED ########
    fft_dot = 32
    use_fft = False
    ##########################
    down_ratio = commen.down_ratio
    init_stride = 10.
    coarse_stride = 4.
    evolve_stride = 1.
    backbone_num_layers = 34
    heads = {'ct_hm': 20, 'wh': commen.points_per_poly * 2}
    evolve_iters = 3
    detect_only = False
    use_GN = False
    detect_type = "CenterNet" #["CenterNet", "FCOS"]
    detect_backbone = "dla34"
    # local moving
    snake = "origin" # ["local", "origin"]
    evolve_name = ""
    local_kernel = 3
    localmoving_embed_dim = 64
    with_low_level_feat = False
    low_level_feat_dim = 0
    moving_range = 5.
    dilated = 1
    # combine net
    combineNet = False
    combine_layer = 1
    combine_n_embd = 256
    combine_heads = 8
    combine_in_c = 64 + 2
    combine_pe_method = "abs"
    combine_rpe_mode = "bias"
    pred_bbox = False
    # local refine: last module
    local_refine = False
    # RCNN
    rcnn = False
    cp_head = {}

class train(object):
    save_ep = 5
    eval_ep = 1
    optimizer = {'name': 'adam', 'lr': 1e-4,
                 'weight_decay': 5e-4,
                 'milestones': [80, 120, ],
                 'gamma': 0.5}
    optimizer_partial = []
    freeze_layer = []
    # scheduler
    scheduler_name = ""
    scheduler_option = {}
    scheduler_update_per_iter = False
    batch_size = 24
    num_workers = 8
    epoch = 150
    with_dml = True
    start_epoch = 10
    weight_dict = {'init': 0.1, 'coarse': 0.1, 'evolve': 1.}
    dataset = 'sbd_train'
    shuffle = True
    from_dist = False
    mda_kpt = False
    grad_acc = 0
    use_Mask_Loss = True
    get_instance_mask = False
    polygon_origin_size = False
    # combine net
    combine_from_dataset = True
    train_combine_only = False
    # boundary head
    train_boundary_head = False
    # weighted smooth l1 loss
    with_wsll = False
    start_module = 3+1  # start from 0. this means the model will do wdml after [n] evolution
    # with mosaic data augmentation
    mosaic_aug = False
    
class test(object):
    test_stage = 'final-dml'  # init, coarse final or final-dml
    test_rescale = None
    ct_score = 0.05
    with_nms = True
    with_post_process = False
    segm_or_bbox = 'segm'
    dataset = 'sbd_val'
    combine_threshold = 1

