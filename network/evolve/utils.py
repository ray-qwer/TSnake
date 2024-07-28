import torch
from utils.extreme_utils import _ext as extreme_utils

def collect_training(poly, ct_01):
    batch_size = ct_01.size(0)
    poly = torch.cat([poly[i][ct_01[i].bool()] for i in range(batch_size)], dim=0)
    return poly

def prepare_training(ret, batch, ro):
    ct_01 = batch['ct_01'].byte()
    init = {}

    init.update({'img_gt_polys': collect_training(batch['img_gt_polys'], ct_01)})
    if 'poly_coarse' in ret:
        init.update({'img_init_polys': ret['poly_coarse'].detach() / ro})
        can_init_polys = img_poly_to_can_poly(ret['poly_coarse'].detach() / ro)
    else:
        init.update({'img_init_polys': ret['poly_init'].detach() / ro})
        can_init_polys = img_poly_to_can_poly(ret['poly_init'].detach() / ro)
    init.update({'can_init_polys': can_init_polys})

    ct_num = batch['meta']['ct_num']
    init.update({'py_ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})
    init.update({'py_ind': init['py_ind']})
    init['py_ind'] = init['py_ind'].to(ct_01.device)

    return init

def img_poly_to_can_poly(img_poly):
    if len(img_poly) == 0:
        return torch.zeros_like(img_poly)
    x_min = torch.min(img_poly[..., 0], dim=-1)[0]
    y_min = torch.min(img_poly[..., 1], dim=-1)[0]
    can_poly = img_poly.clone()
    can_poly[..., 0] = can_poly[..., 0] - x_min[..., None]
    can_poly[..., 1] = can_poly[..., 1] - y_min[..., None]
    return can_poly

def get_gcn_feature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone()
    # normalize to [-1, 1]
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly)[0].permute(1, 0, 2)
        gcn_feature[ind == i] = feature
    return gcn_feature

def prepare_testing_init(polys, ro):
    polys = polys / ro
    can_init_polys = img_poly_to_can_poly(polys)
    img_init_polys = polys
    ind = torch.zeros((img_init_polys.size(0), ), dtype=torch.int32, device=img_init_polys.device)
    init = {'img_init_polys': img_init_polys, 'can_init_polys': can_init_polys, 'py_ind': ind}
    return init

def get_local_feature(cnn_feature, img_poly, ind, h, w, kernel_size=3, \
                      with_cord=False, ro=4., c_it_poly=None, dilated=1):
    # img_poly is 1/ro of absolute coordinates
    img_poly = img_poly.clone()
    gx, gy = torch.meshgrid(torch.arange(-(kernel_size//2), kernel_size//2+1, dilated), torch.arange(-(kernel_size//2), kernel_size//2+1, dilated))
    gxy = torch.cat([gx.unsqueeze(-1), gy.unsqueeze(-1)], dim=2).reshape(-1,2).to(img_poly.device) / ro
    num_poly, num_point, _ = img_poly.shape
    img_poly = img_poly.unsqueeze(2)                        # (num_poly, num_point, 1, 2)
    img_poly_local = (img_poly + gxy).reshape(num_poly, -1, 2)    # num_poly, num_point*(kernel**2), 2
    img_poly_local[..., 0] = img_poly_local[..., 0] / (w / 2.) - 1
    img_poly_local[..., 1] = img_poly_local[..., 1] / (h / 2.) - 1
    

    batch_size = cnn_feature.size(0)
    feat_dim = cnn_feature.size(1)
    local_feature = torch.zeros([num_poly, cnn_feature.size(1), img_poly_local.size(1)]).to(img_poly.device)
    for i in range(batch_size):
        poly = img_poly_local[ind == i].unsqueeze(0)        # (1, k, num_points*kernel**2, 2)
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly, padding_mode="border")[0].permute(1, 0, 2)  # (k, dim, num_points*kernel**2)
        local_feature[ind == i] = feature
    local_feature = local_feature.reshape(num_poly, cnn_feature.size(1), num_point, -1)
    if with_cord:
        assert c_it_poly is not None
        c_it_poly = (c_it_poly * ro).unsqueeze(2)
        c_it_poly = (c_it_poly + gxy).permute(0, 3, 1, 2)   # (num_poly, 2, num_point, kernel**2)
        local_feature = torch.cat([local_feature, c_it_poly], dim=1)
    return local_feature

def get_dilated_feature(cnn_feature, img_poly, ind, h, w, dilated, kernel_size=3):
    # img_poly: absolute position to the origin image
    img_poly = img_poly.clone()
    c_it_poly = img_poly_to_can_poly(img_poly).unsqueeze(2)
    gx, gy = torch.meshgrid(torch.arange(-(kernel_size//2), kernel_size//2+1), torch.arange(-(kernel_size//2), kernel_size//2+1))
    gxy = torch.cat([gx.unsqueeze(-1), gy.unsqueeze(-1)], dim=2).reshape(-1,2).to(img_poly.device) # [9,2]
    num_poly, num_point, _ = img_poly.shape # (B, pts, 2)
    img_poly = img_poly.unsqueeze(2) # (B, pts, 1, 2)
    dilated_kernel = torch.einsum('ij,kl -> ijkl',[dilated, gxy])   # (B, pts, 9, 2)
    img_poly_local = (img_poly + dilated_kernel).reshape(num_poly, -1, 2)
    # normalize to [-1, 1]
    img_poly_local[..., 0] = img_poly_local[..., 0] / (w / 2.) - 1
    img_poly_local[..., 1] = img_poly_local[..., 1] / (h / 2.) - 1

    batch_size = cnn_feature.size(0)
    feat_dim = cnn_feature.size(1)
    local_feature = torch.zeros([num_poly, feat_dim, img_poly_local.size(1)]).to(img_poly.device)
    for i in range(batch_size):
        poly = img_poly_local[ind == i].unsqueeze(0)
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly, padding_mode="border")[0].permute(1, 0, 2)
        local_feature[ind == i] = feature
    local_feature = local_feature.reshape(num_poly, cnn_feature.size(1), num_point, -1) # (B, feat_dim, pts, 9)
    
    c_it_poly = (c_it_poly + dilated_kernel).permute(0, 3, 1, 2)
    local_feature = torch.cat([local_feature, c_it_poly], dim=1)
    return local_feature

def get_average_length(poly):
    # poly shape: (N, pts, 2), torch.tensor
    poly_dis = torch.cat((poly[:, 1:], poly[:, :1]), dim=1)
    vec = poly_dis - poly
    len_1 = torch.sqrt(torch.sum(vec ** 2, dim=2))
    len_2 = torch.cat((len_1[:, 1:], len_1[:, :1]), dim=1)
    ave_len = (len_1 + len_2) / 2
    return ave_len

def prepare_training_rcnn(ret, batch):
    # if ret['cp_wh'].shape[-1] == 256:
    #     # get initial contour by index
    #     wh_pred = ret['cp_wh'] # shape: (cp_num, 256, h, w)
    #     act_01 = batch['act_01']
    #     cp_01 = batch['cp_01'][act_01].byte()   # (cp_num)
    #     cp_ind = batch['ct_ind'][act_01][cp_01] # (cp_num)
    #     print("cp_01",cp_01.shape)
    #     print("cp_ind", cp_ind.shape)
    #     print("act_01", act_01.shape)
    #     print("wh_pred", wh_pred.shape)

    ct_01 = batch['ct_01'].byte() 
    init = {}

    if 'i_it_py' in ret:
        init.update({'i_it_py': ret['i_it_py'].detach()})
        init.update({'c_it_py': img_poly_to_can_poly(ret['i_it_py']).detach()})
        init.update({'i_gt_py': collect_training(batch['i_gt_py'], ct_01)})
    else:
        # these are from the abs coordinate, no need to times 4
        init.update({'i_it_py': collect_training(batch['i_it_py'], ct_01)})
        init.update({'c_it_py': collect_training(batch['c_it_py'], ct_01)})
        init.update({'i_gt_py': collect_training(batch['i_gt_py'], ct_01)})
        init.update({'c_gt_py': collect_training(batch['c_gt_py'], ct_01)})

    ct_num = batch['meta']['ct_num']
    # 4py_ind: indices of bounding box
    init.update({'4py_ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})
    init.update({'py_ind': init['4py_ind']})

    init['4py_ind'] = init['4py_ind'].to(ct_01.device)
    init['py_ind'] = init['py_ind'].to(ct_01.device)

    return init

def get_quadrangle(box):
    x_min, y_min, x_max, y_max = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    quadrangle = [
        (x_min + x_max) / 2., y_min,
        x_min, (y_min + y_max) / 2.,
        (x_min + x_max) / 2., y_max,
        x_max, (y_min + y_max) / 2.
    ]
    quadrangle = torch.stack(quadrangle, dim=2).view(x_min.size(0), x_min.size(1), 4, 2)
    return quadrangle


def uniform_upsample(poly, p_num):
    # 1. assign point number for each edge
    # 2. calculate the coefficient for linear interpolation
    next_poly = torch.roll(poly, -1, 2)
    edge_len = (next_poly - poly).pow(2).sum(3).sqrt()
    edge_num = torch.round(edge_len * p_num / torch.sum(edge_len, dim=2)[..., None]).long()
    edge_num = torch.clamp(edge_num, min=1)
    edge_num_sum = torch.sum(edge_num, dim=2)
    edge_idx_sort = torch.argsort(edge_num, dim=2, descending=True)
    extreme_utils.calculate_edge_num(edge_num, edge_num_sum, edge_idx_sort, p_num)
    edge_num_sum = torch.sum(edge_num, dim=2)
    assert torch.all(edge_num_sum == p_num)

    edge_start_idx = torch.cumsum(edge_num, dim=2) - edge_num
    weight, ind = extreme_utils.calculate_wnp(edge_num, edge_start_idx, p_num)
    poly1 = poly.gather(2, ind[..., 0:1].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly2 = poly.gather(2, ind[..., 1:2].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly = poly1 * (1 - weight) + poly2 * weight

    return poly

def get_octagon(ex):
    w, h = ex[..., 3, 0] - ex[..., 1, 0], ex[..., 2, 1] - ex[..., 0, 1]
    t, l, b, r = ex[..., 0, 1], ex[..., 1, 0], ex[..., 2, 1], ex[..., 3, 0]
    x = 8.

    octagon = [
        ex[..., 2, 0], ex[..., 2, 1],
        torch.min(ex[..., 2, 0] + w / x, r), ex[..., 2, 1],
        ex[..., 3, 0], torch.min(ex[..., 3, 1] + h / x, b),
        ex[..., 3, 0], ex[..., 3, 1],
        ex[..., 3, 0], torch.max(ex[..., 3, 1] - h / x, t),
        torch.min(ex[..., 0, 0] + w / x, r), ex[..., 0, 1],
        ex[..., 0, 0], ex[..., 0, 1],
        torch.max(ex[..., 0, 0] - w / x, l), ex[..., 0, 1],
        ex[..., 1, 0], torch.max(ex[..., 1, 1] - h / x, t),
        ex[..., 1, 0], ex[..., 1, 1],
        ex[..., 1, 0], torch.min(ex[..., 1, 1] + h / x, b),
        torch.max(ex[..., 2, 0] - w / x, l), ex[..., 2, 1]
    ]
    octagon = torch.stack(octagon, dim=2).view(t.size(0), t.size(1), 12, 2)

    return octagon


def prepare_testing_evolve(ex):
    if len(ex) == 0:
        i_it_pys = torch.zeros([0, 128, 2]).to(ex)
        c_it_pys = torch.zeros_like(i_it_pys)
    else:
        #print(ex.shape)  # 1*4*2
        i_it_pys = get_octagon(ex[None])
        #print(i_it_pys.shape) # 1 * 1 * 12 * 2
        i_it_pys = uniform_upsample(i_it_pys, 128)[0]
        # print("i_it_pys", i_it_pys.shape)
        # i_it_pys = uniformsample(i_it_pys, snake_config.poly_num)[0]
        #print(i_it_pys.shape)  # 1 * 128 * 2
        c_it_pys = img_poly_to_can_poly(i_it_pys)
    evolve = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys}
    return evolve