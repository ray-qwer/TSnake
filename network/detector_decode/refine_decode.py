import torch
import torch.nn as nn
from .utils import decode_ct_hm, clip_to_image, get_gcn_feature, decode_FCOS_ctr, decode_FCOS_Det_ctr, \
                    pred_ifft

class Refine(torch.nn.Module):
    def __init__(self, c_in=64, num_point=128, stride=4.):
        super(Refine, self).__init__()
        self.num_point = num_point
        self.stride = stride
        self.trans_feature = torch.nn.Sequential(torch.nn.Conv2d(c_in, 256, kernel_size=3,
                                                                 padding=1, bias=True),
                                                 torch.nn.ReLU(inplace=True),
                                                 torch.nn.Conv2d(256, 64, kernel_size=1,
                                                                 stride=1, padding=0, bias=True))
        self.trans_poly = torch.nn.Linear(in_features=((num_point + 1) * 64),
                                          out_features=num_point * 4, bias=False)
        self.trans_fuse = torch.nn.Linear(in_features=num_point * 4,
                                          out_features=num_point * 2, bias=True)

    def global_deform(self, points_features, init_polys):
        poly_num = init_polys.size(0)
        points_features = self.trans_poly(points_features)
        offsets = self.trans_fuse(points_features).view(poly_num, self.num_point, 2)
        coarse_polys = offsets * self.stride + init_polys.detach()  # TODO: why multiple with stride? no sence
        return coarse_polys

    def forward(self, feature, ct_polys, init_polys, ct_img_idx, ignore=False, hw=None):
        # ct_polys -> center points. It is from gt
        if ignore or len(init_polys) == 0:
            return init_polys
        if hw is None:
            h, w = feature.size(2), feature.size(3)
        else:
            h, w = hw
        poly_num = ct_polys.size(0)
    
        feature = self.trans_feature(feature)

        ct_polys = ct_polys.unsqueeze(1).expand(init_polys.size(0), 1, init_polys.size(2))
        points = torch.cat([ct_polys, init_polys], dim=1)
        feature_points = get_gcn_feature(feature, points, ct_img_idx, h, w).view(poly_num, -1)
        coarse_polys = self.global_deform(feature_points, init_polys)
        return coarse_polys

class Decode(torch.nn.Module):
    def __init__(self, c_in=64, num_point=128, init_stride=10., coarse_stride=4., down_sample=4., 
                    min_ct_score=0.05, use_tanh=False, get_coarse_contour=True, get_ifft=False):
        super(Decode, self).__init__()
        self.stride = init_stride
        self.get_coarse_contour = get_coarse_contour
        self.down_sample = down_sample
        self.min_ct_score = min_ct_score
        if get_coarse_contour:
            self.refine = Refine(c_in=c_in, num_point=num_point, stride=coarse_stride)
        self.use_tanh = use_tanh
        self.get_ifft = get_ifft
        self.num_point = num_point
    def train_decode(self, data_input, output, cnn_feature):
        # data_input: ground truth
        wh_pred = output['wh']
            
        ct_01 = data_input['ct_01'].bool()
        ct_ind = data_input['ct_ind'][ct_01]
        ct_img_idx = data_input['ct_img_idx'][ct_01]
        batch, _, height, width = cnn_feature.size()    
        # batch, _, height, width = data_input['ct_hm'].size()
        ct_x, ct_y = ct_ind % width, torch.div(ct_ind, width, rounding_mode='floor')
        ct_x = ct_x.clip(0, width - 1)
        ct_y = ct_y.clip(0, height - 1)

        if len(wh_pred.shape) == 3: # from FCOS
            wh_pred = wh_pred.reshape(batch, -1, height, width)

        if ct_x.size(0) == 0:   # no instances
            ct_offset = wh_pred[ct_img_idx, :, ct_y, ct_x].view(ct_x.size(0), 1, 2)
        else:   # countable instances
            assert torch.max(ct_img_idx.unique()) <= cnn_feature.size(0), f"ct_img_idx: {ct_img_idx}"
            ct_offset = wh_pred[ct_img_idx, :, ct_y, ct_x].view(ct_x.size(0), -1, 2)
        
        ######### NEW ADDED ##########
        if self.get_ifft:
            output.update({'fft_coef': ct_offset})
            ct_offset = pred_ifft(ct_offset, self.num_point)
            assert ct_offset.shape[1] == self.num_point, f'ct_offset: {ct_offset.shape}'
        ##############################
        
        ct_x, ct_y = ct_x.unsqueeze(1).to(torch.float32), ct_y.unsqueeze(1).to(torch.float32)
        # ct_x, ct_y = ct_x[:, None].to(torch.float32), ct_y[:, None].to(torch.float32)   # [:, None] means unsqueeze
        ct = torch.cat([ct_x, ct_y], dim=1)
        ######## NEW ADDED ##########
        output.update({'ct': ct})
        #############################
        # stride: ratio of minimized
        init_polys = ct_offset * self.stride + ct.unsqueeze(1).expand(ct_offset.size(0),
                                                                      ct_offset.size(1), ct_offset.size(2))
        if self.get_coarse_contour:
            coarse_polys = self.refine(cnn_feature, ct, init_polys, ct_img_idx.clone())
            output.update({'poly_coarse': coarse_polys * self.down_sample})

        output.update({'poly_init': init_polys * self.down_sample})
        return

    def test_decode_FCOS(self, cnn_feature, output, anchors, img_num=1, K=100, min_ct_score=0.05, ignore_gloabal_deform=False):
        cls_pred, ctrness_pred, wh_pred = output["cls_logits"], output["poly_ctrness"], output["wh"]
        det_poly = decode_FCOS_ctr(output, anchors, img_num, min_ct_score=min_ct_score, K=K, stride=self.stride)
        
        if len(det_poly):
            det_poly = det_poly[0]
            batch = 1
        else:
            batch = len(det_poly)

        poly_init = det_poly["poly"]
        scores = det_poly["scores"]
        labels = det_poly["labels"]
        ct = det_poly["ct"]
                
        K, _ = ct.shape
        poly_init = poly_init.view(batch, K, -1, 2)
        scores = scores.view(batch, K, 1)
        labels = labels.view(batch, K, 1).float()
        ct = ct.view(batch, K, 2)

        detection = torch.cat([ct, scores, labels], dim=2)
        # valid, score over some threshold
        valid = detection[0,:, 2] >= self.min_ct_score   # maybe filter too much?
        poly_init, detection = poly_init[0][valid], detection[0][valid]
        
        # initial poly
        
        init_polys = clip_to_image(poly_init, cnn_feature.size(2), cnn_feature.size(3))
        output.update({'poly_init': init_polys * self.down_sample})

        img_id = torch.zeros((len(poly_init), ), dtype=torch.int64)
        poly_coarse = self.refine(cnn_feature, detection[:, :2], poly_init, img_id, ignore=ignore_gloabal_deform)
        coarse_polys = clip_to_image(poly_coarse, cnn_feature.size(2), cnn_feature.size(3))
        output.update({'poly_coarse': coarse_polys * self.down_sample})
        output.update({'detection': detection})
        return
    
    def test_decode(self, cnn_feature, output, K=100, min_ct_score=0.05, ignore_gloabal_deform=False):
        hm_pred, wh_pred = output['ct_hm'], output['wh']
        poly_init, detection = decode_ct_hm(torch.sigmoid(hm_pred), wh_pred,
                                            K=K, stride=self.stride, get_ifft=self.get_ifft, num_point=self.num_point)
        # detection include [instances,(center_x, center_y, scores, classes)]
        """
            TODO: from batch_size = 1 to batch_size = n
                what's new:
                    ct_img_idx: [n_instance, img_idx], img_idx < n
                    ct_01: to show which polys need to be used
        """
        batch = hm_pred.size(0)
        img_idx = torch.arange(batch, dtype=torch.int64, device=cnn_feature.device).repeat_interleave(K)
        num_instance = detection.size(1)
        num_points = poly_init.size(2)
        detection = detection.reshape(batch * num_instance, 4)
        poly_init = poly_init.reshape(batch * num_instance, num_points, 2)
        valid = detection[:, 2] >= min_ct_score
        poly_init, detection = poly_init[valid], detection[valid]
        img_idx = img_idx[valid]

        init_polys = clip_to_image(poly_init, cnn_feature.size(2), cnn_feature.size(3))
        output.update({'poly_init': init_polys * self.down_sample})

        img_id = torch.zeros((len(poly_init), ), dtype=torch.int64)
        
        if self.get_coarse_contour:
            poly_coarse = self.refine(cnn_feature, detection[:, :2], poly_init, img_id, ignore=ignore_gloabal_deform)
            coarse_polys = clip_to_image(poly_coarse, cnn_feature.size(2), cnn_feature.size(3))
            output.update({'poly_coarse': coarse_polys * self.down_sample})
        output.update({'detection': detection})
        output.update({'img_idx': img_idx})
        return

    def forward(self, data_input, cnn_feature, output=None, is_training=True, ignore_gloabal_deform=False):
        if is_training:
            self.train_decode(data_input, output, cnn_feature)
        else:
            self.test_decode(cnn_feature, output, min_ct_score=self.min_ct_score,
                             ignore_gloabal_deform=ignore_gloabal_deform)


class FCOSDecode(Decode):
    """
        This class inherit from Decode
    """
    def __init__(self, c_in=64, num_point=128, init_stride=10., coarse_stride=4.,
                        down_sample=4., min_ct_score=0.05, use_tanh=False, anchor_origin_size=False):
        super(FCOSDecode, self).__init__(c_in=c_in, num_point=num_point, init_stride=init_stride, coarse_stride=coarse_stride, down_sample=down_sample, min_ct_score=min_ct_score)
        self.anchor_origin_size = anchor_origin_size
        self.use_tanh = use_tanh
        self.init_contour_head = nn.Sequential(
            nn.Conv2d(c_in, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_point*2, kernel_size=3, padding=1, bias=True)
        )
        self.init_predictor = nn.Sequential(
            nn.Linear(c_in, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_point*2, bias=False)
        )
        self.coarse_predictor = nn.Sequential(
            nn.Linear(c_in*(num_point+1), 2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_point*2, bias=False)
        )
        """
            regress_range: the range of extreme length that the layer of features could regress
                            if there is only one layer, the regress_range should be (-1, 1e6)
            strides: the down ratio of each layers
        """
        self.regress_ranges = ((-1, 64), (64, 128), (128, 256), (256, 512), (512, 1e6))
        self.strides=[8, 16, 32, 64, 128]
        self.num_point = num_point
    
    def init_weights(self):
        # avoid init_cfg overwrite the initialization of `conv_offset`
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    def extract_features(self, ms_feats, points, img_h, img_w, img_inds, fl_inds):
        """
            points: (x, y)
            fl_inds: feature layer index, as same as ms_ind
            ms_feats: pyramid features    
        """
        num_points = points.size(0)
        ms_points = []
        ms_img_inds = []
        """
            ms_points and ms_img_inds:
                [tensor(), tensor(), ... ] the length is len(feature)
        """
        for i in range(len(ms_feats)):
            ms_points.append(points[fl_inds == i])
            ms_img_inds.append(img_inds[fl_inds == i])
        points_features = torch.zeros([num_points, ms_feats[0].size(1), points.size(1)]).to(ms_feats[0].device)
        
        for i in range(len(ms_feats)):
            ms_points_feature = get_gcn_feature(ms_feats[i], ms_points[i], ms_img_inds[i], img_h, img_w)
            points_features[fl_inds == i] = ms_points_feature
        
        return points_features

    def get_targets(self, gt_contours, gt_centers, gt_whs, contour_proposals, strides):
        gt_centers = gt_centers.unsqueeze(1).repeat(1, self.num_point, 1)
        assert gt_contours.size(1) >= self.num_point and gt_contours.size(1) >= self.num_point
        if self.use_tanh:
            normed_init_offset_target = (gt_contours - gt_centers) / gt_whs.unsqueeze(1)
            normed_global_offset_target = (gt_contours - contour_proposals) / gt_whs.unsqueeze(1)
        else:
            num_instance = len(strides)
            normed_init_offset_target = (gt_contours - gt_centers) / strides.reshape(num_instance, 1, 1)
            normed_global_offset_target = (gt_contours - contour_proposals) / strides.reshape(num_instance, 1, 1)
        return normed_init_offset_target.detach(), normed_global_offset_target.detach()

    def contour_forward(self, feats, centers, whs, img_h, img_w, inds):
        """
            inds: ct_img_inds
        """
        num_instance = centers.size(0)
        if num_instance == 0:
            gt_max_lengths = whs[..., :1]
        else:
            gt_max_lengths = torch.max(whs, dim=-1, keepdim=True)[0]
        regress_ranges = torch.zeros((num_instance, len(self.regress_ranges), 2), dtype=torch.int64, device=centers.device)
        
        for i, regress_range in enumerate(self.regress_ranges):
            regress_ranges[:, i, 0] = regress_range[0]
            regress_ranges[:, i, 1] = regress_range[1]
        
        ms_inds = torch.arange(0, len(feats), device=centers.device,
                               dtype=torch.int64).unsqueeze(0)

        ms_inds = torch.logical_and(regress_ranges[..., 0] < gt_max_lengths,
                                    regress_ranges[..., 1] >= gt_max_lengths).to(torch.int64) * ms_inds # use which layers
        ms_inds = torch.sum(ms_inds, dim=1)
        strides = torch.Tensor(self.strides).to(centers.device)[ms_inds]
        assert centers.size(0) == ms_inds.size(0) and centers.size(1) == 2, f"center:{centers.shape}, ms_ind:{ms_inds.shape}"
        centers_features = self.extract_features(feats, centers.unsqueeze(1), # unsqueeze for get gcn features
                                                 img_h, img_w, inds, ms_inds).squeeze(-1)
        # centers_pred = centers_pred.reshape(num_instance, self.num_point, 2)
        # go through linear layer directly
        centers_pred = self.init_predictor(centers_features).reshape(num_instance, self.num_point, 2)
        if self.use_tanh:
            centers_pred = torch.tanh(centers_pred)
            init_contour = centers.unsqueeze(1) + centers_pred * whs.unsqueeze(1)
        else:
            init_contour = centers.unsqueeze(1) + centers_pred * strides.view(num_instance, 1, 1) # different strides for different layers
        init_contour = init_contour.reshape(num_instance, self.num_point, 2)
        
        coarse_contour = self.refine(feats[0], centers, init_contour, inds, hw=(img_h, img_w))
        coarse_contour_embed = coarse_contour - init_contour.detach()
        
        return init_contour, coarse_contour, centers_pred, coarse_contour_embed, strides

    def train_decode(self, data_input, output, cnn_feature):
        """
            from cnn_features to generate initial contour first
        """
        ct_01 = data_input['ct_01'].bool()
        ct_ind = data_input['ct_ind'][ct_01]
        ct_img_idx = data_input['ct_img_idx'][ct_01]
        img_h, img_w = data_input['inp'].shape[-2:]
        gt_polys = data_input['img_gt_polys'][ct_01]
        gt_bbox = data_input['bbox'] # ground truth, 800*800, N,4
        gt_whs = (gt_bbox[...,2:] - gt_bbox[...,:2]) / 2.
        gt_centers = (gt_bbox[...,2:] + gt_bbox[...,:2]) / 2.
        
        init_contour, coarse_contour, normed_init_offset_pred, normed_coarse_offset_pred, strides = self.contour_forward(cnn_feature, gt_centers, gt_whs, img_h, img_w, ct_img_idx)
        normed_init_offset_target, normed_coarse_offset_target = self.get_targets(gt_polys, gt_centers, gt_whs, init_contour, strides)
        output.update({'poly_init': init_contour})
        output.update({'poly_coarse': coarse_contour})
        output.update({'normed_init_offset_pred': normed_init_offset_pred})
        output.update({'normed_init_offset_target': normed_init_offset_target})
        output.update({'normed_coarse_offset_pred': normed_coarse_offset_pred})
        output.update({'normed_coarse_offset_target': normed_coarse_offset_target})

        # super().train_decode(data_input, output, cnn_feature)
        return

    def test_decode_FCOS(self, cnn_feature, output, anchors, wh, img_num=1, K=100, min_ct_score=0.05, ignore_gloabal_deform=False):
        # wh = self.init_contour_head(cnn_feature)
        # if self.use_tanh:
        #     wh = nn.functional.tanh(wh)
            
        # output.update({'wh': wh})
        img_w, img_h = wh
        det_poly = decode_FCOS_Det_ctr(output, anchors, img_num, min_ct_score=min_ct_score, K=K, stride=self.stride, down_sample=self.down_sample if self.anchor_origin_size else 1)

        if len(det_poly):
            # if det_poly is a list
            det_poly = det_poly[0]
            batch = 1
        else:
            batch = len(det_poly)
        # poly_init = det_poly["poly_init"]
        bbox = det_poly['poly']
        scores = det_poly["scores"]
        labels = det_poly["labels"]
        ct = det_poly["ct"]
        
        K, _ = ct.shape
        scores = scores.view(batch, K, 1)
        labels = labels.view(batch, K, 1).float()
        ct = ct.view(batch, K, 2)
        detection = torch.cat([ct, scores, labels], dim=2)
        valid = detection[0,:,2] >= self.min_ct_score
        detection = detection[0][valid]
        bbox = bbox[valid]
        ct = ct[0, valid]
        inds = torch.zeros(bbox.size(0), dtype=torch.int)
        pred_whs = (bbox[...,1,:] - bbox[...,0,:]) / 2.
        init_contour, coarse_contour, _, _, _ = self.contour_forward(cnn_feature, ct, pred_whs, img_h, img_w, inds)
        
        
        output.update({'poly_init': init_contour})
        output.update({'poly_coarse': coarse_contour})
        output.update({'detection': detection})
        output.update({'bbox_out': bbox})
        return