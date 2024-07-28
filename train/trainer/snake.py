import torch.nn as nn
from ..losses import FocalLoss, DMLoss, sigmoid, DetLoss, FCOSLoss, MaskRasterizationLoss, \
                     CBLoss, DiceLoss, WeightedSmoothL1Loss, ShapeLoss, BMLoss, AngularLoss, \
                     FourierLoss
import torch


class NetworkWrapper(nn.Module):
    def __init__(self, net, with_dml=True, with_wsll=False, start_epoch=10, weight_dict=None, use_Mask_Loss=False, get_instance_mask=False):
        super(NetworkWrapper, self).__init__()
        print("Initial Wrapper Init")
        self.with_dml = with_dml
        self.with_wsll = with_wsll
        self.net = net
        self.ct_crit = FocalLoss()
        self.py_crit = torch.nn.functional.smooth_l1_loss
        self.shape_loss = ShapeLoss()
        self.weight_dict = weight_dict
        self.start_epoch = start_epoch
        if with_dml:
            self.dml_crit = DMLoss(type='smooth_l1', kernel_size=7, sigma=0.6)
        else:
            self.dml_crit = self.py_crit
        
        if with_wsll:
            self.weighted_py_crit = WeightedSmoothL1Loss(kernel_size=7, sigma=0.6)
            
        self.use_Mask_Loss = use_Mask_Loss
        if use_Mask_Loss:
            self.loss_contour_mask = MaskRasterizationLoss()    # maybe change resolution
            self.get_instance_mask = get_instance_mask
        self.train_combine = False
        self.train_bd = False
        if hasattr(self.net, "combine"):
            self.train_combine = True
            self.cb_loss = CBLoss(loss_type="focal")
            self.bbox_loss = torch.nn.functional.smooth_l1_loss
        if hasattr(self.net, "bd_head"):
            self.train_bd = True
            # self.bd_loss = DiceLoss()
            self.bd_loss = FocalLoss(do_sigmoid=True)
            

    def compute_loss_contour_mask(self, polys, gt_masks, gt_bboxes, names):
        if isinstance(polys, list):
            assert isinstance(names, list)
        else:
            if isinstance(names, str):
                names = [names]
            polys = [polys]
        ret = dict()
        num_mask = torch.tensor(len(gt_bboxes), dtype=torch.float, device=gt_bboxes.device)
        ## distributed learning
        # num_mask = max(reduce_mean(num_mask), 1.0)
        for idx, (poly, name) in enumerate(zip(polys, names)):    # polys: [1st iter: tensors(b,n,128,2), 2nd iter: tensors(b,n,128,2), ...]
            ret.update({name: self.loss_contour_mask(poly, gt_masks, gt_bboxes, avg_factor=num_mask, is_gt_mask=self.get_instance_mask)})
        return ret

    def forward(self, batch):
        output = self.net(batch['inp'], batch)
        if 'test' in batch['meta']:
            return output
        epoch = batch['epoch']
        scalar_stats = {}
        loss = 0.

        keyPointsMask = batch['keypoints_mask'][batch['ct_01']]
        
        ct_loss = self.ct_crit(sigmoid(output['ct_hm']), batch['ct_hm'])
        scalar_stats.update({'ct_loss': ct_loss})
        loss += ct_loss * self.weight_dict['ct_loss']

        num_polys = len(output['poly_init'])
        if num_polys == 0:
            init_py_loss = torch.sum(output['poly_init']) * 0.
            coarse_py_loss = torch.sum(output['poly_coarse']) * 0.
        else:
            init_py_loss = self.py_crit(output['poly_init'], output['img_gt_polys'])
            coarse_py_loss = self.py_crit(output['poly_coarse'], output['img_gt_polys'])
        scalar_stats.update({'init_py_loss': init_py_loss})
        scalar_stats.update({'coarse_py_loss': coarse_py_loss})
        loss += init_py_loss * self.weight_dict['init']
        loss += coarse_py_loss * self.weight_dict['coarse']

        py_loss = 0
        shape_loss = 0
        # circular convolution
        n = len(output['py_pred']) -1 if self.with_dml else len(output['py_pred'])
        target_dis = torch.cat( (output['img_gt_polys'][:,1:],  output['img_gt_polys'][:,:1]), dim=1)
        target_shape =  target_dis - output['img_gt_polys']
        for i in range(n):
            if num_polys == 0:
                part_py_loss = torch.sum(output['py_pred'][i]) * 0.0
                shape_part_loss = torch.sum(output['py_pred'][i]) * 0.0
            else:
                if self.with_wsll:
                    part_py_loss = self.weighted_py_crit(output['py_pred'][i], output['img_gt_polys'], keyPointsMask)
                else:
                    part_py_loss = self.py_crit(output['py_pred'][i], output['img_gt_polys'])
                shape_part_loss = self.shape_loss(output['py_pred'][i], target_shape=target_shape)
            py_loss += part_py_loss / len(output['py_pred'])
            shape_loss += shape_part_loss
            scalar_stats.update({'py_loss_{}'.format(i): part_py_loss})

        loss += py_loss * self.weight_dict['evolve']

        # local moving
        if 'py_local_pred' in output:
            n = len(output['py_local_pred'])
            for i in range(n):
                if num_polys == 0:
                    part_py_loss = torch.sum(output['py_local_pred'][i]) * 0.0
                else:
                    part_py_loss = self.py_crit(output['py_local_pred'][i], output['img_gt_polys'])
                py_loss += part_py_loss / len(output['py_local_pred'])
                scalar_stats.update({'py_local_loss_{}'.format(i): part_py_loss})
            loss += py_loss * self.weight_dict['evolve_local']

        if self.with_dml and epoch >= self.start_epoch and num_polys != 0:
            dm_loss = self.dml_crit(output['py_pred'][-2],
                                    output['py_pred'][-1],
                                    output['img_gt_polys'],
                                    keyPointsMask)
            shape_loss += self.shape_loss(output['py_pred'][i], target_shape=target_shape)
            scalar_stats.update({'end_set_loss': dm_loss})

            loss += dm_loss / len(output['py_pred']) * self.weight_dict['evolve']
        else:
            dm_loss = torch.sum(output['py_pred'][-1]) * 0.0
            scalar_stats.update({'end_set_loss': dm_loss})
            loss += dm_loss / len(output['py_pred']) * self.weight_dict['evolve']
        
        shape_loss = shape_loss / (n+1) if self.with_dml and epoch >= self.start_epoch else shape_loss / n
        scalar_stats.update({'shape_loss': shape_loss})
        loss += shape_loss
        
        # mask loss update
        if self.use_Mask_Loss:
            n = len(output['py_local_pred'])
            names = [f'local_mask_loss_{i}' for i in range(1,n+1)]
            if self.get_instance_mask:
                gt_bboxes = batch['bbox']
                gt_masks = batch['mask']
                loss_mask = self.compute_loss_contour_mask(output['py_local_pred'], gt_masks, gt_bboxes, names)
            else:
                gt_polys = output['img_gt_polys']
                gt_bboxes = batch['bbox']
                loss_mask = self.compute_loss_contour_mask(output['py_local_pred'], gt_polys, gt_bboxes, names)
            scalar_stats.update(loss_mask)
            contour_loss = sum(loss_mask.values()) * self.weight_dict['evolve_mask']
            loss += contour_loss
        
        # combine loss
        if self.train_combine:
            if 'ct_01' in batch:
                ct_01 = batch['ct_01']
            else:
                ct_01 = output['ct_01']
            combine_pred = output['combine'][ct_01]
            multicomp_target = batch['multicomp']
            cb_loss = self.cb_loss(combine_pred, multicomp_target, ct_01)
            scalar_stats.update({'cb_loss': cb_loss.item()})
            loss += cb_loss * self.weight_dict['combine']
            if 'bbox' in output:
                pred_bbox = output['bbox']
                if num_polys == 0:
                    bbox_loss = torch.sum(pred_bbox) * 0.0
                # bbox loss
                else:
                    gt_bbox = batch['bbox']
                    bbox_loss = self.bbox_loss(pred_bbox, gt_bbox)
                    scalar_stats.update({'bbox_loss': bbox_loss})
                    loss += bbox_loss
        if self.train_bd:
            gt_bd = batch['bd_map']
            pred_bd = output['bd']
            bd_loss = self.bd_loss(pred_bd, gt_bd)
            scalar_stats.update({'bd_loss': bd_loss.item()})
            loss += bd_loss * self.weight_dict['bd']
        scalar_stats.update({'loss': loss})

        return output, loss, scalar_stats

# class CombineNetworkWrapper(nn.Module):
#     def __init__(self, net, gt_from_dataset=True):
#         super(CombineNetworkWrapper, self).__init__()
#         self.net = net
#         self.cb_crit = FocalLoss()
    
#     def prepare_target(self, comp_target, instance_num):
#         """
#             the same components are in the same index
#             -1 means there is no components, like 0 in ct_01
#             input:
#                 comp_target:
#                     shape: (b, max_len)
#             output:
#                 shape: (instance, max_len)
#         """
#         batch_size, max_len = comp_target.shape
#         comp_matrix = torch.zeros(instance_num, max_len, dtype=torch.float32, device=comp_target.device)
#         idx, row = 0, 0
#         while idx < batch_size:
#             target = comp_target[idx]
#             for tgt in target:
#                 if tgt == -1:
#                     break
#                 comp_matrix[row] = (target == tgt)
#                 row += 1
#             idx += 1
#         return comp_matrix

#     def forward(self, batch):
#         output = self.net(batch['inp'], batch)
#         if 'test' in batch['meta']:
#             return output
#         epoch = batch['epoch']
#         scalar_stats = {}
#         loss = 0.

#         if 'ct_01' in batch:
#             ct_01 = batch['ct_01']
#         else:
#             ct_01 = output['ct_01']
#         instance_num = torch.sum(ct_01)
#         combine_pred = output['combine'][ct_01]
#         multicomp_target = batch['multicomp']

#         multicomp_matrix = self.prepare_target(multicomp_target, instance_num)
#         if instance_num == 0:
#             combine_loss = torch.sum(combine_pred) * 0.0
#         else:
#             combine_loss = self.cb_crit(combine_pred, multicomp_matrix)
#         scalar_stats.update({'cb_loss': combine_loss.item()})
#         loss += combine_loss
#         return output, loss, scalar_stats

class ShareWeightNetworkWrapper(nn.Module):
    def __init__(self, net, weight_dict=None, start_epoch=10, start_module=3+1, num_points=128, num_points_fft=32):
        super().__init__()
        # NOTE: polysnake does have a boundary head to predict boundary
        self.net = net
        self.ct_crit = FocalLoss()
        self.py_crit = torch.nn.functional.smooth_l1_loss
        self.wsll_crit = WeightedSmoothL1Loss(kernel_size=7, sigma=0.6)
        self.bd_crit = FocalLoss()
        self.shape_loss = ShapeLoss()
        self.dml_crit = DMLoss(type='smooth_l1', kernel_size=7, sigma=0.6)  # 0,1 -> dmloss
        self.start_epoch = start_epoch
        assert start_module >= 0
        self.start_module = start_module
        self.ro = 4.
        self.stride = 10.
    def forward(self, batch):
        output = self.net(batch['inp'], batch)
        if 'test' in batch['meta']:
            return output
        
        epoch = batch['epoch']
        scalar_stats = {}
        loss = 0.
        
        keyPointsMask = batch['keypoints_mask'][batch['ct_01']]
        
        ct_loss = self.ct_crit(sigmoid(output['ct_hm']), batch['ct_hm'])
        scalar_stats.update({'ct_loss': ct_loss})
        loss += ct_loss * 1.

        keyPointsMask = batch['keypoints_mask'][batch['ct_01']]
        
        num_polys = len(output['poly_init'])
        if num_polys == 0:
            init_py_loss = torch.sum(output['poly_init']) * 0.
        else:
            init_py_loss = self.py_crit(output['poly_init'], output['img_gt_polys'])
                
        scalar_stats.update({'init_py_loss': init_py_loss})
        loss += init_py_loss * 0.1
        
        # bd loss 
        if 'bd' in output:
            bd_loss = self.bd_crit(torch.sigmoid(output['bd']), batch['bd_map'])
            scalar_stats.update({'bd_loss': bd_loss})
            loss += bd_loss

        py_loss = 0.
        shape_loss = 0.
        dm_loss = 0.
        if 'py_local_pred' in output:
            local_loss = 0.
        # circular convolution
        # n_pred = len(output['py_pred']) - 1 if self.local_refine else len(output['py_pred'])
        n_pred = len(output['py_pred'])
        gt_dis = torch.cat((output['img_gt_polys'][:,1:], output['img_gt_polys'][:,0].unsqueeze(1)), dim=1)
        tar_shape = gt_dis - output['img_gt_polys']
        for i in range(n_pred):
            i_weight = 0.8**(n_pred - i - 1)
            shape_loss += i_weight * self.shape_loss(output['py_pred'][i], target_shape=tar_shape)           
            
            if (epoch + 1) > self.start_epoch:
                if i > self.start_module:
                    dm_loss += i_weight * self.dml_crit(output['py_pred'][i-1], output['py_pred'][i], output['img_gt_polys'], keyPointsMask)
                else:
                    py_loss += i_weight * self.py_crit(output['py_pred'][i], output['img_gt_polys'])
            else:
                py_loss += i_weight * self.py_crit(output['py_pred'][i], output['img_gt_polys'])
        

        if (epoch + 1) > self.start_epoch:
            n_pred_for_py = self.start_module + 1
            py_loss = py_loss / n_pred_for_py
            dm_loss = dm_loss / (n_pred - n_pred_for_py)
        else:
            py_loss = py_loss / n_pred
        # py_loss = py_loss / n_pred

        shape_loss = shape_loss / n_pred
        scalar_stats.update({'py_loss': py_loss})
        scalar_stats.update({'shape_loss': shape_loss})
        scalar_stats.update({'dm_loss': dm_loss})
        if 'py_local_pred' in output:
            # local_loss = local_loss / n_pred
            scalar_stats.update({'local_loss': local_loss})
            loss += local_loss
        loss += py_loss
        loss += shape_loss
        loss += dm_loss
        
        scalar_stats.update({'loss': loss})

        return output, loss, scalar_stats
