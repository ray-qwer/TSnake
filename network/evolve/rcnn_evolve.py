import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .evolve import Evolution_TimeEmbed_Dilated, Evolution_TimeEmbed
from .utils import prepare_training_rcnn as prepare_training, img_poly_to_can_poly, get_gcn_feature,\
                    get_local_feature, get_average_length, get_dilated_feature, get_quadrangle,\
                    prepare_testing_evolve

class Evolution_RCNN_no_dilated(Evolution_TimeEmbed):
    def __init__(self, evolve_iter_num=8, feature_dim=64, evolve_stride=1., ro=4., use_GN=False, low_level_feat_dim=0):
        super().__init__(evolve_iter_num, feature_dim, evolve_stride, ro, use_GN, low_level_feat_dim)

    def prepare_training(self, output, batch):
        init = prepare_training(output, batch)
        output.update({'i_it_py': init['i_it_py']})
        output.update({'i_gt_py': init['i_gt_py']})
        return init

    def get_quadrangle(self, box):
        x_min, y_min, x_max, y_max = box[:,0], box[:,1], box[:,2], box[:,3] 
        a = torch.stack(((x_min + x_max) / 2., y_min), dim=-1)
        b = torch.stack((x_min, (y_min + y_max) / 2.), dim=-1)
        c = torch.stack(((x_min + x_max) / 2., y_max), dim=-1)
        d = torch.stack((x_max, (y_min + y_max) / 2.), dim=-1)
        qua = torch.stack((a,b,c,d), dim=1)

        return qua

    def prepare_testing_init(self, output):
        # ct = torch.cat(((output['cp_box'][None][..., 0] + output['cp_box'][None][...,2]).unsqueeze(-1), (output['cp_box'][None][...,1] + output['cp_box'][None][...,3]).unsqueeze(-1)),-1)/2.
        ## get_init -> 
        i_it_4py = get_quadrangle(output['cp_box'][None])
        ind = output['roi_ind'][output['cp_ind'].long()]
        init = {'ind': ind}
        output.update({'qua': i_it_4py[0]})
        return init

    def prepare_testing_evolve(self, output, h, w):
        ex = output['qua']
        ex[..., 0] = torch.clamp(ex[..., 0], min=0, max=w-1)
        ex[..., 1] = torch.clamp(ex[..., 1], min=0, max=h-1)
        evolve = prepare_testing_evolve(ex)

        output.update({'it_py': evolve['i_it_py']})
        return evolve

    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        i_poly_fea = snake(init_input)
        return i_poly_fea

    def forward(self, output, cnn_feature, batch):
        ret = output
        if batch is not None and 'test' not in batch['meta']:
            with torch.no_grad():
                init = self.prepare_training(output, batch)

            ## py_pred
            py_pred = init['i_it_py'] * self.ro
            py_preds = []

            for i in range(self.iter):
                py_pred = py_pred
                c_py_pred = img_poly_to_can_poly(py_pred)
                time = self.iter - i -1
                py_pred = super().evolve_poly(self.evolve_gcn, cnn_feature, py_pred, c_py_pred, 
                                        init['py_ind'], time=time, stride=self.evolve_stride, )
                py_preds.append(py_pred)

            ret.update({'py_pred': py_preds, 'i_gt_py': output['i_gt_py']})
            
        if not self.training:
            with torch.no_grad():
                init = self.prepare_testing_init(output)
                # print("init", init)
                evolve = self.prepare_testing_evolve(output, cnn_feature.size(2)*4, cnn_feature.size(3)*4)
                py = evolve['i_it_py'] * self.ro
                pys = []


                for i in range(self.iter):
                    c_py = img_poly_to_can_poly(py)
                    time = self.iter - i - 1
                    py = super().evolve_poly(self.evolve_gcn, cnn_feature, py, c_py, init['ind'],
                                    time=time, stride=self.evolve_stride,)
                    pys.append(py)
                
                pys[-1] = self.clip_to_img(pys[-1], cnn_feature.size(2), cnn_feature.size(3))
                ret.update({'py': pys})
                ret.update({'py_ind': init['ind'].detach()})
                
        return output


class Evolution_RCNN(Evolution_TimeEmbed_Dilated):
    def __init__(self, evolve_iter_num=8, feature_dim=64, evolve_stride=1., ro=4., use_GN=False, low_level_feat_dim=0, dilated_size=[5,15,25], restrict=[15,40]):
        super().__init__(evolve_iter_num, feature_dim, evolve_stride, ro, use_GN, low_level_feat_dim, dilated_size, restrict)
    def prepare_training(self, output, batch):
        init = prepare_training(output, batch)
        if output['cp_wh'].shape[-1] == 2:
            output.update({'i_it_py': init['i_it_py']})
        output.update({'i_gt_py': init['i_gt_py']})
        return init

    def get_quadrangle(self, box):
        x_min, y_min, x_max, y_max = box[:,0], box[:,1], box[:,2], box[:,3] 
        a = torch.stack(((x_min + x_max) / 2., y_min), dim=-1)
        b = torch.stack((x_min, (y_min + y_max) / 2.), dim=-1)
        c = torch.stack(((x_min + x_max) / 2., y_max), dim=-1)
        d = torch.stack((x_max, (y_min + y_max) / 2.), dim=-1)
        qua = torch.stack((a,b,c,d), dim=1)

        return qua

    def prepare_testing_init(self, output):
        # ct = torch.cat(((output['cp_box'][None][..., 0] + output['cp_box'][None][...,2]).unsqueeze(-1), (output['cp_box'][None][...,1] + output['cp_box'][None][...,3]).unsqueeze(-1)),-1)/2.
        ## get_init
        init = {}
        if 'i_it_py' in output:     
            init = {'i_it_py': output['i_it_py'].reshape(-1, 128, 2)}
        else:
            i_it_4py = get_quadrangle(output['cp_box'][None])
            output.update({'qua': i_it_4py[0]})
        ind = output['roi_ind'][output['cp_ind'].long()]
        init.update({'ind': ind})
        return init

    def prepare_testing_evolve(self, output, h, w):
        ex = output['qua']
        ex[..., 0] = torch.clamp(ex[..., 0], min=0, max=w-1)
        ex[..., 1] = torch.clamp(ex[..., 1], min=0, max=h-1)
        evolve = prepare_testing_evolve(ex)

        output.update({'i_it_py': evolve['i_it_py']})
        return evolve

    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        i_poly_fea = snake(init_input)
        return i_poly_fea

    def forward(self, output, cnn_feature, batch):
        ret = output
        is_poly = False
        if 'i_it_py' in output:
            is_poly = True
        if batch is not None and 'test' not in batch['meta']:
            with torch.no_grad():
                init = self.prepare_training(output, batch)

            ## py_pred
            if is_poly:
                py_pred = init['i_it_py']
            else:
                py_pred = init['i_it_py'] * self.ro
            py_preds = []
            dilated = self.get_dilated_kernel(py_pred)
            py_pred = self.evolve_dilated_poly(self.dilated_evolve, cnn_feature, py_pred, 
                                            init['py_ind'], dilated=dilated, stride=self.evolve_stride)

            py_preds.append(py_pred)
            for i in range(self.iter):
                py_pred = py_pred
                c_py_pred = img_poly_to_can_poly(py_pred)
                time = self.iter - i -1
                py_pred = super().evolve_poly(self.evolve_gcn, cnn_feature, py_pred, c_py_pred, 
                                        init['py_ind'], time=time, stride=self.evolve_stride, )
                py_preds.append(py_pred)

            ret.update({'py_pred': py_preds, 'i_gt_py': output['i_gt_py']})
            
        if not self.training:
            with torch.no_grad():
                init = self.prepare_testing_init(output)
                # print("init", init)
                if not is_poly:
                    evolve = self.prepare_testing_evolve(output, cnn_feature.size(2)*4, cnn_feature.size(3)*4)
                    py = evolve['i_it_py'] * self.ro
                else:
                    py = init['i_it_py']
                pys = []
                dilated = self.get_dilated_kernel(py)
                py = self.evolve_dilated_poly(self.dilated_evolve, cnn_feature, py, 
                                            init['ind'], dilated=dilated, stride=self.evolve_stride)
                pys.append(py)

                for i in range(self.iter):
                    c_py = img_poly_to_can_poly(py)
                    time = self.iter - i - 1
                    py = super().evolve_poly(self.evolve_gcn, cnn_feature, py, c_py, init['ind'],
                                    time=time, stride=self.evolve_stride,)
                    pys.append(py)
                
                pys[-1] = self.clip_to_img(pys[-1], cnn_feature.size(2), cnn_feature.size(3))
                ret.update({'py': pys})
                ret.update({'py_ind': init['ind'].detach()})
                
        return output

