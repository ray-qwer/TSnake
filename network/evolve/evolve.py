import torch.nn as nn
from .snake import Snake, LocalMovingSnake, LocalMovingSepSnake, AttentiveLocalMoving, TransFeat, \
                    DilatedCNNSnake, dilatedLocalMoving, ResDilatedCNNSnake
from .utils import prepare_training, prepare_testing_init, img_poly_to_can_poly, get_gcn_feature,\
                    get_local_feature, get_average_length, get_dilated_feature
import torch
from .time_embed import timeSnake, dilatedTimeSnake, pos_encoding, trainable_pos_encoding, timeLocalMoving

class Evolution(nn.Module):
    def __init__(self, evolve_iter_num=3, feature_dim=64, evolve_stride=1., ro=4., use_GN=False):
        super(Evolution, self).__init__()
        assert evolve_iter_num >= 1
        self.evolve_stride = evolve_stride
        self.ro = ro
        self.evolve_gcn = Snake(state_dim=feature_dim*2, feature_dim=feature_dim+2, conv_type='dgrid', use_GN=use_GN)
        self.iter = evolve_iter_num - 1
        for i in range(self.iter):
            evolve_gcn = Snake(state_dim=feature_dim*2, feature_dim=feature_dim+2, conv_type='dgrid', use_GN=use_GN)
            self.__setattr__('evolve_gcn'+str(i), evolve_gcn)

        self.module_init()
        
    def module_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def prepare_training(self, output, batch):
        init = prepare_training(output, batch, self.ro)
        return init

    def prepare_testing_init(self, output):
        init = prepare_testing_init(output['poly_coarse'], self.ro)
        return init

    def prepare_testing_evolve(self, output, h, w):
        img_init_polys = output['img_init_polys']
        img_init_polys[..., 0] = torch.clamp(img_init_polys[..., 0], min=0, max=w-1)
        img_init_polys[..., 1] = torch.clamp(img_init_polys[..., 1], min=0, max=h-1)
        output.update({'img_init_polys': img_init_polys})
        return img_init_polys
    
    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind, stride=1., ignore=False, extract_hidden=False):
        if ignore:
            return (i_it_poly * self.ro, None) if extract_hidden else i_it_poly * self.ro
        if len(i_it_poly) == 0:
            return (torch.zeros_like(i_it_poly), None) if extract_hidden else torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        c_it_poly = c_it_poly * self.ro
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        if extract_hidden:
            offset, global_state = snake(init_input, extract_hidden=extract_hidden)
            offset = offset.permute(0, 2, 1)
        else:
            offset = snake(init_input).permute(0, 2, 1)
        i_poly = i_it_poly * self.ro + offset * stride
        return (i_poly, global_state) if extract_hidden else i_poly

    def forward_train(self, output, batch, cnn_feature):
        ret = output
        init = self.prepare_training(output, batch)
        py_pred = self.evolve_poly(self.evolve_gcn, cnn_feature, init['img_init_polys'],
                                   init['can_init_polys'], init['py_ind'], stride=self.evolve_stride)
        py_preds = [py_pred]
        for i in range(self.iter):
            py_pred = py_pred / self.ro
            c_py_pred = img_poly_to_can_poly(py_pred)
            evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
            py_pred = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred,
                                       init['py_ind'], stride=self.evolve_stride)
            py_preds.append(py_pred)
        ret.update({'py_pred': py_preds, 'img_gt_polys': init['img_gt_polys']})
        return output

    def forward_test(self, output, cnn_feature, ignore):
        ret = output
        with torch.no_grad():
            init = self.prepare_testing_init(output)
            img_init_polys = self.prepare_testing_evolve(init, cnn_feature.size(2), cnn_feature.size(3))
            py = self.evolve_poly(self.evolve_gcn, cnn_feature, img_init_polys, init['can_init_polys'], init['py_ind'],
                                  ignore=ignore[0], stride=self.evolve_stride)
            pys = [py, ]
            for i in range(self.iter):
                py = py / self.ro
                c_py = img_poly_to_can_poly(py)
                evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
                py = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py, init['py_ind'],
                                      ignore=ignore[i + 1], stride=self.evolve_stride)
                pys.append(py)
            ret.update({'py': pys})
        return output

    def forward(self, output, cnn_feature, batch=None, test_stage='final-dml'):
        if batch is not None and 'test' not in batch['meta']:
            self.forward_train(output, batch, cnn_feature)
        else:
            ignore = [False] * (self.iter + 1)
            if test_stage == 'coarse' or test_stage == 'init':
                ignore = [True for _ in ignore]
            if test_stage == 'final':
                ignore[-1] = True
            self.forward_test(output, cnn_feature, ignore=ignore)
        return output

class Evolution_TimeEmbed(nn.Module):
    def __init__(self, evolve_iter_num=8, feature_dim=64, evolve_stride=1., ro=4., use_GN=False, low_level_feat_dim=0):
        super().__init__()
        assert evolve_iter_num >= 1
        self.iter = min(evolve_iter_num, 8)

        self.ro = ro
        self.evolve_stride = evolve_stride
        self.evolve_gcn = timeSnake(state_dim=feature_dim*2, feature_dim=feature_dim+2)
        # sinusoid
        self.pos_enc = pos_encoding(channels=feature_dim*2)
        # trainable
        # self.pos_enc = trainable_pos_encoding(self.iter, feature_dim*2)
        self.module_init()

    def module_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def prepare_training(self, output, batch):
        init = prepare_training(output, batch, self.ro)
        return init
    
    def prepare_testing_init(self, output):
        if "poly_coarse" in output:
            init = prepare_testing_init(output['poly_coarse'], self.ro)
        else:
            init = prepare_testing_init(output['poly_init'], self.ro)
        return init

    def prepare_testing_evolve(self, output, h, w):
        img_init_polys = output['img_init_polys']
        img_init_polys[..., 0] = torch.clamp(img_init_polys[..., 0], min=0, max=w-1)
        img_init_polys[..., 1] = torch.clamp(img_init_polys[..., 1], min=0, max=h-1)
        output.update({'img_init_polys': img_init_polys})
        return img_init_polys
    
    def clip_to_img(self, py, h, w):
        ori_h, ori_w = h * self.ro, w * self.ro
        py[..., 0] = torch.clamp(py[..., 0], min=0, max=ori_w-1)
        py[..., 1] = torch.clamp(py[..., 1], min=0, max=ori_h-1)
        return py
        
    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind, time, stride=1., ignore=False):
        """
            NOTE: h, w are hw from cnn_feature, the main usage is to normalize coordinate of i_it_poly 
                into [-1, 1] in "get_gcn_feature", so no need to change
        """
        if ignore:
            return i_it_poly
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(-2) * self.ro, cnn_feature.size(-1) * self.ro
        init_feature = get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        c_it_poly = c_it_poly
        

        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        t_embd = self.pos_enc(time)
        offset = snake(init_input, t_embd).permute(0, 2, 1)
        i_poly = i_it_poly + offset * stride
        return i_poly

    def forward_train(self, output, batch, cnn_feature):
        ret = output
        # prepare can_init_poly and img_init_poly
        init = self.prepare_training(output, batch) # NOTE: DETACH at decode.py
        py_preds = []
        py_pred = init['img_init_polys'] * self.ro
        for i in range(self.iter):
            py_pred = py_pred
            c_py_pred = img_poly_to_can_poly(py_pred)
            time = self.iter - i - 1
            py_pred = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred, c_py_pred, 
                                        init['py_ind'], time=time, stride=self.evolve_stride, 
                                        )
            py_preds.append(py_pred)
        ret.update({'py_pred': py_preds, 'img_gt_polys': init['img_gt_polys']})
        ret.update({'py_ind': init['py_ind'].detach()})
        return output

    def forward_test(self, output, cnn_feature, ignore):
        ret = output
        with torch.no_grad():
            init = self.prepare_testing_init(output)
            img_init_polys = self.prepare_testing_evolve(init, cnn_feature.size(2), cnn_feature.size(3))
            py = img_init_polys * self.ro
            pys = []
            for i in range(self.iter):
                py = py
                c_py = img_poly_to_can_poly(py)
                time = self.iter - i - 1
                # time = i
                py = self.evolve_poly(self.evolve_gcn, cnn_feature, py, c_py, init['py_ind'],
                    time=time, ignore=ignore[i], stride=self.evolve_stride,
                    )
                pys.append(py)
            pys[-1] = self.clip_to_img(pys[-1], cnn_feature.size(2), cnn_feature.size(3))
            ret.update({'py': pys})
            ret.update({'py_ind': init['py_ind'].detach()})
        return output

    def forward(self, output, cnn_feature, batch=None, test_stage='final-dml'):
        if batch is not None and 'test' not in batch['meta']:
            return self.forward_train(output, batch, cnn_feature)
        else:
            ignore = [False] * self.iter
            if test_stage == 'coarse' or test_stage == 'init':
                ignore = [True for _ in ignore]
            if test_stage == 'final':
                ignore[-1] = True
            return self.forward_test(output, cnn_feature, ignore=ignore)

class Evolution_TimeEmbed_Dilated(Evolution_TimeEmbed):
    def __init__(self, evolve_iter_num=8, feature_dim=64, evolve_stride=1., ro=4., use_GN=False, low_level_feat_dim=0, dilated_size=[5,15,25], restrict=[15,40]):
        super().__init__(evolve_iter_num, feature_dim, evolve_stride, ro, use_GN, low_level_feat_dim)
        self.dilated_evolve = DilatedCNNSnake(state_dim=feature_dim*2, feature_dim=feature_dim+2)
        # self.dilated_evolve = ResDilatedCNNSnake(state_dim=feature_dim*2, feature_dim=feature_dim+2)
        assert len(dilated_size) == len(restrict) + 1
        assert restrict == sorted(restrict)
        self.restrict = restrict
        self.dilated_size = dilated_size
        self.kernel = 3

        # sinusoid
        self.pos_enc = pos_encoding(channels=feature_dim*2)
        # trainable
        # self.pos_enc = trainable_pos_encoding(self.iter, feature_dim*2)

        super().module_init()
    def get_dilated_kernel(self, poly):
        ave_len = get_average_length(poly)
        kernels = torch.zeros_like(ave_len)
        if len(self.restrict) == 0:
            kernels = torch.ones_like(ave_len) * self.dilated_size[0]
            return kernels
        for i in range(len(self.restrict)-1):
            if i == 0:
                kernels[ave_len <= self.restrict[i]] = self.dilated_size[i]
            else:
                kernels[ave_len <= self.restrict[i] & ave_len > self.restrict[i-1]] = self.dilated_size[i]
            # print(i, kernels)
        kernels[ave_len > self.restrict[-1]] = self.dilated_size[-1]
        return kernels

    def get_dilated_kernel_by_boundary(self, poly, att, ind):
        # att: boundary directly from boundary map
        h, w = att.size(-2) * self.ro, att.size(-1) * self.ro
        bd_att = get_gcn_feature(torch.sigmoid(att), poly, ind, h, w).squeeze(1)
        kernels_restrict = self.get_dilated_kernel(poly)
        bd_len_kernel = (1 - bd_att) * (kernels_restrict - 1) + 1
        # kernel_att = (1 - bd_att) * (self.longest_dilate - 1) + 1
        # print(bd_len_kernel)
        return bd_len_kernel
        
    def evolve_dilated_poly(self, snake, cnn_feature, i_it_poly, ind, dilated, stride=1., ignore=False):
        # i_it_poly: origin size
        if ignore:
            return i_it_poly
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2) * self.ro, cnn_feature.size(3) * self.ro
        dilated_feature = get_dilated_feature(cnn_feature, i_it_poly, ind, h, w, dilated, self.kernel)
    
        offset = snake(dilated_feature).permute(0,2,1)
        i_poly = i_it_poly + offset * stride
        return i_poly

    def forward_train(self, output, batch, cnn_feature):
        ret = output
        # prepare can_init_poly and img_init_poly
        init = self.prepare_training(output, batch) # NOTE: DETACH at decode.py
        py_preds = []
        py_pred = init['img_init_polys'] * self.ro
        #### TODO: dilated snake ####
        # 1. assign the kernel for each points according to their average length
        # 2. get local features
        # 3. evolve and get new contour
        # bd = output['bd']
        # dilated = self.get_dilated_kernel_by_boundary(py_pred, bd, init['py_ind'])
        dilated = self.get_dilated_kernel(py_pred)
        py_pred = self.evolve_dilated_poly(self.dilated_evolve, cnn_feature, py_pred, 
                                            init['py_ind'], dilated=dilated, stride=self.evolve_stride)
        py_preds.append(py_pred)
        #############################
        for i in range(self.iter):
            py_pred = py_pred
            c_py_pred = img_poly_to_can_poly(py_pred)
            time = self.iter - i - 1
            py_pred = self.evolve_poly(self.evolve_gcn, cnn_feature, py_pred, c_py_pred, 
                                        init['py_ind'], time=time, stride=self.evolve_stride, 
                                       )
            py_preds.append(py_pred)
        ret.update({'py_pred': py_preds, 'img_gt_polys': init['img_gt_polys']})
        ret.update({'py_ind': init['py_ind'].detach()})
        return output
    
    def forward_test(self, output, cnn_feature, ignore):
        ret = output
        with torch.no_grad():
            init = self.prepare_testing_init(output)
            img_init_polys = self.prepare_testing_evolve(init, cnn_feature.size(2), cnn_feature.size(3))
            py = img_init_polys * self.ro
            pys = []
            #### TODO: dilated snake ####
            # 1. assign the kernel for each points according to their average length
            # 2. get local features
            # 3. evolve and get new contour
            # bd = output['bd']
            # dilated = self.get_dilated_kernel_by_boundary(py, bd, init['py_ind'])
            dilated = self.get_dilated_kernel(py)
            py = self.evolve_dilated_poly(self.dilated_evolve, cnn_feature, py, 
                                            init['py_ind'], dilated=dilated, stride=self.evolve_stride)
            pys.append(py)
            #############################
            for i in range(self.iter):
                c_py = img_poly_to_can_poly(py)
                time = self.iter - i - 1
                # time = i
                py = self.evolve_poly(self.evolve_gcn, cnn_feature, py, c_py, init['py_ind'],
                    time=time, ignore=ignore[i], stride=self.evolve_stride,
                    )
                pys.append(py)
            pys[-1] = self.clip_to_img(pys[-1], cnn_feature.size(2), cnn_feature.size(3))
            ret.update({'py': pys})
            ret.update({'py_ind': init['py_ind'].detach()})
        return output
