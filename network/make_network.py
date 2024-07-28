import torch.nn as nn
from .backbone.dla import DLASeg
from .detector_decode.refine_decode import Decode
from .evolve.evolve import Evolution, Evolution_TimeEmbed, Evolution_TimeEmbed_Dilated
from .postprocess.combine import AttentionCombine, MultiLayerAttentionCombine, MultiLayerAttentionCombine_Bbox
import torch
from .detector_decode.utils import decode_ct_hm, clip_to_image
from .ct_rcnn_snake import Network as RCNN_Network

class Network(nn.Module):
    def __init__(self, cfg=None):
        super(Network, self).__init__()
        print("normal network init")
        num_layers = cfg.model.dla_layer
        head_conv = cfg.model.head_conv
        down_ratio = cfg.commen.down_ratio
        heads = cfg.model.heads
        self.test_stage = cfg.test.test_stage
        self.detect_type = cfg.model.detect_type
        self.evolve_name = cfg.model.evolve_name if cfg.model.evolve_name != "" else "Evolution"
        
        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                            pretrained=True,
                            down_ratio=down_ratio,
                            final_kernel=1,
                            last_level=5,
                            head_conv=head_conv, use_dcn=cfg.model.use_dcn, use_GN=cfg.model.use_GN)
        
        self.train_decoder = Decode(num_point=cfg.commen.points_per_poly, init_stride=cfg.model.init_stride,
                                    coarse_stride=cfg.model.coarse_stride, down_sample=cfg.commen.down_ratio,
                                    min_ct_score=cfg.test.ct_score)
        self.gcn = globals()[self.evolve_name](evolve_iter_num=cfg.model.evolve_iters, evolve_stride=cfg.model.evolve_stride,
                             ro=cfg.commen.down_ratio, use_GN=cfg.model.use_GN)

    def forward(self, x, batch=None):
        output, cnn_feature = self.dla(x)   # heatmap and image_features
        
        if batch is not None and 'test' not in batch['meta']:
            self.train_decoder(batch, cnn_feature, output, is_training=True)
        else:
            with torch.no_grad():
                if self.test_stage == 'init':
                    ignore = True
                else:
                    ignore = False
                self.train_decoder(batch, cnn_feature, output, is_training=False, ignore_gloabal_deform=ignore)
        
        output = self.gcn(output, cnn_feature, batch, test_stage=self.test_stage)

        return output

class TimeNetwork(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        print("TimeNetwork Init")
        num_layers = cfg.model.dla_layer
        head_conv = cfg.model.head_conv
        down_ratio = cfg.commen.down_ratio
        heads = cfg.model.heads
        self.get_low_level = cfg.model.with_low_level_feat
        print(self.get_low_level)
        self.test_stage = cfg.test.test_stage
        
        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                            pretrained=True,
                            down_ratio=down_ratio,
                            final_kernel=1,
                            last_level=5,
                            head_conv=head_conv, use_dcn=cfg.model.use_dcn, use_GN=cfg.model.use_GN)
        
        # TODO: Decoder change to no refine one
        self.train_decoder = Decode(num_point=cfg.commen.points_per_poly, init_stride=cfg.model.init_stride,
                                    coarse_stride=cfg.model.coarse_stride, down_sample=cfg.commen.down_ratio,
                                    min_ct_score=cfg.test.ct_score, get_coarse_contour=False)
        # self.gcn = Evolution_TimeEmbed(evolve_iter_num=cfg.model.evolve_iters, evolve_stride=cfg.model.evolve_stride,
        #                      ro=cfg.commen.down_ratio, use_GN=cfg.model.use_GN)
        self.gcn = Evolution_TimeEmbed_Dilated(evolve_iter_num=cfg.model.evolve_iters, evolve_stride=cfg.model.evolve_stride,
                             ro=cfg.commen.down_ratio, use_GN=cfg.model.use_GN,
                             dilated_size=[5], restrict=[])
        
    def forward(self, x, batch=None, with_cnn=False):
        output = self.dla(x, self.get_low_level)
        if self.get_low_level:
            output, cnn_feature, low_level_feature = output
        else:
            output, cnn_feature = output

        if batch is not None and 'test' not in batch['meta']:
            self.train_decoder(batch, cnn_feature, output, is_training=True)
        else:
            with torch.no_grad():
                if self.test_stage == 'init':
                    ignore = True
                else:
                    ignore = False
                self.train_decoder(batch, cnn_feature, output, is_training=False)
        
        if self.get_low_level:
            output = self.gcn(output, cnn_feature, batch, low_level_feat=low_level_feature, test_stage=self.test_stage)
        else:
            output = self.gcn(output, cnn_feature, batch, test_stage=self.test_stage)

        return output if not with_cnn else (output, cnn_feature)
   

class CombineNetwork(Network):
    def __init__(self, cfg):
        super(CombineNetwork, self).__init__(cfg=cfg)
        print("Combine Network Init")
        if cfg.model.combine_layer == 1:
            self.combine = AttentionCombine(cfg.model.combine_in_c, cfg.model.points_per_poly, cfg.model.combine_n_embd, cfg.model.combine_heads)
        else:
            if cfg.model.pred_bbox:
                self.combine = MultiLayerAttentionCombine_Bbox(cfg.model.combine_in_c, cfg.model.points_per_poly, 
                                                cfg.model.combine_n_embd, cfg.model.combine_heads, 
                                                cfg.model.combine_layer, pe_method=cfg.model.combine_pe_method, rpe_mode=cfg.model.combine_rpe_mode)
            else:
                self.combine = MultiLayerAttentionCombine(cfg.model.combine_in_c, cfg.model.points_per_poly, 
                                                cfg.model.combine_n_embd, cfg.model.combine_heads, 
                                                cfg.model.combine_layer, pe_method=cfg.model.combine_pe_method, rpe_mode=cfg.model.combine_rpe_mode)
        # self.eval_net()
        self.from_dataset = cfg.train.combine_from_dataset  # If true, take samples from dataset; else, get samples from model
        self.is_eval = False
        self.pred_bbox = cfg.model.pred_bbox
        if cfg.train.train_combine_only:
            self.freeze_net()
            self.is_eval = self.eval_net()
    
    def train(self, mode=True):
        if mode:
            if self.is_eval:
                self.combine = self.combine.train()
            else:
                self.combine = self.combine.train()
                self.dla = self.dla.train()
                self.train_decoder = self.train_decoder.train()
                self.gcn = self.gcn.train()
        else:
            print("eval")
            self.eval_net()
            self.combine = self.combine.eval()
        return self
    def freeze_net(self):
        freeze_layer = ["dla", "train_decoder", "gcn"]
        for name in freeze_layer:
            partial_net = getattr(self, name)
            for key, value in partial_net.named_parameters():
                value.requires_grad = False
    
    def eval_net(self):
        self.dla = self.dla.eval()
        self.train_decoder = self.train_decoder.eval()
        self.gcn = self.gcn.eval()
        return True

    def forward(self, x, batch=None):
        is_training = False
        if batch is not None and 'test' not in batch['meta']:
            is_training = True
        output, cnn_feature, low_level_feat = self.dla(x, True)
        
        self.train_decoder(batch, cnn_feature, output, is_training=(self.from_dataset and is_training), ignore_gloabal_deform=False)
        output = self.gcn(output, cnn_feature, batch, test_stage=self.test_stage)

        if self.pred_bbox:
            combine, bbox = self.combine(batch, output, cnn_feature, from_dataset=(self.from_dataset and is_training))
            output.update({'combine': combine})
            output.update({'bbox': bbox})
        else:
            combine = self.combine(batch, output, cnn_feature, from_dataset=(self.from_dataset and is_training))
            output.update({'combine': combine})
        
        return output # output will be combined when testing

class TimeCombineNetwork(TimeNetwork):
    def __init__(self, cfg):
        super().__init__(cfg)
        print("Combine Network V2 Init")
        ## module
        if cfg.model.combine_layer == 1:
            self.combine = AttentionCombine(cfg.model.combine_in_c, cfg.model.points_per_poly, cfg.model.combine_n_embd, cfg.model.combine_heads)
        else:
            if cfg.model.pred_bbox:
                self.combine = MultiLayerAttentionCombine_Bbox(cfg.model.combine_in_c, cfg.model.points_per_poly, 
                                                cfg.model.combine_n_embd, cfg.model.combine_heads, 
                                                cfg.model.combine_layer, pe_method=cfg.model.combine_pe_method, rpe_mode=cfg.model.combine_rpe_mode)
            else:
                self.combine = MultiLayerAttentionCombine(cfg.model.combine_in_c, cfg.model.points_per_poly, 
                                                cfg.model.combine_n_embd, cfg.model.combine_heads, 
                                                cfg.model.combine_layer, pe_method=cfg.model.combine_pe_method, rpe_mode=cfg.model.combine_rpe_mode)
        ## argument
        self.from_dataset = cfg.train.combine_from_dataset
    
    def forward(self, x, batch=None):
        is_training = False
        if batch is not None and 'test' not in batch['meta']:
            is_training = True

        output, cnn_feature = super().forward(x, batch, with_cnn=True)
        combine = self.combine(batch, output, cnn_feature, from_dataset=(self.from_dataset and is_training))
        output.update({'combine': combine})
        
        return output

def get_network(cfg):
    if cfg.model.combineNet:
        return CombineNetwork(cfg)
        # return TimeCombineNetwork(cfg)
    if hasattr(cfg.model, "rcnn") and cfg.model.rcnn == True:
        network = RCNN_Network(cfg)
        return network
    if hasattr(cfg.model, "snake") and cfg.model.snake == "shared":
        network = TimeNetwork(cfg)
    else:
        network = Network(cfg)
    return network
