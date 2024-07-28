import torch
import torch.nn as nn
import os

from .backbone.dla import DLASeg

## utils
from .detector_decode.utils import decode_rcnn_ct_hm as decode_ct_hm, clip_to_image


from .detector_decode.cp_head import ComponentDetection
from .evolve.rcnn_evolve import Evolution_RCNN as RAFT, Evolution_RCNN_no_dilated as RAFT_no_dilated

class Network(nn.Module):
    def __init__(self, cfg=None):
        super(Network, self).__init__()
        print("RCNN Network Init")
        num_layers = cfg.model.dla_layer
        head_conv = cfg.model.head_conv
        down_ratio = cfg.commen.down_ratio
        heads = cfg.model.heads
        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv)
        self.cp = ComponentDetection(cfg.model.cp_head)
        self.raft = RAFT(evolve_iter_num=cfg.model.evolve_iters, evolve_stride=cfg.model.evolve_stride,
                            ro=cfg.commen.down_ratio, use_GN=cfg.model.use_GN,
                             dilated_size=[5], restrict=[])
        # self.raft = RAFT_no_dilated(evolve_iter_num=cfg.model.evolve_iters, evolve_stride=cfg.model.evolve_stride,
        #                      ro=cfg.commen.down_ratio, use_GN=cfg.model.use_GN)
    def decode_detection(self, output, h, w):
        ct_hm = output['act_hm']
        wh = output['awh']
        ct, detection = decode_ct_hm(torch.sigmoid(ct_hm), wh)
        detection[..., :4] = clip_to_image(detection[..., :4], h, w)
        output.update({'ct': ct, 'detection': detection})
        return ct, detection

    def forward(self, x, batch=None):
        output, cnn_feature = self.dla(x)
        with torch.no_grad():
            self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3))
        output = self.cp(output, cnn_feature, batch)
        output = self.raft(output, cnn_feature, batch)
        return output