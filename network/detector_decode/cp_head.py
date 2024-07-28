import torch.nn as nn
import torch

# TODO: replace rcnn_snake_utils
from .utils import box_to_roi, decode_cp_detection
from dataset import rcnn_snake_config
# from lib.utils.rcnn_snake import rcnn_snake_config, rcnn_snake_utils
from torchvision.ops import nms, RoIAlign


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ComponentDetection(nn.Module):
    def __init__(self, heads):
        super(ComponentDetection, self).__init__()

        # self.pooler = ROIAlign((rcnn_snake_config.roi_h, rcnn_snake_config.roi_w))
        self.pooler = RoIAlign((rcnn_snake_config.roi_h, rcnn_snake_config.roi_w) ,spatial_scale=1., sampling_ratio=0)

        self.fusion = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                nn.Conv2d(256, classes, kernel_size=1, stride=1)
            )
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def prepare_training(self, cnn_feature, output, batch):
        w = cnn_feature.size(3)
        # take ground truth out
        xs = (batch['act_ind'] % w).float()[..., None]
        ys = (batch['act_ind'] // w).float()[..., None]
        wh = batch['awh']
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
        rois = box_to_roi(bboxes, batch['act_01'].byte())
        roi = self.pooler(cnn_feature, rois)
        output.update({"abox":bboxes.detach()})
        return roi

    def nms_class_box(self, box, score, cls, cls_num):
        box_score_cls = []

        for j in range(cls_num):
            ind = (cls == j).nonzero().view(-1)
            if len(ind) == 0:
                continue

            box_ = box[ind]
            score_ = score[ind]
            # ind = _ext.nms(box_, score_, rcnn_snake_config.max_ct_overlap)  
            ######### replace by torchvision nms
            ind = nms(box_, score_, rcnn_snake_config.max_ct_overlap)

            box_ = box_[ind]
            score_ = score_[ind]

            ind = score_ > rcnn_snake_config.ct_score
            box_ = box_[ind]
            score_ = score_[ind]
            label_ = torch.full([len(box_)], j).to(box_.device).float()

            box_score_cls.append([box_, score_, label_])

        return box_score_cls

    def nms_abox(self, output):
        box = output['detection'][..., :4]
        score = output['detection'][..., 4]
        cls = output['detection'][..., 5]

        batch_size = box.size(0)
        cls_num = output['act_hm'].size(1)

        box_score_cls = []
        for i in range(batch_size):
            box_score_cls_ = self.nms_class_box(box[i], score[i], cls[i], cls_num)
            box_score_cls_ = [torch.cat(d, dim=0) for d in list(zip(*box_score_cls_))]
            box_score_cls.append(box_score_cls_)

        box, score, cls = list(zip(*box_score_cls))
        ind = torch.cat([torch.full([len(box[i])], i) for i in range(len(box))], dim=0)
        box = torch.cat(box, dim=0)
        score = torch.stack(score, dim=1)
        cls = torch.stack(cls, dim=1)

        detection = torch.cat([box, score, cls], dim=1)

        return detection, ind

    def prepare_testing(self, cnn_feature, output):
        if rcnn_snake_config.nms_ct:
            detection, ind = self.nms_abox(output)
        else:
            ind = output['detection'][..., 4] > rcnn_snake_config.ct_score
            detection = output['detection'][ind]
            ind = torch.cat([torch.full([ind[i].sum()], i) for i in range(len(ind))], dim=0)

        ind = ind.to(cnn_feature.device)
        abox = detection[:, :4]
        roi = torch.cat([ind[:, None], abox], dim=1)

        roi = self.pooler(cnn_feature, roi)
        output.update({'detection': detection, 'roi_ind': ind})

        return roi

    def decode_cp_detection(self, cp_hm, cp_wh, output):
        abox = output['detection'][..., :4]
        adet = output['detection']
        ind = output['roi_ind']
        if cp_wh.shape[1] == 2:
            box, cp_ind = decode_cp_detection(torch.sigmoid(cp_hm), cp_wh, abox, adet)
            output.update({'cp_box': box, 'cp_ind': cp_ind})
        else:
            init_poly, box, cp_ind = decode_cp_detection(torch.sigmoid(cp_hm), cp_wh, abox, adet)
            output.update({'i_it_py': init_poly * 4., 'cp_box': box, 'cp_ind': cp_ind})

    def decode_init_contour(self, cp_wh, output, batch):
        # needed: abox, cp_ind, roi_ind
        if cp_wh.shape[1] == 2:
            return
        act_01 = batch['act_01'].byte()
        cp_01 = batch['cp_01'][act_01].byte()
        cp_ind = batch['cp_ind'][act_01][cp_01]
        cp_box_idx = batch['cp_box_idx'][cp_01]
        abox = output['abox'][act_01]
        abox_w, abox_h = abox[..., 2] - abox[..., 0], abox[..., 3] - abox[..., 1]
        
        h, w = cp_wh.shape[-2:]
        cp_x, cp_y = cp_ind % w, torch.div(cp_ind, w, rounding_mode='floor')
        cp_x = cp_x.clip(0, w - 1)        
        cp_y = cp_y.clip(0, h - 1)
        cp_offset = cp_wh[cp_box_idx, :, cp_y, cp_x].view(cp_x.size(0), -1, 2)
        
        # coordinate from rcnn to origin
        xs = cp_x / w * abox_w[cp_box_idx] + abox[cp_box_idx, 0]
        ys = cp_y / h * abox_h[cp_box_idx] + abox[cp_box_idx, 1]
        xs, ys = xs.unsqueeze(1).to(torch.float32), ys.unsqueeze(1).to(torch.float32)
        cp = torch.cat([xs, ys], dim=1)
        init_poly = cp_offset + cp.unsqueeze(1).expand(cp_offset.size(0), cp_offset.size(1), cp_offset.size(2))
        output.update({'i_it_py': init_poly * 4.})

    def prepare_wh_training(self, cp_wh):
        h, w = rcnn_snake_config.cp_h, rcnn_snake_config.cp_w
        cp_wh = torch.sigmoid(cp_wh)
        cp_wh[..., 0] = cp_wh[..., 0] * w
        cp_wh[..., 1] = cp_wh[..., 1] * h
        return cp_wh

    def forward(self, output, cnn_feature, batch=None):
        z = {}

        if batch is not None and 'test' not in batch['meta']:
            # training step
            roi = self.prepare_training(cnn_feature, output, batch)
            roi = self.fusion(roi)
            for head in self.heads:
                z[head] = self.__getattr__(head)(roi)
            ## NOTE: phase2
            # z['cp_wh'] = self.prepare_wh_training(cp_wh)
            self.decode_init_contour(z['cp_wh'], output, batch)
        if not self.training:
            # testing step
            with torch.no_grad():
                roi = self.prepare_testing(cnn_feature, output)
                roi = self.fusion(roi)
                cp_hm = self.cp_hm(roi)
                cp_wh = self.cp_wh(roi)
                ## NOTE: to relative coordinate phase2
                # cp_wh = torch.sigmoid(self.cp_wh(roi))
                self.decode_cp_detection(cp_hm, cp_wh, output)

        output.update(z)

        return output

