from .snake import NetworkWrapper, DetectorWrapper, FCOSNetworkWrapper, \
                    MaskNetworkWrapper, CombineNetworkWrapper, ShareWeightNetworkWrapper
from .trainer import Trainer


def _wrapper_factory(network, cfg):
    if cfg.model.detect_only:
        return DetectorWrapper(network, start_epoch=cfg.train.start_epoch, weight_dict=cfg.train.weight_dict, init_stride=cfg.model.init_stride)
    # elif cfg.model.combineNet:
    #     return CombineNetworkWrapper(network, gt_from_dataset=cfg.train.combine_from_dataset)
    else:
        if hasattr(cfg.model, "detect_type") and "FCOS" in cfg.model.detect_type:
            return FCOSNetworkWrapper(network, with_dml=cfg.train.with_dml,
                          start_epoch=cfg.train.start_epoch, weight_dict=cfg.train.weight_dict, init_stride=cfg.model.init_stride, is_pred_bbox=cfg.model.detect_type=="FCOS_v2",
                          down_ratio=cfg.commen.down_ratio, anchor_origin_size=cfg.model.anchor_origin_size, offset_loss=cfg.train.offset_loss)
        else:
            if cfg.train.use_Mask_Loss and "Local" not in cfg.model.evolve_name:
                return MaskNetworkWrapper(network, start_epoch=cfg.train.start_epoch, 
                                          weight_dict=cfg.train.weight_dict, init_stride=cfg.model.init_stride, get_instance_mask=cfg.train.get_instance_mask)
            else:
                if cfg.model.snake == "shared":
                    return ShareWeightNetworkWrapper(network, weight_dict=cfg.train.weight_dict, start_epoch=cfg.train.start_epoch, start_module=cfg.train.start_module, num_points=cfg.model.points_per_poly, num_points_fft=cfg.model.fft_dot)
                else:
                    return NetworkWrapper(network, with_dml=cfg.train.with_dml, with_wsll=cfg.train.with_wsll,
                          start_epoch=cfg.train.start_epoch, weight_dict=cfg.train.weight_dict, \
                            use_Mask_Loss=cfg.train.use_Mask_Loss, get_instance_mask=cfg.train.get_instance_mask)
        

def make_trainer(network, cfg):
    network = _wrapper_factory(network, cfg)
    acc_iter = int(cfg.train.grad_acc / cfg.train.batch_size)
    if acc_iter == 0:
        acc_iter = 1
    return Trainer(network, acc_iter, cfg.train.scheduler_update_per_iter)
