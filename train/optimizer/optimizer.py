import torch
from bisect import bisect_right

_optimizer_factory = {
    'adamw': torch.optim.AdamW,
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}


def make_optimizer(net, cfg):
    opt_cfg = cfg.train.optimizer
    params = []
    lr = opt_cfg['lr']
    weight_decay = opt_cfg['weight_decay']

    opt_partial = cfg.train.optimizer_partial
    freeze_layer = cfg.train.freeze_layer
    partial_list = []

    if len(freeze_layer) > 0:
        for name in freeze_layer:
            if "." in name:
                names = name.split(".")
            else:
                names = [name]
            partial_net = net
            for i in names:
                partial_net = getattr(partial_net, i)

            for key, value in partial_net.named_parameters():
                value.requires_grad = False

    if len(opt_partial) > 0:
        for i in opt_partial:
            partial_name = i['name']
            partial_lr = i['lr']
            partial_net = getattr(net, partial_name)
            if "exception" in i.keys():
                partial_net_exception = i['exception']
            else:
                partial_net_exception = []
            print(partial_net)
            for key, value in partial_net.named_parameters():
                if not value.requires_grad:
                    continue
                if key in partial_net_exception:
                    print(key)
                    continue
                params += [{"params": [value], "lr": partial_lr, "weight_decay": weight_decay}]
                partial_list.append(partial_name + '.' + key)

    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        if key in partial_list:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if 'adam' in opt_cfg['name']:
        optimizer = _optimizer_factory[opt_cfg['name']](params, lr, weight_decay=weight_decay)
    elif 'adamw' in opt_cfg['name']:
        # beta?
        optimizer = _optimizer_factory[opt_cfg['name']](params, lr, momentum=momentum, weight_decay=weight_decay) 
    else:
        momentum = cfg.train.optimizer["momentum"]
        optimizer = _optimizer_factory[opt_cfg['name']](params, lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer


