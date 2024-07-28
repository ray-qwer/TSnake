import torch
import os
import torch.nn.functional
from termcolor import colored

def load_model(net, optim, scheduler, recorder, model_path, map_location=None):
    # reload the whole settings of training if the training is not finished
    strict = True

    if not os.path.exists(model_path):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0, 0

    print('load model: {}'.format(model_path))
    if map_location is None:
        pretrained_model = torch.load(model_path, map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu',
                                                                'cuda:2': 'cpu', 'cuda:3': 'cpu'})
    else:
        pretrained_model = torch.load(model_path, map_location=map_location)
    net.load_state_dict(pretrained_model['net'], strict=strict)
    # net_weight = net.state_dict()
    # for key in pretrained_model['net'].keys():
    #     if key[:4] == 'net.':
    #         key_ = key[4:]
    #         net_weight.update({key_: pretrained_model['net'][key]})
    # net.load_state_dict(net_weight, strict=strict)
    optim.load_state_dict(pretrained_model['optim'])
    scheduler.load_state_dict(pretrained_model['scheduler'])
    if recorder is not None:
        recorder.load_state_dict(pretrained_model['recorder'])
    if "iters" in pretrained_model and pretrained_model['iters'] > 0:
        iters = pretrained_model["iters"] + 1
    else: 
        iters = 0
    if iters > 0:
        epoch = pretrained_model['epoch']
    else:
        epoch = pretrained_model['epoch'] + 1
    return [epoch, iters]

def save_model(net, optim, scheduler, recorder, epoch, model_dir, iters=0, name=None):
    os.system('mkdir -p {}'.format(model_dir))
    if name is None:
        name = str(epoch)
    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch,
        'iters': iters,
    }, os.path.join(model_dir, '{}.pth'.format(name)))
    return

def save_weight(net, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'net': net.state_dict(),
    }, os.path.join(model_dir, '{}.pth'.format('final')))
    return

def load_network(net, model_dir, strict=True, map_location=None):
    # only load the nentwork?
    if not os.path.exists(model_dir):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    print('load model: {}'.format(model_dir))
    if map_location is None:
        pretrained_model = torch.load(model_dir, map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu',
                                                               'cuda:2': 'cpu', 'cuda:3': 'cpu'})
    else:
        pretrained_model = torch.load(model_dir, map_location=map_location)
    if 'epoch' in pretrained_model.keys():
        epoch = pretrained_model['epoch'] + 1
    else:
        epoch = 0
    pretrained_model = pretrained_model['net']

    net_weight = net.state_dict()

    for key in net_weight.keys():
        if key not in pretrained_model.keys():
            print(key)
            key_ = key.replace('conv_offset', 'conv_offset_mask')
            if key_ in pretrained_model.keys():
                net_weight.update({key: pretrained_model[key_]})
        else:
            if "dla.wh.2" in key:
                net_wh_shape = net_weight[key].shape
                pretrained_wh_shape = pretrained_model[key].shape
                if net_wh_shape != pretrained_wh_shape:
                    print(f"key {key} not match! shape of pretrained: {pretrained_wh_shape}, shape of net: {net_wh_shape}")
                    continue
            key_ = key
            net_weight.update({key: pretrained_model[key_]})
        #net_weight.update({key: pretrained_model[key]})

    net.load_state_dict(net_weight, strict=strict)
    return epoch
