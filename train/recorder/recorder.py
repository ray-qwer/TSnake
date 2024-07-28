from collections import deque, defaultdict
import torch
from tensorboardX import SummaryWriter
import os
import wandb

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class Recorder(object):
    def __init__(self, record_dir, task='e2ec'):
        log_dir = record_dir
        self.writer = SummaryWriter(log_dir=log_dir)

        # scalars
        self.epoch = 0
        self.step = 0
        self.loss_stats = defaultdict(SmoothedValue)
        self.batch_time = SmoothedValue()
        self.data_time = SmoothedValue()

        # images
        self.image_stats = defaultdict(object)
        if 'process_' + task in globals():
            self.processor = globals()['process_' + task]
        else:
            self.processor = None

    def update_loss_stats(self, loss_dict):
        for k, v in loss_dict.items():
            self.loss_stats[k].update(v.detach().cpu())

    def update_image_stats(self, image_stats):
        if self.processor is None:
            return
        image_stats = self.processor(image_stats)
        for k, v in image_stats.items():
            self.image_stats[k] = v.detach().cpu()

    def record(self, prefix, step=-1, loss_stats=None, image_stats=None):
        pattern = prefix + '/{}'
        step = step if step >= 0 else self.step
        loss_stats = loss_stats if loss_stats else self.loss_stats

        for k, v in loss_stats.items():
            if isinstance(v, SmoothedValue):
                self.writer.add_scalar(pattern.format(k), v.median, step)
            else:
                self.writer.add_scalar(pattern.format(k), v, step)

        if self.processor is None:
            return
        image_stats = self.processor(image_stats) if image_stats else self.image_stats
        for k, v in image_stats.items():
            self.writer.add_image(pattern.format(k), v, step)

    def state_dict(self):
        scalar_dict = {}
        scalar_dict['step'] = self.step
        return scalar_dict

    def load_state_dict(self, scalar_dict):
        self.step = scalar_dict['step']

    def __str__(self):
        loss_state = []
        for k, v in self.loss_stats.items():
            loss_state.append('{}: {:.4f}'.format(k, v.avg))
        loss_state = '  '.join(loss_state)

        recording_state = '  '.join(['epoch: {}', 'step: {}', '{}', 'data: {:.4f}', 'batch: {:.4f}'])
        return recording_state.format(self.epoch, self.step, loss_state, self.data_time.avg, self.batch_time.avg)


def make_recorder(record_dir, *args, use_wandb=True, **kwargs):
    
    return  Recorder_WandB(record_dir, *args, **kwargs) if use_wandb else Recorder(record_dir=record_dir)

class Recorder_WandB(object):
    def __init__(self, record_dir, name, cfg, project='e2ec-sbd', **kwargs):
        # initialization
        wandb.login()
        hyper_param_record = {
            "optimizer": cfg.train.optimizer,
            "batch size": cfg.train.batch_size,
            "epoch": cfg.train.epoch,
            "train dataset": cfg.train.dataset,
            "val dataset": cfg.test.dataset,
            "ct from dist": cfg.train.from_dist,
            "weight dict": cfg.train.weight_dict,
            "model": dict((key, value) for key, value in cfg.model.__dict__.items() if not callable(value) and not key.startswith('__')),
            "detect_type": cfg.model.detect_type,
            "backbone": cfg.model.detect_backbone,
            "data": {
                "down ratio": cfg.data.down_ratio,
                "scale": cfg.data.scale,
                "scale range": cfg.data.scale_range,
                "points per poly": cfg.data.points_per_poly
            }
        }
        self.writer = wandb.init(dir=record_dir, project=project, name=name, config=hyper_param_record, resume="allow", **kwargs)
        
        
        # scalars
        self.epoch = 0
        self.step = 0
        self.loss_stats = defaultdict(SmoothedValue)
        self.batch_time = SmoothedValue()
        self.data_time = SmoothedValue()

        # images
        self.image_stats = defaultdict(object)
        if 'process_' + project in globals():
            self.processor = globals()['process_' + project]
        else:
            self.processor = None
    
    def update_loss_stats(self, loss_dict):
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                self.loss_stats[k].update(v.detach().cpu())
            else:
                self.loss_stats[k].update(v) 
    
    def update_image_stats(self, image_stats):
        if self.processor is None:
            return
        image_stats = self.processor(image_stats)
        for k, v in image_stats.items():
            self.image_stats[k] = v.detach().cpu()

    def record_val(self, prefix, epoch=-1, loss_stats=None):
        pattern = prefix + '/{}'
        epoch = epoch if epoch >= 0 else 0
        loss_stats = loss_stats if loss_stats else self.loss_stats

        wandb_log = {}
        for k, v in loss_stats.items():
            if isinstance(v, SmoothedValue):
                wandb_log.update({pattern.format(k): v.median})                
                # self.writer.add_scalar(pattern.format(k), v.median, step)
            else:
                wandb_log.update({"epoch": epoch, pattern.format(k): v})
                # self.writer.add_scalar(pattern.format(k), v, step)
        self.writer.log(wandb_log)

    def record(self, prefix, step=-1, loss_stats=None, image_stats=None):
        pattern = prefix + '/{}'
        step = step if step >= 0 else self.step
        loss_stats = loss_stats if loss_stats else self.loss_stats

        wandb_log = {}
        for k, v in loss_stats.items():
            if isinstance(v, SmoothedValue):
                wandb_log.update({pattern.format(k): v.median})                
                # self.writer.add_scalar(pattern.format(k), v.median, step)
            else:
                wandb_log.update({pattern.format(k): v})
                # self.writer.add_scalar(pattern.format(k), v, step)

        if self.processor is None:
            self.writer.log(wandb_log)
            return
        image_stats = self.processor(image_stats) if image_stats else self.image_stats
        for k, v in image_stats.items():
            wandb_log.update({pattern.format(k): wandb.Image(v)})
            # self.writer.add_image(pattern.format(k), v, step)
        self.writer.log(wandb_log)
        
    def state_dict(self):
        scalar_dict = {}
        scalar_dict['step'] = self.step
        return scalar_dict

    def load_state_dict(self, scalar_dict):
        self.step = scalar_dict['step']

    def __str__(self):
        loss_state = []
        for k, v in self.loss_stats.items():
            loss_state.append('{}: {:.4f}'.format(k, v.avg))
        loss_state = '  '.join(loss_state)

        recording_state = '  '.join(['epoch: {}', 'step: {}', '{}', 'data: {:.4f}', 'batch: {:.4f}'])
        return recording_state.format(self.epoch, self.step, loss_state, self.data_time.avg, self.batch_time.avg)
