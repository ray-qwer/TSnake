from network import make_network
from train.trainer.make_trainer import make_trainer
from train.optimizer.optimizer import make_optimizer
from train.scheduler.scheduler import make_lr_scheduler
from train.recorder.recorder import make_recorder
from dataset.data_loader import make_data_loader
from train.model_utils.utils import load_model, save_model, load_network
from train.trainer.utils import fix_seed
from evaluator.make_evaluator import make_evaluator
import argparse
import importlib
import torch
import os
import gc
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser()

parser.add_argument("config_file")
parser.add_argument("--checkpoint", default="None")
parser.add_argument("--type", default="continue")
parser.add_argument("--bs", default="None")
parser.add_argument("--dml", default="True")
parser.add_argument("--device", default=0, type=int, help='device idx')
parser.add_argument("--exp-name", type=str, help="experiment name", required=True)
parser.add_argument("--no-record", action="store_true" )
parser.add_argument("--detector-only", action="store_true")
parser.add_argument("--model-dir", default="", type=str, help="default: add folder under ./data/model")
parser.add_argument("--start-from-zero", action="store_true")
parser.add_argument("--strict-false", action="store_false")
parser.add_argument("--wandb-id", default="", type=str)
args = parser.parse_args()

def get_cfg(args):
    cfg = importlib.import_module('configs.' + args.config_file)
    if args.bs != 'None':
        cfg.train.batch_size = int(args.bs)
    if args.dml != 'True':
        cfg.train.with_dml = False
    if args.detector_only:
        cfg.model.detector_only = True
    if args.model_dir != "":
        cfg.commen.model_dir = cfg.commen.model_dir + "/" + args.model_dir
    return cfg

class save_model_iters:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.scheduler = None
    def update_scheduler(self, scheduler):
        self.scheduler = scheduler
    def __call__(self, net, optim, recorder, epoch, iters):
        save_model(net, optim, self.scheduler, recorder, epoch, self.model_dir, name="latest", iters=iters)

def train(network, cfg, exp_name):
    save_model_func = save_model_iters(cfg.commen.model_dir)
    train_loader, val_loader = make_data_loader(cfg=cfg)
    iter_per_epoch = len(train_loader)
    trainer = make_trainer(network, cfg)
    optimizer = make_optimizer(network, cfg)
    
    # scheduler update to iters
    scheduler_update_per_iter = cfg.train.scheduler_update_per_iter
    if scheduler_update_per_iter:
        if "milestones" in cfg.train.optimizer:
            cfg.train.optimizer["milestones"] = [i * iter_per_epoch for i in cfg.train.optimizer["milestones"]]
    
    scheduler = make_lr_scheduler(optimizer, cfg, iter_per_epoch)
    if not args.no_record:
        if args.wandb_id !="":
            recorder = make_recorder(cfg.commen.record_dir, name=exp_name, cfg=cfg, use_wandb=True, id=args.wandb_id)
        else:
            recorder = make_recorder(cfg.commen.record_dir, name=exp_name, cfg=cfg, use_wandb=True)
    else:
        recorder = None
    evaluator = make_evaluator(cfg)

    if args.type == 'finetune':
        begin_epoch = load_network(network, model_dir=args.checkpoint, strict=args.strict_false)
        iters = 0
    else:
        begin = load_model(network, optimizer, scheduler, recorder, args.checkpoint)
        begin_epoch, iters = begin
    # start from zero
    begin_epoch = 0 if (args.start_from_zero or args.type=='finetune') else begin_epoch

    
    best_val = 0
    best_epoch = 0
    if os.path.exists(os.path.join(cfg.commen.model_dir, "best")):
        with open(os.path.join(cfg.commen.model_dir, "best"), 'r') as f:
            best_val = float(f.read())

    eval_ep = cfg.test.eval_ep
    for epoch in range(begin_epoch, cfg.train.epoch):
        # save_model_func.update_scheduler(scheduler)
        if recorder is not None:
            recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder, scheduler if scheduler_update_per_iter else None, iters)
        iters = 0
        if not scheduler_update_per_iter:
            scheduler.step()    # update per epoch

        torch.cuda.empty_cache()
        if (epoch + 1) % cfg.train.save_ep == 0:
           save_model(network, optimizer, scheduler, recorder, epoch,
                      cfg.commen.model_dir)

        # saving every step to check which makes the process crash
        save_model(network, optimizer, scheduler, recorder, epoch,
                       cfg.commen.model_dir, name="latest")
        
        
        if (epoch + 1) % eval_ep == 0:
            with torch.no_grad():
                val = trainer.val(epoch, val_loader, evaluator, recorder)
                if val['ap'] > best_val:
                    best_val = val['ap']
                    best_epoch = epoch
                    save_model(network, optimizer, scheduler, recorder, epoch, cfg.commen.model_dir, name="best")
                    with open(os.path.join(cfg.commen.model_dir, "best"), 'w') as f:
                        f.write(str(val['ap']))
        
        print("best val:", best_val)        
        print("best epoch:", best_epoch)
        
        # change to eval every epoch when the best over some threshold
        #if best_val > cfg.test.baseline_val and epoch >= 100:
        #    eval_ep = 1
    return network

def main():
    cfg = get_cfg(args)
    torch.cuda.set_device(args.device)
    fix_seed(42)
    network = make_network.get_network(cfg)
    train(network, cfg, exp_name=args.exp_name)

if __name__ == "__main__":
    main()
