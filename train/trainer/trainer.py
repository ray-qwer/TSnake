import time
import datetime
import torch
import tqdm
import gc
from statistics import mean

class Trainer(object):
    def __init__(self, network, acc_iter=1, scheduler_update_per_iter=False):
        network = network.cuda()
        self.network = network
        assert acc_iter != 0
        self.acc_iter = acc_iter
        self.scheduler_update_per_iter = scheduler_update_per_iter
    
    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: v/ float(self.acc_iter) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple):
                batch[k] = [b.cuda() for b in batch[k]]
            else:
                batch[k] = batch[k].cuda()
        return batch

    def update_grad_stat(self, loss_stats, total_loss_stats):
        if total_loss_stats is None:
            return {k: v for k, v in loss_stats.items()}
        else:
            total_loss_stats = {k: v + total_loss_stats[k] for k, v in loss_stats.items()}
        return total_loss_stats
        
    def clear_total_loss(self, total_loss_stats):
        return {k: 0 for k,_ in total_loss_stats.items()}
    
    def train(self, epoch, data_loader, optimizer, recorder, scheduler=None, iters=0, save_model_func=None):
        saving_iters = 1000
        if self.scheduler_update_per_iter:
            assert scheduler is not None
        max_iter = len(data_loader) // self.acc_iter
        self.network.train()
        end = time.time()
        record = True
        if recorder is None:
            record = False
        print("acc iter:", self.acc_iter)
        acc_iteration = 0
        total_loss_stats = None
        log_flag = True
        grad_acc_flag = False
        data_loader.dataset.update_epoch(epoch)
        for iteration, batch in enumerate(data_loader):
            if iteration < iters:
                continue
            data_time = time.time() - end
            # iteration = iteration + 1
            if record:
                recorder.step += 1
            

            batch = self.to_cuda(batch)
            batch.update({'epoch': epoch})
            output, loss, loss_stats = self.network(batch)
            
            loss = loss.mean()
            loss = loss / self.acc_iter

            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            
            if ((iteration+1) % self.acc_iter == 0) or (iteration + 1 == len(data_loader)):
                acc_iteration += 1
                optimizer.step()
                optimizer.zero_grad()
                if self.scheduler_update_per_iter:
                    scheduler.step()
                log_flag = True
                grad_acc_flag = True

            loss_stats = self.reduce_loss_stats(loss_stats)
            total_loss_stats = self.update_grad_stat(loss_stats, total_loss_stats)
            

            if record and grad_acc_flag:
                recorder.update_loss_stats(total_loss_stats)
                total_loss_stats = self.clear_total_loss(total_loss_stats)
                grad_acc_flag = False
                batch_time = time.time() - end
                end = time.time()
                recorder.batch_time.update(batch_time)
                recorder.data_time.update(data_time)

            if ((acc_iteration + 1) % 20 == 0 or acc_iteration == (max_iter - 1)) and log_flag:
                if record:
                    eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                if record:
                    training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                if record:
                    recorder.record('train')
                log_flag = False
            
            if (iteration + 1) % saving_iters == 0 and save_model_func is not None:
                save_model_func(self.network.net, optimizer, recorder, epoch, iteration)
            # del loss
            # del loss_stats
            # gc.collect()

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        for batch in tqdm.tqdm(data_loader):
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()

            batch.update({'epoch': epoch})
            with torch.no_grad():
                output = self.network(batch)
                if evaluator is not None:
                    evaluator.evaluate(output, batch)

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record_val('val', epoch, val_loss_stats)
        
        return val_loss_stats

