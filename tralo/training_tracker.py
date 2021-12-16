import copy
import gzip
import json
import os
import time
from collections import defaultdict
from os.path import isdir, join, dirname, realpath

import threading

import numpy as np
import torch
from torch import nn
from tralo.log import log
import multiprocessing as mp
from tralo.utils import count_parameters, sha1_hash_object, gzip_write


def compute_metrics(base_path, iteration, metric_values, metric_lock, manager, callback, best_score, model):
    """ defined as function to support multiprocessing """

    output = callback(model)
    metrics_str = ' '.join([f'{k}: {v:.5f}' for k, v in output if k not in {'iterations'}])
    log.info(f'{iteration}: {metrics_str}')

    if output[0][1] > best_score.value:
        best_score.value = output[0][1]
        weight_file = join(base_path, f'weights.pth') if base_path is not None else None
        if weight_file is not None:
            torch.save(model.state_dict(), weight_file)
        
    if iteration not in metric_values:
        metric_values[iteration] = manager.dict()

    metric_values[iteration].update({k: v for k, v in output})

    metric_file = join(base_path, f'metrics.json') if base_path is not None else None
    if metric_file is not None:
        metric_lock.acquire()
        with open(metric_file, 'w') as fh:
            json.dump({k: v.copy() for k, v in metric_values.items()}, fh)
    
        metric_lock.release()


def plot_losses(filename, iterations, losses):
    """ defined as function to support multiprocessing """

    import matplotlib.pyplot as plt
    plt.plot(iterations, losses)
    plt.savefig(filename)
    plt.clf()


class LogUtilization(mp.Process):

    def __init__(self, log_file=None, stats=None, interval=0.1, write_interval=10):
        mp.Process.__init__(self)
        self.log_file = log_file
        self.stats = stats
        self.interval = interval

        # by using mp.Value we allow this variable to be manipulated from outside
        self.step_interval = mp.Value("i", int(write_interval / interval))

        self.p = None
        self.exit = mp.Event()


        stats = dict(cpu=[], gpu=[], mem=[], gpu_mem=[], time=[])
        if self.stats is None:
            self.stats = stats
        else:
            self.stats.update(**stats)        

    def run(self):

        try:
            try:
                import pynvml
                pynvml.nvmlInit()

                dev_id = 0
                if 'CUDA_VISIBLE_DEVICES' in os.environ:
                    if ',' in os.environ['CUDA_VISIBLE_DEVICES']:
                        raise ValueError('Currently utility tracking is only supported for single devices')
                    else:
                        dev_id = int(os.environ['CUDA_VISIBLE_DEVICES'])

                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)

                
            except (ImportError, ValueError):
                gpu_handle = None
                log.warning('Install pynvml to use GPU utilization logging.')

            i = 0
            
            t_start = time.time()

            while not self.exit.is_set():
                time.sleep(self.interval)

                self.stats['time'] += [int((time.time() - t_start)*1000)]

                try:
                    import psutil
                except ImportError:
                    psutil = None

                if psutil is not None:
                    self.stats['cpu'] += [psutil.cpu_percent()]
                    self.stats['mem'] += [psutil.virtual_memory().used]
                
                if gpu_handle is not None:
                    self.stats['gpu'] += [pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu]
                    self.stats['gpu_mem'] += [pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used]

                # if a log file is used
                if self.log_file is not None and i % self.step_interval.value == (self.step_interval.value - 1):
                    if self.p is not None:
                        self.p.join()

                    self.p = threading.Thread(target=gzip_write, args=(self.log_file, dict(self.stats)))
                    self.p.start()

                i += 1

        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):

        # self.write()
        if self.p is not None:
            self.p.join()
               
        self.exit.set()


class TrainingLogger(object):

    def __init__(self, log_dir=None, interval=50, model=None, config=None, 
                 metric=None, async_metric=None, grad_weights=None, grad_interval=None, 
                 plot=False, utilization_iters=200, checkpoint_iters=None):
        """
        Training logger class.

            log_dir:            Folder to save logs and weights.
            interval:           Log interval.
            model:              Reference to model (optional).
            config:             Current training configuration.
            metric:             Tuple of metric function and evaluation interval. The function takes model 
                                as an argument.
            grad_weights:       A list of layers from which you want to obtain gradients
            async_metric:       Must be a tuple (callback, interval). Callback must return a list of tuples (metric_name, score). The
                                first tuple of this list will be used for defining the best weights. 
                                Larger scores are considered better.
            plot                Write training plots.
            utilization_iters:  Number of iterations during which utilization (CPU and GPU) is tracked.
            checkpoint_iters:   List of iterations at which the weights are saved.
            
        """

        # these values can be changed by attribute access, although this should not be necessary in most cases
        self.estimate_duration_iter = 10  # estimate training speed over 10 iterations
        self.fixed_iterations = set([2, 5, 10])  
        self.save_only_trainable_weights = False

        self.model = model
        self.model_params = {n: m for n, m in self.model.named_parameters()} if self.model is not None else {}

        self.plot = plot
        self.interval = interval

        self.checkpoint_iters = checkpoint_iters

        self.grad_interval = interval if grad_interval is None else grad_interval
        assert grad_weights is None or self.grad_interval is not None

        self.mp_manager = mp.Manager()

        if log_dir is None and config is not None:
            log_dir = sha1_hash_object(config)[:10]

        self.stats = dict(start_time=int(time.time()))

        self.base_path = join(f'logs/{log_dir}') if log_dir is not None else None
        if self.base_path is not None:
            os.makedirs(self.base_path, exist_ok=True)
            os.makedirs(join(self.base_path, 'gradients'), exist_ok=True)

            if config is not None:
                json.dump(config, open(join(self.base_path, 'config.json'), 'w'))
                
            with open(join(self.base_path, 'stats.json'), 'w') as fh:            
                json.dump(self.stats, fh)

        # utilization tracking
        if utilization_iters > 0:
            self.utilization_iters = utilization_iters
            
            if self.base_path is not None:
                self.utilization_process = LogUtilization(log_file=join(self.base_path, 'utilization.json.gz'))
            else:
                self.utilization_stats = self.mp_manager.dict()
                self.utilization_process = LogUtilization(stats=self.utilization_stats)
            
            self.utilization_process.start()
        else:
            self.utilization_process = None

        # gradient tracking
        if grad_weights is not None:
            grad_weights = grad_weights if type(grad_weights) in {list, tuple} else list(grad_weights)
            self.grad_weights = []
            self.grad_names = []
            for gw in grad_weights:
                
                # if is pair and first element is string assume its the name
                if type(gw) in {list, tuple} and len(gw) == 2 and type(gw[0]) == str:
                    name = gw[0]
                    gw = gw[1]
                else:
                    name = 'grad'

                # for convenience: transform modules into weights
                if isinstance(gw, nn.Module):
                    if isinstance(gw, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                        self.grad_weights += [gw.weight, gw.bias]
                        self.grad_names += [name + '_weight', name + '_bias']
                    else:
                        raise ValueError('invalid module type. Provide weights instead.')
                elif isinstance(gw, torch.Tensor):
                    self.grad_weights += [gw]
                    self.grad_names += [name]

            self.gradients = [[] for _ in range(len(self.grad_weights))]
            self.gradient_iterations = []
        else:
            self.grad_weights = None
            self.gradients = None
        
        # metrics
        self.metric_values = self.mp_manager.dict()
        self.metrics_lock = torch.multiprocessing.Lock()  # avoid writing from two processes

        if metric is not None:
            self.metric_callback = metric[0]
            self.metric_interval = metric[1]
        else:
            self.metric_interval = None

        if async_metric is not None:
            self.async_metric_callback = async_metric[0]
            self.async_metric_interval = async_metric[1]
        else:
            self.async_metric_interval = None
        # self.metric_callback_async = metric_callback_async

        self.loss_iterations = []
        self.losses = []
        self.loss_cache = []

        self.best_score = self.mp_manager.Value(float, -999)
    
        self.plot_thread = None
        self.running_processes = []
        self.max_processes = 3

        self.time_at_2 = None  # tracks time at iteration 2

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        """ automatically stop processes if used in a context manager """

        if self.utilization_process is not None:
            self.utilization_process.terminate()
        
        self.update_stats()

        for p in self.running_processes:
            p.join()

    def stop(self):
        """ explicit stop function """
        self.__exit__(None, None, None)

    def save_weights(self, only_trainable=False, weight_file='weights.pth'):
        """ convenience function to save weights """

        if self.model is None:
            raise AttributeError('You need to provide a model reference when initializing TrainingTracker to save weights.')

        weights_path = join(self.base_path, weight_file)

        weight_dict = self.model.state_dict()

        if only_trainable:
            weight_dict = {n: weight_dict[n] for n, p in self.model.named_parameters() if p.requires_grad}
        
        torch.save(weight_dict, weights_path)
        log.info(f'Saved weights to {weights_path}')

    def update_stats(self):

        self.stats['stop_time'] = int(time.time())
        self.stats['iterations'] = int(self.loss_iterations[-1]) if len(self.loss_iterations) > 0 else 0

        if self.model:
            self.stats['params'] = int(count_parameters(self.model))
            self.stats['learn_params'] = int(count_parameters(self.model, only_trainable=True))

        try:
            with gzip.open(join(self.base_path, 'utilization.json.gz'), 'rb') as fh:
                util = json.loads(fh.read())
                self.stats['cpu_mean'] = int(np.mean(util['cpu']))
                self.stats['gpu_mean'] = int(np.mean(util['gpu']))
                self.stats['mem_mean'] = int(np.mean(util['mem']))
                self.stats['gpu_mem_mean'] = int(np.mean(util['gpu_mem']))
        except BaseException:
            pass

        if self.base_path is not None:
            with open(join(self.base_path, 'stats.json'), 'w') as fh:            
                json.dump(self.stats, fh)

    def launch_async_metric(self, i):
        # metrics_path = join(self.base_path, f'metrics.json') if self.base_path is not None else None

        if len([1 for p in self.running_processes if p.is_alive()]) > self.max_processes:
            log.info('Too many background processes. joining...')
            for p in self.running_processes:
                p.join()

        from torch.multiprocessing import Process
        
        model_copy = copy.deepcopy(self.model).cpu() if self.model is not None else None
        p = Process(target=compute_metrics, args=(self.base_path, i, self.metric_values, self.metrics_lock, self.mp_manager,
                                                    self.async_metric_callback, self.best_score, model_copy))

        p.start()
        self.running_processes += [p]

    def save_metrics(self, i, **metrics):

        if len(metrics) > 0:

            this_dict = self.mp_manager.dict()
            for metric_name, value in metrics.items():

                if callable(value):
                    value = value()

                this_dict[metric_name] = value
            
            if i not in self.metric_values:
                self.metric_values[i] = this_dict
            else:
                self.metric_values[i].update(this_dict)

            if self.base_path is not None:
                self.metrics_lock.acquire()
                with open(join(self.base_path, f'metrics.json'), 'w') as fh:
                    json.dump({k: v.copy() for k, v in self.metric_values.items()}, fh)
                self.metrics_lock.release()


    def __call__(self, i, loss=None, **extras):
        self.iter(i, loss=loss, **extras)

    def iter(self, i, loss=None, **extras):

        if self.utilization_process is not None and i >= self.utilization_iters:
            log.info('stop utility logging process')
            self.utilization_process.terminate()
            self.utilization_process = None

        if i == 2:
            self.time_at_2 = time.time()

        if i == 2 + self.estimate_duration_iter:
            time_diff = time.time() - self.time_at_2
            time_per_iter = time_diff / self.estimate_duration_iter
            log.info(f'Speed estimates: {time_per_iter:.3f}s/iter or {time_per_iter*1000/60:.1f}min/1000 iter')

        if i % self.grad_interval == self.grad_interval -1:
            if self.grad_weights is not None:
                
                for j, (w, name) in enumerate(zip(self.grad_weights, self.grad_names)):
                    hist = torch.histc(w.grad.cpu(), bins=500)

                    if self.base_path is not None:
                        name = name if name is not None else 'grad'
                        torch.save(hist, join(self.base_path, f'gradients/{i:08d}-{j}-{name}.pth'))
                    else:
                        self.gradients[j] += [hist]
                
                self.gradient_iterations += [i]

        if self.checkpoint_iters is not None and i in self.checkpoint_iters:
            self.save_weights(only_trainable=self.save_only_trainable_weights, 
                              weight_file='weights_{i}.pth')

        # normal metrics
        if self.metric_interval is not None and i % self.metric_interval == self.metric_interval - 1:
            compute_metrics(self.base_path, i, self.metric_values, self.metrics_lock, self.mp_manager, self.metric_callback, self.best_score, self.model)       

        # async metrics
        if self.async_metric_interval is not None and i % self.async_metric_interval == self.async_metric_interval - 1:
            self.launch_async_metric(i)

        # add extras passed via arguments
        loss_str, metrics_str = '', ''
        if len(extras) > 0:
            self.save_metrics(i, **extras)
            if len(self.metric_values) > 0:
                metrics_str = ' '.join([f'{k}: {v:.5f}' for k, v in self.metric_values[i].items() if k not in {'iterations'}])              
            # log.info(f'{i}:{metrics_str}')

        if i % self.interval == self.interval - 1 or i in self.fixed_iterations:
            
            if loss is not None:
                self.loss_cache += [float(loss)]
                current_loss = np.mean(self.loss_cache)
                self.loss_cache = []

                self.losses += [current_loss]
                self.loss_iterations += [i]

            if self.base_path is not None:
                json.dump({'loss': self.losses, 'iterations': self.loss_iterations},
                        open(join(self.base_path, 'losses.json'), 'w'))

            if self.plot and self.base_path is not None:
                self.plot_thread = threading.Thread(target=plot_losses, args=(join(self.base_path + '-loss.pdf'),
                                                                              self.loss_iterations, self.losses))

            if loss is not None:
                loss_str = f' loss: {current_loss:.5f}'
            
            log.info(f'{i}:{loss_str} {metrics_str}')


    def plots(self):

        from tralo.visualize import show_run

        gradients = (self.grad_names, self.grad_weights, self.gradient_iterations, self.gradients) if self.gradients is not None else None

        if self.base_path is not None:
            with gzip.open(join(self.base_path, 'utilization.json.gz'), 'rb') as fh:
                util_stats = json.loads(fh.read())
        elif hasattr(self, 'utilization_stats'):
            util_stats = self.utilization_stats
        else:
            util_stats = None
        
        show_run(self.base_path, self.metric_values, (self.loss_iterations, self.losses), 
                 gradients, util_stats)


# backward compatibility
TrainingTracker = TrainingLogger