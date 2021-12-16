import os
import re
import json
import torch

import numpy as np
from matplotlib import pyplot as plt
from os.path import join, isdir
from tralo.experiments import experiment


def plot_experiment(name):
    results = experiment(name)

    from matplotlib import pyplot as plt

    plt.grid(True)

    plt.title('loss')

    for run in results.run_names():
        losses = json.load(open(join('logs', run, 'losses.json')))
        plt.plot(losses['loss'], label=run)

    # TODO: add metrics

    plt.legend()    


def plot_stats(run, start=0, end=None):

    import gzip, json
    
    with gzip.open(f'{run}/utilization.json') as fh:
        a = json.loads(fh.read())
    
    gpu, cpu, gpu_mem = a['gpu'][start:end], a['cpu'][start:end], a['gpu_mem'][start:end]
    _, ax = plt.subplots(2, 3, figsize=(15, 5))

    ax[0,0].set_title('gpu utility')
    ax[0,0].fill_between(np.arange(0, len(gpu)), gpu, color=plt.cm.tab10(0))
    ax[0,0].set_ylim(0,100)
    ax[0,1].set_title('gpu memory')
    ax[0,1].fill_between(np.arange(0, len(gpu_mem)), np.array(gpu_mem) / 1e9, color=plt.cm.tab10(0))
    ax[0,2].set_title('cpu utility')
    ax[0,2].fill_between(np.arange(0, len(cpu)), cpu, color=plt.cm.tab10(0))
    ax[0,2].set_ylim(0,100)
    
    try:
        l = json.load(open(join(run, 'losses.json')))
        ax[1,0].set_title('loss')
        ax[1,0].plot(l['iterations'], l['loss'])
    except FileNotFoundError:
        pass
    
    try:
        m = json.load(open(join(run, 'metrics.json')))
        m_iter = sorted([int(k) for k in m.keys()])
        metric_names = list(next(iter(m.values())).keys())

        for name in metric_names[:1]:
            ax[1,1].set_title(name)
            ax[1,1].plot(m_iter, [m[str(k)][name] for k in m_iter])
    except FileNotFoundError:
        pass
    plt.tight_layout()
    

def show_run(base_path, metric_values, iter_loss, gradients=None, utility_stats=None):

    loss_iterations, losses = iter_loss

    from matplotlib import pyplot as plt

    iters = sorted(metric_values.keys())

    metric_names = list(set(x for k in metric_values.values() for x in k.keys()))

    _, ax = plt.subplots(1, 1+ len(metric_names), figsize=(3 + 3*len(metric_names),3))

    ax0 = ax[0] if len(metric_names) > 0 else ax
    
    ax0.plot(loss_iterations, losses, marker='.')
    ax0.grid(True)
    ax0.set_title('loss')

    # handles = []
    for i, k in enumerate(metric_names):
        iters = [j for j in metric_values if k in metric_values[j].keys()]
        ax[1+i].grid(True)
        ax[1+i].plot(iters, [metric_values[i][k] for i in iters], marker='.')
        ax[1+i].set_title(k)
        ax[1+i].set_xlim(0, max(metric_values.keys()))
        
    plt.tight_layout()
    plt.show()

    # gradients
    grad_folder = join(base_path, f'gradients') if base_path is not None else None
    if grad_folder is not None or gradients is not None:

        if grad_folder is not None and isdir(grad_folder):
            grad = sorted(os.listdir(grad_folder))
            grad = [(a,) + re.match(r'([0-9]*)-([0-9]*)-(\w*).pth', a).groups() for a in grad]
            grad = [(a[0], int(a[1]), int(a[2]), a[3]) for a in grad]

            grad_weights = sorted(set(a[2] for a in grad))
            grad_names = {w: [a[3] for a in grad if a[2] == w][0] for w in grad_weights}

            grad_files = {w: [torch.load(join(base_path, f'gradients', fn)) for fn, _, gw, _ in grad if gw == w] 
                        for w in grad_weights}
            grad_iters = {w: [iter for _, iter, gw, _ in grad if gw == w] 
                        for w in grad_weights}
        
        elif gradients is not None:
            grad_names, grad_weights, gradient_iterations, grad_files = gradients
            grad_weights = list(range(len(grad_weights)))
            grad_iters = {j: gradient_iterations for j in range(len(grad_weights))}

        _, ax = plt.subplots(1, len(grad_weights), figsize=(3 + 3*len(grad_weights), 3))
        for i, w in enumerate(grad_weights):
            g = torch.stack(grad_files[i]).T
            ax[i].set_title(grad_names[i])
            ax[i].imshow(g, interpolation='nearest', aspect='auto')
            ax[i].set_xticks(range(len(grad_iters[w])))
            ax[i].set_xticklabels(grad_iters[w])

        plt.tight_layout()

    if utility_stats is not None:
        _, ax = plt.subplots(1, 4, figsize=(11,3))

        limits = dict(gpu=(-5,105), cpu=(-5,105))
        for i, k in enumerate(['gpu', 'gpu_mem', 'cpu', 'mem']):
            ax[i].set_title(k)
            if k in limits:
                ax[i].set_ylim(*limits[k])
            ax[i].grid(True)
            ax[i].plot(np.array(utility_stats['time'])/1000, utility_stats[k])
            ax[i].fill_between(np.array(utility_stats['time'])/1000, utility_stats[k])
            ax[i].set_xlabel('time [s]')


    plt.tight_layout()
    plt.show()
