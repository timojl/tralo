# TraLo

TraLo is a lightweight training logger which facilitates tracking and analyzing relevant quantities as well as conducting experiments.
It is easy to integrate into existing code bases and has only one dependency: PyTorch 
(soft dependencies specific functions: torchvision for testdoc, flask for web view, psutil and pynvml for CPU and GPU statistics, matplotlib for notebook plots).

For usage examples see `doctests/`.

## Installation

```bash
pip install git+https://github.com/timojl/tralo
```
(optionally use the `--no-deps` argument to avoid automatically installing pytorch)

## Training

The following code will automatically keep track of loss curves, general statistics and evaluate `my_metric` every 200 iterations. The best model according to the metric will be saved.

```python
def my_metric(model):
  # ... (load your data and calculate your metric here)
  return [('main_score', score1]), ('other_score', score2])]

with TrainingLogger(model=model, 
                    log_dir='logs/my_run',  # place to save the log
                    metric=(my_metric, 200),  # metric callback and interval
                    config=cfg  # dictionary 
                    ) as logger:
  while True:  
    # ... (load your data, model forward and backward pass, compute loss)
    logger(i=i, loss=loss, other_quantity=some_val)
```

More options:
* `async_metric` computes metric in an individual process (same API as metric)
* `utilization_iters`: number of iterations at the beginning during which utilization (CPU and GPU) is tracked.
* `grad_weights` A list of layer names (as strings) from which gradient statistics are extracted.

The following quantities will be saved automatically as files:

- `weights.pth`: The model's weights.
- `losses.json`: The training loss.
- `stats.json`: General information: e.g. training time, number of parameters.
- `metrics.json`: Metrics collected during training (including validation loss).
- `utilization.json.gz`: Utilization.

Additional score files can be placed in the run folder, too.


## CLI
Convenience functions are available through the command line.

- `tralo exp <experiment file>`: Run experiment defined in an experiment configuration.
- `tralo server`: Run local webserver to view logs (requires flask).
- `tralo create <template name> <project name>`: Create a new project based on a template.


## Helper Functions

* `plot_data(data)`: Simple way to plot arbitrary data.  
* `get_batch(dataset, 0, 4)`: Obtain a batch of samples (here first 4 samples of `dataset`).
* `get_from_repository()` Makes sure data from a repository (e.g. a .tar archive) is locally available and extracted.
* ...

## Experiments

Experiments are handled by separating configuration and code. The configuration is provided by a yaml file, see `doctest/mnist_experiment.yaml` for an example.

**Training Configuration**

- `configuration`: General training configurations
  - `trainer`: path to training function (e.g. experiment_setup.train_loop)
  - `scorer`: path to scorer function (e.g. experiment_setup.score)
- `individual_configurations`: Individual training configurations

**Tests Configuration**

- `test_configuration_common`: General test configuration
- `test_configuration`: List of individual test configurations

When the scorer is executed, it will automatically write a file (filename is a hash of the test arguments) containing scores to the log_dir. 

### Plot
Plot in a notebook using `plot_experiment`.


## Web-based Training Visualization

Change into a `logs` folder created after training and run `python -m tralo.server`.
