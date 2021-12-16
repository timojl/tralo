# Example Project

This example project illustrates how Tralo can be used to structure experiments.


### Structure

- `experiment_setup.py`: The code which defines your experiments.
- `experiments`: Yaml definitions of the experiment configurations.
- `datasets`: Here go your custom datasets.
- `models`: Here go your custom models.
- `third_party`: Potential third party dependencies.


### Run Experiment

`nums` selects certain experiment runs (can also be a range, e.g. 2:5).
`retrain` forces to train again, and `retest` forces computation of test scores.

```bash
tralo exp experiments/cifar_example.yaml [--nums 0] [--retrain] [--retest]
```


### Run on HPC

```bash
sbatch hpc_run experiments/cifar_example.yaml
```