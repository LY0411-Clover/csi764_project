# Similarity-Guided Adaptive Expert Allocation for Multi-Domain Continual Learning

Code for the COMPSCI 764 project by Shiqing Guan and Jeff Dong (University of Auckland, 2026).
It extends the public LEAR code (Yu et al., NeurIPS 2025) so that a new task **reuses** an existing expert when its
Mahalanobis distance to a learned task is at most a threshold τ, and **creates** a new expert otherwise.
Every task keeps its own classifier and its own feature statistics, so tasks with different label spaces can share an expert.

## What is in this repository

| Path                                             | Purpose                                                                                                   |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `NeurIPS2025-LEAR-main/models/LEAR.py`           | Our method, **task-agnostic** version (used for the corrected task-agnostic runs).                        |
| `LEAR_task_aware.py`                             | Our method, **task-aware** version (used for the corrected task-aware runs). See "Task-aware runs" below. |
| `NeurIPS2025-LEAR-main/backbone/LEAR.py`         | Backbones, experts (`fcArr`) and classifiers (`classifierArr`), registered as `nn.ModuleList`.            |
| `NeurIPS2025-LEAR-main/utils/training_domain.py` | Training loop and task-agnostic evaluation (routing once per test set).                                   |
| `NeurIPS2025-LEAR-main/datasets/__init__.py`     | The task sequence (`dataset_names` in `get_all_datasets`).                                                |
| `Record/`                                        | Raw logs of every run reported in the paper.                                                              |
| `metrix_calculation.xlsx`                        | Accuracy matrices and AP / BWT / FM for every run.                                                        |

## Requirements

Python 3.10 and the packages in `NeurIPS2025-LEAR-main/requirements.txt`. One GPU with 8 GB of memory is enough
(our runs used an NVIDIA RTX 4070).

## Task sequence

Seven tasks, in this order (set in `datasets/__init__.py`):
`seq-cifar10`, `seq-cifar100`, `seq-eurosat-rgb`, `seq-imagenet-r`, `seq-tinyimg`, `seq-resisc45`, `seq-chestx`.
The `--dataset` argument on the command line is required by the framework but the sequence above is what is trained.
Datasets are downloaded by the dataset classes on first use.

## Running

From `NeurIPS2025-LEAR-main/`:

```bash
python main_domain.py --dataset seq-cifar10 --model LEAR --lr 0.03 --batch_size 32 \
    --n_epochs 2 --num_workers 0 --backbone lear --seed 1
```

**Threshold τ.** Set the default of `threshold` in `begin_task()` in `models/LEAR.py` (the paper's main setting is 33.0;
τ = 0 makes every task create an expert, which reproduces LEAR's always-expand schedule).

**Debug flag.** `debug` at the top of `utils/training_domain.py` is meant for quick end-to-end checks:

| `debug` | Training                | Testing                                   | Used for                              |
| ------- | ----------------------- | ----------------------------------------- | ------------------------------------- |
| `True`  | 50 iterations per epoch | first 3 test batches per task (96 images) | threshold sweeps                      |
| `False` | full epochs             | full test sets                            | all full-evaluation runs in the paper |

**Task-aware runs.** Checkout to the task-aware version, and replace `models/LEAR.py` with `LEAR_task_aware.py`, uncommented `seq-chestx` on around 219 line in the `datasets\_init_.py`.

## Results reported in the paper (seven tasks, τ = 33, full epochs)

| Run                 | Experts K | AP (%) | FM   | Mapping       |
| ------------------- | --------- | ------ | ---- | ------------- |
| Ours, task-agnostic | 3         | 82.53  | 2.63 | 1 1 1 2 1 3 2 |
| Ours, task-aware    | 4         | 83.40  | 2.06 | 1 1 1 2 3 4 2 |
| LEAR (same budget)  | 7         | 81.57  | 2.09 | 1 2 3 4 5 6 7 |

No random seed was fixed for these runs; pass `--seed` for new runs.

## Acknowledgement

Built on the LEAR code by Yu et al. (NeurIPS 2025) and the Mammoth continual-learning framework. See `LICENSE`.
