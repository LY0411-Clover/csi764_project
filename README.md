# Similarity-Guided Adaptive Expert Expansion for Continual Learning

## 1. Overview

This project extends LEAR with a similarity-guided adaptive expert expansion and reuse mechanism for multi-domain continual learning.

The proposed method uses Mahalanobis distance to estimate the similarity between a new task and previously learned task representations. Depending on a predefined threshold, the model either reuses an existing expert or creates a new expert initialized from the most similar existing expert.

Each task is assigned a task-specific classifier, independently of the expert assignment.

## 2. Requirements

- Python: 3.10
- Other dependencies: see `requirements.txt`

## 3. Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```

Create and activate the Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Ensure that the required datasets and pretrained model weights are available before running the experiments.

## 4. Dataset Preparation

This project uses the `seq-cifar10` benchmark.

Download or prepare the dataset according to the dataset implementation in the repository. Ensure that the dataset can be loaded successfully before starting an experiment.

## 5. Running Experiments

### 5.1 Training Configuration

The following example runs LEAR for 10 epochs per task:

```bash
python main_domain.py \
    --dataset seq-cifar10 \
    --model LEAR \
    --lr 0.03 \
    --batch_size 32 \
    --n_epochs 2 \
    --num_workers 0 \
    --backbone lear
```

Use the same dataset, training budget, random seed, and evaluation protocol when comparing the proposed method with the original LEAR baseline.

### 5.2 Threshold Configuration

The expert reuse/expansion threshold controls whether a new task reuses an existing expert or triggers the creation of a new expert.

- If the minimum task-to-expert distance is less than or equal to the threshold, reuse the selected expert.
- If the minimum distance exceeds the threshold, create a new expert initialized from the most similar existing expert.

Please change the threshold in the models\LEAR.py, begin_task() function

If the threshold is currently defined directly in the Python code, modify the corresponding parameter before running the experiment.

### 5.3 Task-Aware Evaluation

In the task-aware setting, the task identity is provided during inference. The model uses the task-to-expert and task-to-classifier mappings to make predictions.

Run the experiment using the task-aware evaluation configuration implemented in the repository.

### 5.4 Task-Agnostic Evaluation

In the task-agnostic setting, the task identity is unavailable during inference. The model estimates task similarity using Mahalanobis distance and selects a task-specific classifier together with its assigned expert.

Run the experiment using the task-agnostic evaluation configuration implemented in the repository.

## 6. Evaluation Metrics

The experiments report:

- Average Performance (AP)
- Backward Transfer (BWT)
- Forgetting Measure (FM)
- Forward Transfer (FWT), when the required baseline results are available
- Task Selection Accuracy for task-agnostic routing
- Number of experts and classifiers
- Model parameter memory and GPU memory usage

Per-task accuracy values should be retained after every task so that continual-learning metrics can be computed from the complete accuracy matrix.

## 7. Experimental Results

Results are saved under the experiment output directory configured by the training framework.

For reproducibility, record the threshold, random seed, dataset configuration, training budget, task-aware/task-agnostic setting, and software environment alongside each run.

## 8. Reproducibility Notes

For a fair comparison:

1. Use identical data splits and task order.
2. Keep the training budget and optimizer configuration consistent.
3. Evaluate the original LEAR baseline and the proposed method under the same protocol.
4. Run multiple random seeds where computational resources permit.
5. Report the mean and standard deviation across runs when multiple seeds are used.

| Dataset         | Domain                           |
| --------------- | -------------------------------- |
| seq-cifar10     | natural images (small)           |
| seq-cub200      | fine-grained birds               |
| seq-eurosat-rgb | satellite imagery                |
| seq-imagenet-r  | natural image distribution shift |
| seq-tinyimg     | large-scale object images        |

| Dataset         | 类型                      | 难度              | 对 LEAR 意义                 |
| --------------- | ------------------------- | ----------------- | ---------------------------- |
| seq-cifar10     | natural image             | 简单              | sanity check                 |
| seq-eurosat-rgb | remote sensing            | domain shift 明显 | 测 domain continual learning |
| seq-imagenet-r  | image style shift         | 较难              | 测 expert reuse              |
| seq-tinyimg     | large-scale natural image | 较难              | 测扩展能力                   |
