# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import copy
import math
import os
import sys
from argparse import Namespace
from time import time
from typing import Iterable, Tuple
import logging
import torch
from tqdm import tqdm

from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset, MammothDatasetWrapper
from datasets.utils.gcl_dataset import GCLDataset
from models.utils.continual_model import ContinualModel
from models.utils.future_model import FutureModel

from utils import disable_logging
from utils.checkpoints import mammoth_load_checkpoint, save_mammoth_checkpoint
from utils.loggers import log_extra_metrics, log_accs, Logger
from utils.schedulers import get_scheduler
from utils.stats import track_system_stats

try:
    import wandb
except ImportError:
    wandb = None

debug = True


@torch.no_grad()
def evaluate(model: ContinualModel, datasets, test_loaders, last=False, return_loss=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.

    The accuracy is evaluated for all the tasks up to the current one, only for the total number of classes seen so far.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand
        last: a boolean indicating whether to evaluate only the last task
        return_loss: a boolean indicating whether to return the loss in addition to the accuracy

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task. If return_loss is True, the loss is also returned as a third element.
    """

    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    task_selection_accs = []
    accs = []

    # loss_fn = dataset.get_loss()
    # avg_loss = 0
    total_len = sum(len(x) for x in test_loaders)

    # expert_index_list = model.task_expert_map
    # classifier_index_list = model.task_classifier_map

    # print('choose experts for evaluate:', expert_index_list)
    # print('choose classifiers for evaluate:', classifier_index_list)

    # if len(test_loaders) > 1:
    #     pbar_choose = tqdm(test_loaders, total=total_len, desc='Choose expert for evaluate',
    #                        disable=model.args.non_verbose,ncols=170)
    #     for j, test_loader in enumerate(test_loaders):
    #         test_iter = iter(test_loader)
    #         #min_idx_list = []
    #         count = 1
    #         sum_distances = [0] * len(test_loaders)
    #         num_choose = 50
    #         while True:
    #             try:
    #                 data = next(test_iter)
    #             except StopIteration:
    #                 break

    #             inputs, labels = data
    #             inputs, labels = inputs.to(model.device), labels.to(model.device)

    #             distances = model.cal_expert_dist(inputs)
    #             sum_distances = [(x + y)/count for x, y in zip(sum_distances, distances)]

    #             min_idx = torch.argmin(torch.tensor(sum_distances)).item()

    #             bar_log = {f'task {j + 1}': min_idx + 1, 'distance': [round(x,2) for x in sum_distances]}

    #             pbar_choose.set_postfix(bar_log, refresh=False)
    #             #pbar_choose.set_description(f"choose expert for task {j + 1}", refresh=False)
    #             pbar_choose.update(1)
    #             count += 1
    #             if count == num_choose:
    #                 break

    #         expert_index_list.append(int(min_idx+1))

    #     pbar_choose.close()
    # else:
    #     expert_index_list = [1]

    pbar = tqdm(test_loaders, total=total_len, desc='Evaluating',
                disable=model.args.non_verbose, ncols=170)
    for k, test_loader in enumerate(test_loaders):

        if last and k < len(test_loaders) - 1:
            continue

        # ============================================================
        # 1. Calculate average distance for the WHOLE test task
        # ============================================================

        num_tasks = len(model.task_distributions)

        sum_distances = torch.zeros(
            num_tasks,
            device=model.device
        )

        num_samples = 0

        print("\n")
        print("=" * 75)
        print(f"[Task-Agnostic Routing] Test Task {k + 1}")
        print("=" * 75)

        model.eval()

        with torch.no_grad():

            for i, data in enumerate(test_loader):

                # if debug and i > 2:
                #     break

                if model.args.debug_mode and i > model.get_debug_iters():
                    break

                inputs, labels = data[0], data[1]

                inputs = inputs.to(model.device)

                # ----------------------------------------------------
                # Distance:
                # [batch_size, num_tasks]
                # ----------------------------------------------------
                distances = model.get_task_distances(inputs)

                # Sum distance of every sample
                sum_distances += distances.sum(dim=0)

                num_samples += inputs.size(0)

        # ============================================================
        # 2. Average distance over the WHOLE test task
        # ============================================================

        avg_distances = sum_distances / num_samples

        # Select ONE task for the whole test task
        selected_task = torch.argmin(
            avg_distances
        ).item()

        selected_expert = model.task_expert_map[selected_task]

        selected_classifier = model.task_classifier_map[selected_task]

        # ============================================================
        # 3. Detailed routing debug
        # ============================================================

        print(
            f"Ground-truth task       : Task {k + 1}"
        )

        print(
            f"Selected task           : Task {selected_task + 1}"
        )

        print(
            f"Selected expert         : Expert {selected_expert + 1}"
        )

        print(
            f"Selected classifier     : "
            f"Classifier {selected_classifier + 1}"
        )

        print("\nTask mappings:")

        for task_idx in range(num_tasks):

            expert_idx = model.task_expert_map[task_idx]

            classifier_idx = model.task_classifier_map[task_idx]

            distance = avg_distances[task_idx].item()

            print(
                f"  Task {task_idx + 1}: "
                f"Expert {expert_idx + 1}, "
                f"Classifier {classifier_idx + 1}, "
                f"Distance = {distance:.4f}"
            )

        print(
            f"\nMinimum distance       : "
            f"{avg_distances[selected_task].item():.4f}"
        )

        print(
            f"Correct task selection : "
            f"{selected_task == k}"
        )

        # ============================================================
        # 4. Show ranking
        # ============================================================

        sorted_tasks = torch.argsort(avg_distances)

        print("\nDistance ranking:")

        for rank, task_idx in enumerate(sorted_tasks):

            task_idx = task_idx.item()

            print(
                f"  #{rank + 1}: "
                f"Task {task_idx + 1} "
                f"(Expert {model.task_expert_map[task_idx] + 1}) "
                f"-> "
                f"{avg_distances[task_idx].item():.4f}"
            )

        print("=" * 75)

        # ============================================================
        # 5. Classification using the ONE selected task
        # ============================================================

        correct = 0
        total = 0

        with torch.no_grad():

            for i, data in enumerate(test_loader):

                # if debug and i > 2:
                #     break

                if model.args.debug_mode and i > model.get_debug_iters():
                    break

                inputs, labels = data[0], data[1]

                inputs = inputs.to(model.device)
                labels = labels.to(model.device)

                # IMPORTANT:
                # Use ONE selected task for the entire test task
                outputs = model.predict_with_task(
                    inputs,
                    selected_task
                )

                _, pred = torch.max(
                    outputs.data,
                    1
                )

                correct += torch.sum(
                    pred == labels
                ).item()

                total += labels.shape[0]

        # ============================================================
        # 6. Results
        # ============================================================

        classification_acc = (
            correct / total * 100
            if total > 0 else 0.0
        )

        task_selection_acc = (
            100.0
            if selected_task == k
            else 0.0
        )

        accs.append(classification_acc)

        task_selection_accs.append(
            task_selection_acc
        )

        print(
            f"\n[Task-Agnostic Evaluation] "
            f"Task {k + 1}"
        )

        print(
            f"  Classification Accuracy : "
            f"{classification_acc:.2f}%"
        )

        print(
            f"  Task Selection Accuracy : "
            f"{task_selection_acc:.2f}%"
        )

        print(
            f"  Correct Classification   : "
            f"{int(correct)}/{int(total)}"
        )

        print(
            f"  Selected Task            : "
            f"Task {selected_task + 1}"
        )

        print(
            f"  Selected Expert          : "
            f"Expert {selected_expert + 1}"
        )

        print(
            f"  Selected Classifier      : "
            f"Classifier {selected_classifier + 1}"
        )

        pbar.update(1)

        pbar.set_postfix(
            {
                f'acc_task_{k + 1}': classification_acc,
                f'task_select_{k + 1}': task_selection_acc
            },
            refresh=False
        )

        pbar.set_description(
            f"Evaluating Task {k + 1}",
            refresh=False
        )

    pbar.close()

    model.net.train(status)

    return accs, task_selection_accs

    # for k, test_loader in enumerate(test_loaders):
    #     if last and k < len(test_loaders) - 1:
    #         continue
    #     correct, correct_mask_classes, total = 0.0, 0.0, 0.0
    #     test_iter = iter(test_loader)
    #     n_classes = datasets[k].N_CLASSES_PER_TASK * datasets[k].N_TASKS
    #     i = 0
    #     while True:
    #         try:
    #             data = next(test_iter)
    #         except StopIteration:
    #             break
    #         if debug:
    #             if i > 2:
    #                 break
    #         if model.args.debug_mode and i > model.get_debug_iters():
    #             break
    #         inputs, labels = data[0], data[1]
    #         inputs, labels = inputs.to(model.device), labels.to(model.device)
    #         if 'class-il' not in model.COMPATIBILITY and 'general-continual' not in model.COMPATIBILITY:
    #             outputs = model(inputs, k)
    #         else:
    #             outputs = model.myPrediction(inputs, expert_index_list[k] - 1)
    #             outputs = model.myPrediction(inputs, expert_index_list[k])
    #             outputs = model.myPrediction(
    #                 inputs,
    #                 expert_index_list[k],
    #                 classifier_index_list[k]
    #             )

    #         _, pred = torch.max(outputs[:, :n_classes].data, 1)
    #         correct += torch.sum(pred == labels).item()
    #         total += labels.shape[0]
    #         i += 1
    #         pbar.set_postfix(
    #             {f'acc_task_{k+1}': max(0, correct / total * 100)}, refresh=False)
    #         pbar.set_description(f"Evaluating Task {k+1}", refresh=False)
    #         pbar.update(1)

    #     accs.append(correct / total * 100
    #                 if 'class-il' in model.COMPATIBILITY or 'general-continual' in model.COMPATIBILITY else 0)
    #     # accs_mask_classes.append(correct_mask_classes / total * 100)
    #     accs_mask_classes.append(0)
    # pbar.close()

    # model.net.train(status)

    # return accs, accs_mask_classes


def initialize_wandb(args: Namespace) -> None:
    """
    Initializes wandb, if installed.

    Args:
        args: the arguments of the current execution
    """
    assert wandb is not None, "Wandb not installed, please install it or run without wandb"
    run_name = args.wandb_name if args.wandb_name is not None else args.model

    run_id = args.conf_jobnum.split('-')[0]
    name = f'{run_name}_{run_id}'
    mode = 'disabled' if os.getenv(
        'MAMMOTH_TEST', '0') == '1' else os.getenv('WANDB_MODE', 'online')
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               config=vars(args), name=name, mode=mode)
    args.wandb_url = wandb.run.get_url()


def _to_device(name: str, x, device):
    if isinstance(x, torch.Tensor):
        if 'label' in name.lower() or 'target' in name.lower():
            return x.to(device, dtype=torch.long)
        return x.to(device)
    return x


def train_single_epoch(model: ContinualModel,
                       train_loader: Iterable,
                       args: Namespace,
                       epoch: int,
                       pbar: tqdm,
                       system_tracker=None,
                       scheduler=None) -> int:
    """
    Trains the model for a single epoch.

    Args:
        model: the model to be trained
        train_loader: the data loader for the training set
        args: the arguments from the command line
        epoch: the current epoch
        system_tracker: the system tracker to monitor the system stats
        scheduler: the scheduler for the current epoch

    Returns:
        the number of iterations performed in the current epoch
    """
    train_iter = iter(train_loader)
    epoch_len = len(train_loader) if hasattr(train_loader, "__len__") else None

    i = 0
    previous_time = time()

    while True:
        try:
            data = next(train_iter)
        except StopIteration:
            break
        if debug:
            if i == 1000:
                break
        if args.debug_mode and i > model.get_debug_iters():
            break
        if args.fitting_mode == 'iters' and model.task_iteration >= model.args.n_iters:
            break

        inputs, labels, not_aug_inputs = data[0], data[1], data[2]
        inputs, labels = inputs.to(model.device), labels.to(
            model.device, dtype=torch.long)
        not_aug_inputs = not_aug_inputs.to(model.device)

        extra_fields = {
            train_loader.dataset.extra_return_fields[k]: _to_device(
                train_loader.dataset.extra_return_fields[k], data[3 + k], model.device)
            for k in range(len(data) - 3)
        }

        loss = model.meta_observe(
            inputs, labels, not_aug_inputs, epoch=epoch, **extra_fields)

        # assert not math.isnan(loss)

        if scheduler is not None and args.scheduler_mode == 'iter':
            scheduler.step()

        if args.code_optimization == 0 and 'cuda' in str(args.device):
            torch.cuda.synchronize()
        system_tracker()
        i += 1

        time_diff = time() - previous_time
        previous_time = time()

        if isinstance(loss, float):
            bar_log = {'loss_ce': loss, 'lr': model.opt.param_groups[0]['lr']}
        elif isinstance(loss, list) and len(loss) == 2:
            bar_log = {
                'loss_ce': loss[0], 'loss_nor': loss[1], 'lr': model.opt.param_groups[0]['lr']}
        else:
            bar_log = {'loss_ce': loss[0], 'loss_kd': loss[1], 'loss_nor': loss[2],
                       'loss_mi': loss[3], 'lr': model.opt.param_groups[0]['lr']}

        if epoch_len:
            ep_h = 3600 / (epoch_len * time_diff)
            bar_log['ep/h'] = ep_h
        pbar.set_postfix(bar_log, refresh=False)
        pbar.update()

    if scheduler is not None and args.scheduler_mode == 'epoch':
        scheduler.step()


def train(model: ContinualModel, datasets,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.

    Args:
        model: the module to be trained
        dataset: the continual dataset at hand
        args: the arguments of the current execution
    """

    print(args)

    if not args.nowand:
        initialize_wandb(args)

    if not args.disable_log:
        logger = Logger(args, datasets[0].SETTING,
                        datasets[0].NAME, model.NAME)

    model.net.to(model.device)
    torch.cuda.empty_cache()

    with track_system_stats(logger) as system_tracker:

        if args.loadcheck is not None:
            model, past_res = mammoth_load_checkpoint(args, model)

            if not args.disable_log and past_res is not None:
                (results, results_mask_classes, csvdump) = past_res
                logger.load(csvdump)

            print('Checkpoint Loaded!')

        print(file=sys.stderr)
        start_task = 0 if args.start_from is None else args.start_from
        # dataset.N_TASKS if args.stop_after is None else args.stop_after
        end_task = len(datasets)

        test_loaders = []

        torch.cuda.empty_cache()
        for t in range(start_task, end_task):

            print('begin task ' + str(t + 1) + ', dataset:' + datasets[t].NAME)

            model.net.train()

            train_loader, test_loader = datasets[t].get_all_data_loaders()

            test_loaders.append(test_loader)

            model.meta_begin_task(datasets[t])

            if not args.inference_only and args.n_epochs > 0:

                # Scheduler is automatically reloaded after each task if defined in the dataset.
                # If the model defines it, it becomes the job of the model to reload it.
                scheduler = get_scheduler(model, args, reload_optim=True) if not hasattr(
                    model, 'scheduler') else model.scheduler

                epoch = 0

                n_iterations = None
                if not isinstance(datasets[0], GCLDataset):
                    n_iterations = model.args.n_epochs * \
                        len(train_loader) if model.args.fitting_mode == 'epochs' else model.args.n_iters
                mininterval = 0.2 if n_iterations is not None and n_iterations > 1000 else 0.1
                train_pbar = tqdm(train_loader, total=n_iterations,  # train_loader is actually ignored, will update the progress bar manually
                                  disable=args.non_verbose, mininterval=mininterval)
                if args.non_verbose:
                    # at least print the task number
                    logging.info(f"Task {t + 1}")

                while True:
                    model.begin_epoch(epoch, datasets[t])

                    train_pbar.set_description(
                        f"Task {t + 1} - Epoch {epoch + 1}")

                    train_single_epoch(model, train_loader, args, pbar=train_pbar, epoch=epoch,
                                       system_tracker=system_tracker, scheduler=scheduler)

                    model.end_epoch(epoch, datasets[t])

                    epoch += 1
                    if args.fitting_mode == 'epochs' and epoch >= model.args.n_epochs:
                        break

                train_pbar.close()

            model.meta_end_task(datasets[t])

            accs = evaluate(model, datasets, test_loaders)

            # logged_accs = eval_dataset.log(args, logger, accs, t, dataset.SETTING)
            log_accs(args, logger, accs, t, datasets[t].SETTING)

        system_tracker.print_stats()

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()
