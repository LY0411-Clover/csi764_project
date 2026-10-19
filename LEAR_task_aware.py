# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import os
from datasets import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
from backbone.utils.hsic import hsic
import torch.nn.functional as F
import random
import copy
from torch.distributions import MultivariateNormal
from scipy.stats import multivariate_normal


class LEAR(ContinualModel):
    NAME = 'LEAR'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--threshold', type=float, default=38.0,
                            help='Mahalanobis threshold for expert allocation.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(LEAR, self).__init__(backbone, loss,
                                   args, transform, dataset=dataset)

        self.train_loader_size = None
        self.iter = 0
        self.task_expert_map = []
        self.task_classifier_map = []

    # ------------------------------------------------------------------
    # Routing helpers. net.distributions has ONE entry per task, so task t
    # must be paired with fcArr[task_expert_map[t]], never with fcArr[t].
    # ------------------------------------------------------------------
    def _class_token(self, x):
        processX = self.net.vitProcess(x)
        f = self.net.local_vitmodel.patch_embed(processX)
        cls = self.net.local_vitmodel.cls_token.expand(f.shape[0], -1, -1)
        f = torch.cat((cls, f), dim=1) + self.net.local_vitmodel.pos_embed
        for block in self.net.local_vitmodel.blocks[:-3]:
            f = block(f)
        f = self.net.Forever_freezed_blocks(f)
        f = self.net.local_vitmodel.norm(f)
        return f[:, 0, :]

    def _task_distances(self, class_token):
        """Mahalanobis distance to every stored TASK distribution -> [B, num_tasks]."""
        out = []
        for t, dist in enumerate(self.net.distributions):
            fc = self.net.fcArr[self.task_expert_map[t]]
            delta = fc(class_token) - dist.mean
            m2 = torch.einsum('bi,ij,bj->b', delta, dist.precision_matrix, delta)
            out.append(m2.clamp_min(0).sqrt())
        return torch.stack(out, dim=1)

    def end_task(self, dataset) -> None:
        # calculate distribution
        train_loader = dataset.train_loader
        num_choose = 100
        with torch.no_grad():
            train_iter = iter(train_loader)

            pbar = tqdm(train_iter, total=num_choose,
                        desc=f"Calculate distribution for task {self.current_task + 1}",
                        disable=False, mininterval=0.5)

            fc_features_list = []

            count = 0
            while count < num_choose:
                try:
                    data = next(train_iter)
                except StopIteration:
                    break

                x = data[0]
                x = x.to(self.device)

                processX = self.net.vitProcess(x)

                features = self.net.local_vitmodel.patch_embed(processX)
                cls_token = self.net.local_vitmodel.cls_token.expand(
                    features.shape[0], -1, -1)
                features = torch.cat((cls_token, features), dim=1)
                features = features + self.net.local_vitmodel.pos_embed

                # forward pass till -3
                for block in self.net.local_vitmodel.blocks[:-3]:
                    features = block(features)

                features = self.net.Forever_freezed_blocks(features)

                features = self.net.local_vitmodel.norm(features)

                class_token = features[:, 0, :]

                fc_features_list.append(
                    self.net.fcArr[self.net.c_expert](class_token))

                count += 1
                pbar.update()

            pbar.close()
            fc_features = torch.cat(fc_features_list, dim=0)  # [num*b,fc_size]
            mu = torch.mean(fc_features, dim=0)
            sigma = torch.cov(fc_features.T)
            self.net.distributions.append(MultivariateNormal(mu, sigma))

        # deal with grad and blocks
        for fc in self.net.fcArr:
            for param in fc.parameters():
                param.requires_grad = False

        for cls in self.net.classifierArr:
            for param in cls.parameters():
                param.requires_grad = False

        self.net.Freezed_local_blocks = copy.deepcopy(
            torch.nn.Sequential(*list(self.net.local_vitmodel.blocks[-3:])))
        self.net.Freezed_global_blocks = copy.deepcopy(
            torch.nn.Sequential(*list(self.net.global_vitmodel.blocks[-3:])))

        for block in self.net.Freezed_local_blocks:
            for param in block.parameters():
                param.requires_grad = False

        for block in self.net.Freezed_global_blocks:
            for param in block.parameters():
                param.requires_grad = False

        self.print_gpu_memory(
            f"After Task {self.current_task + 1}"
        )
        self.print_model_memory()

    def begin_task(self, dataset, threshold=None) -> None:
        if threshold is None:
            threshold = getattr(self.args, 'threshold', 38.0)
        print(f'[CONFIG] threshold = {threshold}')
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        train_loader = dataset.train_loader

        # ============================================================
        # Task 1: always use the first expert
        # ============================================================
        if self.current_task == 0:
            self.net.c_expert = 0

            # Record Task 1 -> Expert 1
            self.task_expert_map.append(self.net.c_expert)
            # Task 1 -> Classifier 0
            self.task_classifier_map.append(0)

            print(
                f"[DEBUG] Task {self.current_task + 1} -> "
                f"Expert {self.net.c_expert + 1}"
            )
            print(
                f"[DEBUG] Task-expert mapping: "
                f"{[x + 1 for x in self.task_expert_map]}"
            )
            print(
                f"[DEBUG] Task-classifier mapping: "
                f"{[x + 1 for x in self.task_classifier_map]}"
            )

        # ============================================================
        # Task 2+: determine whether to reuse or create an expert
        # ============================================================
        else:
            num_choose = 50

            with torch.no_grad():
                train_iter = iter(train_loader)

                pbar = tqdm(
                    train_iter,
                    total=num_choose,
                    desc=f"Choose params for task {self.current_task + 1}",
                    disable=False,
                    mininterval=0.5
                )

                count = 0
                sum_dist = torch.zeros(len(self.net.distributions), device=self.device)
                n_samples = 0

                while count < num_choose:
                    try:
                        data = next(train_iter)
                    except StopIteration:
                        break

                    x = data[0]
                    x = x.to(self.device)

                    processX = self.net.vitProcess(x)

                    features = self.net.local_vitmodel.patch_embed(processX)

                    cls_token = self.net.local_vitmodel.cls_token.expand(
                        features.shape[0], -1, -1
                    )

                    features = torch.cat(
                        (cls_token, features),
                        dim=1
                    )

                    features = (
                        features
                        + self.net.local_vitmodel.pos_embed
                    )

                    # Forward pass till -3
                    for block in self.net.local_vitmodel.blocks[:-3]:
                        features = block(features)

                    features = self.net.Forever_freezed_blocks(features)

                    features = self.net.local_vitmodel.norm(features)

                    class_token = features[:, 0, :]

                    d = self._task_distances(class_token)   # [B, num_tasks]
                    sum_dist += d.sum(dim=0)                # accumulate over ALL batches
                    n_samples += d.shape[0]
                    count += 1

                    pbar.set_postfix(
                        {'distances': [round(v, 2) for v in (sum_dist / n_samples).tolist()]},
                        refresh=False
                    )
                    pbar.update()

                pbar.close()

                # ----------------------------------------------------
                # Find closest existing expert
                # ----------------------------------------------------
                avg_distances = sum_dist / n_samples
                min_task = torch.argmin(avg_distances).item()    # nearest TASK
                nearest_expert = self.task_expert_map[min_task]  # the expert it used
                min_distance = avg_distances[min_task].item()

                print(f"Closest task: {min_task + 1} (expert {nearest_expert + 1})")
                print(f"Minimum Mahalanobis distance: {min_distance:.4f}")

                # ----------------------------------------------------
                # Adaptive expansion
                # ----------------------------------------------------
                if min_distance > threshold:

                    print(
                        f"Distance {min_distance:.4f} > "
                        f"threshold {threshold:.4f}"
                    )

                    print(
                        "Task considered novel. "
                        "Creating new expert."
                    )

                    self.net.CreateNewExper(
                        nearest_expert,
                        # dataset.N_CLASSES
                    )

                    # New expert is the last expert
                    self.net.c_expert = len(self.net.fcArr) - 1
                    # self.net.c_classifier = len(self.net.classifierArr) - 1
                    # Freeze backbone except last 3 blocks
                    self.net.unfreeze_last_blocks()

                    print("\n========== AFTER NEW EXPERT: GRAD STATUS ==========")

                    for name, param in self.net.fcArr[self.net.c_expert].named_parameters():
                        print(
                            f"New Expert - {name}: "
                            f"requires_grad={param.requires_grad}"
                        )

                    for i, block in enumerate(self.net.local_vitmodel.blocks[-3:]):
                        print(
                            f"Local block {-3+i}: "
                            f"{[p.requires_grad for p in block.parameters()]}"
                        )

                    for i, block in enumerate(self.net.global_vitmodel.blocks[-3:]):
                        print(
                            f"Global block {-3+i}: "
                            f"{[p.requires_grad for p in block.parameters()]}"
                        )

                    print("====================================================\n")

                else:

                    print(
                        f"Distance {min_distance:.4f} <= "
                        f"threshold {threshold:.4f}"
                    )

                    print(
                        "Task considered similar. "
                        "Reusing existing expert."
                    )

                    self.net.c_expert = nearest_expert

                    # Freeze reused FC expert
                    for param in self.net.fcArr[nearest_expert].parameters():
                        param.requires_grad = False
                    # Freeze backbone except last 3 blocks
                    self.net.unfreeze_last_blocks()

                    print("\n========== AFTER REUSE: GRAD STATUS ==========")

                    # Expert FC
                    for name, param in self.net.fcArr[nearest_expert].named_parameters():
                        print(
                            f"Expert FC - {name}: requires_grad={param.requires_grad}")

                    # Local last 3 blocks
                    for i, block in enumerate(self.net.local_vitmodel.blocks[-3:]):
                        print(
                            f"Local block {-3+i}: "
                            f"{[p.requires_grad for p in block.parameters()]}"
                        )

                    # Global last 3 blocks
                    for i, block in enumerate(self.net.global_vitmodel.blocks[-3:]):
                        print(
                            f"Global block {-3+i}: "
                            f"{[p.requires_grad for p in block.parameters()]}"
                        )

                    print("===============================================\n")

                    # Create a new classifier for this task
                    # self.net.CreateNewClassifier(
                    #     dataset.N_CLASSES
                    # )

                # ----------------------------------------------------
                # Record task -> expert mapping
                # ----------------------------------------------------
                self.net.CreateNewClassifier(
                    dataset.N_CLASSES
                )

                self.net.c_classifier = len(
                    self.net.classifierArr
                ) - 1

                self.task_expert_map.append(
                    self.net.c_expert
                )

                self.task_classifier_map.append(
                    self.net.c_classifier
                )

                print("\n========== FINAL GRAD STATUS ==========")

                for i, expert in enumerate(self.net.fcArr):
                    print(
                        f"Expert {i+1}: "
                        f"{[p.requires_grad for p in expert.parameters()]}"
                    )

                for i, classifier in enumerate(self.net.classifierArr):
                    print(
                        f"Classifier {i+1}: "
                        f"{[p.requires_grad for p in classifier.parameters()]}"
                    )

                print("=======================================\n")

                print(
                    f"[DEBUG] Task {self.current_task + 1} -> "
                    f"Expert {self.net.c_expert + 1}"
                )

                print(
                    f"[DEBUG] Task-expert mapping: "
                    f"{[x + 1 for x in self.task_expert_map]}"
                )

                print(
                    f"[DEBUG] Task-classifier mapping: "
                    f"{[x + 1 for x in self.task_classifier_map]}"
                )

        # ============================================================
        # Optimizer
        # ============================================================
        self.opt = self.get_optimizer()

        self.print_model_memory()

    # def begin_task(self, dataset, threshold=38.0) -> None:
    #     train_loader = dataset.train_loader
    #     if self.current_task > 0:
    #         num_choose = 50
    #         with torch.no_grad():
    #             train_iter = iter(train_loader)

    #             pbar = tqdm(train_iter, total=num_choose,
    #                         desc=f"Choose params for task {self.current_task + 1}",
    #                         disable=False, mininterval=0.5)

    #             count = 0
    #             while count < num_choose:
    #                 try:
    #                     data = next(train_iter)
    #                 except StopIteration:
    #                     break

    #                 x = data[0]
    #                 x = x.to(self.device)

    #                 processX = self.net.vitProcess(x)

    #                 features = self.net.local_vitmodel.patch_embed(processX)
    #                 cls_token = self.net.local_vitmodel.cls_token.expand(
    #                     features.shape[0], -1, -1)
    #                 features = torch.cat((cls_token, features), dim=1)
    #                 features = features + self.net.local_vitmodel.pos_embed

    #                 # forward pass till -3
    #                 for block in self.net.local_vitmodel.blocks[:-3]:
    #                     features = block(features)

    #                 features = self.net.Forever_freezed_blocks(features)

    #                 features = self.net.local_vitmodel.norm(features)

    #                 class_token = features[:, 0, :]
    #                 distances = [0] * len(self.net.fcArr)
    #                 for t, (fc, dist) in enumerate(zip(self.net.fcArr, self.net.distributions)):
    #                     fc_feature = fc(class_token)
    #                     delta = fc_feature - dist.mean
    #                     inv_cov = torch.linalg.inv(dist.covariance_matrix)
    #                     mahalanobis = torch.sqrt(
    #                         delta @ inv_cov @ delta.T).diagonal()
    #                     distances[t] += mahalanobis.mean()

    #                 count += 1
    #                 bar_log = {'distances': [
    #                     round((x / count).item(), 2) for x in distances]}
    #                 pbar.set_postfix(bar_log, refresh=False)
    #                 pbar.update()
    #             pbar.close()

    #             min_idx = torch.argmin(torch.tensor(distances)).item()
    #             # self.net.CreateNewExper(min_idx, dataset.N_CLASSES)

    #             min_distance = distances[min_idx]

    #             print(
    #                 f"Closest expert: {min_idx}"
    #             )

    #             print(
    #                 f"Minimum Mahalanobis distance: "
    #                 f"{min_distance:.4f}"
    #             )

    #             # -------------------------
    #             # Adaptive expansion
    #             # -------------------------

    #             if min_distance > threshold:
    #                 print(
    #                     f"Distance {min_distance:.4f} > "
    #                     f"threshold {threshold:.4f}"
    #                 )

    #                 print(
    #                     "Task considered novel. "
    #                     "Creating new expert."
    #                 )

    #                 self.net.CreateNewExper(
    #                     min_idx,
    #                     dataset.N_CLASSES
    #                 )
    #                 self.net.c_expert = len(self.net.fcArr) - 1
    #             else:
    #                 print(
    #                     f"Distance {min_distance:.4f} <= "
    #                     f"threshold {threshold:.4f}"
    #                 )

    #                 print(
    #                     "Task considered similar. "
    #                     "Reusing existing expert."
    #                 )
    #                 self.net.c_expert = min_idx
    #             # Record which expert is assigned to this task
    #             self.task_expert_map.append(self.net.c_expert)

    #             print(
    #                 f"[DEBUG] Task {self.current_task + 1} -> "
    #                 f"Expert {self.net.c_expert + 1}"
    #             )
    #             print(
    #                 f"[DEBUG] Task-expert mapping: "
    #                 f"{[x + 1 for x in self.task_expert_map]}"
    #             )

    #     self.opt = self.get_optimizer()

    # def myPrediction(self, x, k):
    #     with torch.no_grad():
    #         # Perform the prediction according to the seloeced expert
    #         out = self.net.myprediction(x, k)
    #     return out
    def myPrediction(self, x, expert_index, classifier_index):
        with torch.no_grad():

            out = self.net.myprediction(
                x,
                expert_index,
                classifier_index
            )
        return out

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        l2_distance = torch.nn.MSELoss()

        self.opt.zero_grad()
        if self.net.Freezed_local_blocks is not None:   # teacher exists from Task 2 on
            outputs, Freezed_global_features, Freezed_local_features, global_features, local_features = self.net(
                inputs, return_features=True)
            loss_kd = kl_loss(local_features, Freezed_local_features)
            # Directly calculate the L2 distance between features is more efficient than calculate MI between prediction, and it's also effective
            loss_mi = l2_distance(global_features, Freezed_global_features)
            loss_hsic = -hsic(global_features, local_features)
            loss_ce = self.loss(outputs, labels)
            loss_tot = loss_ce + loss_kd + loss_hsic + loss_mi
            loss_vis = [loss_ce.item(), loss_kd.item(),
                        loss_hsic.item(), loss_mi.item()]
        else:
            outputs, global_features, local_features = self.net(inputs)
            loss_hsic = -hsic(global_features, local_features)
            loss_ce = self.loss(outputs, labels)
            loss_tot = loss_ce + loss_hsic
            loss_vis = [loss_ce.item(), loss_hsic.item()]

        loss_tot.backward()

        self.opt.step()

        return loss_vis

    def cal_expert_dist(self, x):
        return self._task_distances(self._class_token(x)).mean(dim=0).tolist()

    def print_gpu_memory(self, tag=""):
        if not torch.cuda.is_available():
            print("CUDA is not available.")
            return

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        max_reserved = torch.cuda.max_memory_reserved() / 1024**3

        print(f"\n========== GPU Memory: {tag} ==========")
        print(f"Allocated       : {allocated:.2f} GB")
        print(f"Reserved        : {reserved:.2f} GB")
        print(f"Peak Allocated  : {max_allocated:.2f} GB")
        print(f"Peak Reserved   : {max_reserved:.2f} GB")
        print("========================================\n")

    def print_model_memory(self):
        param_memory = sum(
            p.numel() * p.element_size()
            for p in self.net.parameters()
        ) / 1024**2

        buffer_memory = sum(
            b.numel() * b.element_size()
            for b in self.buffers()
        ) / 1024**2

        print("\n========== Model Memory ==========")
        print(f"Parameters : {param_memory:.2f} MB")
        print(f"Buffers    : {buffer_memory:.2f} MB")
        print(f"Total      : {param_memory + buffer_memory:.2f} MB")
        print("==================================")


def kl_loss(student_feat, teacher_feat):
    student_feat = F.normalize(student_feat, p=2, dim=1)
    teacher_feat = F.normalize(teacher_feat, p=2, dim=1)

    student_prob = (student_feat + 1) / 2
    teacher_prob = (teacher_feat.detach() + 1) / 2

    loss_kld = F.kl_div(
        torch.log(student_prob + 1e-10),
        teacher_prob,
        reduction='batchmean'
    )
    return loss_kld