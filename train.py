"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import now

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", default='./train_configs/MiniGPT_3D/stage_1.yaml',
                        help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    args = parse_args()
    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)


    #####################  show trained param.  #####
    print("")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter {name} will be updated.")

    print()
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Number of trainable parameters: {num_trainable_params}")

    num_trainable_params = sum(p.numel() for p in model.llama_model.parameters() if p.requires_grad)
    print(f"    llama_model: Number of trainable parameters: {num_trainable_params}")

    num_trainable_params = sum(p.numel() for p in model.llama_proj.parameters() if p.requires_grad)
    print(f"    llama_proj: Number of trainable parameters: {num_trainable_params}")

    num_trainable_params = sum(p.numel() for p in model.llama_proj2.parameters() if p.requires_grad)
    print(f"    llama_proj2: Number of trainable parameters: {num_trainable_params}")

    num_trainable_params = sum(p.numel() for p in model.point_2_Qformer_proj.parameters() if p.requires_grad)
    print(f"    point_2_Qformer_proj: Number of trainable parameters: {num_trainable_params}")

    num_trainable_params = sum(p.numel() for p in model.Qformer.parameters() if p.requires_grad)
    print(f"    Qformer: Number of trainable parameters: {num_trainable_params}")


    if model.only_train_MQE:
        if model.query_expert_num > 1:
            num_trainable_params = sum(p.numel() for p in model.query_router.parameters() if p.requires_grad)
            print(f"    query_router: Number of trainable parameters: {num_trainable_params}")

            num_trainable_params = 0
            for i in range(model.query_expert_num):
                num_trainable_params = model.query_expert_dict[str(i)].numel() + num_trainable_params
            print(f"    query_expert: Number of trainable parameters: {num_trainable_params}")

        elif model.query_tokens.requires_grad:
            print(f"    query_tokens: Number of trainable parameters: {model.query_tokens.numel()}")

    print("")



    if cfg.run_cfg.wandb_log:
        wandb.login()
        wandb.init(project="minigpt_3d", name=cfg.run_cfg.job_name)
        wandb.watch(model)


    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
