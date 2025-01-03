# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code adapted from https://github.com/nerfstudio-project/nerfstudio/

from __future__ import annotations

from pathlib import Path
import random
import socket
import traceback
from datetime import timedelta
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# import tyro
import yaml

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion
from nerfstudio.utils import comms, profiler
from nerfstudio.utils.rich_utils import CONSOLE

from nerfacto_v.nerf_trainer import TrainerConfig
from nerfacto_v.nerf_config import nerf_method

DEFAULT_TIMEOUT = timedelta(minutes=30)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore






def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_loop(local_rank: int, world_size: int, config: TrainerConfig, global_rank: int = 0):
    """Main training function that sets up and runs the trainer per process

    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """
    _set_random_seed(config.machine.seed + global_rank)
    trainer = config.setup(local_rank=local_rank, world_size=world_size)
    trainer.setup()
    trainer.train()


def launch(
    main_func: Callable,
    num_devices_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: str = "auto",
    config: Optional[TrainerConfig] = None,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> None:
    """Function that spawns multiple processes to call on main_func

    Args:
        main_func (Callable): function that will be called by the distributed workers
        num_devices_per_machine (int): number of GPUs per machine
        num_machines (int, optional): total number of machines
        machine_rank (int, optional): rank of this machine.
        dist_url (str, optional): url to connect to for distributed jobs.
        config (TrainerConfig, optional): config file specifying training regimen.
        timeout (timedelta, optional): timeout of the distributed workers.
        device_type: type of device to use for training.
    """
    assert config is not None
    world_size = num_machines * num_devices_per_machine
    if world_size == 0:
        raise ValueError("world_size cannot be 0")
    elif world_size == 1:
        main_func(local_rank=0, world_size=world_size, config=config)
    elif world_size > 1:
        raise NotImplementedError


def main(config: TrainerConfig, args) -> None:
    """Main function."""

    if args.data:
        CONSOLE.log("Using --data alias for --data.pipeline.datamanager.data")
        config.pipeline.datamanager.data = Path(args.data)

    if args.load_dir:
        CONSOLE.log("Using --load_dir alias for --config.load_dir")
        CONSOLE.log("Only load model from pretrained one")
        config.load_dir = Path(args.load_dir)
        config.load_step = config.steps_pretrain

    config.pipeline.base_pool_num = args.base_pool_num
    config.pipeline.subset_prop = args.subset_prop
    config.pipeline.lambda_cross_attn = args.lambda_cross_attn
    config.pipeline.seed = args.seed
    config.pipeline.lambda_patch = args.lambda_patch

    config.set_timestamp()

    config.print_to_terminal()
    config.save_config()

    launch(
        main_func=train_loop,
        num_devices_per_machine=config.machine.num_devices,
        device_type=config.machine.device_type,
        num_machines=config.machine.num_machines,
        machine_rank=config.machine.machine_rank,
        dist_url=config.machine.dist_url,
        config=config,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser, Namespace
    import sys
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--load_config', type=str, default=None)
    parser.add_argument('--base_pool_num', type=int, default=5)
    parser.add_argument('--subset_prop', type=float, default=0.4)
    parser.add_argument('--lambda_cross_attn', type=float, default=0.1)
    parser.add_argument('--lambda_patch', type=float, default=0.01)
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)
    config = nerf_method.config
    # print(args)
    main(config, args)
