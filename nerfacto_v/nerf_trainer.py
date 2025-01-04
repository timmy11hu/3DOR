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

import functools
import os
import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Type, cast

from rich import box, style
from rich.panel import Panel
from rich.table import Table
import numpy as np
import torch
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.engine.callbacks import TrainingCallbackLocation
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import check_eval_enabled
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.engine.optimizers import OptimizerConfig

from .render import RenderSpiral
from .nerf_helper import to8b
import imageio
from utils.writer import write_output_json

TRAIN_INTERATION_OUTPUT = Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]


@dataclass
class RMSpropOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with RMSprop"""
    _target: Type = torch.optim.RMSprop


@dataclass
class NerfTrainerConfig(TrainerConfig):
    _target: Type = field(default_factory=lambda: NerfTrainer)
    save_only_latest_checkpoint: bool = False
    steps_per_video: int = 2000
    steps_pretrain: int = 5000
    steps_reftrain: int = 10000
    steps_progtrain: int = 20000
    steps_per_train_image: int = 1000,
    steps_per_eval_image: int = 1000,

    def get_base_dir(self) -> Path:
        """Retrieve the base directory to set relative paths"""
        # check the experiment and method names
        assert self.method_name is not None, "Please set method name in config or via the cli"
        self.set_experiment_name()
        return Path(f"{self.output_dir}/{self.experiment_name}/{self.method_name}")



class NerfTrainer(Trainer):

    def __init__(self, config: NerfTrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        super().__init__(
            config=config, local_rank=local_rank, world_size=world_size
        )
        if self.config.load_dir is None:
            load_dir = Path(f"{self.base_dir}/nerfstudio_models")
            if os.path.exists(load_dir):
                self.config.load_dir = load_dir
        self.config.logging.relative_log_dir = "tensorboard" # tensorboard writer

        self.video_render = RenderSpiral()
        self.stage_pretrain = True
        self.keep_update_inp = True

    def train(self) -> None:
        """Train the model."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"

        if self.pipeline.need_spherify():
            self.config.steps_reftrain = self.config.steps_pretrain

        # don't want to call save_dataparser_transform if pipeline's datamanager does not have a dataparser
        if isinstance(self.pipeline.datamanager, VanillaDataManager):
            self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
                self.base_dir / "dataparser_transforms.json"
            )

        self._init_viewer_state()
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.max_num_iterations + 1
            step = 0
            for step in range(self._start_step, num_iterations):
                while self.training_state == "paused":
                    time.sleep(0.01)
                with self.train_lock:
                    with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                        self.pipeline.train()

                        # training callbacks before the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                            )

                        # time the forward pass
                        loss, loss_dict, metrics_dict = self.train_iteration(step)
                        # print(loss_dict.keys())

                        # training callbacks after the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                            )

                # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.world_size
                        * self.pipeline.datamanager.get_train_rays_per_batch()
                        / max(0.001, train_t.duration),
                        step=step,
                        avg_over_steps=True,
                    )

                self._update_viewer_state(step)

                # a batch of train rays
                if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)
                    # The actual memory allocated by Pytorch. This is likely less than the amount
                    # shown in nvidia-smi since some unused memory can be held by the caching
                    # allocator and some context needs to be created on GPU. See Memory management
                    # (https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
                    # for more details about GPU memory management.
                    writer.put_scalar(
                        name="GPU Memory (MB)", scalar=torch.cuda.max_memory_allocated() / (1024**2), step=step
                    )

                # if step_check(step, self.config.steps_per_save) and step <= self.config.steps_reftrain:
                if step == self.config.steps_pretrain or step == self.config.steps_reftrain:
                    self.save_checkpoint(step)

                if step_check(step, self.config.steps_per_video):
                    self.video_render.render(self.pipeline, self.base_dir, step)

                if not self.stage_pretrain and step >= self.config.steps_reftrain:
                    self.eval_iteration(step)

                # if self.pipeline.need_spherify() and self.stage_pretrain and ((step == self.config.steps_pretrain) or (step-1 == self.config.steps_pretrain)):
                #     self.train_vis_grid()

                # if self.stage_pretrain and step >= self.config.steps_pretrain:
                #     # self.pipeline.predict_pseudo_depth(self.base_dir)
                #     self.reset_model()
                #     self.pipeline.enable_mask_training()
                #     self.pipeline.enable_pseudo_depth(self.base_dir)
                #     self.stage_pretrain = False

                if self.stage_pretrain and step >= self.config.steps_pretrain:
                    self.pipeline.inpaint_base_frame(step, self.base_dir)
                    self.stage_pretrain = False
                    if step < self.config.steps_reftrain:
                        self.reset_model()


                if self.keep_update_inp and step >= self.config.steps_reftrain:
                    self.pipeline.inpaint_left_frames(step, self.base_dir)
                    self.keep_update_inp = False
                    # if not self.pipeline.need_spherify():
                    self.reset_model()

                writer.write_out_storage()

        # save checkpoint at the end of training
        self.save_checkpoint(step)

        writer.write_out_storage()

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("Config File", str(self.config.get_base_dir() / "config.yml"))
        table.add_row("Checkpoint Directory", str(self.checkpoint_dir))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Training Finished :tada:[/bold]", expand=False))

        # after train end callbacks
        for callback in self.callbacks:
            callback.run_callback_at_location(step=step, location=TrainingCallbackLocation.AFTER_TRAIN)

        if not self.config.viewer.quit_on_train_completion:
            self._train_complete_viewer()

    def reset_model(self):
        print("Trainer: reset the model!!!")
        camera_optimizer = copy.deepcopy(self.pipeline.model.camera_optimizer) # keep the camera optimizer but stop undating
        if camera_optimizer.config.mode in ("SO3xR3", "SE3"):
            camera_optimizer.pose_adjustment.requires_grad = False
        vis_grid = copy.deepcopy(self.pipeline.model.vis_grid)
        self.pipeline.model.populate_modules()
        self.pipeline.model.camera_optimizer = camera_optimizer
        self.pipeline.model.vis_grid = vis_grid
        self.pipeline.model.to(self.pipeline.device)
        self.optimizers = self.setup_optimizers()
        from nerfstudio.engine.callbacks import TrainingCallbackAttributes
        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers, grad_scaler=self.grad_scaler, pipeline=self.pipeline, trainer=self
            )
        )

    def is_train_d(self):
        return self.pipeline.model.config.adv_loss_mult > 0.0

    @profiler.time_function
    def backward_and_update_d(self, d_loss):
        d_loss.backward()
        if has_finite_gradients(self.pipeline.model.discriminator):
            self.optimizers.optimizer_step("discriminator")

    @profiler.time_function
    def backward_and_update_nerf(self, step, loss):
        self.grad_scaler.scale(loss).backward()  # type: ignore
        if has_optimizer_for(self.optimizers, "proposal_networks"):
            optimizer_scaler_step(self.optimizers, "proposal_networks", self.grad_scaler)
        optimizer_scaler_step(self.optimizers, "fields", self.grad_scaler)
        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            self.optimizers.scheduler_step_all(step)

    # def train_vis_grid(self):
    #     print("Trainer: training visibility grid...")
    #     self.pipeline.start_vis_training()
    #     for step in range(500):
    #         with self.train_lock:
    #             self.pipeline.train()
    #             # training callbacks before the training iteration
    #             for callback in self.callbacks:
    #                 callback.run_callback_at_location(
    #                     step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
    #                 )
    #             # time the forward pass
    #             loss, loss_dict, metrics_dict = self.train_iteration(step)
    #             # training callbacks after the training iteration
    #             for callback in self.callbacks:
    #                 callback.run_callback_at_location(
    #                     step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
    #                 )
    #     self.pipeline.end_vis_training()


    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """

        self.optimizers.zero_grad_all()
        cpu_or_cuda_str: str = self.device.split(":")[0]
        # assert (
        #     self.gradient_accumulation_steps > 0
        # ), f"gradient_accumulation_steps must be > 0, not {self.gradient_accumulation_steps}"
        for group in self.optimizers.parameters.keys():
            assert (
                    self.gradient_accumulation_steps[group] > 0
            ), f"gradient_accumulation_steps must be > 0, not {self.gradient_accumulation_steps[group]}"
        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            # run forward pass
            output, metrics_dict, batch = self.pipeline.get_prediction(step=step)
            step_d = len(self.pipeline.i_conf) > 1
            # compute loss for discriminator
            if self.is_train_d() and step_d:
                d_loss_dict = self.pipeline.get_discriminator_train_loss_dict(step, output, batch, metrics_dict) # Todo: if use base frame as real sample
                d_loss = functools.reduce(torch.add, d_loss_dict.values())
        # update discriminator
        if self.is_train_d() and step_d:
            self.backward_and_update_d(d_loss)

        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            # compute loss for nerf
            #Todo: modify batch to have real image for base image
            output, loss_dict, metrics_dict = self.pipeline.get_nerf_train_loss_dict(step, output, batch, metrics_dict)

            if self.stage_pretrain and self.pipeline.need_spherify():
                density_loss = self.pipeline.get_object_bbox_train_loss_dict(step)
                loss_dict.update(density_loss)

            loss = functools.reduce(torch.add, loss_dict.values())

        # update nerf
        self.backward_and_update_nerf(step, loss)

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad  # type: ignore
                    total_grad += grad

            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)  # type: ignore

        # Merging loss and metrics dict into a single output.
        if self.is_train_d() and step_d:
            loss_dict.update(d_loss_dict)
        
        return loss, loss_dict, metrics_dict  # type: ignore

    def debug(self, step=0):
        # self.train_iteration(step)
        # self.pipeline.inpaint_base_frame(step, self.base_dir)
        # self.train_iteration(step)
        # self.pipeline.inpaint_left_frames(step, self.base_dir)
        # self.train_iteration(step)
        self.video_render.render(self.pipeline, self.base_dir, step)
        exit()


    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        if step_check(step, self.config.steps_per_train_image):
            ### transet_conf
            output_path = Path(f"{self.base_dir}/{step:05d}/trainset_conf")
            output_path.mkdir(parents=True, exist_ok=True)
            for use_embedding in (True, False):
                suffix = "emb" if use_embedding else "null"
                imgs_tensor, _ = self.pipeline.get_eval_images(step=step,
                                                            num_per_eval=10,
                                                            use_appearance=use_embedding,
                                                            use_conf=True)
                for img_i, image_tensor in imgs_tensor.items():
                    output_fn = f"{output_path}/{img_i:03d}_{suffix}.png"
                    image = image_tensor.cpu().numpy()
                    image = to8b(image)
                    imageio.imwrite(output_fn, image)

            ### transet_supp
            output_path = Path(f"{self.base_dir}/{step:05d}/trainset_supp")
            output_path.mkdir(parents=True, exist_ok=True)
            imgs_tensor, _ = self.pipeline.get_eval_images(step=step,
                                                        num_per_eval=10,
                                                        use_appearance=False,
                                                        use_conf=False)
            for img_i, image_tensor in imgs_tensor.items():
                output_fn = f"{output_path}/{img_i:03d}.png"
                image = image_tensor.cpu().numpy()
                image = to8b(image)
                imageio.imwrite(output_fn, image)


        if step_check(step, self.config.steps_per_eval_image):
            ### testset
            output_path = Path(f"{self.base_dir}/{step:05d}/testset")
            output_path.mkdir(parents=True, exist_ok=True)
            imgs_tensor, metrics_dict, fid = self.pipeline.get_eval_images(step=step,
                                                                      eval_on_trainset=False,
                                                                      num_per_eval=1,
                                                                      use_appearance=False,
                                                                      use_conf=False,
                                                                      fid_tmp_dir=Path(f"{self.base_dir}/tmp_res"))
            for img_i, image_tensor in imgs_tensor.items():
                output_fn = f"{output_path}/{img_i:03d}.png"
                image = image_tensor.cpu().numpy()
                image = to8b(image)
                imageio.imwrite(output_fn, image)

            import json
            psnr, ssim, lpips = [], [], []
            musiq = []
            with open(f"{output_path}/metrics.txt", 'w') as file:
                for img_i, metrics in metrics_dict.items():
                    file.write(f"{img_i:03d}: {json.dumps(metrics)}\n")
                    psnr.append(metrics["psnr"])
                    ssim.append(metrics["ssim"])
                    lpips.append(metrics["lpips"])
                    musiq.append(metrics["musiq"])

            res = {"psnr": np.round(np.mean(psnr), 3),
                   "ssim": np.round(np.mean(ssim), 3),
                   "lpips": np.round(np.mean(lpips), 3),
                   "musiq": np.round(np.mean(musiq), 3),
                   "fid": np.round(fid, 3)}
            with open(f"{self.base_dir}/metrics.txt", 'a') as file:
                file.write(f"{step:05d}: {json.dumps(res)}\n")


    def run_eval_on_all_images(self, step):
        metrics_dict = self.pipeline.get_average_eval_image_metrics(step=step, write_path=self.config.get_base_dir())
        output_path = os.path.join(self.config.get_base_dir(), f"output_{step}.json")
        write_output_json(
            self.config.experiment_name,
            self.config.method_name,
            str(self.config.get_checkpoint_dir()),
            metrics_dict,
            output_path,
        )
        for k, v in metrics_dict.items():
            CONSOLE.print(f"{k}: {v}")
        writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step)

def has_finite_gradients(net, filter=""):
    for n, params in net.named_parameters():
        if filter in n:
            if params.grad is not None and not params.grad.isfinite().all():
                return False
    return True

def optimizer_scaler_step(optimizers, param_group_name, grad_scaler: GradScaler) -> None:
    max_norm = optimizers.config[param_group_name]["optimizer"].max_norm
    if max_norm is not None:
        grad_scaler.unscale_(optimizers.optimizers[param_group_name])
        torch.nn.utils.clip_grad_norm_(optimizers.parameters[param_group_name], max_norm)
    if any(any(p.grad is not None for p in g["params"]) for g in optimizers.optimizers[param_group_name].param_groups):
        grad_scaler.step(optimizers.optimizers[param_group_name])

def has_optimizer_for(optimizers, network_name):
    return network_name in optimizers.optimizers