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

from dataclasses import dataclass, field
from time import time
from typing import Literal, Optional, Type
import os
import math
import shutil
import numpy as np
import scipy
import torch
import cv2
import pyiqa
from PIL import Image
import cleanfid.fid as clean_fid
import torch.nn.init as init
from torch.cuda.amp.grad_scaler import GradScaler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from nerfstudio.utils import profiler
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils.writer import to8b

from .nerf_datamanager import NerfDataManagerConfig
from .nerf_model import NerfModelConfig
from .nerf_helper import to8b
from nerfstudio.model_components.losses import DepthLossType
from .sd_utils import compute_latent_camera
from .nerf_helper import splat_image
import imageio
from .depth_utils import align_depth_with_rgb
from .depth_utils import save_depth
from .nerf_helper import mask2bbox, fit_plane_to_depth
from .depth_utils import align_depth, align_depth_with_rgb, predict_depths
from nerfstudio.cameras.rays import Frustums, RaySamples

from depth_anything.dpt import DepthAnything
import subprocess

LOAD_LOCAL = False

if LOAD_LOCAL:
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    encoder = 'vitl'
    depth_anything_net = DepthAnything(model_configs[encoder])
    depth_anything_net.load_state_dict(torch.load(f'./checkpoints/depth_anything_{encoder}14.pth'))
    depth_anything_net.to("cuda").eval()
else:
    depth_anything_net = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14').to("cuda").eval()


@dataclass
class NerfPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: NerfPipeline)
    """target class to instantiate"""
    datamanager: NerfDataManagerConfig = NerfDataManagerConfig()
    """specifies the datamanager config"""
    model: NerfModelConfig = NerfModelConfig()
    """specifies the model config"""
    seed: int = 2024
    denoise_steps: int = 20
    steps_per_base: int = 10
    base_pool_num: int = 5
    subset_prop: float = 0.4
    lambda_cross_attn: float = 0.2
    lambda_patch: float = 0.01


class NerfPipeline(VanillaPipeline):
    def __init__(
        self,
        config: NerfPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, grad_scaler=grad_scaler
        )
        self.i_conf = []
        self.use_patch: bool = False
        self.zt_std_threshold: float = 0.9
        self.train_vis_only = False
        self.mask_infer_dilate_pad = 10
        self.model.config.inference_near_plane = 1.0

    # def start_vis_training(self):
    #     self.train_vis_only = True
    #     # for param in self.model.super().get_param_groups():
    #     #     param.requires_grad = False

    # def end_vis_training(self):
    #     self.train_vis_only = False

    # def disable_mask_training(self):
    #     self.datamanager.disable_mask_training()
    #
    # def enable_mask_training(self):
    #     self.datamanager.enable_mask_training()
    #
    # def enable_pseudo_depth(self, basedir):
    #     depth_pth = os.path.join(basedir, "depth", "depths.npy")
    #     self.datamanager.train_dataset = self.datamanager.create_train_dataset(depth_pth)
    #     self.datamanager.setup_train()
    #     self.model.config.depth_loss_mult = 1.
    #     self.model.config.depth_loss_type = DepthLossType.URF
    #
    # def disable_pseudo_depth(self):
    #     self.datamanager.train_dataset = self.datamanager.create_train_dataset()
    #     self.datamanager.setup_train()
    #     self.model.config.depth_loss_mult = 1e-2
    #     self.model.config.depth_loss_type = DepthLossType.DS_NERF

    def need_spherify(self):
        return self.datamanager.spherify

    # def find_object_bbox(self, voxel_size=0.05, num_sample=1000):
        from .object_bbox import downsample_mask, adjust_intrinsic_matrix
        from .object_bbox import save_point_clouds_with_colors
        from .object_bbox import mask_to_3d_points, voxel_intersection

        i_train = np.arange(0, self.datamanager.get_train_og_length(), 5)
        cameras = [self.get_camera(i, optimized=False) for i in i_train]
        c2ws = np.stack([cam.camera_to_worlds for cam in cameras])
        Ks = np.stack([cam.get_intrinsics_matrices() for cam in cameras])
        images, masks = [], []
        for img_i in i_train:
            data = self.datamanager.get_train_og_data_tensor(img_i)
            # image = data['image'].cpu().numpy()
            mask = data['mask'].squeeze().cpu().numpy()
            masks.append(mask)

        point_clouds = [
            mask_to_3d_points(downsample_mask(mask), adjust_intrinsic_matrix(K), pose,
                              depth_range=(0.1, 20), num_samples=num_sample,
                              min_bounds=np.array([-1, -1, -1])*2, max_bounds=np.array([1, 1, 1])*2)
            for mask, K, pose in zip(masks, Ks, c2ws)
        ]
        intersection_points = voxel_intersection(point_clouds, voxel_size=voxel_size)
        print(intersection_points)

        # save_point_clouds_with_colors(point_clouds, intersection_points)

        bbox_min = np.min(intersection_points, axis=0) - voxel_size
        bbox_max = np.max(intersection_points, axis=0) + voxel_size
        print(bbox_min, bbox_max)
        return bbox_min, bbox_max


    @profiler.time_function
    def get_prediction(self, step: int):
        # Every 10 steps: contain 1 base, 7 conf, 2 supp
        patches_for_nerf = True
        if not self.use_patch:
            patch_list = []
        elif (step % self.config.steps_per_base == 0 or len(self.i_conf) == 1):
            patch_list = self.i_conf[:1]
        else:
            patch_list = self.i_conf[1:]
            # if step % (self.config.steps_per_base-1) == 0:
            #     patch_list = self.i_supp
            #     patches_for_nerf = False
            # else:
            #     # patches_for_nerf = False
            #     patch_list = self.i_conf[1:]

        ray_bundles, batch = self.datamanager.next_train(step, patch_list=patch_list, patches_for_nerf=patches_for_nerf)
        model_outputs = {
            "patch_size": self.datamanager.config.patch_size,
            "num_patches": self.datamanager.config.num_patches,
            "vis_only": self.train_vis_only,
        }
        for key, ray_bundle in ray_bundles.items():
            if "patches" in key:
                # do not optimize poses with patch based losses
                ray_bundle.origins = ray_bundle.origins.detach()
                ray_bundle.directions = ray_bundle.directions.detach()
            model_output = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
            model_outputs.update({key + "_" + k: v for k, v in model_output.items()})
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        if self.config.datamanager.camera_optimizer is not None:
            assert self.config.datamanager.camera_optimizer.mode == "off"
        
        # if self.config.datamanager.camera_optimizer is not None:
        #     camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
        #     if camera_opt_param_group in self.datamanager.get_param_groups():
        #         # Report the camera optimization metrics
        #         metrics_dict["camera_opt_translation"] = (
        #             self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
        #         )
        #         metrics_dict["camera_opt_rotation"] = (
        #             self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
        #         )

        return model_outputs, metrics_dict, batch

    # @profiler.time_function
    # def get_object_bbox_train_loss_dict(self, step: int, num_points=1024):
        # if self.need_spherify():
        #     (x_min, y_min, z_min), (x_max, y_max, z_max) = self.object_aabb
        #     # x_coords = np.random.uniform(x_min, x_max, num_points)
        #     # y_coords = np.random.uniform(y_min, y_max, num_points)
        #     # z_coords = np.random.uniform(z_min, z_max, num_points)
        #     x_coords = np.clip(np.random.randn(num_points) * (x_max - x_min) / 4. + (x_max + x_min) / 2.,
        #                        x_min, x_max)
        #     y_coords = np.clip(np.random.randn(num_points) * (y_max - y_min) / 4. + (y_max + y_min) / 2.,
        #                        y_min, y_max)
        #     z_coords = np.clip(np.random.randn(num_points) * (z_max - z_min) / 4. + (z_max + z_min) / 2.,
        #                        z_min, z_max)
        #     points = torch.from_numpy(np.vstack((x_coords, y_coords, z_coords)).T)
        #     frustums = Frustums(origins=points, directions=torch.zeros_like(points),
        #                         starts=-0.05/2 * torch.ones((points.shape[0], 1)),
        #                         ends=0.05/2 * torch.ones((points.shape[0], 1)),
        #                         pixel_area=torch.ones((points.shape[0], 1)))
        #     ray_samples = RaySamples(frustums=frustums, camera_indices=torch.zeros((points.shape[0], 1)))
        #     loss_dict = self.model.get_object_bbox_loss_dict(ray_samples)
        # else:
        #     loss_dict = {}
        # return loss_dict

    @profiler.time_function
    def get_discriminator_train_loss_dict(self, step: int, model_outputs, batch, metrics_dict):
        loss_dict = self.model.get_discriminator_loss_dict(step, model_outputs, batch, metrics_dict)
        return loss_dict

    @profiler.time_function
    def get_nerf_train_loss_dict(self, step: int, model_outputs, batch, metrics_dict):
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, loss_dict, metrics_dict

    def get_camera(self, camera_i, optimized=False):
        camera = self.datamanager.get_train_og_camera(int(camera_i))
        assert camera.shape[0] == 1, "Only one camera at a time"
        camera.metadata = {"cam_idx": int(camera_i)}
        if optimized:
            with torch.no_grad():
                camera.camera_to_worlds = self.model.camera_optimizer.apply_to_camera(camera.to(self.device)).cpu()
        return camera[0] # remove batch dimen

    def get_eval_images(self, step, eval_on_trainset=True,
                        num_per_eval=10, use_appearance=False,
                        use_conf=True, fid_tmp_dir=None):
        self.eval()
        fid_metric = pyiqa.create_metric('fid')
        res_tensor = {}
        res_metrics = {}
        if eval_on_trainset:
            pool = self.i_conf if use_conf else self.i_supp
        else:
            pool = self.datamanager.dataparser.i_eval
            num_per_eval = 1
            fid_tmp_dir.mkdir(parents=True, exist_ok=True)
            os.makedirs(os.path.join(fid_tmp_dir, "gt"), exist_ok=True)
            os.makedirs(os.path.join(fid_tmp_dir, "pred"), exist_ok=True)
        for idx, img_i in enumerate(pool):
            if idx % num_per_eval != 0:
                continue
            camera = self.get_camera(int(img_i), optimized=True)
            outputs = self.model.get_outputs_for_camera(camera, use_appearance=use_appearance, cam_i=int(img_i))
            if eval_on_trainset:
                gt_data = self.datamanager.get_train_patch_data(int(img_i))
                gt_rgb = torch.from_numpy(gt_data["image"]).float()
                gt_msk = torch.from_numpy(gt_data["mask"]).float()
            else:
                gt_data = self.datamanager.get_train_og_data_tensor(int(img_i))
                gt_rgb = gt_data['image']
                gt_msk = gt_data['mask']

            batch = {"image": gt_rgb, "mask": gt_msk}
            metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
            # res_tensor[img_i] = images_dict["img"] # concat image
            res_tensor[img_i] = images_dict["rgb"]
            # res_tensor[img_i] = images_dict["depth"] / images_dict["depth"].max()
            res_metrics[img_i] = metrics_dict

            if not eval_on_trainset:
                img_pil1 = Image.fromarray(to8b(images_dict["real"].squeeze().cpu().numpy()))
                img_pil1.save(os.path.join(fid_tmp_dir, f"gt/{img_i:03d}.png"))
                img_pil2 = Image.fromarray(to8b(images_dict["rgb"].squeeze().cpu().numpy()))
                img_pil2.save(os.path.join(fid_tmp_dir, f"pred/{img_i:03d}.png"))

        if not eval_on_trainset:
            fid_score = fid_metric(
                os.path.join(fid_tmp_dir, 'gt'),
                os.path.join(fid_tmp_dir, 'pred')
            )
        else:
            fid_score = 0.

        self.train()
        return res_tensor, res_metrics, fid_score

    # @profiler.time_function
    # def get_average_eval_image_metrics(self, step: Optional[int] = None, write_path=None, eval_on_trainset=False):
    #     """Iterate over all the images in the eval dataset and get the average.
    #     Returns:
    #         metrics_dict: dictionary of metrics
    #     """
    #     with_kid = True
    #     self.eval()
    #     with torch.no_grad():
    #         metrics_dict_list = []
    #         if write_path is not None:
    #             image_dir = os.path.join(write_path, "{}_images_{}".format("train" if eval_on_trainset else "test", step))
    #             os.makedirs(image_dir, exist_ok=True)
    #             if with_kid:
    #                 for _, batch in self.datamanager.fixed_indices_train_dataloader:
    #                     train_rgb_8b = to8b(batch["image"].to(self.device))
    #
    #                     # write train rgb to compute kid
    #                     train_real_dir = os.path.join(image_dir, "train_real")
    #                     os.makedirs(train_real_dir, exist_ok=True)
    #                     h, w, _ = train_rgb_8b.shape
    #                     # devide large images into crops to maintain meaningfull kid computation (on smaller 299x299 resolution)
    #                     n_crops = 1
    #                     if h > 1000:
    #                         n_crops = 4
    #                     h_crop_dim = h // n_crops
    #                     w_crop_dim = w // n_crops
    #                     image_8b_crops = (
    #                         train_rgb_8b.unfold(0, h_crop_dim, h_crop_dim)
    #                         .unfold(1, w_crop_dim, w_crop_dim)
    #                         .permute(0, 1, 3, 4, 2)
    #                         .reshape(-1, h_crop_dim, w_crop_dim, 3)
    #                         .cpu()
    #                         .numpy()
    #                     )
    #                     for n, image_8b in enumerate(image_8b_crops):
    #                         image_file = str(batch["image_idx"]) + "_" + str(n) + ".png"
    #                         cv2.imwrite(
    #                             os.path.join(train_real_dir, image_file), cv2.cvtColor(image_8b, cv2.COLOR_RGB2BGR)
    #                         )
    #         dl = self.datamanager.fixed_indices_train_dataloader if eval_on_trainset else self.datamanager.fixed_indices_eval_dataloader
    #         with Progress(
    #             TextColumn("[progress.description]{task.description}"),
    #             BarColumn(),
    #             TimeElapsedColumn(),
    #             MofNCompleteColumn(),
    #             transient=True,
    #         ) as progress:
    #             task = progress.add_task(
    #                 "[green]Evaluating images...", total=len(dl)
    #             )
    #             for camera_ray_bundle, batch in dl:
    #                 # time this the following line
    #                 inner_start = time()
    #                 height, width = camera_ray_bundle.shape
    #                 num_rays = height * width
    #                 outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
    #                 metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
    #                 # write images
    #                 if write_path is not None:
    #                     for image_name, image in images_dict.items():
    #                         if image_name in [
    #                             "img",
    #                             "depth",
    #                             "rgb",
    #                             "real",
    #                         ]:
    #                             image_8b = to8b(image)
    #                             test_image_type_dir = os.path.join(image_dir, image_name)
    #                             os.makedirs(test_image_type_dir, exist_ok=True)
    #                             image_file = str(batch["image_idx"]) + ".png"
    #                             cv2.imwrite(
    #                                 os.path.join(test_image_type_dir, image_file),
    #                                 cv2.cvtColor(image_8b.cpu().numpy(), cv2.COLOR_RGB2BGR),
    #                             )
    #                             # write crops for kid computation
    #                             if with_kid:
    #                                 test_rendered_dir = os.path.join(image_dir, "test_rendered")
    #                                 os.makedirs(test_rendered_dir, exist_ok=True)
    #                                 if image_name == "rgb":
    #                                     h, w, _ = image_8b.shape
    #                                     image_8b_crops = (
    #                                         image_8b.unfold(0, h_crop_dim, h_crop_dim)
    #                                         .unfold(1, w_crop_dim, w_crop_dim)
    #                                         .permute(0, 1, 3, 4, 2)
    #                                         .reshape(-1, h_crop_dim, w_crop_dim, 3)
    #                                         .cpu()
    #                                         .numpy()
    #                                     )
    #                                     for n, image_8b in enumerate(image_8b_crops):
    #                                         image_file = str(batch["image_idx"]) + "_" + str(n) + ".png"
    #                                         cv2.imwrite(
    #                                             os.path.join(test_rendered_dir, image_file),
    #                                             cv2.cvtColor(image_8b, cv2.COLOR_RGB2BGR),
    #                                         )
    #                 assert "num_rays_per_sec" not in metrics_dict
    #                 metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
    #                 fps_str = f"fps_at_{height}x{width}"
    #                 assert fps_str not in metrics_dict
    #                 metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
    #                 metrics_dict_list.append(metrics_dict)
    #                 progress.advance(task)
    #         # average the metrics list
    #         metrics_dict = {}
    #         for key in metrics_dict_list[0].keys():
    #             metrics_dict[key] = float(
    #                 torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
    #             )
    #         self.train()
    #         if with_kid:
    #             metrics_dict["kid"] = clean_fid.compute_kid(
    #                 test_rendered_dir,
    #                 train_real_dir,
    #             )
    #             shutil.rmtree(train_real_dir)
    #             shutil.rmtree(test_rendered_dir)
    #     return metrics_dict

    def render_view(self, img_i: int):
        self.eval()
        camera = self.get_camera(img_i, optimized=True)
        outputs = self.model.get_outputs_for_camera(camera)
        self.train()
        return outputs

    def predict_pseudo_depth(self, base_dir=None):
        self.eval()
        from PIL import Image
        from nerfstudio.utils import colormaps
        savedir = create_savedir(base_dir, 'depth')
        self.i_train = range(self.datamanager.get_train_og_length())
        self.cameras = [self.get_camera(i, optimized=True) for i in self.i_train]
        depths = []
        for img_i in self.i_train:
            print(img_i)
            outputs = self.render_view(img_i)
            depth = outputs['expected_depth'].cpu().numpy()
            depths.append(depth)
            # depth = colormaps.apply_depth_colormap(
            #     outputs['expected_depth'],
            #     accumulation=outputs["accumulation"],
            #     near_plane=None, far_plane=None,
            #     colormap_options=colormaps.ColormapOptions(),
            #     ).cpu().numpy()
            # rgb = outputs['rgb'].cpu().numpy()
            # img = Image.fromarray(to8b(np.concatenate([rgb, depth])))
            # img.save(os.path.join(savedir, 'i{:03d}.png'.format(img_i)))
        depths = np.stack(depths)
        np.save(os.path.join(savedir, "depths.npy"), depths)
        return depths

    def render_view_reproject(self, img_i_src: int, img_i_tgt: int):
        camera_src = self.cameras[img_i_src]
        camera_tgt = self.cameras[img_i_tgt]

        camera_src_lat = compute_latent_camera(self.pipe, camera_src)
        camera_tgt_lat = compute_latent_camera(self.pipe, camera_tgt)

        data_src = self.datamanager.get_train_patch_data(img_i_src)
        zt_src_numpy = data_src['zt_np']
        zt_src_tensor = torch.from_numpy(zt_src_numpy).float().permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model.get_outputs_for_reprojection(camera_src_lat, camera_tgt_lat, zt_src_tensor)

        zt_tgt_numpy = outputs['rgb'].cpu().numpy()
        return zt_tgt_numpy

    def render_view_warp(self, img_i_src: int, img_i_tgt: int):
        from .sd_utils import to_rgb, to8b
        camera_src = self.cameras[img_i_src]
        camera_tgt = self.cameras[img_i_tgt]

        # camera_src_lat = compute_latent_camera(self.pipe, camera_src)
        # camera_tgt_lat = compute_latent_camera(self.pipe, camera_tgt)

        camera_src_lat = camera_src
        camera_tgt_lat = camera_tgt

        data_src = self.datamanager.get_train_patch_data(img_i_src)
        zt_src_numpy = data_src['zt_np']
        loc, scale = zt_src_numpy.min(), (zt_src_numpy.max() - zt_src_numpy.min())
        depth_src = data_src['depth']
        # depth_src = cv2.resize(depth_src,
        #                       dsize=(camera_src_lat.width.item(), camera_src_lat.height.item()),
        #                       interpolation=cv2.INTER_LINEAR)

        zt_src_numpy_prime = cv2.resize(to8b((zt_src_numpy-loc)/scale),
                               dsize=(camera_src_lat.width.item(), camera_src_lat.height.item()),
                               interpolation=cv2.INTER_NEAREST_EXACT)


        def reverse_transformations(c2w):
            """Reverses specific transformations applied to a camera-to-world matrix."""
            c2w_new = np.copy(c2w)
            c2w_new[0:3, 1:3] *= -1
            return c2w_new

        zt_tgt_numpy, warp_mask = splat_image(c2w_src=reverse_transformations(camera_src_lat.camera_to_worlds),
                                              c2w_tgt=reverse_transformations(camera_tgt_lat.camera_to_worlds),
                                              init_depth=depth_src, init_image=zt_src_numpy_prime,
                                              H=camera_tgt_lat.height.item(), W=camera_tgt_lat.width.item(),
                                              K=np.array([[camera_src_lat.fx.item(), 0, camera_src_lat.cx.item()],
                                                          [0, camera_src_lat.fy.item(), camera_src_lat.cy.item()],
                                                          [0, 0, 1]]),
                                              device='cpu')

        zt_tgt_numpy = zt_tgt_numpy/255. * scale + loc
        zt_tgt_numpy = zt_tgt_numpy * warp_mask + (1 - warp_mask) * np.random.normal(0., 1., zt_tgt_numpy.shape)
        zt_tgt_numpy = cv2.resize(zt_tgt_numpy, dsize=(zt_src_numpy.shape[1], zt_src_numpy.shape[0]),
                                  interpolation=cv2.INTER_NEAREST_EXACT)

        # rgb_src_numpy = data_src['image']
        # # rgb_src_numpy = cv2.resize(rgb_src_numpy,
        # #                       dsize=(camera_src_lat.width.item(), camera_src_lat.height.item()),
        # #                       interpolation=cv2.INTER_LINEAR)
        # rgb_tgt_numpy, _ = splat_image(c2w_src=reverse_transformations(camera_src_lat.camera_to_worlds),
        #                                c2w_tgt=reverse_transformations(camera_tgt_lat.camera_to_worlds),
        #                                init_depth=depth_src, init_image=rgb_src_numpy,
        #                                H=camera_tgt_lat.height.item(), W=camera_tgt_lat.width.item(),
        #                                K=np.array([[camera_src_lat.fx.item(), 0, camera_src_lat.cx.item()],
        #                                                   [0, camera_src_lat.fy.item(), camera_src_lat.cy.item()],
        #                                                   [0, 0, 1]]),
        #                                       device='cpu')
        #
        # rgb_tgt_numpy = cv2.resize(rgb_src_numpy,dsize=(zt_tgt_numpy.shape[1], zt_tgt_numpy.shape[0]),
        #                            interpolation=cv2.INTER_LINEAR)
        #
        # res = to_rgb(torch.from_numpy(zt_tgt_numpy).permute(2,0,1).float().unsqueeze(0)).squeeze(0).cpu().permute(1,2,0).numpy()
        # res = np.concatenate([rgb_tgt_numpy, res])
        # res = Image.fromarray(to8b(res))
        # res.save(os.path.join("demo_" + str(img_i_tgt) + ".png"))

        return zt_tgt_numpy


    def infer_depth(self, img_i: int, image, mask, relative=False):
        output_nerf = self.render_view(img_i)
        depth_nerf = align_depth_with_rgb(image, output_nerf["expected_depth"].squeeze().cpu().numpy(), 1 - mask)

        if self.need_spherify():
            h, w = depth_nerf.shape
            depth_nerf = (output_nerf["expected_depth"].squeeze().cpu().numpy() + depth_nerf) / 2.
            normalizer = np.max(depth_nerf)
            depth_to_inp = depth_nerf / normalizer
            depth_to_inp = cv2.resize(depth_to_inp, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)

            depth_nerf[mask.squeeze()] = normalizer

            mask_for_inp = np.copy(mask)
            mask_for_inp = cv2.resize(mask_for_inp.astype(np.uint8).squeeze(), dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
            kernel_size = 5
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            mask_for_inp = cv2.morphologyEx(mask_for_inp.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            mask_for_inp = cv2.morphologyEx(mask_for_inp, cv2.MORPH_CLOSE, kernel)
            mask_for_inp = scipy.ndimage.binary_dilation(mask_for_inp, structure=np.ones((kernel_size, kernel_size)), iterations=3)

            depth_aligned = predict_depths(np.stack([depth_to_inp]*3, axis=-1), mask_for_inp) / 255.
            depth_aligned = cv2.resize(depth_aligned[..., 0], dsize=(w, h), interpolation=cv2.INTER_LINEAR) * normalizer

            # bbox = mask2bbox(np.squeeze(mask).astype(int))
            # depth_aligned = fit_plane_to_depth(depth_nerf, np.squeeze(mask).astype(int), bbox, expansion=100)

        else:
            depth_rel = predict_depths([image], depth_anything_net)[0] # (H, W)
            assert len(mask.shape) == 2 or len(mask.shape) == 3
            if len(mask.shape) == 3:
                mask = np.squeeze(mask)

            depth_aligned = align_depth_with_rgb(image, align_depth(depth_rel, depth_nerf, mask <= 0, intercept=~relative, weighted=True))
        return depth_nerf, depth_aligned

    def clean_noisy_mask(self, mask, kernel_size=5):

        input_shape = mask.shape
        if len(input_shape) == 3:
            mask = np.squeeze(mask)
        elif len(input_shape) == 2:
            pass
        else:
            raise NotImplementedError

        # res = scipy.ndimage.median_filter(res, size=2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = largest_connected_component_dilation(mask, pad=self.mask_infer_dilate_pad, iterations=self.mask_infer_dilate_iter)

        mask = scipy.ndimage.binary_fill_holes(mask)
        mask = scipy.ndimage.binary_closing(mask)

        return mask.astype(np.uint8).reshape(input_shape)

    def infer_mask_and_image(self, img_i: int, image, mask):
        output_nerf = self.render_view(img_i)
        vis = output_nerf["visibility"].cpu().numpy()
        assert vis.shape == mask.shape
        vis[~mask] = 1

        mask_nerf = 1 - vis # object -> (0, 1]
        mask_nerf = self.clean_noisy_mask(mask_nerf)
        mask_diff = (1*mask - mask_nerf).astype(bool)

        rgb = output_nerf["rgb"].squeeze().cpu().numpy()
        assert rgb.shape == image.shape
        if self.need_spherify():
            image = np.where(mask_nerf, 0, rgb)
        else:
            image[mask.squeeze()] = 0.
            image[mask_diff.squeeze()] = rgb[mask_diff.squeeze()]

        return mask_nerf, image

    def update_base_depth(self):
        img_i = self.img_i_base
        data_base = self.datamanager.get_train_patch_data(self.img_i_base)
        mask = data_base['mask']
        depth_nerf, depth_base = self.infer_depth(img_i, data_base['image'], mask, relative=False)
        self.datamanager.update_train_patch_map(img_i, 'depth', depth_base)

    def inpaint_base_frame(self, step, base_dir):
        import numpy as np
        from diffusers import StableDiffusionInpaintPipeline
        from .nerf_helper import find_base_frame

        if LOAD_LOCAL:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "checkpoints/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16).to("cuda")
        else:
            # subprocess.run(['ping', '-c', '4', 'huggingface.co'], check=True, timeout=10)
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                # "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16).to("cuda")

        self.i_train = np.arange(self.datamanager.get_train_og_length())
        self.cameras = [self.get_camera(i, optimized=True) for i in self.i_train]
        masks_og = []
        images, masks = [], []

        savedir = create_savedir(base_dir, 'nerf_masks')
        mask_filenames = self.datamanager.train_dataparser_outputs.mask_filenames
        for img_i in self.i_train:
            data = self.datamanager.get_train_og_data_tensor(img_i)
            mask = data['mask'].cpu().numpy()

            if self.need_spherify():
                mask_fname = os.path.join(savedir, 'mask_{:03d}.png'.format(img_i))
                # mask_fname = os.path.join(savedir, mask_filenames[img_i].name)
                image_fname = os.path.join(savedir, 'image_{:03d}.png'.format(img_i))

                if os.path.isfile(mask_fname) and os.path.isfile(image_fname):
                    mask_nerf = imageio.imread(mask_fname) / 255.
                    image_nerf = imageio.imread(image_fname) / 255.
                    mask_nerf = mask_nerf[..., np.newaxis].astype(bool)

                    if self.datamanager.dataparser.config.data.name in ('kitchen',):
                        kernel_size = 11 # turn on for kitchen
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                        mask_nerf = cv2.morphologyEx(mask_nerf.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                        mask_nerf = cv2.morphologyEx(mask_nerf, cv2.MORPH_CLOSE, kernel)
                        mask_nerf = scipy.ndimage.binary_dilation(
                            mask_nerf, structure=np.ones((kernel_size, kernel_size)))

                else:
                    image = data['image'].cpu().numpy()
                    mask_nerf, image_nerf = self.infer_mask_and_image(img_i, image, mask)
                    imageio.imwrite(mask_fname, to8b(np.squeeze(mask_nerf)))
                    imageio.imwrite(image_fname, to8b(image_nerf))
            else:
                image_nerf = data['image'].cpu().numpy()
                mask_nerf = mask

            self.datamanager.update_train_patch_data(img_i, {'image': image_nerf, 'mask': mask_nerf})
            images.append(image_nerf)
            masks.append(mask_nerf)
            masks_og.append(mask) # need for depth reason

        images = np.stack(images)
        masks = np.stack(masks)

        savedir = create_savedir(base_dir, 'inpaint')
        prompt = self.datamanager.get_prompt()
        zT_base, image_base, img_i_base = find_base_frame(np.array(self.i_train), self.cameras, images, masks, self.pipe, self.config.seed,
                                                          prompt, self.config.denoise_steps, savedir, step, num=self.config.base_pool_num,
                                                          strength=1.0, spherify=self.need_spherify())
        img_i_base = int(img_i_base)
        depth_nerf, depth_base = self.infer_depth(img_i_base, image_base,
                                                  masks[img_i_base] if self.need_spherify() else masks_og[img_i_base],
                                                  relative=False)
        save_depth(np.concatenate([np.squeeze(depth_nerf), depth_base]), savedir, img_i_base)
        # update_inpainted_dataset
        self.datamanager.update_train_patch_data(img_i_base,
                                                 {'image': image_base,
                                                  'mask': masks[img_i_base],
                                                  # 'mask_og': masks_og[img_i_base]
                                                  'depth': depth_base,
                                                  'zt_np': zT_base.squeeze().permute(1, 2, 0).cpu().numpy()})

        self.img_i_base = img_i_base
        self.i_conf = [img_i_base] # update
        self.i_supp = [img_i_base]

        _ = create_savedir(base_dir, 'inv_inpaint', need_clean=False)  # clean up the directory
        self.use_patch = True
        self.model.config.depth_loss_type = DepthLossType.DS_NERF


    def inpaint_left_frames(self, step, base_dir):
        self.eval()
        from .nerf_helper import calculate_min_distance, mutual_inpaint_image, compute_similarity, inpaint_image
        savedir = create_savedir(base_dir, 'inv_inpaint')
        depthdir = create_savedir(base_dir, 'depth')


        # if not self.need_spherify():
        from .masactrl import MutualSelfAttentionControl, AttentionBase, regiter_attention_editor_diffusers
        STEP, LAYER = 0, 0
        editor = MutualSelfAttentionControl(STEP, LAYER, total_steps=self.config.denoise_steps, model_type="SD",
                                            # lambda_attn=1.0)
                                            lambda_attn=0.2 if self.need_spherify() else self.config.lambda_cross_attn)
        regiter_attention_editor_diffusers(self.pipe, editor)

        def invert_matrix(matrix):
            """ Invert a 4x4 transformation matrix. """
            R = matrix[0:3, 0:3]
            T = matrix[0:3, 3]
            R_inv = R.T
            T_inv = -R_inv @ T
            M_inv = np.eye(4)
            M_inv[0:3, 0:3] = R_inv
            M_inv[0:3, 3] = T_inv
            return M_inv

        def group_cameras(c2w_matrices, anchor_index, num_sectors=4, num_z_levels=3):
            anchor_matrix = c2w_matrices[anchor_index]
            anchor_inv = invert_matrix(anchor_matrix)

            groups = {(i, j): [] for i in range(num_sectors) for j in range(num_z_levels)}
            z_values = c2w_matrices[:, 0:3, 3][:, 2]  # Z values of all cameras
            z_min, z_max = np.min(z_values), np.max(z_values)
            z_range = z_max - z_min
            z_step = z_range / num_z_levels

            for idx, c2w in enumerate(c2w_matrices):
                if idx == anchor_index: continue
                world_pos = np.append(c2w[0:3, 3], 1)  # Homogeneous coordinates
                local_pos = anchor_inv @ world_pos

                # Calculate angle and determine sector
                angle = np.arctan2(local_pos[1], local_pos[0])  # Angle in radians
                sector_index = int((angle + np.pi) / (2 * np.pi / num_sectors)) % num_sectors

                # Determine z level
                z_level = int((local_pos[2] - z_min) / z_step)
                z_level = max(0, min(z_level, num_z_levels - 1))  # Ensure z_level is within the valid range

                groups[(sector_index, z_level)].append(idx)

            return groups

        c2w_matrices = np.stack([cam.camera_to_worlds for cam in self.cameras])
        quantile_dict = group_cameras(c2w_matrices, self.img_i_base)

        ref_sim_dict = {}
        data_base = self.datamanager.get_train_patch_data(self.img_i_base)
        zt_base_np = data_base['zt_np']
        zt_base_tensor = torch.from_numpy(zt_base_np).unsqueeze(0).permute(0, 3, 1, 2)
        image_base = data_base['image']
        mask_base = data_base['mask']

        count = 0
        for img_i_tgt in self.i_train:
            if img_i_tgt == self.img_i_base:
                continue

            data_tgt = self.datamanager.get_train_patch_data(img_i_tgt)
            mask_tgt = data_tgt['mask']

            fname = os.path.join(savedir, '{:03d}.png'.format(img_i_tgt))
            if os.path.isfile(fname):
                inpainted = imageio.imread(fname) / 255.
            else:
                zt_tgt_np = self.render_view_warp(self.img_i_base, img_i_tgt)
                # zt_tgt_np = self.render_view_reproject(self.img_i_base, img_i_tgt) # (H, W, C)
                mask_tgt_zt = cv2.resize(mask_tgt.squeeze().astype(int), (zt_tgt_np.shape[1], zt_tgt_np.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)

                mean_sample = np.mean(zt_tgt_np[mask_tgt_zt >= 1]).item()
                std_sample = np.std(zt_tgt_np[mask_tgt_zt >= 1], ddof=1).item()  # divided by (n-1)
                print(img_i_tgt, " noise initial mean and std", mean_sample, std_sample)
                # if std_sample < self.zt_std_threshold: continue

                if std_sample < 0.9:  # add _compmore gaussian to realize N(0,1)
                    if mean_sample >= 0.1: zt_tgt_np -= mean_sample  # make it at mean=0
                    std_comple = np.sqrt(np.square(0.9) - np.square(std_sample))
                    zt_tgt_np += np.random.normal(0., std_comple, zt_tgt_np.shape)
                    print(img_i_tgt, " aug mean and std",
                          np.mean(zt_tgt_np[mask_tgt_zt >= 1]),
                          np.std(zt_tgt_np[mask_tgt_zt >= 1], ddof=1))

                if self.need_spherify():
                    zt_tgt_np = np.where(mask_tgt_zt[..., np.newaxis] >= 1, zt_tgt_np,
                                         np.random.normal(0., 1., zt_tgt_np.shape))
                else:
                    zt_tgt_np = np.where(mask_tgt_zt[..., np.newaxis] >= 1, zt_tgt_np, zt_base_np)

                zt_tgt_tensor = torch.from_numpy(zt_tgt_np).unsqueeze(0).permute(0, 3, 1, 2).to(self.pipe.dtype)

                strength = 1.0 if self.need_spherify() else 0.9
                if not self.need_spherify():
                    render_outputs = self.render_view(int(img_i_tgt))  # (H, W, C)
                    render_rgb = render_outputs["rgb"].cpu().numpy()
                    # input_tgt = np.where(mask_tgt, render_rgb, data_tgt['image'].cpu().numpy())
                    input_tgt = np.where(mask_tgt, render_rgb, data_tgt['image'])
                else:
                    input_tgt = data_tgt['image']

                print("Pipeline: reset editor......")
                editor.reset()
                inpainted = mutual_inpaint_image(image_ref=image_base, mask_ref=mask_base,
                                                latent_ref=zt_base_tensor,  # use the reference one
                                                image_tgt=input_tgt, mask_tgt=mask_tgt, latent_tgt=zt_tgt_tensor,
                                                pipe=self.pipe, prompt=self.datamanager.get_prompt(),
                                                savedir=savedir, seed=self.config.seed,
                                                fn='e{:05d}_i{:03d}_rt{:03d}'.format(step, img_i_tgt, self.img_i_base),
                                                strength=strength, num_inference_steps=self.config.denoise_steps)
                # else:
                #     inpainted = inpaint_image(pipe=self.pipe, image=input_tgt, mask=mask_tgt, latent_noise=None,
                #                               prompt=self.datamanager.get_prompt(), savedir=savedir,
                #                               fn='e{:05d}_i{:03d}_rt{:03d}'.format(step, img_i_tgt, self.img_i_base),
                #                               strength=strength, num_inference_steps=self.config.denoise_steps)
                    # h, w, _ = input_tgt.shape
                    # img_to_inp = cv2.resize(input_tgt, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
                    # mask_for_inp = cv2.resize(mask_tgt.astype(np.uint8).squeeze(), dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
                    # mask_for_inp = scipy.ndimage.binary_dilation(mask_for_inp, structure=np.ones((2, 2)), iterations=5)
                    #
                    # inpainted = predict(img_to_inp, mask_for_inp).squeeze() / 255.
                    # inpainted = cv2.resize(inpainted, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

                imageio.imwrite(fname, to8b(inpainted))

            self.datamanager.update_train_patch_data(img_i_tgt, {'image': inpainted,
                                                                 'mask': mask_tgt,
                                                                 # 'depth': depth_tgt,
                                                                 # 'zt_np': zt_tgt_np
                                                                 })
            # if self.need_spherify():
            #     fname = os.path.join(depthdir, '{:03d}.npy'.format(img_i_tgt))
            #     if os.path.isfile(fname):
            #         depth_tgt = np.load(fname)
            #     else:
            #         _, depth_tgt = self.infer_depth(img_i_tgt, inpainted, mask_tgt, relative=False)
            #         np.save(fname, depth_tgt)
            #     self.datamanager.update_train_patch_map(img_i_tgt, "depth", depth_tgt)
            #     save_depth(depth_tgt, depthdir, img_i_tgt)

            lpips_to_base = compute_similarity(img_src=image_base, msk_src=mask_base.astype(int),
                                               img_tgt=inpainted, msk_tgt=mask_tgt.astype(int))

            # self.lpips_ref_scores[img_i_tgt] = lpips_to_base
            ref_sim_dict[img_i_tgt] = lpips_to_base
            count += 1
            # if count >= 10: break

        with open(os.path.join(savedir, f"lpips.txt"), "a") as f:
            f.write(f'epoch: {step}\n')
            for k, v in ref_sim_dict.items():
                f.write(f'{k}: {v}\n')

        self.i_conf = [self.img_i_base]
        for q, img_quatile_group in quantile_dict.items():
            scores = [ref_sim_dict[img_i] for img_i in img_quatile_group]
            priority = np.argsort(scores)
            picked = np.array(img_quatile_group)[priority[:int(len(img_quatile_group)*self.config.subset_prop)]]
            # picked = np.array(img_quatile_group)[priority[:6]]
            # print(img_quatile_group)
            # print(f"pipeline: quantile{q:02d} picked ", picked)
            self.i_conf.extend(picked)

        # sorted_i = sorted(ref_sim_dict, key=ref_sim_dict.get)
        self.i_supp = np.setdiff1d(self.i_train, self.i_conf) # support set not using photoloss
        # self.i_supp = self.i_conf
        print("\nCurrent iconf: ", self.i_conf)

        with open(os.path.join(savedir, f"i_conf.txt"), "a") as f:
            f.write(f'epoch: {step}, thres: {self.zt_std_threshold:0.2f}\n')
            f.write(' '.join(map(str, self.i_conf)) + '\n')

        # self.model.config.depth_loss_type = DepthLossType.URF
        self.model.config.l1_loss_mult = self.config.lambda_patch
        self.model.config.vgg_loss_mult = self.config.lambda_patch
        self.model.config.adv_loss_mult = self.config.lambda_patch
        self.model.config.depth_loss_mult *= self.config.steps_per_base

        # if self.need_spherify():
        #     self.model.config.l1_loss_mult = 1.0
        #     self.model.config.adv_loss_mult = 1e-3
        #     self.model.config.vgg_loss_mult = 1e-3
        #     self.i_supp = self.i_conf

        self.train()


def create_savedir(basedir, map, epoch=None, need_clean=False):
    if epoch is not None:
        savedir = os.path.join(basedir, str(map) + '_{:05d}'.format(epoch))
    else:
        savedir = os.path.join(basedir, str(map))

    if need_clean and os.path.exists(savedir):
        shutil.rmtree(savedir)  # Remove the directory and all its contents
    os.makedirs(savedir, exist_ok=True)
    return savedir


def largest_connected_component_dilation(mask, pad, iterations):
    from scipy.ndimage import label, binary_closing, binary_fill_holes, binary_dilation
    """
    Keep the largest connected component in a binary mask.

    Parameters:
    mask (np.ndarray): Binary mask of shape (H, W) with values 0 and 1

    Returns:
    np.ndarray: Binary mask with only the largest connected component
    """
    # Label connected components
    labeled_array, num_features = label(mask)

    # Find the size of each component
    component_sizes = np.bincount(labeled_array.ravel())

    # Ignore the background component at index 0
    component_sizes[0] = 0

    # Find the largest component
    largest_component_label = component_sizes.argmax()

    # Create a mask for the largest component
    largest_component_mask = (labeled_array == largest_component_label).astype(np.uint8)

    res = binary_closing(binary_fill_holes(largest_component_mask))
    res = binary_closing(res)
    res = binary_dilation(res, structure=np.ones((pad, pad)), iterations=iterations)

    return res

"""
    def inpaint_left_frames(self, step, base_dir):
        self.eval()
        from .nerf_helper import calculate_min_distance, mutual_inpaint_image, compute_similarity

        savedir = create_savedir(base_dir, 'inv_inpaint')

        img_and_dist = []
        base_pose_pool = [self.cameras[img_i].camera_to_worlds for img_i in self.i_used]
        for cam in self.cameras:
            local_i, dist = calculate_min_distance(cam.camera_to_worlds, base_pose_pool)
            img_and_dist.append((self.i_used[local_i], dist))

        ref_sim_dict = {}
        data_base = self.datamanager.get_train_patch_data(self.img_i_base)
        zt_base_np = data_base['zt_np']
        zt_base_tensor = torch.from_numpy(zt_base_np).unsqueeze(0).permute(0, 3, 1, 2)
        image_base = data_base['image']
        mask_base = data_base['mask']

        lpips_threshold = np.minimum(1.3 * np.max(self.lpips_ref_scores[self.i_conf]), 0.4) if len(self.i_conf) > 1 else 1.
        find_flag = False
        count = 0
        while not find_flag:
            for img_i_tgt in np.setdiff1d(self.i_train, self.i_filtered):
                img_i_src, _ = img_and_dist[img_i_tgt]

                zt_tgt_np = self.render_view_reproject(img_i_src, img_i_tgt) # (H, W, C)

                data_tgt = self.datamanager.get_train_og_data_tensor(img_i_tgt)
                mask_tgt = data_tgt['mask'].cpu().numpy()
                mask_tgt_zt = cv2.resize(mask_tgt.squeeze().astype(int), (zt_tgt_np.shape[1], zt_tgt_np.shape[0]), interpolation=cv2.INTER_NEAREST)

                mean_sample = np.mean(zt_tgt_np[mask_tgt_zt >= 1]).item()
                std_sample = np.std(zt_tgt_np[mask_tgt_zt >= 1], ddof=1).item()  # divided by (n-1)
                print(img_i_tgt, " noise initial mean and std", mean_sample, std_sample)
                # if std_sample < self.zt_std_threshold: continue

                if std_sample < 0.9:  # add _compmore gaussian to realize N(0,1)
                    if mean_sample >= 0.1: zt_tgt_np -= mean_sample  # make it at mean=0
                    std_comple = np.sqrt(np.square(0.9) - np.square(std_sample))
                    zt_tgt_np += np.random.normal(0., std_comple, zt_tgt_np.shape)
                    print(img_i_tgt, " aug mean and std",
                          np.mean(zt_tgt_np[mask_tgt_zt >= 1]),
                          np.std(zt_tgt_np[mask_tgt_zt >= 1], ddof=1))

                render_outputs = self.render_view(int(img_i_tgt)) # (H, W, C)
                render_rgb = render_outputs["rgb"].cpu().numpy()
                input_tgt = np.where(mask_tgt, render_rgb, data_tgt['image'].cpu().numpy())

                zt_tgt_np = np.where(mask_tgt_zt[..., np.newaxis] >= 1, zt_tgt_np, zt_base_np)
                zt_tgt_tensor = torch.from_numpy(zt_tgt_np).unsqueeze(0).permute(0, 3, 1, 2)

                # strength = 1.0 if std_sample < 0.9 else 0.9
                inpainted = mutual_inpaint_image(image_ref=image_base, mask_ref=mask_base,
                                                 latent_ref=zt_base_tensor,  # use the reference one
                                                 image_tgt=input_tgt, mask_tgt=mask_tgt, latent_tgt=zt_tgt_tensor,
                                                 pipe=self.pipe, prompt=self.datamanager.get_prompt(),
                                                 savedir=savedir, seed=self.config.seed,
                                                 fn='e{:05d}_i{:03d}_rt{:03d}'.format(step, img_i_tgt, img_i_src),
                                                 strength=0.9, num_inference_steps=10)

                self.datamanager.update_train_patch_data(img_i_tgt, {'image': inpainted,
                                                                     'mask': mask_tgt,
                                                                     'zt_np': zt_tgt_np})

                lpips_to_base = compute_similarity(img_src=image_base, msk_src=mask_base.astype(int),
                                                   img_tgt=inpainted, msk_tgt=mask_tgt.astype(int))

                if lpips_to_base < lpips_threshold:  # if larger then can not use the inpainted image
                    self.lpips_ref_scores[img_i_tgt] = lpips_to_base
                    ref_sim_dict[img_i_tgt] = lpips_to_base
                    count += 1
                # if count >= 40: break

            if len(ref_sim_dict) < 2:
                self.zt_std_threshold -= 0.05
                print("Lower the std threshold to ", self.zt_std_threshold)
                if self.zt_std_threshold < 0.5: 1 / 0
                with open(os.path.join(savedir, f"lpips.txt"), "a") as f:
                    f.write(f"epoch {step} : lower the std threshold to {self.zt_std_threshold:0.2f} ..........\n")
            else:
                #########################  update iconf #######################
                find_flag = True

                # if lpips_threshold == 1.:  # clean up for first round
                #     mean_value = sum(ref_sim_dict.values()) / len(ref_sim_dict)
                #     ref_sim_dict = {k: v for k, v in ref_sim_dict.items() if v <= mean_value}

                with open(os.path.join(savedir, f"lpips.txt"), "a") as f:
                    f.write(f'epoch: {step}\n')
                    for k, v in ref_sim_dict.items():
                        f.write(f'{k}: {v}\n')

                # num_candidate = len(ref_sim_dict)
                # num_adding = min(self.config.num_support, math.ceil(num_candidate / 2))
                #
                # if self.config.drop_mode == "none" or len(self.i_conf) <= 1:
                #     pass
                # else:
                #     if self.config.drop_mode == "render":
                #         num_keeping = min(self.config.num_support + self.config.num_support - num_adding, len(self.i_conf) - 1)
                #         scores = []
                #         for img_i in self.i_conf[1:]:
                #             data = self.datamanager.get_train_patch_data(img_i)
                #             outputs = self.render_view(int(img_i))
                #             rgb_render = outputs['rgb'].cpu().numpy()
                #             m = data['mask']
                #             scores.append(np.square(rgb_render[m] - data['image'][m]).mean())
                #
                #         sorted_indices_local = np.argsort(scores)  # Sort indices by distance
                #         keep_idx = np.array(self.i_conf[1:])[sorted_indices_local[:num_keeping]]
                #         self.i_conf = self.i_conf[:1]
                #         self.i_conf.extend(keep_idx)
                #     else:
                #         raise NotImplementedError
                print("Pipeline: add quarter of the left image to the lists..")
                num_adding = len(ref_sim_dict) // 1 # quarter =//2

                sorted_i = sorted(ref_sim_dict, key=ref_sim_dict.get)
                self.i_conf.extend(sorted_i[:num_adding])
                self.i_used.extend(sorted_i[:num_adding])
                self.i_filtered.extend(sorted_i)

                print("\nCurrent iconf: ", self.i_conf)
                with open(os.path.join(savedir, f"i_conf.txt"), "a") as f:
                    f.write(f'epoch: {step}, thres: {self.zt_std_threshold:0.2f}\n')
                    f.write(' '.join(map(str, self.i_conf)) + '\n')

        if lpips_threshold == 1.:
            depth_rel = self.infer_depth(self.img_i_base, image_base, mask_base, relative=True)
            self.datamanager.update_train_patch_map(idx=self.img_i_base, map_name='depth', data=depth_rel)

            from nerfstudio.model_components.losses import DepthLossType
            self.model.config.depth_loss_type = DepthLossType.SPARSENERF_RANKING

            self.prog_train = True
        self.train()
"""


