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
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Literal

import random
import numpy as np
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import OrientedBox, SceneBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.losses import DepthLossType, depth_loss, depth_ranking_loss
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.utils import colormaps
# from nerfstudio.models.nerfacto import NerfactoModelConfig, NerfactoModel
from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig, DepthNerfactoModel

from .nerfacto_field import NerfField
# from utils.vgg_loss import VggLoss
from generator.stylegan2 import Discriminator
from .nerf_datamanager import get_mask_bbox
import pyiqa


@dataclass
class NerfModelConfig(DepthNerfactoModelConfig):
    _target: Type = field(default_factory=lambda: NerfModel)
    # background_color: Literal["random", "last_sample", "black", "white"] = "black"

    # visibility_threshold: float = 0.9

    distortion_loss_mult: float = 0.0

    depth_loss_mult: float = 0.01

    tv_loss_mult: float = 0.001

    # is_euclidean_depth: bool = False
    # """Whether input depth maps are Euclidean distances (or z-distances)."""
    # depth_sigma: float = 0.01 # 0.01
    # """Uncertainty around depth values in meters (defaults to 1cm)."""
    # should_decay_sigma: bool = False
    # """Whether to exponentially decay sigma."""
    # starting_depth_sigma: float = 0.2
    # """Starting uncertainty around depth values in meters (defaults to 0.2m)."""
    # sigma_decay_rate: float = 0.99985
    # """Rate of exponential decay."""
    # depth_loss_type: DepthLossType = DepthLossType.DS_NERF
    # """Depth loss type."""

    l1_loss_mult: float = 1e-2
    """L1 loss multiplier."""
    vgg_loss_mult: float = 1e-2
    """Vgg loss multiplier."""
    adv_loss_mult: float = 1e-2
    """Adversarial loss multiplier."""
    gp_loss_mult: float = 0.2
    """Gradient penalty loss multiplier."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    appearance_embed_dim: int = 0
    """Dimension of the appearance embedding."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    # camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    # """Config of the camera optimizer to use"""
    # nll_loss_mult: float = 1e-6
    # """Appearance loss multiplier."""

    # near_plane: float = 0.5
    # """How far along the ray to start sampling."""
    far_plane: float = 100.0
    """How far along the ray to stop sampling."""
    inference_near_plane: float = 1.0
    """How far along the ray to stop sampling."""


class NerfModel(DepthNerfactoModel):
    """NeRF model"""

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules() # call the orginal nerfacto

        ###### reinitialize the field with vis_grid
        # self.field = NerfField(
        #     self.scene_box.aabb,
        #     hidden_dim=self.config.hidden_dim,
        #     num_levels=self.config.num_levels,
        #     max_res=self.config.max_res,
        #     base_res=self.config.base_res,
        #     features_per_level=self.config.features_per_level,
        #     log2_hashmap_size=self.config.log2_hashmap_size,
        #     hidden_dim_color=self.config.hidden_dim_color,
        #     hidden_dim_transient=self.config.hidden_dim_transient,
        #     spatial_distortion=None if self.config.disable_scene_contraction else SceneContraction(order=float("inf")),
        #     num_images=self.num_train_data,
        #     use_pred_normals=self.config.predict_normals,
        #     use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        #     appearance_embedding_dim=self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0,
        #     average_init_density=self.config.average_init_density,
        #     implementation=self.config.implementation,
        # )
        # from .mlp import MLPWithDVGO
        # self.field.mlp_base = MLPWithDVGO(num_levels=self.config.num_levels,
        #                                   features_per_level=self.config.features_per_level,
        #                                   out_dim=1+15,
        #                                   implementation="torch")

        from . import grid
        from nerfstudio.model_components.renderers import UncertaintyRenderer
        assert not self.config.disable_scene_contraction # otherwise need to reset xyz_min and xyz_max
        self.vis_grid = grid.create_grid('DenseGrid', channels=1, world_size=[256]*3, xyz_min=[0.]*3, xyz_max=[1.]*3)
        self.renderer_vis = UncertaintyRenderer()

        # Discriminator
        if self.config.adv_loss_mult > 0.0:
            self.discriminator = Discriminator(
                block_kwargs={},
                mapping_kwargs={},
                epilogue_kwargs={"mbstd_group_size": 4},
                channel_base=16384,
                channel_max=512,
                num_fp16_res=4,
                conv_clamp=256,
                img_channels=3,
                c_dim=0,
                img_resolution=64,
            )

        # additional losses
        # self.vgg_loss = VggLoss(device=torch.device("cuda"))
        import lpips as lpips_lib
        self.vgg_loss = lpips_lib.LPIPS(net='vgg').cuda()
        # from DISTS_pytorch import DISTS
        # self.vgg_loss = DISTS()
        # print("nerf_model: if use average embedding?", self.field.use_average_appearance_embedding)
        self.musiq_metric = pyiqa.create_metric('musiq', device="cuda")


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        # param_groups = {}
        # param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        # param_groups["fields"] = self.field.get_field_parameters_list()
        # self.camera_optimizer.get_param_groups(param_groups=param_groups)
        param_groups = super().get_param_groups()
        param_groups['fields'] += list(self.vis_grid.parameters())
        # if self.config.use_appearance_embedding:
        #     param_groups["embedding_appearance"] = self.field.get_embedding_parameters_list()
        if self.config.adv_loss_mult > 0.0:
            param_groups["discriminator"] = list(self.discriminator.parameters())
        return param_groups

    # def get_visibility(self, ray_samples_list):
    #     visibility_list = [] # B, N, 1
    #     for ray_samples in ray_samples_list:
    #         """Computes and returns the visibilities."""
    #         if self.field.spatial_distortion is not None:
    #             positions = ray_samples.frustums.get_positions()
    #             positions = self.field.spatial_distortion(positions)
    #             positions = (positions + 2.0) / 4.0
    #         else:
    #             positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.field.aabb)
    #         # Make sure the tcnn gets inputs between 0 and 1.
    #         selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
    #         positions = positions * selector[..., None]
    #         self.field._sample_locations = positions
    #         if not self.field._sample_locations.requires_grad:
    #             self.field._sample_locations.requires_grad = True
    #         positions_flat = positions.view(-1, 3) #[0,1]
    #         visibility = self.vis_grid(positions_flat).view(*ray_samples.frustums.shape, -1)
    #         visibility_list.append(visibility)
    #     return torch.cat(visibility_list, dim=-2)

    def get_outputs(self, ray_bundle: RayBundle):
        # outputs = super().get_outputs(ray_bundle)
        # return outputs # might need output embedded_appearance from self.field
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        # visibilities = self.get_visibility(ray_samples_list[-1:]).clamp(0, 1)
        # visibility = self.renderer_vis(betas=visibilities, weights=weights.clone().detach())
        # print(visibility.min(), visibility.max())
        # if not self.training:
        #     print(visibility.min(), visibility.max())
        #     visibility[visibility <= self.config.visibility_threshold] = 0.
        #     visibility[visibility >= self.config.visibility_threshold] = 1.

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            # "embedded_appearance": field_outputs["embedded_appearance"],
            "density": field_outputs[FieldHeadNames.DENSITY],
            # "visibility": visibility
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]

        return outputs

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras,
                               obb_box: Optional[OrientedBox] = None,
                               use_appearance: bool = False,
                               cam_i: int = 0) -> Dict[str, torch.Tensor]:
        ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box).to(self.device)
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)
            ray_bundle.nears = self.config.inference_near_plane * torch.ones_like(ray_bundle.camera_indices) # for clean purpose
            print("model: currently clip near plane to ", self.config.inference_near_plane)
        # if not use_appearance:
        #     ray_bundle.camera_indices = None
        # else:
        #     ray_bundle.camera_indices = torch.ones_like(ray_bundle.camera_indices) * cam_i
        return self.get_outputs_for_camera_ray_bundle(ray_bundle)

    def get_outputs_for_reprojection_with_depth(self, camera_src, camera_tgt, input_src, depth_tgt):
        ray_bundle = camera_tgt.generate_rays(camera_indices=0, keep_shape=False, obb_box=None) # for forward usage so set keep_shape to false
        ray_bundle.camera_indices = None
        ray_bundle = ray_bundle.to(self.device)

        delta = 2.
        ray_bundle.nears = depth_tgt.reshape(-1, 1).to(ray_bundle.origins) - delta
        ray_bundle.fars = depth_tgt.reshape(-1, 1).to(ray_bundle.origins) + delta

        ray_samples = self.proposal_sampler.initial_sampler(ray_bundle, num_samples=5) #uniform
        t = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
        weights = torch.exp(-torch.abs(t-depth_tgt.view(-1, 1, 1).to(t)))
        # m = torch.sum(weights, dim=-2).squeeze() > 0.
        # weights[m] /= torch.sum(weights[m], dim=-2, keepdim=True)
        weights /= torch.sum(weights, dim=-2, keepdim=True)
        assert not torch.isnan(weights).any()

        # src intrinsic
        focal = torch.concat([camera_src.fx[..., None], -camera_src.fy[..., None]], dim=-1).to(self.device)  # (1, 2)
        c = torch.concat([camera_src.cx[..., None], camera_src.cy[..., None]], dim=-1).to(self.device)  # (1, 2)

        # src extrinsic
        c2w_src = camera_src.camera_to_worlds
        c2w_src = c2w_src.view([1] + list(c2w_src.shape[-2:]))
        rot = c2w_src[:, :3, :3].transpose(1, 2)  # (B, N, 3, 3)
        trans = -torch.bmm(rot, c2w_src[:, :3, 3:])
        w2c_src = torch.cat((rot, trans), dim=-1)
        w2c_src = w2c_src.view([1] + list(w2c_src.shape[-2:]))
        w2c_src = w2c_src.to(self.device)

        # projection to src camera space
        xyzs = ray_samples.frustums.get_positions().unsqueeze(0)  # (B, HW, 128, 3)
        xyzs = torch.matmul(w2c_src[:, None, :3, :3], xyzs.unsqueeze(-1))[..., 0] + w2c_src[:, None, :3, 3]
        # print("nerf_model: ", xyzs.shape) # (B, HW, 128, 3) ? yes

        # projection to src image space
        uvs = torch.where(xyzs[..., 2:] == 0.0, torch.tensor([0.5, 0.5], device=xyzs.device), -xyzs[..., :2] / xyzs[..., 2:])  # (B, HW, 128, 2)
        uvs *= focal  # (B, HW, 128, 2)
        uvs += c  # (B, HW, 128, 2)
        uvs = 2 * uvs / torch.tensor([camera_src.width, camera_src.height]).to(uvs.device) - 1.0  # (B, HW, 128, 2)

        # 'bilinear' | 'nearest' | 'bicubic'
        text_map = F.interpolate(input_src, scale_factor=2, mode='nearest').squeeze().to(uvs.device)
        colors = F.grid_sample(text_map.unsqueeze(0), uvs,  # map [1, 3/4, H,W], uvs [1, HW, 128, 2]
                                align_corners=False, mode='nearest', padding_mode="zeros")  # (B, C, Hout=HW, Wout=128)
        colors = colors.squeeze().permute(1, 2, 0)  # (B, C, HW, 128) -> (HW, 128, C)
        colors = colors.reshape(list(weights.shape[:-1]) + [text_map.shape[0]])

        image_height, image_width = camera_tgt.height, camera_tgt.width

        with torch.no_grad():
            # rgb = self.renderer_rgb(rgb=colors, weights=weights) # can not use it since channel is 4
            rgb = torch.sum(colors * weights, dim=-2).reshape(image_height, image_width, colors.shape[-1])
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples).reshape(image_height, image_width, 1)
            expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples).reshape(image_height, image_width, 1)
            # accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            # "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }
        return outputs


    def get_outputs_for_reprojection(self, camera_src, camera_tgt, input_src):
        ray_bundle = camera_tgt.generate_rays(camera_indices=0, keep_shape=False, obb_box=None) # for forward usage so set keep_shape to false
        # ray_bundle.camera_indices = None
        ray_bundle = ray_bundle.to(self.device)

        if self.collider is not None: # not using if in inference: set collider to False
            ray_bundle = self.collider(ray_bundle)

        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        # with torch.no_grad():
        #     # depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        #     expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        #
        # ray_samples_z = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
        # ray_samples_mu = expected_depth.unsqueeze(1).expand(-1, ray_samples_z.shape[1], -1)
        # # ray_samples_std = torch.std(weights, dim=1, keepdim=True).expand(-1, ray_samples_z.shape[1], -1)
        # ray_samples_std = torch.ones_like(ray_samples_mu)
        # distribution = torch.distributions.Laplace(ray_samples_mu, ray_samples_std) # or Laplace/Normal
        # probs = torch.exp(distribution.log_prob(ray_samples_z))

        # weights = probs.to(weights)
        weights = torch.square(weights)
        m = torch.sum(weights, dim=-2).squeeze() > 0.
        weights[m] /= torch.sum(weights[m], dim=-2, keepdim=True)
        assert not torch.isnan(weights).any()

        # src intrinsic
        focal = torch.concat([camera_src.fx[..., None], -camera_src.fy[..., None]], dim=-1).to(self.device)  # (1, 2)
        c = torch.concat([camera_src.cx[..., None], camera_src.cy[..., None]], dim=-1).to(self.device)  # (1, 2)

        # src extrinsic
        c2w_src = camera_src.camera_to_worlds
        c2w_src = c2w_src.view([1] + list(c2w_src.shape[-2:]))
        rot = c2w_src[:, :3, :3].transpose(1, 2)  # (B, N, 3, 3)
        trans = -torch.bmm(rot, c2w_src[:, :3, 3:])
        w2c_src = torch.cat((rot, trans), dim=-1)
        w2c_src = w2c_src.view([1] + list(w2c_src.shape[-2:]))
        w2c_src = w2c_src.to(self.device)

        # projection to src camera space
        xyzs = ray_samples.frustums.get_positions().unsqueeze(0)  # (B, HW, 128, 3)
        xyzs = torch.matmul(w2c_src[:, None, :3, :3], xyzs.unsqueeze(-1))[..., 0] + w2c_src[:, None, :3, 3]
        # print("nerf_model: ", xyzs.shape) # (B, HW, 128, 3) ? yes

        # projection to src image space
        uvs = torch.where(xyzs[..., 2:] == 0.0, torch.tensor([0.5, 0.5], device=xyzs.device), -xyzs[..., :2] / xyzs[..., 2:])  # (B, HW, 128, 2)
        uvs *= focal  # (B, HW, 128, 2)
        uvs += c  # (B, HW, 128, 2)
        uvs = 2 * uvs / torch.tensor([camera_src.width, camera_src.height]).to(uvs.device) - 1.0  # (B, HW, 128, 2)

        # 'bilinear' | 'nearest' | 'bicubic'
        text_map = F.interpolate(input_src, scale_factor=2, mode='nearest').squeeze().to(uvs.device)
        colors = F.grid_sample(text_map.unsqueeze(0), uvs,  # map [1, 3/4, H,W], uvs [1, HW, 128, 2]
                                align_corners=False, mode='nearest', padding_mode="zeros")  # (B, C, Hout=HW, Wout=128)
        colors = colors.squeeze().permute(1, 2, 0)  # (B, C, HW, 128) -> (HW, 128, C)
        colors = colors.reshape(list(weights.shape[:-1]) + [text_map.shape[0]])

        image_height, image_width = camera_tgt.height, camera_tgt.width

        with torch.no_grad():
            # rgb = self.renderer_rgb(rgb=colors, weights=weights) # can not use it since channel is 4
            rgb = torch.sum(colors * weights, dim=-2).reshape(image_height, image_width, colors.shape[-1])
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples).reshape(image_height, image_width, 1)
            expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples).reshape(image_height, image_width, 1)
            # accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            # "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }
        return outputs


    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        metrics_dict["distortion"] = 0.

        compute_on = "random_rays"
        # gt_rgb = batch[compute_on + "_image"].to(self.device) # RGB or RGBA image
        # gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        # metrics_dict["psnr"] = self.psnr(outputs[compute_on + "_rgb"], gt_rgb)
        if self.training:
            metrics_dict["distortion"] += distortion_loss(
                outputs[compute_on + "_weights_list"], outputs[compute_on + "_ray_samples_list"]
            )
        self.camera_optimizer.get_metrics_dict(metrics_dict)

        if "patches_indices" in batch.keys():
            compute_on = "patches"
            if self.training:
                metrics_dict["distortion"] += 1. * distortion_loss(
                    outputs[compute_on + "_weights_list"], outputs[compute_on + "_ray_samples_list"]
                )
                metrics_dict["distortion"] /= 2.

            # if "patches_depth_image" in batch.keys():
            #     # if self.config.depth_loss_type in (DepthLossType.DS_NERF, DepthLossType.URF):
            #     metrics_dict["depth_loss"] = 0.0
            #     sigma = self._get_sigma().to(self.device)
            #     termination_depth = batch[compute_on+"_depth_image"].to(self.device).reshape_as(outputs[compute_on+"_expected_depth"])
            #
            #     # mask_dpt = batch[compute_on + '_mask'].to(self.device).view(-1)
            #     # mask_dpt = torch.ones_like(mask_dpt, dtype=torch.bool)
            #     mask_dpt = torch.rand_like(batch[compute_on + '_mask'].float().to(self.device).view(-1)) < 0.1
            #
            #     # patch_size = outputs["patch_size"]
            #     # real_depth = batch[compute_on+"_depth_image"].to(self.device).reshape(patch_size, patch_size, 1)
            #     # fake_depth = outputs[compute_on+"_expected_depth"].to(self.device).reshape(patch_size, patch_size, 1)
            #     # d_mse = torch.sqrt((real_depth - fake_depth) ** 2)
            #     # mask_dpt = d_mse.to(self.device).view(-1) > 0.0001
            #
            #     for i in range(len(outputs[compute_on+"_weights_list"])):
            #         metrics_dict["depth_loss"] += depth_loss(
            #             weights=outputs[compute_on+"_weights_list"][i][mask_dpt],
            #             ray_samples=outputs[compute_on+"_ray_samples_list"][i][mask_dpt],
            #             termination_depth=termination_depth[mask_dpt],
            #             predicted_depth=outputs[compute_on+"_expected_depth"][mask_dpt],
            #             sigma=sigma,
            #             directions_norm=outputs[compute_on+"_directions_norm"][mask_dpt],
            #             is_euclidean=self.config.is_euclidean_depth,
            #             depth_loss_type=self.config.depth_loss_type,
            #         ) / len(outputs[compute_on+"_weights_list"])
            #
            #     # elif self.config.depth_loss_type in (DepthLossType.SPARSENERF_RANKING,):
            #     # metrics_dict["depth_ranking"] = depth_ranking_loss(
            #     #     outputs[compute_on+"_expected_depth"].reshape_as(batch[compute_on+"_depth_image"]),
            #     #     batch[compute_on+"_depth_image"].to(self.device)
            #     # )
            #     # else:
            #     #     raise NotImplementedError(f"Unknown depth loss type {self.config.depth_loss_type}")

        return metrics_dict

    def get_patches(self, outputs, batch, patch_type="patches", fake_only=False, posneg1=False):
        real_patch_key = patch_type + "_image"
        fake_patch_key = patch_type + "_rgb"
        bs = outputs["num_patches"]
        patch_size = outputs["patch_size"]
        if not fake_only:
            real_patches = batch[real_patch_key].to(self.device).reshape(bs, patch_size, patch_size, 3)
            real_patches = real_patches.permute(0, 3, 1, 2).contiguous()
            if posneg1:
                real_patches = real_patches * 2.0 - 1.0
        else:
            real_patches = None
        fake_patches = outputs[fake_patch_key].reshape(bs, patch_size, patch_size, 3)
        fake_patches = fake_patches.permute(0, 3, 1, 2).contiguous()
        if posneg1:
            fake_patches = fake_patches * 2.0 - 1.0
        return real_patches, fake_patches

    def run_discriminator(self, x):
        bs, c, h, w = x.shape
        if self.training:
            input = x.contiguous()
        else:
            input = x
        # reshape
        d_img_resolution = self.discriminator.img_resolution
        if d_img_resolution < h:
            input = (
                input.unfold(2, d_img_resolution, d_img_resolution)
                .unfold(3, d_img_resolution, d_img_resolution)
                .permute(0, 2, 3, 1, 4, 5)
                .reshape(-1, c, d_img_resolution, d_img_resolution)
            )
        # call
        output = self.discriminator(input)
        # reshape
        if d_img_resolution < h:
            output = output.reshape(bs, 1, -1)
        return output

    def compute_adv_loss(self, fake_patch):
        pred_fake = self.run_discriminator(fake_patch)
        adv_loss = torch.nn.functional.softplus(-pred_fake).mean()
        return adv_loss

    def compute_simple_gradient_penalty(self, x):
        x.requires_grad_(True)
        pred_real = self.run_discriminator(x)
        gradients = torch.autograd.grad(outputs=[pred_real.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]
        r1_penalty = gradients.square().sum([1, 2, 3]).mean()
        return r1_penalty / 2

    def compute_d_loss(self, real_patch, fake_patch):
        pred_real = self.run_discriminator(real_patch)
        loss_d_real = torch.nn.functional.softplus(-pred_real).mean()
        gradient_penalty = self.config.gp_loss_mult * self.compute_simple_gradient_penalty(real_patch)
        loss_d = loss_d_real
        if fake_patch is not None:
            pred_fake = self.run_discriminator(fake_patch.detach())
            loss_d_fake = torch.nn.functional.softplus(pred_fake).mean()
            loss_d += loss_d_fake
            # print("model: loss_d real and fake", loss_d_real, loss_d_fake)
        return loss_d, gradient_penalty

    def get_discriminator_loss_dict(self, step, outputs, batch, metrics_dict=None):
        loss_dict = {}
        if self.config.gp_loss_mult > 0:
            real_patches, fake_patches = self.get_patches(outputs, batch, posneg1=True)
            # discriminator loss
            set_requires_grad(self.discriminator, True)
            if batch['patches_is_base']: fake_patches = None
            loss_dict["d_loss"], loss_dict["gp"] = self.compute_d_loss(real_patches, fake_patches)
        return loss_dict

    def get_object_bbox_loss_dict(self, ray_samples: RaySamples):
        ray_samples = ray_samples.to(self.device)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        density = field_outputs[FieldHeadNames.DENSITY]
        # rgb = field_outputs[FieldHeadNames.RGB]
        density_loss = density.mean() #shrink towards 0
        # density_loss = (10.-density).abs().mean()
        return {"density_loss": 0.001*density_loss}


    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}

        # if outputs["vis_only"] and self.training:
        #     loss_dict["visibility_loss"] = 100 * torch.mean((1. - outputs["random_rays_visibility"].clamp(0, 1)) ** 2)  # visibility loss, only unmasked is sampled
        #     return loss_dict

        # losses on random rays
        compute_on = "random_rays"
        # gt_image = batch[compute_on + "_image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs[compute_on + "_rgb"],
            pred_accumulation=outputs[compute_on + "_accumulation"],
            gt_image=batch[compute_on + "_image"].to(self.device),
        )
        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs[compute_on + "_weights_list"], outputs[compute_on + "_ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            self.camera_optimizer.get_loss_dict(loss_dict)

            if "random_rays_depth_image" in batch.keys() and self.config.depth_loss_mult > 0.:
                metrics_dict["depth_loss"] = 0.0
                termination_depth = batch[compute_on + "_depth_image"].to(self.device).reshape_as(
                    outputs[compute_on + "_expected_depth"])
                predicted_depth = outputs[compute_on + "_expected_depth"]
                d_mse = torch.mean((predicted_depth - termination_depth) ** 2)
                loss_dict["depth_loss"] = self.config.depth_loss_mult * d_mse

        if "patches_indices" in batch.keys(): # means use patches
            compute_on = "patches"
            if self.training:
                loss_dict["interlevel_loss"] += self.config.interlevel_loss_mult * interlevel_loss(
                    outputs[compute_on + "_weights_list"], outputs[compute_on + "_ray_samples_list"]
                )
                loss_dict["interlevel_loss"] /= 2.
                if loss_dict.get("camera_opt_regularizer", None):
                    del loss_dict["camera_opt_regularizer"]

            if batch['patches_is_base']:
                # gt_rgb = batch[compute_on + "_image"].to(self.device)
                pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
                    pred_image=outputs[compute_on + "_rgb"],
                    pred_accumulation=outputs[compute_on + "_accumulation"],
                    gt_image=batch[compute_on + "_image"].to(self.device),
                )
                loss_dict["rgb_loss"] += self.rgb_loss(gt_rgb, pred_rgb.reshape_as(gt_rgb))
            else:
                if batch["patches_for_nerf"]:
                    # photo metric loss
                    gt_rgb = batch[compute_on + "_image"].to(self.device)
                    pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
                        pred_image=outputs[compute_on + "_rgb"],
                        pred_accumulation=outputs[compute_on + "_accumulation"],
                        gt_image=gt_rgb,
                    )
                    loss_dict["rgb_loss"] += self.config.l1_loss_mult * F.l1_loss(gt_rgb, pred_rgb.reshape_as(gt_rgb))

                # perceptual loss
                if self.config.vgg_loss_mult > 0:
                    real_patches, fake_patches = self.get_patches(outputs, batch)
                    include_first = outputs["num_patches"]
                    real_patches = real_patches[:include_first]
                    fake_patches = fake_patches[:include_first]
                    loss_dict["vgg_loss"] = self.config.vgg_loss_mult * self.vgg_loss(fake_patches, real_patches)
                    # loss_dict["vgg_loss"] = self.config.vgg_loss_mult * self.vgg_loss(fake_patches, real_patches, require_grad=True, batch_average=True)

                # adversarial loss
                if self.config.adv_loss_mult > 0:
                    # compute the loss for each scale
                    set_requires_grad(self.discriminator, False)
                    _, fake_patches = self.get_patches(outputs, batch, fake_only=True, posneg1=True)
                    loss_dict["adv_loss"] = self.config.adv_loss_mult * self.compute_adv_loss(fake_patches)

                # # likelihood loss
                # if self.config.nll_loss_mult > 0 and outputs[compute_on + "_embedded_appearance"] is not None:
                #     embeddings = outputs[compute_on + "_embedded_appearance"]
                #     embeddings = embeddings.reshape((-1, embeddings.shape[-1]))
                #     log_var = torch.tensor(0.)
                #     nll_loss = torch.mean(-0.5 * torch.sum(1 + log_var - embeddings ** 2 - log_var.exp(), dim=1), dim=0)
                #     loss_dict['nll_loss'] = self.config.nll_loss_mult * nll_loss  # 0.0001 shrink to [0,3] 0.00001 to [4, 13]

            # if "patches_depth_image" in batch.keys():
                # loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]
                # loss_dict["depth_ranking"] = (self.config.depth_loss_mult * 0.1 * np.interp(self.step, [0, 2000], [0, 0.2]) * metrics_dict["depth_ranking"])

            patch_size = outputs["patch_size"]
            fake_depth = outputs[compute_on + "_expected_depth"].to(self.device).reshape(patch_size, patch_size, 1)

            w_variance = torch.mean(torch.pow(fake_depth[:, :-1] - fake_depth[:, 1:], 2))
            h_variance = torch.mean(torch.pow(fake_depth[:-1, :] - fake_depth[1:, :], 2))
            tv_loss = h_variance + w_variance
            loss_dict["depth_tv_loss"] = self.config.tv_loss_mult * tv_loss

            if "patches_depth_image" in batch.keys() and self.config.depth_loss_mult > 0.:
                if self.config.depth_loss_type == DepthLossType.DS_NERF:
                    metrics_dict["depth_loss"] = 0.0
                    sigma = self._get_sigma().to(self.device)
                    termination_depth = batch[compute_on + "_depth_image"].to(self.device).reshape_as(
                        outputs[compute_on + "_expected_depth"])

                    mask_dpt = torch.rand_like(batch[compute_on + '_mask'].float().to(self.device).view(-1)) < 0.1
                    # mask_dpt = batch[compute_on + '_mask'].to(self.device).view(-1)
                    for i in range(len(outputs[compute_on + "_weights_list"])):
                        metrics_dict["depth_loss"] += depth_loss(
                            weights=outputs[compute_on + "_weights_list"][i][mask_dpt],
                            ray_samples=outputs[compute_on + "_ray_samples_list"][i][mask_dpt],
                            termination_depth=termination_depth[mask_dpt],
                            predicted_depth=outputs[compute_on + "_expected_depth"][mask_dpt],
                            sigma=sigma,
                            directions_norm=outputs[compute_on + "_directions_norm"][mask_dpt],
                            is_euclidean=self.config.is_euclidean_depth,
                            depth_loss_type=self.config.depth_loss_type,
                        ) / len(outputs[compute_on + "_weights_list"])
                    loss_dict["depth_loss"] = 0.1 * self.config.depth_loss_mult * metrics_dict["depth_loss"]
                elif self.config.depth_loss_type == DepthLossType.URF:
                    real_depth = batch[compute_on + "_depth_image"].to(self.device).reshape(patch_size, patch_size, 1)
                    mask_dpt = torch.rand_like(batch[compute_on + '_mask'].float().to(self.device).reshape(patch_size, patch_size, 1)) < 0.2
                    # mask_dpt = batch[compute_on + '_mask'].to(self.device).reshape(patch_size, patch_size, 1)
                    d_mse = torch.mean((real_depth[mask_dpt] - fake_depth[mask_dpt]) ** 2)
                    loss_dict["depth_loss"] = self.config.depth_loss_mult * d_mse
                    print(f"model: depth loss {d_mse.item():.5f}, tv loss {tv_loss.item():.5f}")
                else:
                    raise NotImplementedError(f"Unknown depth loss type {self.config.depth_loss_type}")

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        if "mask" in batch:
            print("model: evaluate metrics on the masked region")
            gt_msk = batch["mask"].to(self.device)
            bbox = get_mask_bbox(gt_msk)
            left, right, top, bottom = bbox['left'], bbox['right'], bbox['top'], bbox['bottom']
            gt_rgb = gt_rgb[top:bottom, left:right]
            predicted_rgb = predicted_rgb[top:bottom, left:right]

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb_1chw = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        if min(gt_rgb_1chw.shape[2:]) < 32:
            _h, _w = gt_rgb.shape[2] * 2, gt_rgb.shape[3] * 2
            gt_rgb_1chw = F.interpolate(gt_rgb_1chw, size=(_h, _w), mode='bilinear', align_corners=False)
            predicted_rgb = F.interpolate(predicted_rgb, size=(_h, _w), mode='bilinear', align_corners=False)

        psnr = self.psnr(gt_rgb_1chw, predicted_rgb)
        ssim = self.ssim(gt_rgb_1chw, predicted_rgb)
        lpips = self.lpips(gt_rgb_1chw, predicted_rgb.clamp(0.0, 1.0))
        musiq = self.musiq_metric(predicted_rgb).mean()

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)
        metrics_dict["musiq"] = float(musiq)

        images_dict = {
            "rgb": outputs["rgb"],
            "real": gt_rgb, 
            "img": combined_rgb, 
            "accumulation": combined_acc,
            "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict


def set_requires_grad(net, requires_grad=False):
    for param in net.parameters():
        param.requires_grad = requires_grad