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
from typing import (
    Dict,
    Literal,
    Tuple,
    Type,
    Union,
)
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F

# from nerfstudio.data.pixel_samplers import PatchPixelSamplerConfig
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig

from .nerf_sampler import MaskPixelSamplerConfig
from .nerf_dataset import PseudoDepthDataset

@dataclass
class NerfDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: NerfDataManager)
    num_patches: int = 1
    """Number of patches to sample."""
    pixel_sampler: MaskPixelSamplerConfig = MaskPixelSamplerConfig()


class NerfDataManager(VanillaDataManager):

    def __init__(
        self,
        config: NerfDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
        self.prompt = self.dataparser._get_prompt()
        self.spherify = self.dataparser.spherify
        print("datamanager: prompt: ", self.prompt)

    def create_train_dataset(self, depths=None):
        if depths is None:
            """Sets up the data loaders for training"""
            return self.dataset_type(
                dataparser_outputs=self.train_dataparser_outputs,
                scale_factor=self.config.camera_res_scale_factor,
            )
        else:
            return PseudoDepthDataset(
                dataparser_outputs=self.train_dataparser_outputs,
                scale_factor=self.config.camera_res_scale_factor,
                depths=depths
            )

    def disable_mask_training(self):
        self.train_pixel_sampler.config.ignore_mask = True

    def enable_mask_training(self):
        self.train_pixel_sampler.config.ignore_mask = False

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self.config.pixel_sampler.setup(num_rays_per_batch=self.config.train_num_rays_per_batch)
        if self.config.num_patches > 0:
            # patch_size = self.config.patch_size
            # patch_pixel_sampler_config = MaskPatchPixelSampler(patch_size=patch_size)
            # num_patch_rays = patch_size * patch_size * self.config.num_patches
            # self.train_patch_pixel_sampler = patch_pixel_sampler_config.setup(patch_size=patch_size, num_rays_per_batch=num_patch_rays)
            print(self.config.num_patches)
        # self.train_camera_optimizer = self.config.camera_optimizer.setup(
        #     num_cameras=self.train_dataset.cameras.size, device=self.device
        # )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            # self.train_camera_optimizer,
        )
        self.train_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.train_patch_data = [{} for _ in range(len(self.train_dataset))]

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)
        # self.eval_camera_optimizer = self.config.camera_optimizer.setup(
        #     num_cameras=self.eval_dataset.cameras.size, device=self.device
        # )
        self.eval_ray_generator = RayGenerator(
            self.eval_dataset.cameras.to(self.device),
            # self.eval_camera_optimizer,
        )
        # for loading full images
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.fixed_indices_train_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        ) # used to compute kid between input images and rendered test images
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    def next_train(self, step: int, patch_list: list = [], patches_for_nerf: bool = True) -> Tuple[Dict, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        ray_bundles = dict()
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        # assert not 'depth_image' in batch.keys()
        ray_indices = batch["indices"]
        batch = {"random_rays_" + k: v for k, v in batch.items()}
        ray_bundle = self.train_ray_generator(ray_indices)
        # ray_bundle.camera_indices = None
        ray_bundles["random_rays"] = ray_bundle
        if len(patch_list) > 0 and self.config.num_patches > 0:
            if len(patch_list) == 1:
                is_base = True
                img_i = patch_list[0]
            else:
                is_base = False
                img_i = np.random.choice(patch_list)

            s = 1 # s = np.random.choice([0.5, 1.0, 2.0], size=1, p=[0.2, 0.5, 0.3])[0]
            patch_ray_indices, image_patch, mask_patch, depth_patch = self.get_mask_patch_pixel(img_i, s)

            # camera = copy.deepcopy(self.get_train_og_camera(img_i))
            # camera.rescale_output_resolution(s)
            # y = patch_ray_indices[:, 1]  # row indices
            # x = patch_ray_indices[:, 2]  # col indices
            # patch_ray_bundle = camera.generate_rays(
            #     camera_indices=0,
            #     coords=camera.get_image_coords()[y, x],
            # ).to(self.device)
            patch_ray_bundle = self.train_ray_generator(patch_ray_indices)
            if is_base:
                # patch_ray_bundle.camera_indices = None
                batch["patches_is_base"] = True
            else:
                # patch_ray_bundle.camera_indices = None
                batch["patches_is_base"] = False # for execute of non mse loss

            batch["patches_for_nerf"] = patches_for_nerf
            # print("datamanager: ", img_i, patch_ray_bundle.camera_indices, batch["patches_is_base"])

            patch_batch = {"image": image_patch, "indices": patch_ray_indices, "mask": mask_patch}
            if depth_patch is not None:
                patch_batch["depth_image"] = depth_patch

            ray_bundles["patches"] = patch_ray_bundle
            batch.update({"patches_" + k: v for k, v in patch_batch.items()})
        return ray_bundles, batch

    def get_train_og_camera(self, idx: int):
        return self.train_dataset.cameras[idx: idx+1]

    def get_train_og_data_tensor(self, idx: int):
        data = self.train_dataset.__getitem__(idx)
        # data['mask'] = dilate_mask(data['mask'], kernel_size=9, iterations=11)
        return data

    def get_train_og_length(self): 
        return len(self.train_dataset)

    def get_prompt(self):
        return self.prompt

    def get_train_patch_data(self, idx: int):
        # assert 'mask' not in data.keys()
        return self.train_patch_data[idx]

    def update_train_patch_data(self, idx: int, data):
        # assert 'mask' not in data.keys()
        self.train_patch_data[idx] = data
        print("datamanager: Succefully upadate patch data: ", idx)

    def update_train_patch_map(self, idx: int, map_name, data):
        # assert map_name != 'mask'
        self.train_patch_data[idx][map_name] = data
        print("datamanager: Succefully upadate patch map: ", idx)

    def get_mask_patch_pixel(self, img_i: int, scale: float):
        data = self.train_patch_data[img_i]
        image = torch.from_numpy(data['image']).float()
        mask = torch.from_numpy(data['mask']).bool()
        image_height, image_width = image.shape[:2]

        if scale != 1.0:
            image_height = int(image_height * scale)
            image_width = int(image_width * scale)
            image = F.interpolate(image.permute(2, 0, 1).unsqueeze(0), size=(image_height, image_width), mode='area').squeeze(0).permute(1,2,0)
            mask = F.interpolate(mask.float().permute(2, 0, 1).unsqueeze(0), size=(image_height, image_width), mode='nearest').squeeze(0).permute(1, 2, 0).bool()


        bbox = get_mask_bbox(mask)
        center_row, center_col = bbox['center_row'], bbox['center_col']

        half_patch_size = int(self.config.patch_size / 2)
        pad = max(half_patch_size // 8, (self.config.patch_size - min(bbox['height'], bbox['width'])) // 2)  # 16
        center_row += random.randint(-pad, pad)
        center_col += random.randint(-pad, pad)
        center_row = min(max(center_row, half_patch_size), image_height-half_patch_size) # make sure in the frame
        center_col = min(max(center_col, half_patch_size), image_width-half_patch_size)
        image_patch = image[center_row-half_patch_size:center_row+half_patch_size,
                      center_col-half_patch_size: center_col+half_patch_size]
        mask_patch = mask[center_row-half_patch_size:center_row+half_patch_size,
                     center_col-half_patch_size: center_col+half_patch_size]

        if data.get('depth', None) is not None:
            depth = torch.from_numpy(data['depth']).float()
            depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(image_height, image_width), mode='area').squeeze()
            depth_patch = depth[center_row-half_patch_size:center_row+half_patch_size,
                          center_col-half_patch_size: center_col+half_patch_size]
        else:
            depth_patch = None

        # Generate all x (columns) and y (rows) coordinates within the bounding box
        cols = torch.arange(center_col - half_patch_size, center_col + half_patch_size)  # right inclusive
        rows = torch.arange(center_row - half_patch_size, center_row + half_patch_size)  # bottom inclusive

        # Create a grid of x and y coordinates
        yy, xx = torch.meshgrid(rows, cols, indexing='ij')

        # Stack the coordinates to list each pair; reshape to [num_points, 2]
        patch_ray_indices = torch.stack((torch.full_like(xx.flatten(), img_i), yy.flatten(), xx.flatten()), dim=1)
        # image_indices = torch.stack((xx.flatten(), yy.flatten()), dim=1)
        # camera_indices = torch.full_like(xx.flatten(), img_i)

        patch_ray_indices = torch.floor(patch_ray_indices).long()
        # image_indices = indices.flatten(0, 2)

        return patch_ray_indices, image_patch, mask_patch, depth_patch


def get_mask_bbox(mask):
    if len(mask.shape) > 2:
        mask = mask.squeeze()
        assert len(mask.shape) == 2

    if not mask.dtype == torch.bool:
        mask = mask.bool()

    # Get the indices where the mask is True
    nonzero_indices = torch.nonzero(mask, as_tuple=True)
    rows = nonzero_indices[0]
    cols = nonzero_indices[1]

    # Calculate limits
    top = torch.min(rows).item() if len(rows) > 0 else None
    bottom = torch.max(rows).item() if len(rows) > 0 else None
    left = torch.min(cols).item() if len(cols) > 0 else None
    right = torch.max(cols).item() if len(cols) > 0 else None
    return {'left': left, 'right': right, 'top': top, 'bottom': bottom,
            'center_col': (left+right)//2, 'center_row': (top+bottom)//2, 'height': bottom-top, 'width': right-left}


def dilate_mask(mask, kernel_size=3, iterations=5):
    if mask.dim() != 3 or mask.size(2) != 1:
        raise ValueError("mask should be of shape [H, W, 1]")
    if mask.dtype != torch.bool:
        raise ValueError("mask should be a boolean tensor")

    dilation_kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32, device=mask.device)
    padding_size = kernel_size // 2

    mask_float = mask.float().squeeze(-1)  # Remove the last dimension to get [H, W]
    mask_float = mask_float.unsqueeze(0).unsqueeze(0)  # Now shape is [1, 1, H, W]

    for _ in range(iterations):
        mask_float = F.conv2d(mask_float, dilation_kernel, padding=padding_size)
        mask_float = (mask_float > 0).float()  # Apply threshold and convert back to float for next iteration

        # Convert final output back to boolean tensor and restore original shape
    dilated_mask = mask_float.squeeze(0).squeeze(0).to(torch.bool).unsqueeze(-1)

    return dilated_mask



# def dilate_mask_near_left_edge(mask, kernel_size=3, iterations=5, dilation_width=50):
#     """
#     Dilates a given boolean mask tensor near its leftmost edge.
#
#     Parameters:
#         mask (torch.Tensor): A boolean tensor of shape [H, W, 1].
#         kernel_size (int): The size of the dilation kernel, must be an odd number.
#         iterations (int): Number of times the dilation should be applied.
#         dilation_width (int): Width of the dilation area from the leftmost edge.
#
#     Returns:
#         torch.Tensor: A tensor where only the area near the left edge has been dilated.
#     """
#     if mask.dim() != 3 or mask.size(2) != 1:
#         raise ValueError("mask should be of shape [H, W, 1]")
#     if mask.dtype != torch.bool:
#         raise ValueError("mask should be a boolean tensor")
#
#     # Finding the leftmost edge of non-zero elements in the mask
#     non_zero_columns = torch.any(mask.squeeze(-1), dim=0)
#     leftmost_edge = torch.nonzero(non_zero_columns, as_tuple=True)[0][0].item()
#
#     # Define the range to apply dilation
#     dilation_start = max(0, leftmost_edge)
#     dilation_end = min(mask.size(1), dilation_start + dilation_width)
#
#     # Prepare the dilation kernel
#     dilation_kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32, device=mask.device)
#     padding_size = kernel_size // 2
#
#     # Prepare mask for conv2d
#     mask_float = mask.float().squeeze(-1)  # Remove the last dimension to get [H, W]
#     mask_float = mask_float.unsqueeze(0).unsqueeze(0)  # Now shape is [1, 1, H, W]
#
#     # Apply dilation iteratively
#     for _ in range(iterations):
#         mask_float = F.conv2d(mask_float, dilation_kernel, padding=padding_size)
#         mask_float = (mask_float > 0).float()  # Apply threshold and convert back to float for next iteration
#
#     # Convert final output back to boolean tensor
#     dilated_mask = mask_float.squeeze(0).squeeze(0).to(torch.bool)
#
#     # Restore regions outside the dilation zone
#     original_mask_flat = mask.squeeze(-1)
#     dilated_mask[:, :dilation_start] = original_mask_flat[:, :dilation_start]
#     dilated_mask[:, dilation_end:] = original_mask_flat[:, dilation_end:]
#
#     return dilated_mask.unsqueeze(-1)  # Restore original shape [H, W, 1]

