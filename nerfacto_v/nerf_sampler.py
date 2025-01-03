import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Type, Union

import torch
from jaxtyping import Int
from torch import Tensor

from nerfstudio.data.pixel_samplers import PixelSampler, PixelSamplerConfig


@dataclass
class MaskPixelSamplerConfig(PixelSamplerConfig):
    _target: Type = field(default_factory=lambda: MaskPixelSampler)



class MaskPixelSampler(PixelSampler):
    """Samples 'pixel_batch's from 'image_batch's. Samples square patches
    from the images randomly. Useful for patch-based losses.

    Args:
        config: the PatchPixelSamplerConfig used to instantiate class
    """

    config: MaskPixelSamplerConfig

    # overrides base method

    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        """
        Naive pixel sampler, uniformly samples across all possible pixels of all possible images.

        Args:
            batch_size: number of samples in a batch
            num_images: number of images to sample over
            mask: mask of possible pixels in an image to sample from.
        """
        indices = (
            torch.rand((batch_size, 3), device=device)
            * torch.tensor([num_images, image_height, image_width], device=device)
        ).long()

        if isinstance(mask, torch.Tensor) and not self.config.ignore_mask:
            mask = ~mask.bool()
            if self.config.rejection_sample_mask:
                num_valid = 0
                for _ in range(self.config.max_num_iterations):
                    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
                    chosen_indices_validity = mask[..., 0][c, y, x].bool()
                    num_valid = int(torch.sum(chosen_indices_validity).item())
                    if num_valid == batch_size:
                        break
                    else:
                        replacement_indices = (
                            torch.rand((batch_size - num_valid, 3), device=device)
                            * torch.tensor([num_images, image_height, image_width], device=device)
                        ).long()
                        indices[~chosen_indices_validity] = replacement_indices

                if num_valid != batch_size:
                    warnings.warn(
                        """
                        Masked sampling failed, mask is either empty or mostly empty.
                        Reverting behavior to non-rejection sampling. Consider setting
                        pipeline.datamanager.pixel-sampler.rejection-sample-mask to False
                        or increasing pipeline.datamanager.pixel-sampler.max-num-iterations
                        """
                    )
                    self.config.rejection_sample_mask = False
                    nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
                    chosen_indices = random.sample(range(len(nonzero_indices)), k=batch_size)
                    indices = nonzero_indices[chosen_indices]
            else:
                nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
                chosen_indices = random.sample(range(len(nonzero_indices)), k=batch_size)
                indices = nonzero_indices[chosen_indices]

        return indices



# @dataclass
# class MaskPatchPixelSamplerConfig(PixelSamplerConfig):
#     """Config dataclass for PatchPixelSampler."""
#
#     _target: Type = field(default_factory=lambda: MaskPatchPixelSampler)
#     """Target class to instantiate."""
#     patch_size: int = 256
#     """Side length of patch. This must be consistent in the method
#     config in order for samples to be reshaped into patches correctly."""
#
#
# class MaskPatchPixelSampler(PixelSampler):
#     """Samples 'pixel_batch's from 'image_batch's. Samples square patches
#     from the images randomly. Useful for patch-based losses.
#
#     Args:
#         config: the PatchPixelSamplerConfig used to instantiate class
#     """
#
#     config: MaskPatchPixelSamplerConfig
#
#     def set_num_rays_per_batch(self, num_rays_per_batch: int):
#         """Set the number of rays to sample per batch. Overridden to deal with patch-based sampling.
#
#         Args:
#             num_rays_per_batch: number of rays to sample per batch
#         """
#         self.num_rays_per_batch = (num_rays_per_batch // (self.config.patch_size**2)) * (self.config.patch_size**2)
#
#     def find_mask_bbox(self, mask):
#         if len(mask.shape) > 2:
#             mask = mask.squeeze()
#             assert len(mask.shape) == 2
#
#         if not mask.dtype == torch.bool:
#             mask = mask.bool()
#
#         # Get the indices where the mask is True
#         nonzero_indices = torch.nonzero(mask, as_tuple=True)
#         rows = nonzero_indices[0]
#         cols = nonzero_indices[1]
#
#         # Calculate limits
#         top = torch.min(rows).item() if len(rows) > 0 else None
#         bottom = torch.max(rows).item() if len(rows) > 0 else None
#         left = torch.min(cols).item() if len(cols) > 0 else None
#         right = torch.max(cols).item() if len(cols) > 0 else None
#         return {'left': left, 'right': right, 'top': top, 'bottom': bottom,
#                 'center_x': (left+right)//2, 'center_y': (top+bottom)//2, 'height': bottom-top, 'width': right-left}
#
#     # overrides base method
#     def sample_method(
#         self,
#         batch_size: int,
#         num_images: int,
#         image_height: int,
#         image_width: int,
#         mask: Optional[Tensor] = None,
#         device: Union[torch.device, str] = "cpu",
#     ) -> Int[Tensor, "batch_size 3"]:
#         if isinstance(mask, Tensor) and not self.config.ignore_mask:
#             sub_bs = batch_size // (self.config.patch_size**2)
#             assert sub_bs == 1
#             half_patch_size = int(self.config.patch_size / 2)
#
#             bbox = self.find_mask_bbox(mask)
#             center_x, center_y = bbox['center_x'], bbox['center_y']
#             pad = max(half_patch_size // 4, (self.config.patch_size-min(bbox['height'], bbox['weight']))//2) # 32
#             center_x += random.randint(-pad, pad)
#             center_y += random.randint(-pad, pad)
#
#             # m = erode_mask(mask.permute(0, 3, 1, 2).float(), pixel_radius=half_patch_size)
#             # nonzero_indices = torch.nonzero(m[:, 0], as_tuple=False).to(device)
#             # chosen_indices = random.sample(range(len(nonzero_indices)), k=sub_bs)
#             # indices = nonzero_indices[chosen_indices]
#
#             m = torch.zeros_like(mask.squeeze())
#             m[center_x-half_patch_size:center_x+half_patch_size, center_y-half_patch_size: center_y+half_patch_size] = 1
#             indices = torch.nonzero(m, as_tuple=False).to(device)
#
#             indices = (
#                 indices.view(sub_bs, 1, 1, 3)
#                 .broadcast_to(sub_bs, self.config.patch_size, self.config.patch_size, 3)
#                 .clone()
#             )
#
#             yys, xxs = torch.meshgrid(
#                 torch.arange(self.config.patch_size, device=device), torch.arange(self.config.patch_size, device=device)
#             )
#             indices[:, ..., 1] += yys - half_patch_size
#             indices[:, ..., 2] += xxs - half_patch_size
#
#             indices = torch.floor(indices).long()
#             indices = indices.flatten(0, 2)
#         else:
#             raise NotImplementedError
#             # sub_bs = batch_size // (self.config.patch_size**2)
#             # indices = torch.rand((sub_bs, 3), device=device) * torch.tensor(
#             #     [num_images, image_height - self.config.patch_size, image_width - self.config.patch_size],
#             #     device=device,
#             # )
#             #
#             # indices = (
#             #     indices.view(sub_bs, 1, 1, 3)
#             #     .broadcast_to(sub_bs, self.config.patch_size, self.config.patch_size, 3)
#             #     .clone()
#             # )
#             #
#             # yys, xxs = torch.meshgrid(
#             #     torch.arange(self.config.patch_size, device=device), torch.arange(self.config.patch_size, device=device)
#             # )
#             # indices[:, ..., 1] += yys
#             # indices[:, ..., 2] += xxs
#             #
#             # indices = torch.floor(indices).long()
#             # indices = indices.flatten(0, 2)
#
#         return indices