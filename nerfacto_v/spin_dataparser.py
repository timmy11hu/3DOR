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
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Literal, Optional, Type

import numpy as np
import torch
from PIL import Image
from rich.prompt import Confirm

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)
from nerfstudio.process_data.colmap_utils import parse_colmap_camera_params
from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.scripts import run_command

MAX_AUTO_RESOLUTION = 1600

from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig, ColmapDataParser


prompt_dict = {
    "1": "A stone bench, a bush in the background, the bench is grey with a rectangular shape in perspective, photorealistic 8k",
    "2": "A wooden tree trunk on dirt, photorealistic 8k",
    "3": "A red fence, photorealistic 8k",
    "4": "Stone stairs, photorealistic 8k",
    "7": "A circular lid made of rusty iron on a grass ground, photorealistic 8k",
    "9": "A corner of a brick wall, photorealistic 8k",
    "10": "A wooden bench in front of a white fence, photorealistic 8k",
    "12": "An image of nature with grass, bushes in the background, photorealistic 8k",
    "trash": "A brick wall, photorealistic 8k",
    "book": "A desk in front of a brick wall with an iron pipe, photorealistic 8k"
}


@dataclass
class SpinDataParserConfig(ColmapDataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: SpinDataParser)
    downscale_factor: Optional[int] = 4
    images_path: Path = Path("images")
    masks_path: Optional[Path] = Path("masks")
    colmap_path: Path = Path("sparse/0")
    load_3D_points: bool = False
    assume_colmap_world_coordinate_convention: bool = False

    # """target class to instantiate"""
    # data: Path = Path()
    # """Directory or explicit json file path specifying location of data."""
    # scale_factor: float = 1.0
    # """How much to scale the camera origins by."""
    # downscale_factor: Optional[int] = 4
    # """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    # downscale_rounding_mode: Literal["floor", "round", "ceil"] = "floor"
    # """How to round downscale image height and Image width."""
    # scene_scale: float = 1.0
    # """How much to scale the region of interest by."""
    # orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    # """The method to use for orientation."""
    # center_method: Literal["poses", "focus", "none"] = "poses"
    # """The method to use to center the poses."""
    # auto_scale_poses: bool = True
    # """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    # assume_colmap_world_coordinate_convention: bool = True
    # """Colmap optimized world often have y direction of the first camera pointing towards down direction,
    # while nerfstudio world set z direction to be up direction for viewer. Therefore, we usually need to apply an extra
    # transform when orientation_method=none. This parameter has no effects if orientation_method is set other than none.
    # When this parameter is set to False, no extra transform is applied when reading data from colmap.
    # """
    # eval_mode: Literal["fraction", "filename", "interval", "all"] = "interval"
    # """
    # The method to use for splitting the dataset into train and eval.
    # Fraction splits based on a percentage for train and the remaining for eval.
    # Filename splits based on filenames containing train/eval.
    # Interval uses every nth frame for eval (used by most academic papers, e.g. MipNerf360, GSplat).
    # All uses all the images for any split.
    # """
    # train_split_fraction: float = 0.9
    # """The fraction of images to use for training. The remaining images are for eval."""
    # eval_interval: int = 8
    # """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    # depth_unit_scale_factor: float = 1e-3
    # """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    #
    # images_path: Path = Path("images")
    # """Path to images directory relative to the data path."""
    # masks_path: Optional[Path] = Path("masks")
    # """Path to masks directory. If not set, masks are not loaded."""
    # depths_path: Optional[Path] = None
    # """Path to depth maps directory. If not set, depths are not loaded."""
    # colmap_path: Path = Path("sparse/0")
    # """Path to the colmap reconstruction directory relative to the data path."""
    # load_3D_points: bool = True
    # """Whether to load the 3D points from the colmap reconstruction. This is helpful for Gaussian splatting and
    # generally unused otherwise, but it's typically harmless so we default to True."""
    # max_2D_matches_per_3D_point: int = 0
    # """Maximum number of 2D matches per 3D point. If set to -1, all 2D matches are loaded. If set to 0, no 2D matches are loaded."""


class SpinDataParser(ColmapDataParser):
    """COLMAP DatasetParser.
    Expects a folder with the following structure:
        images/ # folder containing images used to create the COLMAP model
        sparse/0 # folder containing the COLMAP reconstruction (either TEXT or BINARY format)
        masks/ # (OPTIONAL) folder containing masks for each image
        depths/ # (OPTIONAL) folder containing depth maps for each image
    The paths can be different and can be specified in the config. (e.g., sparse/0 -> sparse)
    Currently, most COLMAP camera models are supported except for the FULL_OPENCV and THIN_PRISM_FISHEYE models.

    The dataparser loads the downscaled images from folders with `_{downscale_factor}` suffix.
    If these folders do not exist, the user can choose to automatically downscale the images and
    create these folders.

    The loader is compatible with the datasets processed using the ns-process-data script and
    can be used as a drop-in replacement. It further supports datasets like Mip-NeRF 360 (although
    in the case of Mip-NeRF 360 the downsampled images may have a different resolution because they
    use different rounding when computing the image resolution).
    """

    config: SpinDataParserConfig

    def __init__(self, config: SpinDataParserConfig):
        super().__init__(config)
        self.config = config
        self._downscale_factor = None
        self.spherify = False

    def _get_all_images_and_cameras(self, recon_dir: Path):
        if (recon_dir / "cameras.txt").exists():
            cam_id_to_camera = colmap_utils.read_cameras_text(recon_dir / "cameras.txt")
            im_id_to_image = colmap_utils.read_images_text(recon_dir / "images.txt")
        elif (recon_dir / "cameras.bin").exists():
            cam_id_to_camera = colmap_utils.read_cameras_binary(recon_dir / "cameras.bin")
            im_id_to_image = colmap_utils.read_images_binary(recon_dir / "images.bin")
        else:
            raise ValueError(f"Could not find cameras.txt or cameras.bin in {recon_dir}")

        cameras = {}
        frames = []
        camera_model = None

        # Parse cameras
        for cam_id, cam_data in cam_id_to_camera.items():
            cameras[cam_id] = parse_colmap_camera_params(cam_data)

        # Parse frames
        # we want to sort all images based on im_id
        ordered_im_id = sorted(im_id_to_image.keys())
        for im_id in ordered_im_id:
            im_data = im_id_to_image[im_id]
            # NB: COLMAP uses Eigen / scalar-first quaternions
            # * https://colmap.github.io/format.html
            # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
            # the `rotation_matrix()` handles that format for us.
            rotation = colmap_utils.qvec2rotmat(im_data.qvec)
            translation = im_data.tvec.reshape(3, 1)
            w2c = np.concatenate([rotation, translation], 1)
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
            c2w = np.linalg.inv(w2c)
            # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
            c2w[0:3, 1:3] *= -1
            if self.config.assume_colmap_world_coordinate_convention:
                # world coordinate transform: map colmap gravity guess (-y) to nerfstudio convention (+z)
                c2w = c2w[np.array([0, 2, 1, 3]), :]
                c2w[2, :] *= -1

            frame = {
                "file_path": (self.config.data / self.config.images_path / im_data.name).with_suffix(".png").as_posix(), # modified
                "transform_matrix": c2w,
                "colmap_im_id": im_id,
            }
            frame.update(cameras[im_data.camera_id])
            if self.config.masks_path is not None:
                frame["mask_path"] = (
                    (self.config.data / self.config.masks_path / im_data.name).with_suffix(".png").as_posix() # modified
                )
            if self.config.depths_path is not None:
                frame["depth_path"] = (
                    (self.config.data / self.config.depths_path / im_data.name).with_suffix(".png").as_posix()
                )
            frames.append(frame)
            if camera_model is not None:
                assert camera_model == frame["camera_model"], "Multiple camera models are not supported"
            else:
                camera_model = frame["camera_model"]

        out = {}
        out["frames"] = frames
        if self.config.assume_colmap_world_coordinate_convention:
            # world coordinate transform: map colmap gravity guess (-y) to nerfstudio convention (+z)
            applied_transform = np.eye(4)[:3, :]
            applied_transform = applied_transform[np.array([0, 2, 1]), :]
            applied_transform[2, :] *= -1
            out["applied_transform"] = applied_transform.tolist()
        out["camera_model"] = camera_model
        assert len(frames) > 0, "No images found in the colmap model"
        return out

    def _get_image_indices(self, image_filenames, split):
        # has_split_files_spec = (
        #     (self.config.data / "train_list.txt").exists()
        #     or (self.config.data / "test_list.txt").exists()
        #     or (self.config.data / "validation_list.txt").exists()
        # )
        # if (self.config.data / f"{split}_list.txt").exists():
        #     CONSOLE.log(f"Using {split}_list.txt to get indices for split {split}.")
        #     with (self.config.data / f"{split}_list.txt").open("r", encoding="utf8") as f:
        #         filenames = f.read().splitlines()
        #     # Validate split first
        #     split_filenames = set(self.config.data / self.config.images_path / x for x in filenames)
        #     unmatched_filenames = split_filenames.difference(image_filenames)
        #     if unmatched_filenames:
        #         raise RuntimeError(
        #             f"Some filenames for split {split} were not found: {set(map(str, unmatched_filenames))}."
        #         )
        #
        #     indices = [i for i, path in enumerate(image_filenames) if path in split_filenames]
        #     CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
        #     indices = np.array(indices, dtype=np.int32)
        # elif has_split_files_spec:
        #     raise RuntimeError(f"The dataset's list of filenames for split {split} is missing.")
        # else:
            # # find train and eval indices based on the eval_mode specified
            # if self.config.eval_mode == "fraction":
            #     i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
            # elif self.config.eval_mode == "filename":
            #     i_train, i_eval = get_train_eval_split_filename(image_filenames)
            # elif self.config.eval_mode == "interval":
            #     i_train, i_eval = get_train_eval_split_interval(image_filenames, self.config.eval_interval)
            # elif self.config.eval_mode == "all":
            #     CONSOLE.log(
            #         "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
            #     )
            #     i_train, i_eval = get_train_eval_split_all(image_filenames)
            # else:
            #     raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

        i_eval = np.arange(40)
        self.i_eval = i_eval
        i_train = np.arange(40, len(image_filenames))

        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")
        return indices

    def _get_prompt(self):
        dst = self.config.data.name
        return prompt_dict[dst]

