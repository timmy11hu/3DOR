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
import sys, os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Literal, Optional, Type

import numpy as np
import torch
from PIL import Image
from rich.prompt import Confirm

from .camera_utils import transform_poses_pca, generate_ellipse_path, generate_spiral_path
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
    "vasedeck": "wooden deck, photorealistic 8k",
    "pinecone": "stone ground, photorealistic 8k",
    "kitchen": "a wooden floor partially covered by two rugs, photorealistic 8k",
    "garden": "a wooden table, photorealistic 8k",
    "paper": "A brick wall, photorealistic 8k",
    "stander": "A brick wall, photorealistic 8k",
    "window": "A brick wall, photorealistic 8k",
    "trash1": "A brick floor, photorealistic 8k",
    "trash2": "A brick wall, photorealistic 8k",
    "trash3": "A wooden bench arm, photorealistic 8k",
    "trash4": "A brick wall and wooden window, photorealistic 8k",
    "trash5": "A brick wall, photorealistic 8k",
    "trash6": "A corner of a brick wall with wooden window frame, photorealistic 8k",
    "sofa": "A grey sofa, photorealistic 8k",
    "chair": "A grey plane sofa chair, photorealistic 8k",
    "plant": "a white pot centered on a patterned tablecloth, photorealistic 8k",
    "cup":  "A golden cup, photorealistic 8k",
    "toy": "Bottom of a white clay toy, photorealistic 8k"

}

train_test_split = {"paper": 75,
                    "trash1": 20,
                    "trash2": 28,
                    "trash3": 35,
                    "trash4": 33,
                    "trash5": 25,
                    "trash6": 36,
                    "sofa": 44}

@dataclass
class LLFFDataParserConfig(ColmapDataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: LLFFDataParser)
    downscale_factor: Optional[int] = 4
    images_path: Path = Path("images")
    masks_path: Optional[Path] = Path("masks")
    colmap_path: Path = Path("sparse/0")
    load_3D_points: bool = False
    assume_colmap_world_coordinate_convention: bool = False
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"

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


class LLFFDataParser(ColmapDataParser):
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

    config: LLFFDataParserConfig

    def __init__(self, config: LLFFDataParserConfig):
        super().__init__(config)
        self.config = config
        self._downscale_factor = None
        if self.config.data.name in ('vasedeck', 'pinecone', "kitchen", "garden"):
            self.spherify = True
        else:
            self.spherify = False

        if self.config.data.name in ('chair', "sofa"):
            self.config.downscale_factor = 1
        # if self.config.data.name in ('vasedeck', 'pinecone'):
        #     self.config.orientation_method = "up"

    def _get_all_images_and_cameras(self, recon_dir: Path):
        if (recon_dir / "cameras.txt").exists():
            cam_id_to_camera = colmap_utils.read_cameras_text(recon_dir / "cameras.txt")
            im_id_to_image = colmap_utils.read_images_text(recon_dir / "images.txt")
        elif (recon_dir / "cameras.bin").exists():
            cam_id_to_camera = colmap_utils.read_cameras_binary(recon_dir / "cameras.bin")
            im_id_to_image = colmap_utils.read_images_binary(recon_dir / "images.bin")
        else:
            raise ValueError(f"Could not find cameras.txt or cameras.bin in {recon_dir}")

        # Load bounds if possible (only used in forward facing scenes).
        posefile = recon_dir.parent.parent / 'poses_bounds.npy'
        poses_arr = np.load(posefile)
        self.bounds = poses_arr[:, -2:]

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
                "file_path": (self.config.data / self.config.images_path / im_data.name).as_posix(),
                # "file_path": (self.config.data / self.config.images_path / im_data.name).with_suffix(".png").as_posix(), # modified
                "transform_matrix": c2w,
                "colmap_im_id": im_id,
            }
            frame.update(cameras[im_data.camera_id])
            if self.config.masks_path is not None:
                frame["mask_path"] = (
                    (self.config.data / self.config.masks_path / im_data.name).with_suffix(".png").as_posix()
                # modified
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

        if self.config.data.name in train_test_split.keys():
            i_train = np.arange(train_test_split[self.config.data.name])
            # i_train = np.arange(len(image_filenames))
            i_eval = np.arange(train_test_split[self.config.data.name], len(image_filenames))
            self.i_eval = i_eval
        else:
            i_train = np.arange(len(image_filenames))
            # i_eval = [int(len(image_filenames) / 10 * i) for i in range(10)]  # evenly sample 10
            i_eval = i_train
            self.i_eval = i_eval

        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")
        return indices

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        colmap_path = self.config.data / self.config.colmap_path
        assert colmap_path.exists(), f"Colmap path {colmap_path} does not exist."

        meta = self._get_all_images_and_cameras(colmap_path)
        camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in meta["frames"]:
            fx.append(float(frame["fl_x"]))
            fy.append(float(frame["fl_y"]))
            cx.append(float(frame["cx"]))
            cy.append(float(frame["cy"]))
            height.append(int(frame["h"]))
            width.append(int(frame["w"]))
            distort.append(
                camera_utils.get_distortion_params(
                    k1=float(frame["k1"]) if "k1" in frame else 0.0,
                    k2=float(frame["k2"]) if "k2" in frame else 0.0,
                    k3=float(frame["k3"]) if "k3" in frame else 0.0,
                    k4=float(frame["k4"]) if "k4" in frame else 0.0,
                    p1=float(frame["p1"]) if "p1" in frame else 0.0,
                    p2=float(frame["p2"]) if "p2" in frame else 0.0,
                )
            )

            image_filenames.append(Path(frame["file_path"]))
            poses.append(frame["transform_matrix"])
            if "mask_path" in frame:
                mask_filenames.append(Path(frame["mask_path"]))
            if "depth_path" in frame:
                depth_filenames.append(Path(frame["depth_path"]))

        assert len(mask_filenames) == 0 or (len(mask_filenames) == len(image_filenames)), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (len(depth_filenames) == len(image_filenames)), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # poses, transform = transform_poses_pca(poses)

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        poses[:, :3, 3] *= scale_factor
        self.bounds *= scale_factor

        if self.spherify:
            if self.config.data.name in ('vasedeck', 'pinecone', 'garden'):
                # from .nerf_helper import normalize, viewmatrix
                # rad = 1.2
                # centroid = np.mean(poses.cpu().numpy()[:, :3, 3], 0)
                # # bbox_min, bbox_max = pipeline.object_aabb
                # # centroid = (bbox_min + bbox_max) / 2.
                # # centroid[2] = 0
                # zh = centroid[2]  # ==0.
                # radcircle = np.sqrt(rad ** 2 - zh ** 2)
                # new_poses = []
                # for th in np.linspace(0., 2*np.pi, 120):
                #     camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh]) + centroid
                #     # lookat = camorigin - np.array([0., 0., -1.])
                #     lookat = camorigin - np.array([centroid[0], centroid[1], -1.])
                #     up = np.array([0, 0., -1.])
                #     vec2 = normalize(lookat)
                #     vec0 = normalize(np.cross(vec2, up))
                #     vec1 = normalize(np.cross(vec2, vec0))
                #     pos = camorigin
                #     p = np.stack([vec0, vec1, vec2, pos], 1)
                #     new_poses.append(p)
                # self.render_poses = np.stack(new_poses, 0)
                poses, self.render_poses, _ = spherify_poses(poses.cpu().numpy(), self.bounds)
                poses = torch.from_numpy(np.array(poses).astype(np.float32))
                poses = poses[:, :3, :4]
                self.render_poses = self.render_poses[:, :3, :4]
            else:
                self.render_poses = generate_ellipse_path(
                    poses.cpu().numpy(),
                    n_frames=120,
                    z_variation=0.,
                    z_phase=0.)
        else:
            # https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/main/internal/datasets.py
            self.render_poses = generate_spiral_path(
                poses, self.bounds, n_frames=120)

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        indices = self._get_image_indices(image_filenames, split)
        image_filenames, mask_filenames, depth_filenames, downscale_factor = self._setup_downscale_factor(
            image_filenames, mask_filenames, depth_filenames
        )

        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        fx = torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = torch.tensor(width, dtype=torch.int32)[idx_tensor]
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        cameras.rescale_output_resolution(
            scaling_factor=1.0 / downscale_factor, scale_rounding_mode=self.config.downscale_rounding_mode
        )

        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
            transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        metadata = {}
        if self.config.load_3D_points:
            # Load 3D points
            metadata.update(self._load_3D_points(colmap_path, transform_matrix, scale_factor))

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                **metadata,
            },
        )
        return dataparser_outputs

    def _get_prompt(self):
        dst = self.config.data.name
        return prompt_dict[dst]


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])  # 1, 4
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)  # 4,4
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])  # N,1,4
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)  # N,4,4

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def spherify_poses(poses, bds):
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []

    for th in np.linspace(0., 2. * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, new_poses, bds