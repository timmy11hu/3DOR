import numpy as np
import torch
import time
import imageio
import cv2
from tqdm import tqdm
import torch.nn.functional as Fu
from pytorch3d.implicitron.tools.point_cloud_utils import get_rgbd_point_cloud
from pytorch3d.renderer import (AlphaCompositor, MultinomialRaysampler,
                                NDCMultinomialRaysampler,
                                PointsRasterizationSettings, PointsRasterizer,
                                ray_bundle_to_ray_points)
from pytorch3d.renderer.cameras import CamerasBase, get_ndc_to_screen_transform
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.utils import cameras_from_opencv_projection
from scipy.spatial import ConvexHull
from splatting import splatting_function

from functools import lru_cache
from skimage.draw import line, polygon, polygon2mask
from easydict import EasyDict as edict


def draw_line(s1,s2,e1,e2):
    # draw 2D line from start (s) to end (e)
    return line(s1,s2,e1,e2)


def adjust_scale_center(pts_world_fg, pts_world_fg_filter=None):
    try:
        # remove outliers
        if pts_world_fg_filter is None:
            from sklearn.neighbors import LocalOutlierFactor
            pts_world_fg_filter = LocalOutlierFactor(n_neighbors=20, contamination=0.3).fit_predict(pts_world_fg)
    except:
        breakpoint()
    pts_world_fg_filtered = pts_world_fg[pts_world_fg_filter == 1]
    
    # find bounds
    min_bound, max_bound = pts_world_fg_filtered.min(dim=0)[0], pts_world_fg_filtered.max(dim=0)[0]
    world_scale = (max_bound - min_bound).max()
    world_center = (max_bound + min_bound) / 2

    return world_scale, world_center, pts_world_fg_filter


def render_forward_splat_co3d(src_frame, tgt_frame, rgb_key="image_rgb", depth_key="depth_map", filter_depth=False, splat_depth=False, return_angles=False, filter_bg=False, epi_bg=False, max_rays=2000, black_bg=False, splat_params=False, adjust_mono=False):
    """
    src_frame: source frame containing rgb image, depth, and camera
    tgt_frame: target frame containing camera
    rgb_key: key for rgb image in src_frame
    depth_key: key for depth image in src_frame

    filter_depth: if True, replaces 0 depth values with max non-zero depth value
    splat_depth: if True, returns splatted depth
    return_angles: if True, returns ray angles between source and target rays
    filter_bg: if True, sets weights of background points to -inf during softmax (splatting)
    epi_bg: if True, draws epipolar rays from source to target image
    max_rays: maximum number of rays to draw for epi_bg
    black_bg: if True, sets background to black instead of white
    splat_params: if True, returns flow and weights used for splatting
    adjust_mono: if True, adjusts mono depth to fit in [-1, 1] (range used during rendering objaverse assets)

    """

    start_time = time.time()
    # unproject to world
    imh, imw = getattr(src_frame, rgb_key).shape[-2:]
    src_rgb, src_depth = getattr(src_frame, rgb_key), getattr(src_frame, depth_key)
    if filter_depth:
        src_depth[src_depth == 0] = src_depth.unique()[-1]  # replace 0 with max non-zero depth

    # generates raybundle using camera intrinsics and extrinsics
    src_ray_bundle = NDCMultinomialRaysampler(
        image_width=imw,
        image_height=imh,
        n_pts_per_ray=1,
        min_depth=1.0,
        max_depth=1.0,
    )(src_frame.camera)

    # get points in world space
    pts_world = ray_bundle_to_ray_points(
      src_ray_bundle._replace(lengths=src_depth[:, 0, ..., None])
    ).squeeze(-2)
    z = pts_world[0].cpu().numpy()


    rays_time = time.time()

    bg_mask = torch.logical_or(src_depth <= 0, src_depth >= 10)

    if adjust_mono:
        # ignore background points
        pts_world_fg = pts_world[~bg_mask.squeeze(1), :]

        # find adjust params
        world_scale, world_center, pts_filter = adjust_scale_center(pts_world_fg)

        # scale and translate points b/w [-1, 1]
        pts_world = 2.0 * (pts_world - world_center) / world_scale
        
    # new_bounds = pts_world[~bg_mask.squeeze(1), :].reshape(-1,3).min(dim=0)[0], pts_world[~bg_mask.squeeze(1), :].reshape(-1,3).max(dim=0)[0]
    # print(f"readjusted mono depth, new bounds: {new_bounds}")

    # get rays from source and target camera to world points
    src_rays = pts_world - src_frame.camera.get_camera_center()
    src_rays = src_rays / src_rays.norm(dim=-1, keepdim=True)
    target_rays = pts_world - tgt_frame.camera.get_camera_center()
    target_rays = target_rays / target_rays.norm(dim=-1, keepdim=True)

    # print min and max pts_world
    # print(f"min: {pts_world.reshape(-1,3).min(0).values.cpu().tolist()}, max: {pts_world.reshape(-1,3).max(0).values.cpu().tolist()}")

    # compute angle between rays
    angle = torch.acos((src_rays * target_rays).sum(dim=-1))

    # replace nans with 0
    angle[angle != angle] = 0
    
    # convert to degrees
    ray_angles = angle * 180 / np.pi
    
    # move to source screen space (pixel indices)
    src_pts_screen = src_frame.camera.transform_points_screen(pts_world.squeeze(), image_size=(imh, imw))

    # move to screen camera view space (debug)
    # src_pts_cam = src_frame.camera.get_world_to_view_transform().transform_points(pts_world.squeeze())

    # move to target screen space
    tgt_pts_screen = tgt_frame.camera.transform_points_screen(pts_world.squeeze(), image_size=(imh, imw))

    # move to target camera view space
    tgt_pts_cam = tgt_frame.camera.get_world_to_view_transform().transform_points(pts_world.squeeze())

    # depths in camera view space
    new_z = tgt_pts_cam[None, :,:,2:3].permute(0, 3, 1, 2)

    # flow of points in screen space
    flow = tgt_pts_screen[None, :,:, :2] - src_pts_screen[None, :,:, :2]
    flow = flow.permute(0, 3, 1, 2)
    flow_time = time.time()
    
    # calculate importance weights
    importance = 1. / (new_z)
    importance_min = importance.amin((1, 2, 3), keepdim=True)
    importance_max = importance.amax((1, 2, 3), keepdim=True)
    weights = (importance - importance_min) / (importance_max - importance_min + 1e-6) * 20 - 10

    # make weights of background points -inf
    if filter_bg:
        new_bg_mask = torch.logical_or(new_z <= -10, new_z >= 10)
        weights[new_bg_mask] = -1e10

    # run splatting
    num_channels = src_rgb.shape[1]
    src_mask_ = torch.ones_like(new_z)
    
    input_data = torch.cat([src_rgb, (1. / (new_z)), src_mask_, ray_angles.unsqueeze(1), src_depth], dim=1)
    
    if "diffuse_image" in src_frame:
        input_data = torch.cat([input_data, src_frame.diffuse_image.unsqueeze(0)], dim=1)
    
    if "normal_image" in src_frame:
        input_data = torch.cat([input_data, src_frame.normal_image.unsqueeze(0)], dim=1)
    
    if "gloss_image" in src_frame:
        input_data = torch.cat([input_data, src_frame.gloss_image.unsqueeze(0)], dim=1)
    
    output_data = splatting_function('softmax', input_data, flow.float(), weights.detach())
    splat_time = time.time()

    # process splat output
    warp_feature = output_data[:, 0:num_channels, Ellipsis]
    warp_disp = output_data[:, num_channels:num_channels + 1, Ellipsis]
    warp_mask = output_data[:, num_channels + 1:num_channels + 2, Ellipsis]
    warp_ray_angles = output_data[:, num_channels + 2:num_channels + 3, Ellipsis]
    warp_depth = output_data[:, num_channels + 3:num_channels + 4, Ellipsis]

    warp_diffuse, warp_normal, warp_gloss = None, None, None
        
    if "diffuse_image" in src_frame:
        warp_diffuse = output_data[:, num_channels + 4:num_channels + 7, Ellipsis]
    
    if "normal_image" in src_frame:
        warp_normal = output_data[:, num_channels + 7:num_channels + 10, Ellipsis]

    if "gloss_image" in src_frame:
        warp_gloss = output_data[:, num_channels + 10:num_channels + 13, Ellipsis]

    # print(f"Time taken, rays: {rays_time - start_time:.3f}, flow: {flow_time - rays_time:.3f}, splat: {splat_time - flow_time:.3f}")

    if epi_bg and not torch.eq(src_frame.camera.T, tgt_frame.camera.T).all():
        epi_time = time.time()
        nudge = 0.01
        
        pts_world_nudged = ray_bundle_to_ray_points(src_ray_bundle._replace(lengths=src_depth[:, 0, ..., None] + nudge)).squeeze(-2)

        if adjust_mono:
            # ignore background points
            # pts_world_nudged_fg = pts_world_nudged[~bg_mask.squeeze(1), :]

            # # find adjust params using same filter as mono depth
            # world_scale, world_center, _ = adjust_scale_center(pts_world_nudged_fg, pts_filter)

            # scale and translate points b/w [-1, 1]
            pts_world_nudged = 2.0 * (pts_world_nudged - world_center) / world_scale

        tgt_pts_screen_nudged = tgt_frame.camera.transform_points_screen(pts_world_nudged.squeeze(), image_size=(imh, imw))
        
        tgt_pts_screen_flow = tgt_pts_screen_nudged[:,:, :2] - tgt_pts_screen[:,:, :2]

        # normalize flow
        tgt_pts_screen_flow = tgt_pts_screen_flow / tgt_pts_screen_flow.norm(dim=-1, keepdim=True)
        
        # get slope and intercept of rays
        slope = tgt_pts_screen_flow[:,:,0:1] / tgt_pts_screen_flow[:,:,1:2]
        intercept = tgt_pts_screen_nudged[:,:, 0:1] - slope * tgt_pts_screen_nudged[:,:, 1:2]
        
        # find intersection of rays with tgt screen (x = 0, x = imw, y = 0, y = imh)
        left = slope * 0 + intercept
        left = torch.cat([left, torch.zeros_like(left)], dim=-1)

        right = slope * imw + intercept
        right = torch.cat([right, torch.ones_like(right) * imw - 1], dim=-1)

        top = (0 - intercept) / slope
        top = torch.cat([torch.zeros_like(top), top], dim=-1)

        bottom = (imh - intercept) / slope
        bottom = torch.cat([torch.ones_like(bottom) * imh - 1, bottom], dim=-1)

        # based on sign of slope, find intersection with left/right or top/bottom
        lr_intersect = torch.where(tgt_pts_screen_flow[:,:,1:2] < 0, left, right)
        tb_intersect = torch.where(tgt_pts_screen_flow[:,:,0:1] < 0, top, bottom)

        # based on distance to intersection, find which one is closer from tgt_pts_screen
        dist_lr_intersect = (tgt_pts_screen[:,:,:2] - lr_intersect).norm(dim=-1, keepdim=True)
        dist_tb_intersect = (tgt_pts_screen[:,:,:2] - tb_intersect).norm(dim=-1, keepdim=True)
        ray_intersect = torch.where(dist_lr_intersect < dist_tb_intersect, lr_intersect, tb_intersect)

        # draw rays where interesections arent nan, and starts are in image bounds
        nan_indices = (
            (~torch.isnan(ray_intersect.mean(dim=-1)))
            .logical_and(tgt_pts_screen[...,0] < imh)
            .logical_and(tgt_pts_screen[...,0] > 0)
            .logical_and(tgt_pts_screen[:,:,1] < imw)
            .logical_and(tgt_pts_screen[:,:,1] > 0)
        )
            
        ray_start = tgt_pts_screen[nan_indices][:,:2].long()
        ray_end = ray_intersect[nan_indices][:,:2].long()
        
        # assert bounds
        ray_end[:, 0][ray_end[:,0] >= imh] = imh - 1
        ray_end[:, 1][ray_end[:,1] >= imw] = imw - 1

        # sample evenly placed rays
        if ray_start.shape[0] > max_rays:
            ray_start = ray_start[::round(ray_start.shape[0] / max_rays)]
            ray_end = ray_end[::round(ray_end.shape[0] / max_rays)]

        if len(ray_start) > 0:
            try:
                # reshape for editing
                warp_feature = warp_feature.squeeze().permute(1, 2, 0)

                # set all background to white
                if not black_bg:
                    warp_feature = torch.where(warp_feature.mean(dim=-1, keepdim=True) == 0, torch.tensor([255, 255, 255], dtype=warp_feature.dtype), warp_feature)
                else:
                    warp_feature = torch.where(warp_feature.mean(dim=-1, keepdim=True) == 0, torch.tensor([0, 0, 0], dtype=warp_feature.dtype), warp_feature)

                # draw lines
                rrs, ccs = [], []
                for rs, re in tqdm(zip(ray_start, ray_end), total=len(ray_start), desc="Drawing rays", disable=True):
                    rr, cc = draw_line(int(rs[1]), int(rs[0]), int(re[1]), int(re[0]))
                    rrs.append(rr)
                    ccs.append(cc)
                
                # find unique pixels
                rrs, ccs = np.concatenate(rrs), np.concatenate(ccs)

                # apply mask
                if not black_bg:
                    warp_feature[rrs, ccs] = torch.where(warp_feature[rrs, ccs].mean(dim=-1, keepdim=True) == 255, torch.tensor([0, 0, 0], dtype=warp_feature.dtype), warp_feature[rrs, ccs])
                else:
                    warp_feature[rrs, ccs] = torch.where(warp_feature[rrs, ccs].mean(dim=-1, keepdim=True) == 0, torch.tensor([-1, -1, -1], dtype=warp_feature.dtype), warp_feature[rrs, ccs])

                warp_feature = warp_feature.permute(2, 0, 1).unsqueeze(0)
            except Exception as e:
                pass
                print(f"Exception: {e}, src_viewpoint: {src_frame.viewpoint.T}, tgt_viewpoint: {tgt_frame.viewpoint.T}, diff: {src_frame.viewpoint.T - tgt_frame.viewpoint.T}")
                breakpoint()

        # print(f"epi taken: {time.time() - epi_time:.3f}, no of rays: {ray_start.shape[0]}")
    else:
        pass
        # print(f"splat rays skipped, viewpoints same or epi_bg: {epi_bg}.")
        # breakpoint()


    # print(f"time taken: {time.time() - start_time:.3f}")
    # registry.prev_count = registry.prev_count if hasattr(registry, "prev_count") else 1
    # if (registry.get("cache_count", -1) // 10000) %  registry.prev_count == 0:
    #     registry.prev_count = registry.prev_count + 1 if hasattr(registry, "prev_count") else 1
    # print(draw_line.cache_info())

    output = [warp_feature, warp_disp, warp_mask]

    if return_angles:
        output.append(warp_ray_angles)
    
    if splat_depth:
        output.append(warp_depth)
    
    if splat_params:
        output.extend([flow, weights])

    if warp_diffuse is not None:
        output.append(warp_diffuse)
    
    if warp_normal is not None:
        output.append(warp_normal)
    
    if warp_gloss is not None:
        output.append(warp_gloss)
    
    return output


def main():
    # source viewpoint (sR is rotation, sT is translation, sP is intrinsic)
    # imh and imw are image height and width
    src_viewpoint = cameras_from_opencv_projection(sR, sT, sP, torch.tensor([imh, imw]).float().unsqueeze(0))

    # target viewpoint (tR is rotation, tT is translation, tP is intrinsic)
    # imh and imw are image height and width
    tgt_viewpoint = cameras_from_opencv_projection(tR, tT, tP, torch.tensor([imh, imw]).float().unsqueeze(0))

    # read RGB image and depth image in CHW format
    image = torch.tensor(imageio.imread("image.png")).float() / 255.0
    depth = torch.tensor(imageio.imread("depth.png")).float()

    # unsqueeze batch dim
    image = image.unsqueeze(0); depth = depth.unsqueeze(0)

    # set background depth to -1.0, so that it is not splatted
    bg_depth_mask = None
    depth[bg_depth_mask] = -1.0

    # create source frame which contains rgb image, mono depth, and camera
    src_frame = edict({
        "camera": src_viewpoint,
        "image_rgb": image,
        "depth_map": depth,
    })

    # create target frame which contains only camera
    tgt_frame = edict({
        "camera": tcamera,
    })

    # run softmax-splatting (reproject source image to target image)
    splat_outputs = render_forward_splat_co3d(src_frame, tgt_frame, filter_depth=False, return_angles=False, filter_bg=True, epi_bg=False, max_rays=-1, black_bg=False, splat_depth=False, splat_params=False, adjust_mono=False)


if __name__ == "__main__":
    main()

"""
NOTE: this code was a part of larger codebase, and the main function is just a placeholder to show how to use the splatting function. it is not runnable as is and may require some minor modifications.


Installation:

# ensure pytorch and other dependencies are installed

# install softmax-splatting
cd ~ && git clone https://github.com/hperrot/splatting
cd splatting && pip install -e .

# install pytorch3d
pip uninstall -y pytorch3d
apt install clang
MACOSX_DEPLOYMENT_TARGET=10.14 CC=clang CXX=clang++ pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# easydict
pip install easydict



Running: 

- ideally, test on a renders of a synthetic scene (e.g. blender) with known camera poses and ground truth depth. 
- your splatted / reprojected image should blend perfectly with the target image; if not, consider debugging the following:
    - camera conversion from opencv to pytorch3d (most likely culprit)
    - camera intrinsics / extrinsics
    - print bounds of unprojected points in world space

"""
