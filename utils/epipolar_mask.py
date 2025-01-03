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
from scipy.spatial import ConvexHull

from functools import lru_cache
from skimage.draw import line, polygon, polygon2mask
from easydict import EasyDict as edict


def draw_line(s1,s2,e1,e2):
    # draw 2D line from start (s) to end (e)
    return line(s1,s2,e1,e2)


def compute_epipolar_mask(src_frame, tgt_frame, imh, imw, use_depth=True, debug_video=False, merge_mask=True):
    """
    src_frame: source frame containing camera
    tgt_frame: target frame containing camera
    debug_depth: if True, uses depth map to compute epipolar lines on target image
    debug_video: if True, saves a video showing the epipolar lines for each source pixel

    """

    start_time = time.time()
    # generates raybundle using camera intrinsics and extrinsics
    src_ray_bundle = NDCMultinomialRaysampler(
        image_width=imw,
        image_height=imh,
        n_pts_per_ray=1,
        min_depth=1.0,
        max_depth=1.0,
    )(src_frame.camera)
    
    if use_depth:
        src_depth = getattr(src_frame, "depth_map")
        src_depth = src_depth[:, 0, ..., None]
    else:
        # get points in world space (at fixed depth)
        src_depth = 3.5 * torch.ones((1, imh, imw, 1), dtype=torch.float32, device=src_frame.camera.device)

    pts_world = ray_bundle_to_ray_points(
      src_ray_bundle._replace(lengths=src_depth)
    ).squeeze(-2)
    # print(f"world points bounds: {pts_world.reshape(-1,3).min(dim=0)[0]} to {pts_world.reshape(-1,3).max(dim=0)[0]}")
    rays_time = time.time()

    # move source points to target screen space
    tgt_pts_screen = tgt_frame.camera.transform_points_screen(pts_world.squeeze(), image_size=(imh, imw))

    print("src camera world pos: ", src_frame.camera.get_camera_center())
    print("src center pixel world pos: ", pts_world[0,128,128])

    # move source camera center to target screen space
    src_center_tgt_screen = tgt_frame.camera.transform_points_screen(src_frame.camera.get_camera_center(), image_size=(imh, imw)).squeeze()

    # build epipolar mask (draw lines from source camera center to source points in target screen space)
    # start: source camera center, end: source points in target screen space

    # get flow of points 
    camera_to_pts_flow = tgt_pts_screen[...,:2] - src_center_tgt_screen[...,:2]

    # normalize flow
    camera_to_pts_flow = camera_to_pts_flow / camera_to_pts_flow.norm(dim=-1, keepdim=True)

    # get slope and intercept of lines
    slope = camera_to_pts_flow[:,:,0:1] / camera_to_pts_flow[:,:,1:2]
    intercept = tgt_pts_screen[:,:, 0:1] - slope * tgt_pts_screen[:,:, 1:2]

    # find intersection of lines with tgt screen (x = 0, x = imw, y = 0, y = imh)
    left = slope * 0 + intercept
    left_sane = (left <= imh) & (0 <= left)
    left = torch.cat([left, torch.zeros_like(left)], dim=-1)

    right = slope * imw + intercept
    right_sane = (right <= imh) & (0 <= right)
    right = torch.cat([right, torch.ones_like(right) * imw], dim=-1)

    top = (0 - intercept) / slope
    top_sane = (top <= imw) & (0 <= top)
    top = torch.cat([torch.zeros_like(top), top], dim=-1)

    bottom = (imh - intercept) / slope
    bottom_sane = (bottom <= imw) & (0 <= bottom)
    bottom = torch.cat([torch.ones_like(bottom) * imh, bottom], dim=-1)

    # find intersection of lines
    points_one = torch.zeros_like(left)
    points_two = torch.zeros_like(left)

    # collect points from [left, right, bottom, top] in sequence
    points_one = torch.where(left_sane.repeat(1,1,2), left, points_one)

    points_one_zero = (points_one.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_one = torch.where(right_sane.repeat(1,1,2) & points_one_zero, right, points_one)

    points_one_zero = (points_one.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_one = torch.where(bottom_sane.repeat(1,1,2) & points_one_zero, bottom, points_one)

    points_one_zero = (points_one.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_one = torch.where(top_sane.repeat(1,1,2) & points_one_zero, top, points_one)

    # collect points from [top, bottom, right, left] in sequence (opposite)
    points_two = torch.where(top_sane.repeat(1,1,2), top, points_two)

    points_two_zero = (points_two.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_two = torch.where(bottom_sane.repeat(1,1,2) & points_two_zero, bottom, points_two)

    points_two_zero = (points_two.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_two = torch.where(right_sane.repeat(1,1,2) & points_two_zero, right, points_two)

    points_two_zero = (points_two.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_two = torch.where(left_sane.repeat(1,1,2) & points_two_zero, left, points_two)

    # if source point lies inside target screen (find only one intersection)
    if (imh >= src_center_tgt_screen[0] >= 0) and (imw >= src_center_tgt_screen[1] >= 0):
        points_one_flow = points_one - src_center_tgt_screen[:2]
        points_one_flow_direction = (points_one_flow > 0)

        points_two_flow = points_two - src_center_tgt_screen[:2]
        points_two_flow_direction = (points_two_flow > 0)

        orig_flow_direction = (center_to_pts_flow > 0)

        # if flow direction is same as orig flow direction, pick points_one, else points_two
        points_one_alinged = (points_one_flow_direction == orig_flow_direction).all(dim=-1).unsqueeze(-1).repeat(1,1,2)
        points_one = torch.where(points_one_alinged, points_one, points_two)

        # points two is source camera center
        points_two = points_two * 0 + src_center_tgt_screen[:2]
    
    # if debug terminate with depth
    if use_depth:
        # remove points that are out of bounds (in target screen space)
        tgt_pts_screen_mask = (tgt_pts_screen[...,:2] < 0) | (tgt_pts_screen[...,:2] > imh)
        tgt_pts_screen_mask = ~tgt_pts_screen_mask.any(dim=-1, keepdim=True)

        depth_dist = torch.norm(src_center_tgt_screen[:2] - tgt_pts_screen[...,:2], dim=-1, keepdim=True)
        points_one_dist = torch.norm(src_center_tgt_screen[:2] - points_one, dim=-1, keepdim=True)
        points_two_dist = torch.norm(src_center_tgt_screen[:2] - points_two, dim=-1, keepdim=True)

        # replace where reprojected point is closer to source camera on target screen
        points_one = torch.where((depth_dist < points_one_dist) & tgt_pts_screen_mask, tgt_pts_screen[...,:2], points_one)
        points_two = torch.where((depth_dist < points_two_dist) & tgt_pts_screen_mask, tgt_pts_screen[...,:2], points_two)

    # build epipolar mask
    attention_mask = torch.zeros((imh * imw, imh, imw), dtype=torch.bool, device=src_frame.camera.device)

    # quantize points to pixel indices
    points_one = (points_one - 0.5).reshape(-1,2).long().cpu().numpy()
    points_two = (points_two - 0.5).reshape(-1,2).long().cpu().numpy()
    
    if not (imh == 32 and imw == 32):
        # iterate over points_one and points_two together and draw lines
        for idx, (p1, p2) in enumerate(zip(points_one, points_two)):
            # skip out of bounds points
            if p1.sum() == 0 and p2.sum() == 0:
                continue
            
            # draw lines from all neighbors of p1 to neighbors of p2 (mask dilation)
            rrs, ccs = [], []
            for dx, dy in [(0,0), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]:  # 8 neighbors
                _p1 = [min(max(p1[0] + dy, 0), imh - 1), min(max(p1[1] + dx, 0), imw - 1)]
                _p2 = [min(max(p2[0] + dy, 0), imh - 1), min(max(p2[1] + dx, 0), imw - 1)]
                rr, cc = line(int(_p1[1]), int(_p1[0]), int(_p2[1]), int(_p2[0]))
                rrs.append(rr); ccs.append(cc)
            rrs, ccs = np.concatenate(rrs), np.concatenate(ccs)
            attention_mask[idx, rrs.astype(np.int32), ccs.astype(np.int32)] = True

    else:
        points_one_y, points_one_x = points_one[:,0], points_one[:,1]
        points_two_y, points_two_x = points_two[:,0], points_two[:,1]
        attention_mask = registry.masks32[points_one_y, points_one_x, points_two_y, points_two_x]
        attention_mask = torch.from_numpy(attention_mask).to(src_frame.camera.device)

    # reshape to (imh, imw, imh, imw)
    attention_mask = attention_mask.reshape(imh, imw, imh, imw)

    # create a video visualizing location of source pixel and correponding epipolar line in target image
    if debug_video:
        am_video = []
        attention_mask = attention_mask.reshape(imh, imw, imh, imw)
        for i in range(0, imh):
            for j in  range(0, imw):
                am_img = (attention_mask[i,j].squeeze().unsqueeze(-1).repeat(1,1,3).float().cpu().numpy() * 255).astype(np.uint8)
                am_video.append(am_img)
        imageio.mimsave("am.gif", am_video)

    if merge_mask:
        valid_depth = (src_depth.squeeze() > 0).cpu().numpy()
        union_mask = torch.zeros((imh, imw), dtype=torch.bool, device=attention_mask.device)
        for x in range(imh):
            for y in range(imw):
                if valid_depth[x, y]:
                    # Update the union mask with a logical OR operation
                    union_mask = union_mask | attention_mask[x, y]
        return union_mask

    return attention_mask

def main():
    # source viewpoint (sR is rotation, sT is translation, sP is intrinsic)
    # imh and imw are image height and width
    src_viewpoint = cameras_from_opencv_projection(sR, sT, sP, torch.tensor([imh, imw]).float().unsqueeze(0))

    # target viewpoint (tR is rotation, tT is translation, tP is intrinsic)
    # imh and imw are image height and width
    tgt_viewpoint = cameras_from_opencv_projection(tR, tT, tP, torch.tensor([imh, imw]).float().unsqueeze(0))

    # create source frame which contains rgb image, mono depth, and camera
    src_frame = edict({
        "camera": src_viewpoint,
    })

    # create target frame which contains only camera
    tgt_frame = edict({
        "camera": tcamera,
    })

    # create epipolar mask
    splat_outputs = compute_epipolar_mask(src_frame, tgt_frame, imh, imw)


if __name__ == "__main__":
    main()

"""
NOTE: this code was a part of larger codebase, and the main function is just a placeholder to show how to use the epipolar masking function. it is not runnable as is and may require some minor modifications.


Installation:

# ensure pytorch and other dependencies are installed

# install pytorch3d
pip uninstall -y pytorch3d
apt install clang
MACOSX_DEPLOYMENT_TARGET=10.14 CC=clang CXX=clang++ pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# easydict
pip install easydict

Running: 

- ideally, test on a renders of a synthetic scene (e.g. blender) with known camera poses and ground truth depth. 
- your debug videos and depth-based mask should look reasonable, if not try debugging these:
    - camera conversion from opencv/blender to pytorch3d (most likely culprit)
    - camera intrinsics / extrinsics
    - print bounds of unprojected points in world space

"""
