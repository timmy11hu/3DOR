import os, sys, shutil
import copy
import cv2
import numpy as np
import torch
import scipy
import imageio
import imagehash
from PIL import Image
from .sd_utils import sample_noise_latent, get_latent_timestep
from .masactrl import MutualSelfAttentionControl, AttentionBase, regiter_attention_editor_diffusers


to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)



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
def find_central_cams(c2ws):
    pose = poses_avg(c2ws)
    central_location = pose[:3, 3]
    positions = np.array([c2w[:3, 3] for c2w in c2ws])
    distances = np.linalg.norm(positions - central_location, axis=1)
    sorted_indices = np.argsort(distances)  # Sort indices by distance
    centralest_indices = sorted_indices
    # centralest_index = np.argmin(distances)
    return centralest_indices


def find_highest_z_cam(c2ws):
    # Extract the z-values of the camera positions
    z_values = np.array([c2w[2, 3] for c2w in c2ws])
    # Sort indices by z-values in descending order
    sorted_indices = np.argsort(z_values)[::-1]
    return sorted_indices

@torch.no_grad()
def encode_latent_from_np(image, pipe, crops_coords=None, resize_mode="default", generator=None):
    img = Image.fromarray(to8b(image))
    width, height = img.size  # Get dimensions
    new_w, new_h = 8 * (width // 8), 8 * (height // 8)
    img = img.resize((new_w, new_h))
    image = pipe.image_processor.preprocess(img, height=new_h, width=new_w,
                                            crops_coords=crops_coords, resize_mode=resize_mode)
    image = image.clone().to(pipe.device).to(pipe.dtype)
    latent = pipe._encode_vae_image(image, generator=generator)
    return latent


@torch.no_grad()
def inpaint_image(pipe, image, mask, latent_noise, prompt, savedir, fn,
                  guidance_scale=7.5, strength=1.0, num_inference_steps=20, soft_mask=False):
    assert len(image.shape) == 3
    assert len(mask.shape) == 2 or len(mask.shape) == 3
    if len(mask.shape) == 3:
        mask = np.squeeze(mask)

    img = Image.fromarray(to8b(image))
    msk = Image.fromarray(to8b(mask))
    raw, raw_cp = copy.deepcopy(img), copy.deepcopy(img)
    width, height = img.size  # Get dimensions
    new_w, new_h = 8 * (width // 8), 8 * (height // 8)
    img = img.resize((new_w, new_h))
    if soft_mask:
        msk = msk.resize((new_w, new_h))
    else:
        msk = msk.resize((new_w, new_h), resample=Image.NEAREST)

    msk.save('demo.png')
    inpainted = pipe(prompt=prompt, image=img, mask_image=msk,
                     height=new_h, width=new_w,
                     latents=latent_noise,
                     guidance_scale=guidance_scale, strength=strength, num_inference_steps=num_inference_steps).images[0]
    # img.paste(inpainted, (0, 0), mask=msk)
    # raw.paste(img, (left, top))
    # inpainted = raw
    # img.paste(inpainted, (0, 0), mask=msk)
    # inpainted = img.resize((width, height))
    inpainted = inpainted.resize((width, height))
    image = Image.new("RGB", (width, 2 * height), "white")
    image.paste(raw_cp, (0, 0))
    image.paste(inpainted, (0, height))
    image.save(os.path.join(savedir, fn+'.png'))
    return np.array(inpainted) / 255.

@torch.no_grad()
def mutual_inpaint_image(image_ref, mask_ref, latent_ref,
                         image_tgt, mask_tgt, latent_tgt,
                         pipe, prompt, savedir, fn, seed,
                         guidance_scale=7.5, strength=1.0,
                         num_inference_steps=20, soft_mask=False):

    image_raw = copy.deepcopy(image_tgt)
    for i, (img, msk, zt) in enumerate([(image_ref, mask_ref, latent_ref), (image_tgt, mask_tgt, latent_tgt)]):
        assert len(img.shape) == 3 and len(zt.shape) == 4 # zt need batch channel
        assert len(msk.shape) == 3 or len(msk.shape) == 2
        if len(msk.shape) == 3:
            msk = np.squeeze(msk)

        img_np = np.copy(img)
        img = Image.fromarray(to8b(img))
        msk = Image.fromarray(to8b(msk))
        image_raw = copy.deepcopy(img) # copy the raw tgt image
        width, height = img.size  # Get dimensions
        new_w, new_h = 8 * (width // 8), 8 * (height // 8)
        img = img.resize((new_w, new_h))
        if soft_mask:
            msk = msk.resize((new_w, new_h))
        else:
            msk = msk.resize((new_w, new_h), resample=Image.NEAREST)

        if strength < 1.0:
            # raise NotImplementedError
            if i == 0:
                # inside: latents = self.scheduler.add_noise(image_latents, noise, timestep)
                noise, latent = sample_noise_latent(img_np, pipe, seed, new_h, new_w, strength=strength,
                                                    num_inference_steps=num_inference_steps)
            elif i == 1:
                latent_image = encode_latent_from_np(img_np, pipe)
                timestep = get_latent_timestep(pipe, strength=strength, num_inference_steps=num_inference_steps)
                latent = pipe.scheduler.add_noise(latent_image, zt.to(pipe.device), timestep)
            else:
                raise NotImplementedError
            zt = latent / pipe.scheduler.init_noise_sigma # for bugs

        if i == 0:
            image_ref, mask_ref, latent_ref = img, msk, zt
        elif i == 1:
            image_tgt, mask_tgt, latent_tgt = img, msk, zt

    images = [image_ref, image_tgt]
    mask_images = [mask_ref, mask_tgt]
    prompts = [prompt]*2

    latents = torch.concat([latent_ref, latent_tgt.to(pipe.dtype)], dim=0).to(pipe.dtype).to(pipe.device)

    # STEP, LAYER = 0, 0
    # editor = MutualSelfAttentionControl(STEP, LAYER, total_steps=num_inference_steps, model_type="SD")
    # regiter_attention_editor_diffusers(pipe, editor)

    inpainted = pipe(prompt=prompts, image=images, mask_image=mask_images,
                     height=new_h, width=new_w, latents=latents,
                     guidance_scale= guidance_scale, strength=strength,
                     num_inference_steps=num_inference_steps).images[1]

    # image_tgt.paste(inpainted, (0, 0), mask=msk)
    # inpainted = image_tgt.resize((width, height))
    inpainted = inpainted.resize((width, height))
    image = Image.new("RGB", (width, 2 * height), "white")
    image.paste(image_raw, (0, 0))
    image.paste(inpainted, (0, height))
    image.save(os.path.join(savedir, fn+'.png'))
    return np.array(inpainted) / 255.



def apply_mask(image, mask):
    """Apply a binary mask to an image. Where mask == 0, set the image to white.
    Assumes image and mask are numpy arrays with values in [0, 1]."""
    # Scale image from [0, 1] to [0, 255] and convert to integers
    scaled_image = (image * 255).astype(np.uint8)
    # Apply mask, set background to white
    masked_image = np.where(mask[:, :, None] == 1, scaled_image, 255)
    return masked_image
def calculate_hashes(images, masks):
    """Calculate perceptual hashes for a list of images using their masks."""
    image_hashes = []
    for image, mask in zip(images, masks):
        if len(mask.shape) == 3:
            mask = np.squeeze(mask)
        # Apply the mask and convert to a PIL image
        img_pil = Image.fromarray(apply_mask(image, mask).astype('uint8'))
        # Calculate the perceptual hash of the masked image
        img_hash = imagehash.phash(img_pil)
        image_hashes.append(img_hash)
    return image_hashes
def find_most_similar_image(image_hashes):
    """Find the image with the highest overall similarity to all other images."""
    num_images = len(image_hashes)
    min_distance_sums = np.inf
    most_similar_index = -1
    # Calculate pairwise distances
    for i in range(num_images):
        sum_distances = 0
        for j in range(num_images):
            if i != j:
                distance = image_hashes[i] - image_hashes[j]
                sum_distances += distance
        # Update the most similar image if the current sum of distances is smaller
        if sum_distances < min_distance_sums:
            min_distance_sums = sum_distances
            most_similar_index = i
    return most_similar_index, min_distance_sums


def find_base_frame(i_train, cameras, images, masks, pipe, seed,
                    prompt, denoise_steps, savedir, i, num=1, strength=1.0,
                    spherify=False):
    # sdxl requires strength=0.99 so the stored noise is zT, fed latent is zt
    poses = np.stack([cam.camera_to_worlds for cam in cameras])
    masks = np.squeeze(masks)
    H, W = images.shape[1:3] # B, H, W, 3
    if spherify:
        i_local = find_highest_z_cam(poses[i_train])
        skip = 1
    else:
        i_local = find_central_cams(poses[i_train])
        skip = 3
    if num == 1:
        img_i_ref = i_train[i_local][0]
    else:
        candidates = i_train[i_local[np.arange(0, num * skip, skip)]]
        print("Candidates: ", candidates)
        images_inpainted = []
        for img_i in candidates:
            zT_ref, _ = sample_noise_latent(images[img_i], pipe, seed, 8 * (H // 8), 8 * (W // 8),
                                                 strength=strength, num_inference_steps=denoise_steps)
            # scale according to https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_inpaint.py#L815
            # if strength != 1.0: zt_ref /= pipe.scheduler.init_noise_sigma
            image_ref = inpaint_image(pipe, image=images[img_i], mask=masks[img_i], latent_noise=zT_ref,
                                      prompt=prompt, savedir=savedir, fn='i{:03d}_e{:05d}'.format(img_i, i),
                                      num_inference_steps=denoise_steps, strength=strength)
            images_inpainted.append(image_ref)
        image_hashes = calculate_hashes(images_inpainted, masks[candidates])
        most_similar_index, total_min_distance = find_most_similar_image(image_hashes)
        img_i_ref = candidates[most_similar_index]

    print("Base: ", img_i_ref)
    zT_ref, _ = sample_noise_latent(images[img_i_ref], pipe, seed, 8 * (H // 8), 8 * (W // 8),
                                         strength=strength, num_inference_steps=denoise_steps)
    # if strength != 1.0: zt_ref /= pipe.scheduler.init_noise_sigma
    image_ref = inpaint_image(pipe, image=images[img_i_ref], mask=masks[img_i_ref], latent_noise=zT_ref,
                              prompt=prompt, savedir=savedir, fn='i{:03d}_e{:05d}'.format(img_i_ref, i),
                              num_inference_steps=denoise_steps, strength=strength)
    return zT_ref, image_ref, img_i_ref


def calculate_min_distance(src_pose, tgt_poses):
    distances = [np.linalg.norm(src_pose[:3, 3]-tgt[:3, 3]) for tgt in tgt_poses]
    local_i = np.argmin(distances)
    return (local_i, distances[local_i])


def mask2ct_idx(ob_mask, div=1):
    ob_ct = scipy.ndimage.binary_dilation(ob_mask).astype(ob_mask.dtype) - ob_mask
    # ob_ct[:H//2, :] = 0
    ob_ct_idxs = np.argwhere(ob_ct == 1.)
    return ob_ct_idxs
def mask2bbox(ob_mask):
    # ret, thresh = cv2.threshold(imread(m), 127, 255, cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # coord = contours[0]  # N, 1, xy
    ob_ct_idxs = mask2ct_idx(ob_mask)
    upper, bottom = np.min(ob_ct_idxs[..., 0]), np.max(ob_ct_idxs[..., 0])
    left, right = np.min(ob_ct_idxs[..., 1]), np.max(ob_ct_idxs[..., 1])
    return left, right, upper, bottom

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
lpips_fn = LearnedPerceptualImagePatchSimilarity(normalize=True)
@torch.no_grad()
def compute_similarity(img_src, img_tgt, msk_src, msk_tgt):
    left, right, upper, bottom = mask2bbox(msk_src)
    img_src_msk = img_src[upper:bottom, left:right]
    left, right, upper, bottom = mask2bbox(msk_tgt)
    img_tgt_msk = img_tgt[upper:bottom, left:right]
    h, w = img_src_msk.shape[0], img_src_msk.shape[1]
    if min(h,w) < 32: h,w  = 2*h, 2*w
    img_src_msk = cv2.resize(img_src_msk, dsize=(h, w), interpolation=cv2.INTER_LINEAR)
    img_tgt_msk = cv2.resize(img_tgt_msk, dsize=(h, w), interpolation=cv2.INTER_LINEAR)
    img_src_msk= torch.moveaxis(torch.from_numpy(img_src_msk).float(), -1, 0)[None, ...]
    img_tgt_msk = torch.moveaxis(torch.from_numpy(img_tgt_msk).float(), -1, 0)[None, ...]
    lpips = lpips_fn(img_src_msk, img_tgt_msk).cpu().numpy().item()
    return lpips


"""
img_i = len(poses)//2
warp_image = splat_image(c2w_src=poses[img_i_ref], c2w_tgt=poses[img_i],
                         init_depth=depth_ref, init_image=image_ref,
                         H=H, W=W, K=K, device='cpu')
mask = masks[img_i][..., None]
warp_image = images[img_i] * (1-mask) + warp_image * mask
warp_image = np.concatenate([images[img_i], warp_image])
res = Image.fromarray((warp_image * 255.).astype(np.uint8))
res.save(os.path.join(savedir, "splat" + str(img_i) + ".png"))
"""
from pytorch3d.utils import cameras_from_opencv_projection
from easydict import EasyDict as edict
from utils.splat import render_forward_splat_co3d
def splat_image(c2w_src, c2w_tgt, init_depth, init_image, H, W, K, device):
    c2w_src = np.vstack([c2w_src, np.array([[0, 0, 0, 1]])])
    c2w_tgt = np.vstack([c2w_tgt, np.array([[0, 0, 0, 1]])])
    w2c_src = torch.from_numpy(np.linalg.inv(c2w_src)).to(device).float().unsqueeze(0)
    w2c_tgt = torch.from_numpy(np.linalg.inv(c2w_tgt)).to(device).float().unsqueeze(0)

    k = torch.from_numpy(K).unsqueeze(0).to(device).float()

    # output would yield to source intrisics
    src_viewpoint = cameras_from_opencv_projection(w2c_src[:, :3, :3], w2c_src[:, :3, 3], k,
                                                   torch.tensor([H, W]).float().unsqueeze(0))
    tgt_viewpoint = cameras_from_opencv_projection(w2c_tgt[:, :3, :3], w2c_tgt[:, :3, 3], k,
                                                   torch.tensor([H, W]).float().unsqueeze(0))

    depth = torch.from_numpy(init_depth).float().unsqueeze(0).unsqueeze(0)
    image_towarp = torch.from_numpy(init_image).permute(2,0,1).float().unsqueeze(0)
    # image_towarp[depth.expand(-1, 3, -1, -1) <= 0.] = bg
    src_frame = edict({"camera": src_viewpoint, "image_rgb": image_towarp, "depth_map": depth, })
    tgt_frame = edict({"camera": tgt_viewpoint, })

    warp_image, warp_disp, warp_mask = render_forward_splat_co3d(src_frame, tgt_frame,
                                                                 filter_depth=False, return_angles=False,
                                                                 filter_bg=True, epi_bg=False, max_rays=-1,
                                                                 black_bg=False, splat_depth=False,
                                                                 splat_params=False, adjust_mono=False)
    # warp_image = warp_image * warp_mask + (1 - warp_mask)
    return warp_image.squeeze(0).cpu().permute(1,2,0).numpy(), warp_mask.squeeze(0).cpu().permute(1,2,0).numpy()


from sklearn.linear_model import LinearRegression


def fit_plane_to_depth(depth_map, mask, bbox, expansion=5):
    left, right, upper, bottom = bbox
    # Optionally expand the bounding box
    left = max(left - expansion, 0)
    right = min(right + expansion, mask.shape[1] - 1)
    upper = max(upper - expansion, 0)
    bottom = min(bottom + expansion, mask.shape[0] - 1)

    # Extract the bounding box regions
    mask_bbox = mask[upper:bottom+1, left:right+1]
    depth_bbox = depth_map[upper:bottom+1, left:right+1]

    # Use zero-value mask points to fit the plane
    zero_y, zero_x = np.where(mask_bbox == 0)
    zero_depths = depth_bbox[zero_y, zero_x]

    # Prepare data for plane fitting
    X_train = np.column_stack((zero_x, zero_y, np.ones_like(zero_x)))
    y_train = zero_depths

    # Fit a linear regression model to the zero-masked depths
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict depths where mask is one
    one_y, one_x = np.where(mask_bbox == 1)
    if len(one_y) == 0:
        return depth_map  # No mask value of 1 in the area, return original depth map

    X_predict = np.column_stack((one_x, one_y, np.ones_like(one_x)))
    predicted_depths = model.predict(X_predict)

    # Update the depth map with the predicted values
    new_depth_map = np.copy(depth_map)
    for idx, depth in enumerate(predicted_depths):
        new_depth_map[upper + one_y[idx], left + one_x[idx]] = depth

    return new_depth_map