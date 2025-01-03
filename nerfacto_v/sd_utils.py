import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import copy
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# latent_to_rgb_matrix = np.array([[0.298, 0.207, 0.208],  # L1
#                         [0.187, 0.286, 0.173],  # L2
#                         [-0.158, 0.189, 0.264],  # L3
#                         [-0.184, -0.271, -0.473]])  # L4]
#
# def latent2rgb(latent):
#     sh = list(latent.shape[:-1])
#     img = (np.dot(latent.reshape(-1, latent.shape[-1]), latent_to_rgb_matrix))
#     img = img.reshape(sh+[3])
#     img = (img.clip(-1., 1.) + 1.) / 2. #[-1,1] to [0,1]
#     img = img.clip(0., 1.)
#     return img
#
#
# def rgb2latent(rgb):
#     rgb = 2. * rgb - 1. #[0,1] to [-1,1]
#     sh = list(rgb.shape[:-1])
#     latent = (np.dot(rgb.reshape(-1, rgb.shape[-1]), np.linalg.pinv(latent_to_rgb_matrix)))
#     latent = latent.reshape(sh+[4])
#     print("latent statistics: ", latent.min(), latent.mean(), latent.max())
#     return latent


@torch.no_grad()
def encode_latent_from_np(image, pipe, crops_coords = None, resize_mode = "default", generator=None):
    # H, W = image.shape[:2]
    # new_w, new_h = 8 * (W // 8), 8 * (H // 8)
    # image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
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
def decode_latent_to_np(latents, pipe, generator=None):
    condition_kwargs = {}
    # if isinstance(pipe.vae, AsymmetricAutoencoderKL):
    #     init_image = init_image.to(device=device, dtype=masked_image_latents.dtype)
    #     init_image_condition = init_image.clone()
    #     init_image = self._encode_vae_image(init_image, generator=generator)
    #     mask_condition = mask_condition.to(device=device, dtype=masked_image_latents.dtype)
    #     condition_kwargs = {"image": init_image_condition, "mask": mask_condition}
    # print("dec1", latents.min(), latents.max())
    image = pipe.vae.decode(
        latents.unsqueeze(0).to(pipe.device).to(pipe.dtype) / pipe.vae.config.scaling_factor,
        return_dict=False, generator=generator, **condition_kwargs
    )[0]
    # print("dec2", image.min(), image.max())
    image, has_nsfw_concept = pipe.run_safety_checker(image, pipe.device, pipe.dtype)
    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
    image = pipe.image_processor.postprocess(image, output_type='np', do_denormalize=do_denormalize)
    # print("dec3", image.min(), image.max())
    return np.squeeze(image.clip(0,1))


def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    **kwargs,
):
    import inspect
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# def update_latent_model_optimizers(model):
#     from utils_nerfacto.mlp import MLP
#     import torch.nn as nn
#     from utils_nerfacto.optimizers import Optimizers
#     from utils_nerfacto.optimizers import AdamOptimizerConfig
#     from utils_nerfacto.schedulers import ExponentialDecaySchedulerConfig
#     model.mlp_head = MLP(
#         in_dim=model.mlp_head.in_dim,
#         num_layers=model.num_layers_color,
#         layer_width=model.hidden_dim_color,
#         out_dim=4,
#         activation=nn.ReLU(),
#         out_activation=None,
#         implementation=model.implementation,
#     )
#     param_groups = model.get_param_groups()
#     optim_config = {k: {"optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
#                         "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=20000)}
#                     for k in param_groups.keys()}
#     optimizers = Optimizers(optim_config, param_groups)
#     return model, optimizers


def get_latent_timestep(pipe, strength=1.0, num_inference_steps=50):
    timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, num_inference_steps, pipe.device, None)
    timesteps, num_inference_steps = pipe.get_timesteps(num_inference_steps=num_inference_steps,
                                                        strength=strength, device=pipe.device)
    latent_timestep = timesteps[:1].repeat(1)
    return latent_timestep


def get_sqrt_alpha_prod_and_sqrt_one_minus_alpha_prod(pipe, latent_timestep):
    sqrt_alpha_prod = pipe.scheduler.alphas_cumprod.to(pipe.device)[latent_timestep.to(pipe.device)] ** 0.5
    sqrt_one_minus_alpha_prod = (1. - pipe.scheduler.alphas_cumprod.to(pipe.device)[latent_timestep.to(pipe.device)]) ** 0.5
    return sqrt_alpha_prod, sqrt_one_minus_alpha_prod


def get_timestep_from_sqrt_one_minus_alpha_prod(pipe, sqrt_one_minus_alpha_prod):
    alpha_cumprod = 1. - (sqrt_one_minus_alpha_prod) ** 2
    differences = torch.abs(alpha_cumprod - pipe.scheduler.alphas_cumprod)
    timestep_close_idx =  torch.argmin(differences)
    timestep = timestep_close_idx
    return timestep


def get_strength_from_timestep(pipe, timestep, num_inference_steps=50):
    timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, num_inference_steps, pipe.device, None)
    differences = torch.abs(timesteps - timestep)
    close_timestep_idx = torch.argmin(differences)
    strength = 1.0 - (close_timestep_idx+1) / len(timesteps)
    return strength


def get_stength_from_sigma(pipe, sigma, num_inference_steps=50):
    timestep = get_timestep_from_sqrt_one_minus_alpha_prod(pipe, sigma)
    strength_approx = get_strength_from_timestep(pipe, timestep, num_inference_steps=num_inference_steps)
    return strength_approx


@torch.no_grad()
def sample_noise_latent(image, pipe, seed, height, width, strength=1.0, num_inference_steps=20):
    batch_size, num_images_per_prompt = 1, 1
    num_channels_latents = pipe.vae.config.latent_channels
    generator = None
    latents = None

    latent_timestep = get_latent_timestep(pipe, strength=strength, num_inference_steps=num_inference_steps)
    is_strength_max = strength == 1.0
    # return_image_latents = pipe.unet.config.in_channels == 4

    if isinstance(image, np.ndarray):
        print("Convert image to latent space")
        image = encode_latent_from_np(image, pipe)

    torch.manual_seed(seed)
    latents_outputs = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        pipe.dtype,
        pipe.device,
        generator,
        latents,
        image=image,
        timestep=latent_timestep,
        is_strength_max=is_strength_max,
        return_noise=True,
        return_image_latents=False,
    )

    # if timestep of strength is not the full then they are different
    # else initial latent = noise
    latents, noise = latents_outputs
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py
    # noise is the randn tensor, latents = noise * self.scheduler.init_noise_sigma
    return noise, latents


def compute_latent_intrinsics(pipe, H, W, K):
    new_w, new_h = 8 * (W // 8) // pipe.vae_scale_factor, 8 * (H // 8) // pipe.vae_scale_factor
    s_x, s_y = new_w / W, new_h / H
    K[0, 0] *= s_x  # Adjust f_x
    K[1, 1] *= s_y  # Adjust f_y
    K[0, 2] *= s_x  # Adjust c_x
    K[1, 2] *= s_y  # Adjust c_y
    return new_h, new_w, K

def compute_latent_camera(pipe, camera):

    W = camera.width
    H = camera.height

    camera_new = copy.deepcopy(camera)
    new_w, new_h = 8 * (W // 8) // pipe.vae_scale_factor, 8 * (H // 8) // pipe.vae_scale_factor

    camera_new.width = new_w
    camera_new.height = new_h

    s_x, s_y = new_w / W, new_h / H
    camera_new.fx *= s_x  # Adjust f_x
    camera_new.fy *= s_y  # Adjust f_y
    camera_new.cx *= s_x  # Adjust c_x
    camera_new.cy *= s_y  # Adjust c_y
    return camera_new

# @torch.no_grad()
# def prepare_latent_nerf_vae(pipe, images, masks, H, W, K, extra, model=None):
#     images_latent = []
#     masks_latent = []
#     # process images and K
#     new_h, new_w, K = compute_latent_intrinsics(pipe, H, W, K)
#     for i, img in enumerate(images):
#         latent = encode_latent_from_np(img, pipe).cpu().squeeze().permute(1, 2, 0).numpy()
#         mask_latent = cv2.resize(masks[i], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
#         assert latent.shape[:2] == mask_latent.shape[:2]
#         images_latent.append(latent)
#         masks_latent.append(mask_latent)
#
#     images = np.stack(images_latent, dtype=images.dtype)
#     masks = np.stack(masks_latent)
#     print(images.shape, masks.shape)
#
#     extra['depths'] = [cv2.resize(d, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR) for d in extra['depths']]
#     extra['depth_valids'] = [cv2.resize(d, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR) for d in extra['depth_valids']]
#
#     if model is not None:
#         model, optimizers = update_latent_model_optimizers(model)
#     else:
#         optimizers = None
#
#     return model, optimizers, images, masks, new_h,  new_w, K, extra


# @torch.no_grad()
# def prepare_latent_nerf_linear(pipe, images, masks, H, W, K, extra, downsample=True, model=None):
#     images_latent = []
#     masks_latent = []
#     # process images and K
#     if downsample:
#         new_w, new_h = 8 * (W // 8)// pipe.vae_scale_factor, 8 * (H // 8)// pipe.vae_scale_factor
#         # Adjust the intrinsic parameters
#         s_x, s_y = new_w / W, new_h / H
#         K[0, 0] *= s_x  # Adjust f_x
#         K[1, 1] *= s_y  # Adjust f_y
#         K[0, 2] *= s_x  # Adjust c_x
#         K[1, 2] *= s_y  # Adjust c_y
#         extra['depths'] = [cv2.resize(d, dsize=(new_w // pipe.vae_scale_factor, new_h // pipe.vae_scale_factor),
#                                       interpolation=cv2.INTER_LINEAR) for d in extra['depths']]
#         extra['depth_valids'] = [cv2.resize(d, dsize=(new_w // pipe.vae_scale_factor, new_h // pipe.vae_scale_factor),
#                                             interpolation=cv2.INTER_LINEAR) for d in extra['depth_valids']]
#     else:
#         new_w, new_h = W, H
#
#     for i, img in enumerate(images):
#         img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#         lat = rgb2latent(img)
#         mask_lat = cv2.resize(masks[i], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
#         images_latent.append(lat)
#         masks_latent.append(mask_lat)
#
#     images = np.stack(images_latent, dtype=images.dtype)
#     masks = np.stack(masks_latent)
#     print(images.shape, masks.shape)
#
#     if model is not None:
#         model, optimizers = update_latent_model_optimizers(model)
#     else:
#         optimizers = None
#
#     return model, optimizers, images, masks, new_h, new_w, K, extra


@torch.no_grad()
def to_rgb(x):
    x = x.float()
    colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
    x = nn.functional.conv2d(x, weight=colorize)
    x = (x - x.min()) / (x.max() - x.min())
    return x


if __name__ == '__main__':
    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-2-inpainting",
    #     # "runwayml/stable-diffusion-inpainting",
    #     torch_dtype=torch.float16).to("cuda")
    # sigma = 1.0
    # timestep = get_timestep_from_sqrt_one_minus_alpha_prod(pipe, sigma)
    # print(timestep)
    # strength = get_strength_from_timestep(pipe, timestep, num_inference_steps=50)
    # print(strength)
    # latent_timestep = get_latent_timestep(pipe, strength=strength, num_inference_steps=50)
    # print(latent_timestep)
    # sqrt_alpha_prod, sqrt_one_minus_alpha_prod = get_sqrt_alpha_prod_and_sqrt_one_minus_alpha_prod(pipe, latent_timestep)
    # print(sigma, sqrt_one_minus_alpha_prod)

    z = np.random.randint(0, 256, (3, 4, 3), dtype=np.uint8)
    from PIL import Image
    img = Image.fromarray(z)
    new_dimensions = (256, 192)  # Width: 256, Height: 512
    high_res_img = img.resize(new_dimensions, Image.NEAREST)
    high_res_img.save('z.png')



    """
    sigma = round_to_hundredth_nearest_05(std_sample)
    # strength = round_to_tenth(get_stength_from_sigma(pipe=pipe, sigma=sigma, num_inference_steps=50).item())
    latent_image = encode_latent_from_np(image, pipe)
    # noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    latent_tensor = ((1. - sigma ** 2) ** 0.5) * latent_image + latent_tensor  
    """


