import math
import os, sys, shutil
import copy
import cv2
import numpy as np
import torch
import scipy
import imageio
from sklearn.linear_model import HuberRegressor


to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


def align_depth_with_rgb(image, depth, confidence=None):

    solver = cv2.ximgproc.createFastBilateralSolverFilter(to8b(image), sigma_spatial=8, sigma_luma=8,
                                                          sigma_chroma=8, num_iter=25, max_tol=1e-5)
    max_depth = np.max(depth)  # Determine the max depth from the map
    depth_map_noisy = to8b(depth / max_depth)
    if confidence is None:
        confidence = np.ones_like(depth_map_noisy, dtype=np.float32)
    else:
        confidence = confidence.astype(np.float32)
    filtered_depth = solver.filter(depth_map_noisy, confidence=confidence)
    filtered_depth = filtered_depth.astype(np.float32) / 255.0 * max_depth
    return filtered_depth


def align_depth(pred_depth, nerf_depth, valid_mask, weighted=False, robust=True, intercept=True):
    mask = ~scipy.ndimage.binary_erosion(~valid_mask) # mask: 1 = valid depth

    valid_N = nerf_depth[mask]
    valid_P = pred_depth[mask]
    if weighted:
        mask_indices = np.argwhere(~mask) #object area
        mask_center_y = mask_indices[:, 0].mean()
        mask_center_x = mask_indices[:, 1].mean()
        y_coords, x_coords = np.indices(mask.shape)
        distances = np.sqrt((y_coords - mask_center_y) ** 2 + (x_coords - mask_center_x) ** 2)
        weights = 1. / (distances)
        weights = np.clip(weights / weights[mask].max(), 0., 1.)
        # imageio.imwrite("w.png", to8b(weights))
        weights = weights[mask]
    else:
        weights = np.ones_like(valid_P).astype(np.float16)

    valid_P = valid_P.reshape(-1, 1)
    valid_N = valid_N.reshape(-1)
    weights = weights.reshape(-1)

    if intercept:
        # Prepare the design matrix X with an intercept term
        X = np.hstack([np.ones((valid_P.shape[0], 1)), valid_P])
    else:
        X = valid_P

    if robust:
        huber = HuberRegressor(epsilon=1.35, alpha=0.0)  # Set alpha to 0 for no regularization
        huber.fit(X, valid_N, sample_weight=weights)
        # beta = np.hstack([huber.intercept_, huber.coef_[1:]])
        print(huber.intercept_, huber.coef_[1:])
        # Using beta, predict Y values for the given or new X values
        # Assuming 'pred_depth' is defined elsewhere
        X_pred = pred_depth.reshape(-1, 1)
        if intercept:
            X_pred = np.hstack([np.ones_like(X_pred), X_pred])
        Y_pred = huber.predict(X_pred).reshape(pred_depth.shape)
    else:
        # Weight adjustments are made directly to the design matrix X and response vector Y
        X_weighted = X * weights[:, np.newaxis]  # Apply weights to each observation in X
        Y_weighted = valid_N * weights  # Apply weights to Y

        # Compute beta coefficients using the correct WLS formula
        beta = np.linalg.inv(X_weighted.T @ X_weighted) @ (X_weighted.T @ Y_weighted)

        # Using beta, predict Y values for the given or new X values
        Y_pred = beta[0] + pred_depth * beta[1]

    print("Depth average: ", np.mean(Y_pred))
    return Y_pred


def save_depth(depth, savedir, i):
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    imageio.imwrite(os.path.join(savedir, 'i{:03d}_dpt.png'.format(i)), depth)



from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

@torch.no_grad()
def predict_depths(images, depth_anything_net, savedir=None):
    import torch.nn.functional as F
    depths = []
    for i, image in enumerate(images):
        h, w = image.shape[:2]

        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to("cuda")

        depth = depth_anything_net(image)
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depths.append(np.copy(depth.cpu().numpy()))

        # if savedir is not None:
        #     save_depth(depth, savedir, i)
    return depths
