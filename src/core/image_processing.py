import logging
from typing import List

import cv2
import numpy as np
from numba import jit
import torch
import torchvision.transforms.functional as F
import time

logger = logging.getLogger(__name__)


@jit(nopython=True, cache=True)
def _fast_skin_mask_ycrcb(ycrcb_data, lower_cr, upper_cr, lower_cb, upper_cb):
    mask = np.zeros((ycrcb_data.shape[0], ycrcb_data.shape[1]), dtype=np.uint8)
    for i in range(ycrcb_data.shape[0]):
        for j in range(ycrcb_data.shape[1]):
            cr, cb = ycrcb_data[i, j, 1], ycrcb_data[i, j, 2]
            if lower_cr <= cr <= upper_cr and lower_cb <= cb <= upper_cb:
                mask[i, j] = 255
    return mask

# TODO: can't segment properly (segments facial area also instead of just background). Fix or remove.
def apply_adaptive_skin_segmentation(frame: np.ndarray, method: str = 'ycrcb_adaptive') -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame

    if method == 'ycrcb_adaptive':
        ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)

        cr_mean = np.mean(ycrcb_frame[:, :, 1])
        cb_mean = np.mean(ycrcb_frame[:, :, 2])

        cr_std = np.std(ycrcb_frame[:, :, 1])
        cb_std = np.std(ycrcb_frame[:, :, 2])

        lower_cr = max(77, int(cr_mean - 1.5 * cr_std))
        upper_cr = min(227, int(cr_mean + 1.5 * cr_std))
        lower_cb = max(133, int(cb_mean - 1.5 * cb_std))
        upper_cb = min(173, int(cb_mean + 1.5 * cb_std))

        skin_mask = _fast_skin_mask_ycrcb(
            ycrcb_frame, lower_cr, upper_cr, lower_cb, upper_cb
        )
    else:
        ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
        skin_mask = _fast_skin_mask_ycrcb(
            ycrcb_frame, 133, 173, 77, 127
        )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return cv2.bitwise_and(frame, frame, mask=skin_mask)


def scale_frame_by_reference(frame, target_size):
    """
    Scales a frame DOWN while preserving aspect ratio, using a robust,
    orientation-agnostic approach.
    """
    target_w, target_h = target_size
    original_h, original_w = frame.shape[:2]

    # Sort dimensions to handle different orientations correctly
    orig_dims_sorted = sorted((original_w, original_h))
    target_dims_sorted = sorted((target_w, target_h))

    # If the original is smaller than the target on either its short or long side,
    #  return the original frame to prevent any upscaling.
    if orig_dims_sorted[0] < target_dims_sorted[0] or \
       orig_dims_sorted[1] < target_dims_sorted[1]:
        return frame

    # Calculate the scale ratio by comparing short-side-to-short-side and
    #  long-side-to-long-side.
    scale_ratio = max(
        target_dims_sorted[0] / orig_dims_sorted[0],
        target_dims_sorted[1] / orig_dims_sorted[1]
    )

    # Calculate new dimensions and resize
    new_width = int(round(original_w * scale_ratio))
    new_height = int(round(original_h * scale_ratio))

    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


# TODO: fix or remove
def preprocess_batch_grayscale(bgr_tensor_batch: torch.Tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tensor_on_device = bgr_tensor_batch.to(device)

        # BGR to RGB
        rgb_tensor = tensor_on_device.flip(-1)

        # (B, C, H, W) for torchvision
        rgb_tensor_chw = rgb_tensor.permute(0, 3, 1, 2)

        if rgb_tensor_chw.dtype == torch.uint8:
            rgb_tensor_chw = rgb_tensor_chw.float() / 255.0

        grayscale_batch = F.rgb_to_grayscale(rgb_tensor_chw)

        return grayscale_batch.cpu()
    except Exception as e:
        print(f"Error when converting to grayscale on GPU-u: {e}. Fallback to CPU...")
        cpu_device = torch.device("cpu")
        tensor_on_cpu = bgr_tensor_batch.to(cpu_device)
        rgb_tensor = tensor_on_cpu.flip(-1)
        rgb_tensor_chw = rgb_tensor.permute(0, 3, 1, 2)
        if rgb_tensor_chw.dtype == torch.uint8:
            rgb_tensor_chw = rgb_tensor_chw.float() / 255.0
        grayscale_batch_cpu = F.rgb_to_grayscale(rgb_tensor_chw)
        return grayscale_batch_cpu


def process_frame(frame: np.ndarray, flags: List[str]):

    if frame is None or frame.size == 0 or flags is None:
        return frame, 0.0

    start_time = time.perf_counter()
    try:
        if 'median_filter' in flags:
            frame = cv2.medianBlur(frame, 3)  # Using a default kernel size of 3

        if 'grayscale' in flags:
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if 'clahe' in flags:
            # CLAHE requires a grayscale image. Convert if not already done.
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            frame = clahe.apply(frame)

    except (cv2.error, IndexError, ValueError) as e:
        logger.warning(f"Frame processing error: {e}")

    elapsed_time = time.perf_counter() - start_time

    return frame, elapsed_time


def apply_skin_segmentation(frame: np.ndarray) -> np.ndarray:
    return apply_adaptive_skin_segmentation(frame)


def apply_clahe(gray_frame: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    if gray_frame is None or gray_frame.size == 0:
        return gray_frame

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray_frame)

