import numpy as np
import cv2
from typing import List

from src.core.constants import LandmarkMethod, DetectionMethod

def prepare_images_for_detector(
    rgb_frames: List[np.ndarray],
    method: str
) -> List[np.ndarray]:
    """
    Prepares a batch of input frames for a specific detection method by handling
    color space conversion and ensuring a contiguous memory layout.

    This function strictly assumes the input frames are in RGB format.

    Args:
        rgb_frames (List[np.ndarray]): A list of input frames in RGB format.
        method (str): The name of the detection method (e.g., "YOLO", "LBF").

    Returns:
        List[np.ndarray]: A list of processed images ready for the specified detector.
    """
    grayscale_methods = {
        DetectionMethod.OPENCV.value,
        DetectionMethod.DLIB.value,
        LandmarkMethod.LBF.value
    }
    bgr_methods = {
        DetectionMethod.YUNET.value
    }
    # other assume rgb

    images = []
    for frame in rgb_frames:
        processed_frame = frame

        if method in grayscale_methods:
            if frame.ndim == 3 and frame.shape[2] == 3:
                processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        elif method in bgr_methods:
            if frame.ndim == 3 and frame.shape[2] == 3:
                processed_frame = frame[..., ::-1]
        else:
            if frame.ndim == 2:
                processed_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Always ensure the final image has a contiguous memory layout to prevent errors
        # with underlying C++ libraries (like MediaPipe).
        images.append(np.ascontiguousarray(processed_frame))
        del frame

    return images
