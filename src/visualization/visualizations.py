import hashlib
import logging
import os
import time
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

from src.config import ExperimentConfig
from src.core.constants import AppConstants, UI, LandmarkIndices

logger = logging.getLogger(__name__)

bool_indicators = ['is_eye_closed_threshold', 'is_eye_closed', 'is_yawn', 'is_eye_closed_signal',
                   'is_microsleeping']

def get_cache_folder(config: ExperimentConfig, are_calibration_frames: bool, label: int) -> Optional[str]:
    final_cache_path, _, calib_cache_path = config.get_cache_filenames()
    cache_basename = os.path.splitext(os.path.basename(
        final_cache_path if not are_calibration_frames else calib_cache_path
    ))[0]
    run_specific_viz_dir = os.path.join(config.DEBUG_VIZ_DIR, cache_basename, f"label_{label}")
    os.makedirs(run_specific_viz_dir, exist_ok=True)
    return run_specific_viz_dir


def save_landmark_custom_failure_debug_image(
        debug_path: str,
        frame: np.ndarray,
        face_rect: np.ndarray,
        landmarks: np.ndarray,
        reason: str
):
    """Saves a debug image when landmark validation fails."""
    if not debug_path:
        return
    try:
        output_dir = os.path.join(
            debug_path,
            "LANDMARK_FAILURES",
            "LANDMARKER_DETECTOR_CUSTOM",
        )
        os.makedirs(output_dir, exist_ok=True)

        viz_frame = frame.copy()
        x, y, w, h = face_rect
        cv2.rectangle(viz_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if landmarks is not None and landmarks.size > 0:
            for (lx, ly) in landmarks:
                cv2.circle(viz_frame, (int(lx), int(ly)), 2, (0, 0, 255), -1)
        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()[:8]
        filename = f"failure_{reason}_{int(time.time() * 1000)}_{frame_hash}.jpg"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, viz_frame)
    except Exception as e:
        logger.warning(f"Could not save landmark failure debug image: {e}", exc_info=True)



def save_landmark_failure_debug_image(
        frame: np.ndarray,
        frame_idx: int,
        face_rect: np.ndarray,
        video_path: str,
        failure_reason: str,
        config: ExperimentConfig
    ):
    """Saves a debug image when landmark validation fails."""
    try:
        if not config.SAVE_DEBUG_IMAGES_ON_FAILURE:
            return

        output_dir = os.path.join(
            config.DEBUG_VIZ_DIR,
            "LANDMARK_FAILURES",
            config.SETUP_LANDMARK_DETECTION_METHOD.value,
            f"{config.SETUP_FRAME_SCALE}"
        )
        os.makedirs(output_dir, exist_ok=True)

        viz_frame = frame.copy()
        x, y, w, h = face_rect
        cv2.rectangle(viz_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        subject_id = os.path.basename(os.path.dirname(video_path))
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_str = f"{frame_idx}"
        filename = f"{subject_id}_{video_name}_frame_{frame_str}_failure-{failure_reason}.jpg"

        output_path = os.path.join(output_dir, filename)

        cv2.imwrite(output_path, viz_frame)
    except Exception as e:
        logger.warning(f"Could not save landmark failure debug image: {e}", exc_info=True)


def save_face_detection_debug_image_on_failure(
        frame: np.ndarray,
        frame_idx: int,
        video_path: str,
        failure_reason: str,
        config: ExperimentConfig
):
    """
    Saves a composite debug image when a failure occurs.
    """
    try:
        if not config.SAVE_DEBUG_IMAGES_ON_FAILURE:
            return

        output_dir = os.path.join(
            config.DEBUG_VIZ_DIR,
            "DETECTION_FAILURES",
            config.SETUP_FACE_DETECTION_METHOD.value,
            f"{config.SETUP_FRAME_SCALE}"
        )
        os.makedirs(output_dir, exist_ok=True)

        subject_id = os.path.basename(os.path.dirname(video_path))
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_str = f"{frame_idx}"
        filename = f"{subject_id}_{video_name}_frame_{frame_str}_failure-{failure_reason}.jpg"

        output_path = os.path.join(output_dir, filename)

        cv2.imwrite(output_path, frame)

    except Exception as e:
        logger.error(f"Could not save debug image for {video_path}: {e}", exc_info=True)


def visualize_features(
        label: int,
        frame_idx: int,
        subject_id: str,
        config: ExperimentConfig,
        original_frame: np.ndarray | None,
        landmarks: np.ndarray,
        frame: np.ndarray | None,
        numerical_features: Dict,
        visual_features: Dict,
        visualize_features_config: List[str],
        face_coords: np.ndarray = None,
        are_calibration_frames: bool = False,
        save_as_file: bool = True
):
    try:
        debug_dir = os.path.join(get_cache_folder(config, are_calibration_frames, label))
        os.makedirs(debug_dir, exist_ok=True)
        debug_img = create_debug_visualization(
            original_frame=original_frame,
            landmarks=landmarks,
            frame=frame,
            numerical_features=numerical_features,
            visual_features=visual_features,
            face_coords=face_coords
        )
        if save_as_file:
            filename = f"{subject_id}_label_{label}_frame_{frame_idx}.jpg"

            if not visualize_features_config or 'all' in visualize_features_config:
                save_path = os.path.join(debug_dir, filename)
                cv2.imwrite(save_path, debug_img)

            ear_value = numerical_features.get('ear')
            if ear_value is not None:
                if ear_value > 0.4:
                    high_ear_dir = os.path.join(debug_dir, 'ear_raw_high')
                    os.makedirs(high_ear_dir, exist_ok=True)
                    save_path = os.path.join(high_ear_dir, filename)
                    cv2.imwrite(save_path, debug_img)
                    logger.warning(f"High EAR ({ear_value:.3f}) at frame {frame_idx}. Saved to {high_ear_dir}")
                # elif ear_value < 0.01:
                #     low_ear_dir = os.path.join(debug_dir, 'ear_raw_low')
                #     os.makedirs(low_ear_dir, exist_ok=True)
                #     save_path = os.path.join(low_ear_dir, filename)
                #     cv2.imwrite(save_path, debug_img)
                #     logger.warning(f"Low EAR ({ear_value:.3f}) at frame {frame_idx}. Saved to {low_ear_dir}")

            if visualize_features_config:
                if numerical_features.get('is_eye_closed') and 'is_eye_closed' in visualize_features_config:
                    closed_dir = os.path.join(debug_dir, 'is_eye_closed')
                    os.makedirs(closed_dir, exist_ok=True)

                    save_path = os.path.join(closed_dir, filename)
                    cv2.imwrite(save_path, debug_img)

                if numerical_features.get('is_eye_closed_threshold') and 'is_eye_closed_threshold' in visualize_features_config:
                    closed_signal_dir = os.path.join(debug_dir, 'is_eye_closed_threshold')
                    os.makedirs(closed_signal_dir, exist_ok=True)

                    save_path = os.path.join(closed_signal_dir, filename)
                    cv2.imwrite(save_path, debug_img)

                if numerical_features.get('is_eye_closed_signal') and 'is_eye_closed_signal' in visualize_features_config:
                    closed_signal_dir = os.path.join(debug_dir, 'is_eye_closed_signal')
                    os.makedirs(closed_signal_dir, exist_ok=True)

                    save_path = os.path.join(closed_signal_dir, filename)
                    cv2.imwrite(save_path, debug_img)

                if numerical_features.get('is_microsleeping') and 'is_microsleeping' in visualize_features_config:
                    microsleeping_dir = os.path.join(debug_dir, 'is_microsleeping')
                    os.makedirs(microsleeping_dir, exist_ok=True)

                    save_path = os.path.join(microsleeping_dir, filename)
                    cv2.imwrite(save_path, debug_img)


        return debug_img
    except Exception as viz_e:
        logger.warning(f"Visualization failed for frame {frame_idx}: {viz_e}", exc_info=True)


# TODO: implement; e.g. in eyes closed folder: would be good to see frame before and frame after eye closed classification.,
#  rather than having to manually search
def save_event_sequence_images(
        base_output_dir: str,
        event_category: str,
        frames: List[np.ndarray],
        event_indices: List[int],
        subject_id: str,
        label: str,
        context_frames: int = 1
):
    """
    Saves images for a specific event type (e.g., 'blinks', 'microsleeps')
    into a dedicated subfolder.

    Args:
        base_output_dir: The main debug directory for the video.
        event_category: The name of the event, used for the subfolder (e.g., "blinks").
        frames: A list of all frames for the video.
        event_indices: A list of frame indices where the event occurred.
        subject_id: The ID of the subject.
        label: The label of the video.
        context_frames: Number of frames to save before and after each event.
    """
    if not event_indices:
        return

    # Create the specific subfolder for this event type
    category_dir = os.path.join(base_output_dir, event_category)
    os.makedirs(category_dir, exist_ok=True)

    saved_indices = set()
    n_total_frames = len(frames)

    for event_idx in event_indices:
        start_idx = max(0, event_idx - context_frames)
        end_idx = min(n_total_frames - 1, event_idx + context_frames)

        for i in range(start_idx, end_idx + 1):
            if i in saved_indices:
                continue

            frame_to_save = frames[i]
            if frame_to_save is not None:
                filename = f"{subject_id}_{label}_{event_category}_frame_{i:06d}.jpg"
                save_path = os.path.join(category_dir, filename)
                cv2.imwrite(save_path, frame_to_save)
                saved_indices.add(i)


def create_debug_panel(visualizations: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Creates a single horizontal panel from a dictionary of visualization images."""
    if not visualizations:
        return None
    panel_components = []
    target_height = AppConstants.DEBUG_PANEL_HEIGHT
    for title, img in visualizations.items():
        if img is None or img.size == 0:
            continue
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        original_h, original_w = img.shape[:2]
        if original_h == 0: continue
        aspect_ratio = original_w / original_h
        target_w = int(target_height * aspect_ratio)
        if target_w == 0: continue
        resized_img = cv2.resize(img, (target_w, target_height))
        cv2.putText(resized_img, title, (10, 25), UI.FONT, 0.6, UI.OUTLINE_COLOR, 3, cv2.LINE_AA)
        cv2.putText(resized_img, title, (10, 25), UI.FONT, 0.6, UI.DEFAULT_COLOR, 1, cv2.LINE_AA)
        panel_components.append(resized_img)
    if not panel_components:
        return None
    return cv2.hconcat(panel_components)


def create_debug_visualization(
    original_frame: np.ndarray,
    landmarks: np.ndarray,
    frame: np.ndarray,
    numerical_features: Dict[str, any],
    visual_features: Dict[str, np.ndarray] = None,
    face_coords: np.ndarray = None,
    min_detail_item_width: int = 300,
    top_panel_width: int = 250,
    top_panel_height: int = 300
) -> Optional[np.ndarray]:
    """
    Creates a composite debug image with a single-row top panel (all major views side by side)
    and a bottom row with detail images. All resizing preserves aspect ratio.
    """
    PADDING = 10
    TOP_TARGET_SIZE = (top_panel_width, top_panel_height)
    DETAIL_IMG_HEIGHT = 120

    try:
        # --- 1. Top Panel: single horizontal row ---
        top_panels = []

        # Panel 1: Original Frame
        if original_frame is not None:
            top_panels.append(_resize_and_pad(_to_displayable_bgr(original_frame), TOP_TARGET_SIZE))

        # Panel 2: Processed + Face + Landmarks
        if frame is not None:
            processed_displayable = _to_displayable_bgr(frame)
            if landmarks is not None:
                for (x, y) in np.asarray(landmarks).astype(int):
                    cv2.circle(processed_displayable, (x, y), 1, (0, 255, 0), -1)
            top_panels.append(_resize_and_pad(processed_displayable, TOP_TARGET_SIZE))
        if visual_features and 'face_crop' in visual_features:
            face_crop = _to_displayable_bgr(visual_features['face_crop']).copy() # copy to prevent drawing on ROIs
            top_panels.append(_resize_and_pad(face_crop, TOP_TARGET_SIZE))

            face_crop_landmarks = _to_displayable_bgr(visual_features['face_crop']).copy()
            if landmarks is not None and face_coords is not None:
                # Translate landmarks to local coordinates using face_coords
                offset_x, offset_y, _, _ = face_coords
                local_landmarks = landmarks - np.array([offset_x, offset_y])
                for (x, y) in local_landmarks.astype(int):
                    if 0 <= x < face_crop_landmarks.shape[1] and 0 <= y < face_crop_landmarks.shape[0]:  # Ensure within bounds
                        cv2.circle(face_crop_landmarks, (x, y), 1, (0, 255, 0), -1)

            top_panels.append(_resize_and_pad(face_crop_landmarks, TOP_TARGET_SIZE))

        # Panel 3: Head Pose
        if visual_features and 'head_pose' in visual_features:
            top_panels.append(_resize_and_pad(_to_displayable_bgr(visual_features['head_pose']), TOP_TARGET_SIZE))

        # Panel 4: Metrics
        metrics_panel = _create_metrics_panel(numerical_features, (600, top_panel_height))
        top_panels.append(metrics_panel)

        # Horizontally stack all top panels
        if not top_panels:
            return None
        top_row_montage = cv2.hconcat(top_panels)

        # --- 2. Bottom panel (detail images) ---
        bottom_montages = []
        if visual_features:
            detail_items = {k: v for k, v in visual_features.items() if k != 'head_pose'}
            labeled_images = []
            for name, img in detail_items.items():
                img = _to_displayable_bgr(img)
                labeled_img = _create_labeled_detail_image(
                    name.replace("_", " ").title(), img, DETAIL_IMG_HEIGHT, min_detail_item_width
                )
                if labeled_img is not None:
                    labeled_images.append(labeled_img)
            # Wrap items into rows of max 4
            for i in range(0, len(labeled_images), 4):
                row_items = labeled_images[i:i + 4]
                if row_items:
                    bottom_montages.append(cv2.hconcat(row_items))

        # --- 3. Stack final montage and pad to max width ---
        all_montages = [top_row_montage] + bottom_montages
        if not all_montages:
            return None

        canvas_w = max(m.shape[1] for m in all_montages)
        padded_montages = []
        for montage in all_montages:
            h, w, _ = montage.shape
            pad_total = canvas_w - w
            pad_left = pad_total // 2
            padded_montages.append(
                cv2.copyMakeBorder(montage, 0, 0, pad_left, pad_total - pad_left, cv2.BORDER_CONSTANT, value=(0, 0, 0)))

        final_montages_with_padding = []
        for i, montage in enumerate(padded_montages):
            final_montages_with_padding.append(montage)
            if i < len(padded_montages) - 1:
                final_montages_with_padding.append(np.zeros((PADDING, canvas_w, 3), dtype=np.uint8))

        return cv2.vconcat(final_montages_with_padding)

    except (cv2.error, IndexError, ValueError) as e:
        logger.warning(f"Failed to create debug visualization - {e}", exc_info=True)
        return None

# TODO: remove, deprecated
def create_rotation_debug_image(
        rotation_data: List[Tuple[Optional[int], np.ndarray]],
        best_face_details: Optional[Tuple[np.ndarray, list, np.ndarray]],
        all_detections: List[List[dict]]
) -> np.ndarray:
    """Creates a 2x2 montage of rotated frames with all face detections visualized."""
    if len(rotation_data) != 4:
        return rotation_data[0][1] if rotation_data else np.zeros((480, 640, 3), dtype=np.uint8)
    max_h = max(frame.shape[0] for _, frame in rotation_data)
    max_w = max(frame.shape[1] for _, frame in rotation_data)
    best_landmarks, best_face_box, best_frame = best_face_details if best_face_details else (None, None, None)
    padded_frames = []
    for rotation_idx, (rotation_code, frame) in enumerate(rotation_data):
        canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        h, w, _ = frame.shape
        y_offset = (max_h - h) // 2
        x_offset = (max_w - w) // 2
        canvas[y_offset:y_offset + h, x_offset:x_offset + w] = frame
        rotation_detections = all_detections[rotation_idx] if rotation_idx < len(all_detections) else []
        for detection in rotation_detections:
            face_box = detection['face_box']
            landmarks = detection['landmarks']
            x, y, w, h = face_box
            is_best_detection = (best_face_details is not None and np.array_equal(frame,
                                                                                  best_frame) and face_box == best_face_box)
            bbox_color, landmark_color, thickness = ((0, 255, 0), (0, 255, 0), 3) if is_best_detection else (
            (0, 0, 255), (0, 0, 255), 2)
            cv2.rectangle(canvas, (x + x_offset, y + y_offset), (x + w + x_offset, y + h + y_offset), bbox_color,
                          thickness)
            if landmarks is not None:
                for (lx, ly) in landmarks:
                    cv2.circle(canvas, (int(lx) + x_offset, int(ly) + y_offset), 2, landmark_color, -1)
            confidence = detection.get('confidence', 0.0)
            score = detection.get('score', 0.0)
            text = f"C:{confidence:.2f} S:{score:.2f}"
            text_y = max(y + y_offset - 10, 20)
            cv2.putText(canvas, text, (x + x_offset, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 1)
        padded_frames.append(canvas)
    top_row = cv2.hconcat([padded_frames[0], padded_frames[1]])
    bottom_row = cv2.hconcat([padded_frames[2], padded_frames[3]])
    montage = cv2.vconcat([top_row, bottom_row])
    return montage


def _create_labeled_detail_image(
        name: str,
        image: np.ndarray,
        target_panel_height: int,
        target_panel_width: int
) -> Optional[np.ndarray]:
    """
    Helper function to resize an image into a fixed-size panel, preserving
    aspect ratio, and adding a labeled bar below with its original dimensions.
    """
    if image is None or image.size == 0:
        return None

    LABEL_BAR_HEIGHT = 20
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    TEXT_COLOR = (255, 255, 255)

    image_canvas_height = target_panel_height - LABEL_BAR_HEIGHT

    img_bgr = _to_displayable_bgr(image)
    image_panel = _resize_and_pad(img_bgr, (target_panel_width, image_canvas_height))

    label_bar = np.zeros((LABEL_BAR_HEIGHT, target_panel_width, 3), dtype=np.uint8)

    original_h, original_w = image.shape[:2]
    dim_text = f"({original_w}x{original_h})"

    cv2.putText(label_bar, name, (5, 15), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    text_size, _ = cv2.getTextSize(dim_text, FONT, FONT_SCALE, FONT_THICKNESS)
    text_x = target_panel_width - text_size[0] - 5
    cv2.putText(label_bar, dim_text, (text_x, 15), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return cv2.vconcat([image_panel, label_bar])


def _create_metrics_panel(
    numerical_features: Dict[str, any],
    target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Creates a metrics panel with two vertical columns, splitting the features evenly.
    Respects target_size (width, height).
    """
    target_w, target_h = target_size
    panel = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    y_start = 25
    line_spacing = 25
    items_per_column = 9
    column_width = target_w // 2

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    default_color = (255, 255, 255)
    red_color = (0, 0, 255)
    green_color = (0, 255, 0)

    feature_items = list(numerical_features.items())

    for i, (key, value) in enumerate(feature_items):
        # Stop after filling two columns
        if i >= items_per_column * 2:
            break

        # Determine column (0 or 1) and row index within that column
        column_index = i // items_per_column
        row_index = i % items_per_column

        # Calculate x and y position for the text
        x_pos = 15 + (column_index * column_width)
        y_pos = y_start + (row_index * line_spacing)

        display_key = key.replace('_', ' ').title()
        if isinstance(value, float):
            display_value = f"{value:.2f}s" if 'timestamp' in key.lower() else f"{value:.4f}"
        elif isinstance(value, bool) or (isinstance(value, int) or (isinstance(value, float)) and key in bool_indicators):
            display_value = "Yes" if value else "No"
        else:
            display_value = str(value)
        text = f"{display_key}: {display_value}"

        color = default_color
        if key in bool_indicators:
            color = red_color if value else green_color

        cv2.putText(panel, text, (x_pos, y_pos), FONT, font_scale, color, thickness, cv2.LINE_AA)

    return panel


def _create_metrics_minimal_vertical(
        numerical_features: Dict[str, any],
        target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Metrics panel in a **single column**. Respects target_size (width, height).
    """
    target_w, target_h = target_size
    panel = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    y_pos = 25
    line_spacing = 25

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    default_color = (255, 255, 255)
    red_color = (0, 0, 255)
    green_color = (0, 255, 0)

    for key, value in numerical_features.items():
        display_key = key.replace('_', ' ').title()
        if isinstance(value, float):
            display_value = f"{value:.2f}s" if 'timestamp' in key.lower() else f"{value:.4f}"
        elif isinstance(value, bool) or (isinstance(value, int) and key in [bool_indicators]):
            display_value = "Yes" if value else "No"
        else:
            display_value = str(value)
        text = f"{display_key}: {display_value}"

        color = default_color
        if key in bool_indicators:
            color = red_color if value else green_color

        cv2.putText(panel, text, (15, y_pos), FONT, font_scale, color, thickness, cv2.LINE_AA)
        y_pos += line_spacing  # Adjust spacing as needed

    return panel

def _concat_frames_horizontally_with_padding(frames_to_pad: list, padding_color=(0, 0, 0)):
    """Concatenates images horizontally, padding smaller images to match the tallest."""
    frames = [f for f in frames_to_pad if f is not None]
    if not frames:
        return None
    max_height = max(frame.shape[0] for frame in frames)
    padded_frames = []
    for frame in frames:
        h, w = frame.shape[:2]
        if h < max_height:
            padding_height = max_height - h
            top_padding = padding_height // 2
            bottom_padding = padding_height - top_padding
            padded_frame = cv2.copyMakeBorder(frame, top_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT,
                                              value=padding_color)
            padded_frames.append(padded_frame)
        else:
            padded_frames.append(frame)
    return cv2.hconcat(padded_frames)


def _to_displayable_bgr(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Converts any image into a displayable 3-channel BGR uint8 format."""
    if img is None or img.size == 0:
        return None
    if np.issubdtype(img.dtype, np.floating):
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _resize_and_pad(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resizes an image to fit target dimensions while preserving aspect ratio."""
    target_w, target_h = target_size
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset, x_offset = (target_h - new_h) // 2, (target_w - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def _create_info_panel(
        scalar_features: Dict[str, any],
        target_size: Tuple[int, int]
) -> np.ndarray:
    """Creates a black panel with text for scalar features like EAR, MAR, timestamp."""
    target_w, target_h = target_size
    panel = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_pos = 30
    for key, value in scalar_features.items():
        if isinstance(value, float):
            text = f"{key.replace('_', ' ').title()}: {value:.3f}"
        else:
            text = f"{key.replace('_', ' ').title()}: {value}"
        cv2.putText(panel, text, (15, y_pos), UI.FONT, 0.7, UI.DEFAULT_COLOR, 1, cv2.LINE_AA)
        y_pos += 30
    return panel


def _resize_and_pad_with_ratio(image: np.ndarray, target_aspect_ratio: float, background_color=(0, 0, 0)) -> np.ndarray:
    """
    Resizes an image to fit a target aspect ratio by adding padding.
    Does not distort the image.

    Args:
        image: The input image as a NumPy array.
        target_aspect_ratio: The desired aspect ratio (width / height).
        background_color: The color for the padding (default is black).

    Returns:
        The padded image as a NumPy array.
    """
    img_h, img_w = image.shape[:2]
    img_aspect_ratio = img_w / img_h

    # If the aspect ratios are already very close, no need to pad.
    if abs(img_aspect_ratio - target_aspect_ratio) < 0.01:
        return image

    # Case 1: The image is WIDER than the target (needs vertical padding)
    if img_aspect_ratio > target_aspect_ratio:
        new_h = int(img_w / target_aspect_ratio)
        pad_top = (new_h - img_h) // 2
        pad_bottom = new_h - img_h - pad_top
        padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, 0, 0,
                                          cv2.BORDER_CONSTANT, value=background_color)
    # Case 2: The image is TALLER than the target (needs horizontal padding)
    else:
        new_w = int(img_h * target_aspect_ratio)
        pad_left = (new_w - img_w) // 2
        pad_right = new_w - img_w - pad_left
        padded_image = cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right,
                                          cv2.BORDER_CONSTANT, value=background_color)

    return padded_image
