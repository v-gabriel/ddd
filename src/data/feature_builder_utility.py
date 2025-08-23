import logging
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
import pandas as pd
from skimage import exposure
from skimage.feature import hog

from src.config import ExperimentConfig
from src.core.constants import LandmarkMethod, LandmarkIndices
from src.data.metric_extractor import MetricExtractor

logger = logging.getLogger(__name__)

class FeatureBuilderUtility:

    def __init__(self,
                 landmark_method: LandmarkMethod,
                 config: ExperimentConfig):
        self.metric_extractor = MetricExtractor(landmark_method)

        self.config = config

        self.left_eye_indices, self.right_eye_indices, self.mouth_indices = None, None, None
        self.nose_tip_indice, self.chin_indice, self.left_eye_corner_indice, self.right_eye_corner_indice = None, None, None, None
        if landmark_method == LandmarkMethod.LBF:
            self.left_eye_indices = LandmarkIndices.LBF_LEFT_EYE
            self.right_eye_indices = LandmarkIndices.LBF_RIGHT_EYE
            self.mouth_indices = LandmarkIndices.LBF_MOUTH
        elif landmark_method == LandmarkMethod.MEDIAPIPE:
            self.left_eye_indices = LandmarkIndices.MEDIAPIPE_LEFT_EYE
            self.right_eye_indices = LandmarkIndices.MEDIAPIPE_RIGHT_EYE
            self.mouth_indices = LandmarkIndices.MEDIAPIPE_MOUTH
            self.nose_tip_indice = 1
            self.chin_indice = 18
            self.left_eye_corner_indice = 33
            self.right_eye_corner_indice = 263
        else:
            raise ValueError(f"Unsupported landmark method: {landmark_method.value}")

        self.all_eye_indices = np.concatenate([self.left_eye_indices, self.right_eye_indices])
        self.roi_definitions = {
            'left_eye': (self.left_eye_indices, 0.2),
            'right_eye': (self.right_eye_indices, 0.2),
            'mouth': (self.mouth_indices, 0.3)
        }

    def get_padded_roi(
            self,
            points_global: np.ndarray,
            base_pad: float,
            offset_x: int,
            offset_y: int,
            crop_w: int,
            crop_h: int,
            face_crop: np.ndarray
    ) -> np.ndarray | None:
        """
        Helper to extract a single padded ROI from the face_crop.
        """
        if points_global is None or points_global.size == 0:
            return None

        points_local = points_global - np.array([offset_x, offset_y])

        x_min, y_min = np.min(points_local, axis=0)
        x_max, y_max = np.max(points_local, axis=0)
        w, h = x_max - x_min, y_max - y_min

        if w <= 0 or h <= 0:
            return None

        pad_w = int(w * base_pad)
        pad_h = int(h * base_pad)

        x1_unclamped = int(x_min - pad_w)
        y1_unclamped = int(y_min - pad_h)
        x2_unclamped = int(x_max + pad_w)
        y2_unclamped = int(y_max + pad_h)

        x1_clamped = max(0, x1_unclamped)
        y1_clamped = max(0, y1_unclamped)
        x2_clamped = min(crop_w, x2_unclamped)
        y2_clamped = min(crop_h, y2_unclamped)

        if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
            return None

        return face_crop[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

    def build_rois_from_crop(
            self,
            face_crop: np.ndarray,
            landmarks: np.ndarray,
            face_coords: tuple,
    ) -> dict:
        if face_crop is None or landmarks is None or face_coords is None:
            return {'rois': {}}

        offset_x, offset_y, _, _ = face_coords
        crop_h, crop_w = face_crop.shape[:2]
        rois_data = {}

        for name, (indices, base_pad) in self.roi_definitions.items():
            points = landmarks[indices]
            rois_data[name] = self.get_padded_roi(
                points, base_pad, offset_x, offset_y, crop_w, crop_h, face_crop
            )

        all_eye_points = landmarks[self.all_eye_indices]
        rois_data['eye_region'] = self.get_padded_roi(
            all_eye_points, 0.25, offset_x, offset_y, crop_w, crop_h, face_crop
        )

        return {'rois': rois_data}

    def process_cnn_features(self, rois: dict, face_crop: np.array) -> Dict:
        """Takes base features and prepares them for a CNN model."""

        cnn_face = self.resize_with_padding(face_crop, self.config.SETUP_FACE_ROI_SIZE_FACE) \
            if face_crop is not None and face_crop.size > 0 else None
        cnn_eye_left =  self.resize_with_padding(rois.get("left_eye"), self.config.SETUP_FACE_ROI_SIZE_EYE) \
            if rois.get("left_eye") is not None and rois.get("left_eye").size > 0 else None
        cnn_eye_right =  self.resize_with_padding(rois.get("right_eye"), self.config.SETUP_FACE_ROI_SIZE_EYE) \
            if rois.get("right_eye") is not None and rois.get("right_eye").size > 0 else None
        cnn_eye_region =  self.resize_with_padding(rois.get("eye_region"), self.config.SETUP_FACE_ROI_SIZE_EYE_REGION) \
            if rois.get("eye_region") is not None and rois.get("eye_region").size > 0 else None
        cnn_mouth =  self.resize_with_padding(rois.get("mouth"), self.config.SETUP_FACE_ROI_SIZE_MOUTH) \
            if rois.get("mouth") is not None and rois.get("mouth").size > 0 else None

        # cnn_face = cv2.resize(face_crop , self.config.FACE_ROI_SIZE_FACE, interpolation=cv2.INTER_AREA) \
        #     if face_crop is not None and face_crop.size > 0 else None
        # cnn_eye_left = cv2.resize(rois.get("left_eye"), self.config.FACE_ROI_SIZE_EYE, interpolation=cv2.INTER_AREA) \
        #     if rois.get("left_eye") is not None and rois.get("left_eye").size > 0 else None
        # cnn_eye_right = cv2.resize(rois.get("right_eye"), self.config.FACE_ROI_SIZE_EYE, interpolation=cv2.INTER_AREA) \
        #     if rois.get("right_eye") is not None and rois.get("right_eye").size > 0 else None
        # cnn_eye_region = cv2.resize(rois.get("eye_region"), self.config.FACE_ROI_SIZE_EYE, interpolation=cv2.INTER_AREA) \
        #     if rois.get("eye_region") is not None and rois.get("eye_region").size > 0 else None
        # cnn_mouth = cv2.resize(rois.get("mouth"), self.config.FACE_ROI_SIZE_MOUTH, interpolation=cv2.INTER_AREA) \
        #     if rois.get("mouth") is not None and rois.get("mouth").size > 0 else None

        cnn_features = {
            'cnn_mouth': cnn_mouth,
            'cnn_eye_right': cnn_eye_right,
            'cnn_eye_left': cnn_eye_left,
            'cnn_eye_region': cnn_eye_region,
            'cnn_face': cnn_face,
        }

        return cnn_features

    def process_hog_features(self, rois: dict, visualize: bool = False) -> Tuple[Dict, Optional[Dict]]:
        """Takes base features and calculates HOG features."""

        hog_dict, hog_visualization = self.extract_hog_features(rois, visualize)
        return hog_dict, hog_visualization

    def process_head_pose_features(self, df: pd.Series, visualize: bool = False):
        """Takes base features and calculates head pose."""

        head_pose, head_pose_visualization = self.extract_head_pose_features(
            landmarks=df['landmarks'],
            face_crop=df.get('face_crop', None) if visualize else None,
            face_coords=df.get('face_coords', None) if visualize else None,
            visualize=visualize
        )
        return head_pose, head_pose_visualization

    def calculate_perclos_from_array(self, is_closed: np.ndarray, timestamp_array: np.ndarray) -> np.ndarray:
        """
            Calculates PERCLOS over the is_closed array using a rolling window.
        """
        n_frames = len(is_closed)

        window_duration_seconds = self.config.SETUP_PERCLOS_WINDOW_SECONDS

        time_diffs_sec = np.diff(timestamp_array)
        time_diffs_sec[time_diffs_sec == 0] = 1e-6
        median_frame_time = np.median(time_diffs_sec)

        perclos_window_frames = int(window_duration_seconds / median_frame_time) if time_diffs_sec.size > 0 else 0
        if perclos_window_frames == 0:
            return np.zeros(n_frames)

        df = pd.DataFrame({'is_closed': is_closed.astype(float)})
        perclos = df['is_closed'].rolling(
            window=perclos_window_frames,
            min_periods=1  # Start calculating even if the window isn't full
        ).mean().to_numpy()

        return perclos

    def calculate_eye_closed_signal(self,
        ear_norm: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        timestamp: np.ndarray,
        calib_velocities: List[float]
    ):

        n_frames = len(ear_norm)

        if calib_velocities is not None:
            mean_vel = np.mean(calib_velocities)
            std_vel = np.std(calib_velocities) or 0.1
        else:
            logger.warning("[calculate_eye_closed_signal] No calibration velocities were passed.")
            mean_vel, std_vel = 0.0, 0.1

        norm_velocity = (velocity - mean_vel) / std_vel
        # norm_velocity = velocity

        # -1.50     -1.50
        #  1.00      1.00
        # -1.50     -1.25
        #  1.00      1.00
        v_close_thresh = -1.5 * std_vel
        v_open_thresh = 1.00 * std_vel
        a_thresh = -1.25 * std_vel
        v_swing_min = 1.00 * std_vel

        max_blink_duration_sec = 1.00

        is_closing = (norm_velocity < v_close_thresh) & (acceleration < a_thresh) & (ear_norm < self.config.SETUP_CLOSURE_EAR_THRESHOLD)
        is_opening = (norm_velocity > v_open_thresh) | (acceleration > -a_thresh)
        stable_closed = (ear_norm < self.config.SETUP_EAR_NORMALIZED_THRESHOLD) & (np.abs(norm_velocity) < v_swing_min)
        stable_open = (ear_norm >= self.config.SETUP_EAR_NORMALIZED_THRESHOLD) & (np.abs(norm_velocity) < v_swing_min)

        is_eye_closed_signal = (is_closing | stable_closed).astype(int)

        sign_changes = np.diff(np.sign(norm_velocity), prepend=0)
        closure_starts = np.where(sign_changes < 0)[0]
        for start in closure_starts:
            if np.abs(norm_velocity[start]) >= v_swing_min:
                end = start + 1
                closure_start_time = timestamp[start]
                while end < n_frames:
                    duration = timestamp[end] - closure_start_time
                    if duration > max_blink_duration_sec or is_opening[end] or stable_open[end] or ear_norm[
                        end] >= self.config.SETUP_EAR_NORMALIZED_THRESHOLD:
                        break  # Explicit break on EAR recovery or max duration
                    is_eye_closed_signal[end] = 1
                    end += 1

        return is_eye_closed_signal, is_closing, is_opening, stable_closed, stable_open

    def calculate_eye_closure_acceleration(self, velocity: np.ndarray, timestamp: np.ndarray) -> np.ndarray:
        """
        Calculates the instantaneous acceleration of eye closure (derivative of velocity).

        Args:
            velocity: Array of eye closure velocities.
            timestamp: Array of timestamps (in seconds) for each frame.

        Returns:
            np.ndarray: Acceleration array for each frame.
        """
        if len(velocity) < 2:
            return np.zeros_like(velocity, dtype=float)

        d_vel = np.diff(velocity)
        d_time = np.diff(timestamp)
        d_time[d_time == 0] = 1e-6

        # Acceleration is d(velocity)/d(t)
        acceleration = d_vel / d_time

        full_acceleration = np.concatenate(([0.0], acceleration))

        return full_acceleration

    def calculate_eye_closure_velocity(self, ear_array: np.ndarray, timestamp_array: np.ndarray) -> np.ndarray:
        """
            Calculates the instantaneous velocity of eye closure.
        """
        if len(ear_array) < 2:
            return np.zeros_like(ear_array, dtype=float)

        d_ear = np.diff(ear_array)
        d_time_sec = np.diff(timestamp_array)
        d_time_sec[d_time_sec == 0] = 1e-6

        # Velocity is d(EAR)/d(t)
        velocity = d_ear / d_time_sec

        full_velocity = np.concatenate(([0.0], velocity))

        df = pd.DataFrame({'velocity': full_velocity})
        return df.to_numpy()

    def analyze_blinks_from_array(self, is_closed: np.ndarray, timestamp_array: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Analyzes blink patterns, calculating a robust blink rate per minute.
        This function assumes the input timestamp_array is in SECONDS.
        """
        n_frames = len(is_closed)
        if n_frames < 2:
            return np.zeros(n_frames), np.zeros(n_frames)

        time_diffs_sec = np.diff(timestamp_array)
        time_diffs_sec[time_diffs_sec == 0] = 1e-6
        median_frame_time = np.median(time_diffs_sec)

        window_duration_seconds = self.config.SETUP_BLINK_WINDOW_SECONDS
        blink_window_frames = int(window_duration_seconds / median_frame_time) if time_diffs_sec.size > 0 else 0
        if blink_window_frames == 0:
            return np.zeros(n_frames), np.zeros(n_frames)

        changes = np.diff(is_closed.astype(int), prepend=is_closed[0])
        blink_starts = np.where(changes == 1)[0]
        blink_ends = np.where(changes == -1)[0]

        if len(blink_starts) == 0:
            return np.zeros(n_frames), np.zeros(n_frames)

        # Correctly pair up blink start and end events
        paired_blinks = []
        end_idx = 0
        for start in blink_starts:
            possible_ends = blink_ends[blink_ends > start]
            if len(possible_ends) > 0:
                paired_blinks.append((start, possible_ends[0]))

        if not paired_blinks:
            return np.zeros(n_frames), np.zeros(n_frames)

        blink_end_indices = np.array([end for _, end in paired_blinks])
        blink_durations_sec = np.array([timestamp_array[end] - timestamp_array[start] for start, end in paired_blinks])

        blink_count_signal = np.zeros(n_frames)
        blink_duration_signal = np.zeros(n_frames)

        np.add.at(blink_count_signal, blink_end_indices, 1)
        np.add.at(blink_duration_signal, blink_end_indices, blink_durations_sec)

        df = pd.DataFrame({'count': blink_count_signal, 'duration': blink_duration_signal})
        rolling_sum = df.rolling(window=blink_window_frames, min_periods=1).sum()

        # Calculate blink rate per MINUTE
        blink_rate_per_min = (rolling_sum['count'] / window_duration_seconds) * 60

        # Calculate average blink duration in the window
        avg_blink_dur = (rolling_sum['duration'] / rolling_sum['count']).fillna(0)

        return blink_rate_per_min.to_numpy(), avg_blink_dur.to_numpy()

    def calculate_microsleeps_from_array(self, is_closed: np.ndarray, timestamp_array: np.ndarray) -> np.ndarray:
        """
            Detects microsleeps (long eye closures) in the is_closed signal.
        """
        microsleeps = np.zeros(len(is_closed))
        microsleep_duration_sec = 0.5  # Standard 500ms threshold, per literature

        changes = np.diff(is_closed.astype(int), prepend=0)
        closure_starts = np.where(changes == 1)[0]
        closure_ends = np.where(changes == -1)[0]

        if len(closure_starts) == 0 or len(closure_ends) == 0:
            return microsleeps

        for start in closure_starts:
            end_candidates = closure_ends[closure_ends > start]
            if len(end_candidates) > 0:
                end = end_candidates[0]
                duration = timestamp_array[end] - timestamp_array[start]
                if duration >= microsleep_duration_sec:
                    # Mark all frames during this microsleep as 1.0
                    microsleeps[start:end] = 1.0

        return microsleeps.astype(int)

    def resize_with_padding(self, image: np.ndarray, target_size: tuple) -> np.ndarray | None:
        """
        Resizes an image to a target size by scaling it while preserving the
        aspect ratio and padding the remaining space with black pixels.

        Args:
            image: The input image (ROI).
            target_size: A tuple (width, height) for the final output image.

        Returns:
            The resized and padded image, or None if the input is invalid.
        """
        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]
        if w <= 0 or h <= 0:
            return None

        target_w, target_h = target_size

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w <= 0 or new_h <= 0:
            return None

        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if resized_image.ndim == 2:
            padded_image = np.zeros((target_h, target_w), dtype=resized_image.dtype)
        else:
            padded_image = np.zeros((target_h, target_w, resized_image.shape[2]), dtype=resized_image.dtype)

        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        padded_image[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_image

        return padded_image

    def extract_head_pose_features(
            self,
            landmarks: np.ndarray,
            face_crop: np.ndarray,
            face_coords: np.ndarray,
            visualize: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Approximates 3D head pose and optionally returns a visualization.
        This function uses the landmark indices defined in the __init__ method.

        Args:
            landmarks: The array of 3D or 2D facial landmarks (2D landmarks may not be precise or be able to extract all features).
            face_crop: Detected face crop.
            face_coords: Face coordinates in relation to the original frame.
            visualize: If True, a visualization image is created and returned.

        Returns:
            A tuple containing:
            - A NumPy array with the 3 normalized pose features [roll, pitch, yaw].
            - The visualization frame (or None if visualize=False).
        """
        yaw_p1, yaw_p2, yaw_p3 = None, None, None

        # E.g. MTCNN, advanced with more indices / 3D view
        if self.nose_tip_indice is not None:
            nose_tip = landmarks[self.nose_tip_indice]
            left_eye_corner = landmarks[self.left_eye_corner_indice]
            right_eye_corner = landmarks[self.right_eye_corner_indice]
            chin = landmarks[self.chin_indice]

            eye_line_vector = right_eye_corner - left_eye_corner
            nose_chin_vector = chin - nose_tip
            head_tilt = np.arctan2(eye_line_vector[1], eye_line_vector[0])
            head_nod = np.arctan2(nose_chin_vector[1], nose_chin_vector[0])
            head_turn = np.linalg.norm(left_eye_corner - nose_tip) - np.linalg.norm(right_eye_corner - nose_tip)

            roll_p1, roll_p2 = left_eye_corner.astype(int), right_eye_corner.astype(int) # valjanje
            pitch_p1, pitch_p2 = nose_tip.astype(int), chin.astype(int) # nagib
            yaw_p1, yaw_p2, yaw_p3 = nose_tip.astype(int), left_eye_corner.astype(int), right_eye_corner.astype(int) # zakretanje
        else:
            left_eye_center = np.mean(landmarks[self.left_eye_indices], axis=0)
            right_eye_center = np.mean(landmarks[self.right_eye_indices], axis=0)
            mouth_center = np.mean(landmarks[self.mouth_indices], axis=0)
            face_center_top = (left_eye_center + right_eye_center) / 2

            eye_line_vector = right_eye_center - left_eye_center
            face_center_vector = mouth_center - face_center_top
            head_tilt = np.arctan2(eye_line_vector[1], eye_line_vector[0])
            head_nod = np.arctan2(face_center_vector[1], face_center_vector[0])
            head_turn = 0.0

            roll_p1, roll_p2 = left_eye_center.astype(int), right_eye_center.astype(int)
            pitch_p1, pitch_p2 = face_center_top.astype(int), mouth_center.astype(int)

        # Normalize features to a consistent [-1, 1] range
        head_tilt_norm = np.clip(head_tilt, -np.pi / 2, np.pi / 2) / (np.pi / 2)
        head_nod_norm = np.clip(head_nod, -np.pi / 2, np.pi / 2) / (np.pi / 2)
        head_turn_norm = np.clip(head_turn, -50, 50) / 50 if head_turn != 0.0 else 0.0

        vis_frame = None
        if visualize:
            offset_x, offset_y, _, _ = face_coords

            def translate_point(p):
                if p is None: return None
                x_local = int(p[0] - offset_x)
                y_local = int(p[1] - offset_y)

                if 0 <= x_local < face_crop.shape[1] and 0 <= y_local < face_crop.shape[0]:
                    return (x_local, y_local)
                return None

            # Translate all vector points using the calculated offset
            roll_p1_local = translate_point(roll_p1)
            roll_p2_local = translate_point(roll_p2)
            pitch_p1_local = translate_point(pitch_p1)
            pitch_p2_local = translate_point(pitch_p2)
            yaw_p1_local = translate_point(yaw_p1)
            yaw_p2_local = translate_point(yaw_p2)
            yaw_p3_local = translate_point(yaw_p3)

            vis_frame = face_crop.copy()

            if roll_p1_local and roll_p2_local:
                cv2.line(vis_frame, roll_p1_local, roll_p2_local, (0, 255, 0), 2)

            if pitch_p1_local and pitch_p2_local:
                cv2.line(vis_frame, pitch_p1_local, pitch_p2_local, (255, 0, 0), 2)

            if yaw_p1_local and yaw_p2_local and yaw_p3_local:
                cv2.line(vis_frame, yaw_p1_local, yaw_p2_local, (0, 0, 255), 1)
                cv2.line(vis_frame, yaw_p1_local, yaw_p3_local, (0, 0, 255), 1)

            # y_pos, font_scale = 20, 0.5
            # cv2.putText(vis_frame, f"Roll: {head_tilt_norm:.2f}", (5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)
            # y_pos += 20
            # cv2.putText(vis_frame, f"Pitch: {head_nod_norm:.2f}", (5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1)
            # y_pos += 20
            # cv2.putText(vis_frame, f"Yaw: {head_turn_norm:.2f}", (5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)

        feature_vector = np.array([head_tilt_norm, head_nod_norm, head_turn_norm])
        return feature_vector, vis_frame

    def extract_hog_features(self, rois: Dict[str, np.ndarray], visualize: bool = False) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Extracts HOG features for each specified ROI without combining them.

        The function resizes each ROI to a configurable size to ensure all
        output vectors for a given ROI type have a consistent length.

        Args:
            rois: A dictionary of ROI images, e.g., {"left_eye": img, "right_eye": img}.
            visualize: If True, generates visualization images for the HOGs.

        Returns:
            A tuple containing:
            - A dictionary of HOG feature vectors, e.g., {"hog_eye_left": vector}.
            - A dictionary of HOG visualization images.
        """

        hog_params = {
            'orientations': self.config.SETUP_HOG_ORIENTATIONS,
            'pixels_per_cell': self.config.SETUP_HOG_PIXELS_PER_CELL,
            'cells_per_block': self.config.SETUP_HOG_CELLS_PER_BLOCK,
            'block_norm': 'L2-Hys', 'feature_vector': True
        }

        hog_vectors = {}
        hog_visuals = {}

        roi_definitions = {
            'eye_left': (rois.get("left_eye"), self.config.SETUP_FACE_ROI_SIZE_EYE),
            'eye_right': (rois.get("right_eye"), self.config.SETUP_FACE_ROI_SIZE_EYE),
            'mouth': (rois.get("mouth"), self.config.SETUP_FACE_ROI_SIZE_MOUTH)
        }

        for name, (roi_img, target_size) in roi_definitions.items():
            key_name = f"hog_{name}"

            cy, cx = hog_params['pixels_per_cell']
            by, bx = hog_params['cells_per_block']
            min_height = cy * by
            min_width = cx * bx

            h, w = target_size
            new_h = max(h, min_height)
            new_w = max(w, min_width)

            if new_h > h or new_w > w:
                logger.warning(f"ROI bumped from ({h}, {w}) to min required ({new_h}, {new_w}) for HOG params.")
                target_size = (new_h, new_w)

            def handle_blank():
                # If the ROI is missing, calculate the expected HOG vector length
                #  and create a zero-vector to ensure data consistency.

                h, w = target_size
                cy, cx = hog_params['pixels_per_cell']
                by, bx = hog_params['cells_per_block']
                nbins = hog_params['orientations']

                n_cells_y = h // cy
                n_cells_x = w // cx
                n_blocks_y = (n_cells_y - by) + 1
                n_blocks_x = (n_cells_x - bx) + 1

                hog_vector_length = n_blocks_y * n_blocks_x * by * bx * nbins
                hog_vectors[key_name] = np.zeros(hog_vector_length, dtype=np.float32)

            if roi_img is None or roi_img.size == 0:
                handle_blank()
                continue

            if len(roi_img.shape) == 3:
                roi_img = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)

            # preprocessed_roi = cv2.resize(roi_img, target_size, interpolation=cv2.INTER_AREA)
            preprocessed_roi = self.resize_with_padding(roi_img, target_size)

            if preprocessed_roi is None or preprocessed_roi.size == 0:
                handle_blank()
                continue

            preprocessed_roi = cv2.GaussianBlur(preprocessed_roi, (3, 3), 0)

            if visualize:
                fd, hog_image = hog(preprocessed_roi, **hog_params, visualize=True)
                hog_visuals[f"hog_{name}"] = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            else:
                fd = hog(preprocessed_roi, **hog_params, visualize=False)

            hog_vectors[key_name] = fd

        return hog_vectors if hog_vectors else None, hog_visuals

    # TODO: reimplement sobel + circ hough
    def get_sobel_prewitt_area(self, eye_region: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        if eye_region is None or eye_region.size == 0:
            return 0.0, None

        gray = eye_region if len(eye_region.shape) == 2 else cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

        prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        prewitt_x = cv2.filter2D(blurred, cv2.CV_64F, prewitt_kernel_x)
        prewitt_y = cv2.filter2D(blurred, cv2.CV_64F, prewitt_kernel_y)

        edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2 + prewitt_x ** 2 + prewitt_y ** 2)
        edge_mag_normalized = cv2.normalize(edge_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, edge_thresh = cv2.threshold(edge_mag_normalized, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(edge_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area = sum(cv2.contourArea(cnt) for cnt in contours)
        visualization_image = cv2.cvtColor(edge_mag_normalized, cv2.COLOR_GRAY2BGR)

        return area, visualization_image