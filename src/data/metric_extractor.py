import logging
from typing import Optional, Tuple, Dict

import numpy as np

from src.core.constants import LandmarkIndices, LandmarkMethod

logger = logging.getLogger(__name__)


class MetricExtractor:
    def __init__(self,
                 landmark_method: LandmarkMethod):

        self.landmark_method = landmark_method

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

    def map_raw_features(self,
                             frame: np.ndarray,
                             landmarks: np.ndarray,
                             face_box: np.ndarray,
                             timestamp: float,
                             frame_idx: int
                             ) ->  Optional[Dict]:
        """
        Performs the initial, lightweight extraction of essential features and raw image regions.
        This data serves as the input for all downstream feature engineering.
        """
        try:
            x, y, w, h = face_box
            face_crop = frame[y:y + h, x:x + w]

            base_features = {
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'face_crop': face_crop,
                'landmarks': landmarks,
                'face_coords': face_box
            }
            return base_features

        except Exception as e:
            logger.error(f"Error during base feature extraction: {e}", exc_info=True)
            return None


    def extract_all_rois(self, gray_frame: np.ndarray, landmarks: np.ndarray) -> Tuple[
        Dict[str, Optional[np.ndarray]], Dict[str, Optional[np.ndarray]]]:
        """
        Extracts facial ROIs, returning both the adaptively padded versions
        and the raw, unpadded versions for comparison.

        Returns:
            A tuple containing two dictionaries: (padded_rois, unpadded_rois).
        """
        frame_h, frame_w = gray_frame.shape

        def get_roi_and_coords(points: np.ndarray, base_pad: float = 0.2) -> Tuple[
            Optional[np.ndarray], Optional[Tuple]]:
            """
            Helper to calculate padded/unpadded coordinates and return the padded crop.
            Returns: A tuple of (padded_crop, unpadded_coords).
            """
            if points is None or points.size == 0:
                return None, None

            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            w, h = x_max - x_min, y_max - y_min

            if w == 0 or h == 0:
                return None, None

            aspect_ratio = w / h
            pad_x = int(w * base_pad * (1.0 + 0.1 * (1.0 - aspect_ratio)))
            pad_y = int(h * base_pad * (1.0 + 0.1 * aspect_ratio))

            x1_pad = max(0, x_min - pad_x)
            y1_pad = max(0, y_min - pad_y)
            x2_pad = min(frame_w, x_max + pad_x)
            y2_pad = min(frame_h, y_max + pad_y)

            padded_crop = gray_frame[y1_pad:y2_pad, x1_pad:x2_pad]

            unpadded_coords = (int(x_min), int(y_min), int(x_max), int(y_max))

            return padded_crop, unpadded_coords

        padded_rois = {}
        unpadded_rois = {}

        roi_definitions = {
            'left_eye': (landmarks[self.left_eye_indices], 0.2),
            'right_eye': (landmarks[self.right_eye_indices], 0.2),
            'mouth': (landmarks[self.mouth_indices], 0.3)
        }

        for name, (points, base_pad) in roi_definitions.items():
            padded_crop, unpadded_coords = get_roi_and_coords(points, base_pad)

            padded_rois[name] = padded_crop

            if unpadded_coords:
                x1, y1, x2, y2 = unpadded_coords
                if x2 > x1 and y2 > y1:
                    unpadded_rois[name] = gray_frame[y1:y2, x1:x2]
                else:
                    unpadded_rois[name] = None
            else:
                unpadded_rois[name] = None

        return padded_rois, unpadded_rois


    def _calc_ear_mediapipe(self, landmarks: np.ndarray, eye_side: str = 'left') -> float:

        if eye_side == 'left':
            default_pairs = [
                # inner to outer
                (382, 398),
                (381, 384),
                (380, 385),
                (374, 386),
                (373, 387),
                (390, 388),
                (249, 466)
            ]
            default_horiz = (263, 362)
        elif eye_side == 'right':
            default_pairs = [
                # inner to outer
                (155, 173),
                (154, 157),
                (153, 158),
                (145, 159),
                (144, 160),
                (163, 161),
                (7, 246),
            ]
            default_horiz = (33, 133)
        else:
            raise ValueError('eye_side must be "left" or "right"')

        verticals = [np.linalg.norm(landmarks[i] - landmarks[j]) for i, j in default_pairs]
        h = np.linalg.norm(landmarks[default_horiz[0]] - landmarks[default_horiz[1]])
        if h < 1e-6:
            return 0.0
        ear = sum(verticals) / (len(verticals) * h)
        return ear

    def _calc_ear_lbf(self, landmarks: np.ndarray, eye_side: str = 'left') -> float:

        if eye_side == 'left':
            default_pairs = [(37, 41), (38, 40)]  # 2 classic verticals
            default_horiz = (36, 39)
        elif eye_side == 'right':
            default_pairs = [(43, 47), (44, 46)]  # 2 classic verticals
            default_horiz = (42, 45)
        else:
            raise ValueError('eye_side must be "left" or "right"')

        verticals = [np.linalg.norm(landmarks[i] - landmarks[j]) for i, j in default_pairs]
        h = np.linalg.norm(landmarks[default_horiz[0]] - landmarks[default_horiz[1]])
        if h < 1e-6:
            return 0.0
        ear = sum(verticals) / (len(verticals) * h)
        return ear


    def extract_ear(self, landmarks: np.ndarray) -> float:

        if self.landmark_method == LandmarkMethod.LBF:
            right_ear = self._calc_ear_lbf(landmarks, eye_side='right')
            left_ear = self._calc_ear_lbf(landmarks, eye_side='left')
        elif self.landmark_method == LandmarkMethod.MEDIAPIPE:
            right_ear = self._calc_ear_mediapipe(landmarks, eye_side='right')
            left_ear = self._calc_ear_mediapipe(landmarks, eye_side='left')
        else:
            raise ValueError(f"[MetricExtractor] Invalid landmark method: {self.landmark_method.value}")

        valid_ears = []

        # A single eye's EAR should not be absurdly high. This filters out errors from head turning.
        # TODO: remove? in latest runs seems to be fixed (for MediaPipe Face Mesh at least)
        MAX_INDIVIDUAL_EAR = 0.5

        if 0.0 < left_ear < MAX_INDIVIDUAL_EAR:
            valid_ears.append(left_ear)

        if 0.0 < right_ear < MAX_INDIVIDUAL_EAR:
            valid_ears.append(right_ear)

        if valid_ears:
            return np.mean(valid_ears) if len(valid_ears) > 0 else 0.0

        return 0.0


    def extract_mar(self, landmarks: np.ndarray) -> float:
        if self.landmark_method == LandmarkMethod.LBF:
            return self._extract_mar_lbf(landmarks)
        elif self.landmark_method == LandmarkMethod.MEDIAPIPE:
            return self._extract_mar_mediapipe(landmarks)
        else:
            raise NotImplementedError(f"MAR calculation not implemented for {self.landmark_method.value}")

    def _extract_mar_lbf(self, landmarks: np.ndarray) -> float:
        mouth_points = landmarks[LandmarkIndices.LBF_MOUTH]
        vertical_dist1 = np.linalg.norm(mouth_points[3] - mouth_points[9])
        vertical_dist2 = np.linalg.norm(mouth_points[2] - mouth_points[10])
        horizontal_dist = np.linalg.norm(mouth_points[0] - mouth_points[6])
        if horizontal_dist == 0:
            return 0.0
        return (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)

    def _extract_mar_mediapipe(self, landmarks: np.ndarray) -> float:
        mouth_corner_left_idx, mouth_corner_right_idx = 61, 291
        inner_lip_upper_center_idx, inner_lip_lower_center_idx = 13, 14
        inner_lip_upper_left_idx, inner_lip_lower_left_idx = 82, 87
        inner_lip_upper_right_idx, inner_lip_lower_right_idx = 312, 317

        vertical_dist1 = np.linalg.norm(landmarks[inner_lip_upper_center_idx] - landmarks[inner_lip_lower_center_idx])
        vertical_dist2 = np.linalg.norm(landmarks[inner_lip_upper_left_idx] - landmarks[inner_lip_lower_left_idx])
        vertical_dist3 = np.linalg.norm(landmarks[inner_lip_upper_right_idx] - landmarks[inner_lip_lower_right_idx])
        horizontal_dist = np.linalg.norm(landmarks[mouth_corner_left_idx] - landmarks[mouth_corner_right_idx])

        if horizontal_dist == 0:
            return 0.0
        return (vertical_dist1 + vertical_dist2 + vertical_dist3) / (3.0 * horizontal_dist)


    def extract_eye_region(self, frame: np.ndarray, landmarks: np.ndarray) -> Optional[np.ndarray]:
        left_eye_lm = landmarks[self.left_eye_indices]
        right_eye_lm = landmarks[self.right_eye_indices]

        eye_boxes = []
        for eye_lm in [left_eye_lm, right_eye_lm]:
            x_coords = eye_lm[:, 0]
            y_coords = eye_lm[:, 1]
            x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
            y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

            pad_x = int((x_max - x_min) * 1) # 0.5
            pad_y = int((y_max - y_min) * 1) # 0.7

            x1 = max(0, x_min - pad_x)
            y1 = max(0, y_min - pad_y)
            x2 = min(frame.shape[1], x_max + pad_x)
            y2 = min(frame.shape[0], y_max + pad_y)
            eye_boxes.append((x1, y1, x2, y2))

        if not eye_boxes:
            return None

        combined_xmin = min(b[0] for b in eye_boxes)
        combined_ymin = min(b[1] for b in eye_boxes)
        combined_xmax = max(b[2] for b in eye_boxes)
        combined_ymax = max(b[3] for b in eye_boxes)

        eye_region = frame[combined_ymin:combined_ymax, combined_xmin:combined_xmax]
        if eye_region.size == 0:
            return None
        return eye_region
