import concurrent.futures
import logging
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple

from src.core.constants import Paths, LandmarkMethod
from src.visualization.visualizations import save_landmark_custom_failure_debug_image

logger = logging.getLogger(__name__)

class LandmarkDetector:
    """
    Detects facial landmarks using a selectable backend (LBF or MediaPipe).
    """

    def __init__(self, method: LandmarkMethod = LandmarkMethod.LBF):
        """
        Initialize the landmark detector with the specified method.

        Args:
            method: The backend to use, either "lbf" or "mediapipe".
        """
        self.total_detection_time = 0.0
        self.detection_calls = 0
        self.detection_failures = 0
        self.debug_path = None

        self.method: LandmarkMethod = method
        if self.method == LandmarkMethod.LBF:
            self._initialize_lbf()
        elif self.method == LandmarkMethod.MEDIAPIPE:
            self._initialize_mediapipe()
        else:
            raise ValueError(f"Unsupported landmark detection method: {method}")

    def _initialize_lbf(self):
        """Initializes the OpenCV LBF landmark detector."""
        self.facemark = cv2.face.createFacemarkLBF()
        try:
            self.facemark.loadModel(Paths.LBF_LANDMARKER_MODEL)
        except Exception as e:
            raise RuntimeError(f"Failed to load LBF model from {Paths.LBF_LANDMARKER_MODEL}: {e}")

    def _initialize_mediapipe(self):
        """Initializes the MediaPipe Face Landmarker."""
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if not os.path.exists(Paths.MEDIAPIPE_LANDMARKER_MODEL):
            raise FileNotFoundError(f"MediaPipe model not found at {Paths.MEDIAPIPE_LANDMARKER_MODEL}")

        base_options = python.BaseOptions(model_asset_path=Paths.MEDIAPIPE_LANDMARKER_MODEL)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options, num_faces=1,
            output_face_blendshapes=False, output_facial_transformation_matrixes=False
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        # TODO: remove or try on Linux (not supported on Windows)
        # base_options = python.BaseOptions(model_asset_path=Paths.MEDIAPIPE_LANDMARKER_TASK,
        #                                       delegate=python.BaseOptions.Delegate.GPU)  # Enable GPU
        # options = vision.FaceLandmarkerOptions(
        #     base_options=base_options,
        #     running_mode=vision.RunningMode.IMAGE,  # Use IMAGE mode for batch-like processing
        #     num_faces=1,
        #     output_face_blendshapes=False,
        #     output_facial_transformation_matrixes=False
        # )
        # self.landmarker = vision.FaceLandmarker.create_from_options(options)


    def detect_landmarks_batch(self,
                               frames: np.ndarray[np.ndarray],
                               face_rects: np.ndarray[np.ndarray],
                               face_rois: np.ndarray[np.ndarray],
                               max_workers: Optional[int] = max(1, int((os.cpu_count() or 1))),
                               force_sequential: bool = False,
                               debug_path: Optional[str] = None
                               ) -> List[Optional[np.ndarray]]:
        """
        Detects landmarks on a batch of frames in parallel using a ThreadPoolExecutor.

        Args:
            frames (np.ndarray[np.ndarray]): Array of full frames.
            face_rects (np.ndarray[np.ndarray]): Array of face bounding boxes [x, y, w, h].
            face_rois (np.ndarray[np.ndarray]): Array of face rois.
            max_workers (int): The number of threads to use for processing.
            force_sequential (bool): Skips the usage of ThreadPoolExecutor, processes using a simple for loop.
            debug_path (str): Debug path if validation fails or landmarks are not detected.

        Returns:
            A list containing detected landmarks (np.ndarray) or None for each input frame.
        """
        if len(frames) != len(face_rects):
            raise ValueError("The number of frames must match the number of face rectangles.")

        start_time = time.perf_counter()

        tasks = zip(frames, face_rects, face_rois)

        batch_results: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        if force_sequential:
            for task in tasks:
                result = self._detect_landmarks_task(task)
                batch_results.append(result)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # executor.map processes tasks in parallel and returns results in order
                batch_results = list(executor.map(self._detect_landmarks_task, tasks))

        duration = time.perf_counter() - start_time

        final_landmarks_list = []
        for i, (frame, landmarks, face_rect) in enumerate(batch_results):
            final_landmarks_list.append(landmarks)

            if landmarks is None:
                self.detection_failures += 1
                logger.warning("[LandmarkDetector] Landmark detection returned no result.")

            if debug_path is not None:
                self._validate_landmarks(frame, face_rect, landmarks, debug_path)

        num_frames = len(frames)
        self.total_detection_time += duration
        self.detection_calls += num_frames

        avg_time_ms = (duration / num_frames) * 1000 if num_frames > 0 else 0
        logger.debug(
            f"Processed landmark batch of {num_frames} in parallel ({max_workers} workers) in {duration:.2f}s ({avg_time_ms:.2f} ms/frame).")

        return final_landmarks_list

    def _detect_landmarks_task(self, args: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """
        Worker function for a single frame.
        Raw - no logging.
        Use in batch calls.
        """
        frame, face_rect, face_roi = args

        landmarks = self._detect_landmarks(frame, face_rect, face_roi)

        return frame, landmarks, face_rect


    def detect_landmarks(self, frame: np.ndarray, face_rect: np.ndarray, face_roi, debug_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Detects landmarks by dispatching to the configured backend.
        """
        start_time = time.perf_counter()

        landmarks = self._detect_landmarks(frame, face_rect, face_roi)

        duration = time.perf_counter() - start_time
        self.total_detection_time += duration
        self.detection_calls += 1

        if landmarks is None:
            self.detection_failures += 1
            logger.warning("[LandmarkDetector] Landmark detection returned no result.")

        if debug_path is not None:
            self._validate_landmarks(frame, face_rect, landmarks, debug_path)

        return landmarks

    def _detect_landmarks(self, frame: np.ndarray, face_rect: np.ndarray, face_roi: np.ndarray):
        """
        Detects landmarks by dispatching to the configured backend. Raw - no logging.
        """

        if self.method == LandmarkMethod.LBF:
            landmarks = self._detect_landmarks_lbf(frame, face_rect)
        elif self.method == LandmarkMethod.MEDIAPIPE:
            landmarks = self._detect_landmarks_mediapipe(face_roi, face_rect)
        else:
            raise ValueError(f"Unsupported landmark detection method: {self.method.value}")

        return landmarks


    def _validate_landmarks(self, frame: np.ndarray, face_rect: np.ndarray, landmarks: np.ndarray, debug_path: str):
        logger.warning(f"[Landmarks] Validation is active. Possible overhead.")

        failure_reason = None
        if landmarks is None:
            failure_reason = "no_result"
        elif not self._are_landmarks_geometrically_valid(face_rect, landmarks):
            failure_reason = "bad_geometry"
        elif not self._has_valid_aspect_ratio(landmarks):
            failure_reason = "bad_aspect_ratio"

        if failure_reason and failure_reason != "no_result":
            debug_dir = os.path.join(debug_path, self.method.value)
            os.makedirs(debug_dir, exist_ok=True)
            save_landmark_custom_failure_debug_image(debug_path, frame, face_rect, landmarks, failure_reason)
            return None

    def _are_landmarks_geometrically_valid(self, face_rect: np.ndarray, landmarks: np.ndarray,
                                           min_area_ratio: float = 0.25) -> bool:
        """
        Checks if the area of the landmark cluster is a reasonable fraction of the face box area.

        Args:
            face_rect: The (x, y, w, h) of the input face detection.
            landmarks: The detected landmark points.
            min_area_ratio: The minimum required ratio of (landmark_area / face_area).

        Returns:
            True if the geometry is considered valid, False otherwise.
        """
        if landmarks is None or landmarks.size == 0:
            return False

        # Calculate area of the original face bounding box
        _, _, face_w, face_h = face_rect
        face_area = face_w * face_h
        if face_area == 0:
            return False

        lx, ly, lw, lh = self.get_landmark_bbox(landmarks)
        landmark_area = lw * lh

        area_ratio = landmark_area / face_area
        if area_ratio < min_area_ratio:
            logger.warning(
                f"Landmark validation failed: Landmark area ({landmark_area}) is only {area_ratio:.2%} of face area ({face_area}) (requires: {min_area_ratio}).")
            return False

        return True

    def _has_valid_aspect_ratio(self, landmarks: np.ndarray, min_ratio: float = 0.45, max_ratio: float = 2.0) -> bool:
        """Checks if the aspect ratio of the landmark cluster is 'face-like'."""
        if landmarks is None or landmarks.size == 0:
            return False

        lx, ly, lw, lh = self.get_landmark_bbox(landmarks)

        if lw == 0 or lh == 0:
            return False

        aspect_ratio = lh / lw

        if not (min_ratio <= aspect_ratio <= max_ratio):
            logger.warning(f"Landmark validation failed: Unrealistic aspect ratio of {aspect_ratio:.2f}.")
            return False

        return True


    def log_final_metrics(self, clear: Optional[bool] = False):
        """Logs the summary of landmark performance."""
        if self.detection_calls == 0:
            logger.info("No landmark calls were made.")
            return

        avg_time_ms = (self.total_detection_time / self.detection_calls) * 1000
        failure_rate = (self.detection_failures / self.detection_calls) * 100

        logger.info("--- Landmark Performance Metrics ---")
        logger.info(f"Method: {self.method.name}")
        logger.info(f"Total Calls: {self.detection_calls}")
        logger.info(f"Total Failures: {self.detection_failures} ({failure_rate:.2f}%)")
        logger.info(f"Total Time: {self.total_detection_time:.2f} seconds")
        logger.info(f"Average Time per Call: {avg_time_ms:.2f} ms")
        logger.info("----------------------------------------")

        res = {
            'calls': self.detection_calls,
            'failures': self.detection_failures,
            'time': self.total_detection_time
        }

        if clear:
            self.detection_calls = 0
            self.detection_failures = 0
            self.total_detection_time = 0

        return res


    def get_landmark_bbox(self, landmarks: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Compute (x, y, w, h) bounding box from a set of 2D landmarks.

        Args:
            landmarks: np.ndarray of shape (N, 2), where each row is (x, y)

        Returns:
            A tuple (x, y, w, h) representing the top-left corner and width/height of the bounding box,
            or None if landmarks are invalid (e.g. any negative coordinates).
        """
        if landmarks is None or landmarks.size == 0:
            return None

        try:
            x_coords = landmarks[:, 0]
            y_coords = landmarks[:, 1]

            x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
            y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

            width = x_max - x_min
            height = y_max - y_min

            return (x_min, y_min, width, height)

        except (IndexError, ValueError) as e:
            logger.warning(f"[LandmarksBBOX] Failed to calculate landmark bbox: {e}", exc_info=True)
            return None


    def _detect_landmarks_lbf(self, frame: np.ndarray, face_rect: np.array) -> Optional[np.ndarray]:
        """LBF-specific landmark detection."""
        try:
            face_rect_np = np.array(face_rect)
            if np.any(face_rect_np < 0): return None

            success, landmarks = self.facemark.fit(frame, np.array([face_rect]))
            if success and landmarks is not None and len(landmarks) > 0:
                return landmarks[0][0].astype(np.int32)  # Returns (68, 2) array
            return None
        except Exception as e:
            logger.warning(f"[LandmarksLBF] Failed to detect landmarks: {e}", exc_info=True)
            return None

    def _detect_landmarks_mediapipe(self, face_rgb_roi: np.ndarray, face_rect: np.ndarray) -> Optional[np.ndarray]:
        """
            MediaPipe-specific landmark detection.
            Face ROI needs to be a contiguouos array (np.ascontiguousarray(face_roi))
        """
        try:
            if face_rgb_roi.size == 0:
                return None

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb_roi)
            results = self.landmarker.detect(mp_image)

            if not results or not results.face_landmarks: return None

            face_landmarks = results.face_landmarks[0]

            x, y, w, h = face_rect
            roi_h, roi_w, _ = face_rgb_roi.shape

            landmarks = np.array([
                (int(lm.x * roi_w + x), int(lm.y * roi_h + y))
                for lm in face_landmarks
            ], dtype=np.int32)
            return landmarks  # (468, 2)
        except Exception as e:
            logger.warning(f"[LandmarksMediapipe] Failed to detect landmarks: {e}", exc_info=True)
            return None
