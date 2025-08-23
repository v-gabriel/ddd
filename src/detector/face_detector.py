import logging
import time
from typing import List, Tuple, Optional

import cv2
import dlib
import numpy as np
import torch
from facenet_pytorch import MTCNN
from ultralytics import YOLO

from src.core.constants import DetectionMethod, Thresholds, Paths
from src.detector.landmark_detector import LandmarkDetector

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Detects faces in images using various detection backends.
    """

    def __init__(self, method: DetectionMethod):
        """
        Initialize with specified detection method.
        Args:
            method: Detection method name from DetectionMethod enum values
        """
        valid_methods = [m.value for m in DetectionMethod]
        if method.value not in valid_methods:
            raise ValueError(f"Unsupported detection method: {method}. Supported methods are: {valid_methods}")

        self.method = method
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing FaceDetector with method '{self.method}' on device '{self.device}'")
        logger.info(f"Cv2 cuda available: '{cv2.cuda.getCudaEnabledDeviceCount()}'")
        logger.info(f"Dlib cuda available: '{dlib.DLIB_USE_CUDA}'")

        self.total_detection_time = 0.0
        self.detection_calls = 0
        self.detection_failures = 0

        self.detector = None
        self._init_detector()


    def _init_detector(self):
        """Initializes the specific face detector model based on the selected method."""
        if self.method == DetectionMethod.YOLO:
            self.detector = YOLO(Paths.FACE_DETECTOR_YOLO)
            self.detector.to(self.device)
        elif self.method == DetectionMethod.MTCNN:
            self.detector = MTCNN(device=self.device)
            self.detector.to(self.device)
        elif self.method == DetectionMethod.YUNET:
            self.detector = cv2.FaceDetectorYN.create(
                model=Paths.FACE_DETECTOR_YUNET,
                config="",
                input_size=(640, 640),
                score_threshold=0.7,
                nms_threshold=0.3,
                top_k=1,
                backend_id=cv2.dnn.DNN_BACKEND_CUDA,
                target_id=cv2.dnn.DNN_TARGET_CUDA
            )
        elif self.method == DetectionMethod.OPENCV:
            self.detector = cv2.CascadeClassifier(Paths.FACE_DETECTOR_HAARCASCADE)

        elif self.method == DetectionMethod.MEDIAPIPE:
            logger.warning("GPU Delegate is not yet supported for Windows")
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(
                model_asset_path=Paths.FACE_DETECTOR_MEDIAPIPE,
                delegate=python.BaseOptions.Delegate.GPU
            )
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=Thresholds.FACE_DETECTION_CONFIDENCE
            )
            self.detector = vision.FaceDetector.create_from_options(options)
        elif self.method == DetectionMethod.DLIB:
            self.detector = dlib.get_frontal_face_detector() # HOG + Linear SVM
        elif self.method == DetectionMethod.DLIB_CNN:
            self.detector = dlib.cnn_face_detection_model_v1(Paths.FACE_DETECTOR_DLIB_CNN) # MMOD CNN
        else:
            logger.warning(
                f"Method '{self.method}' is defined but not implemented. Fallback to OpenCV.")
            self.method = DetectionMethod.OPENCV
            self.detector = cv2.CascadeClassifier(Paths.FACE_DETECTOR_HAARCASCADE)

        if self.method in [DetectionMethod.OPENCV, DetectionMethod.DLIB]:
            logger.info(f"GPU not supported for selected FaceDetector ({self.method}). Using CPU.")

    def log_final_metrics(self, clear: Optional[bool] = False):
        """Logs the summary of face detection performance."""
        if self.detection_calls == 0:
            logger.info("No face detection calls were made.")
            return

        avg_time_ms = (self.total_detection_time / self.detection_calls) * 1000
        failure_rate = (self.detection_failures / self.detection_calls) * 100

        logger.info("--- Face Detection Performance Metrics ---")
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

    def detect_faces_batch(
            self,
            frames: List[np.ndarray],
            force_sequential: bool = False,
            apply_orientation_logic: bool = False,
    ) -> List[Tuple[np.ndarray, float, Optional[int]]]:
        """
        Detects faces in a batch of frames.

        Args:
            frames: A list of frames to process.
            force_sequential: Skips batch load in supported models, processes using a simple for loop.
            apply_orientation_logic: If True, exhaustively tests four rotations for each frame
                                     and selects the best detection based on a scoring metric.
                                     If False, performs a single, fast detection pass with no
                                     rotation handling.

        Returns:
            A list of tuples, where each tuple contains the bounding boxes, confidence score,
            and rotation code for a frame.
        """
        if not frames:
            return []

        start_time = time.perf_counter()

        if apply_orientation_logic:
            final_results = self._detect_faces_batch_with_orientation(frames, force_sequential=force_sequential)
        else:
            final_results = self._detect_faces_batch(frames, force_sequential=force_sequential)

        duration = time.perf_counter() - start_time
        self.total_detection_time += duration

        num_frames = len(frames)
        self.detection_calls += num_frames
        failures = sum(1 for res in final_results if len(res[0]) == 0)
        self.detection_failures += failures

        return final_results

    def detect_face(
            self,
            frame: np.ndarray,
            landmark_detector: 'LandmarkDetector',
            debug_output_dir: str | None,
            apply_orientation_logic: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[np.ndarray]]:
        """
        Finds the most prominent, valid face and extracts landmarks.
        """
        batch_results = self.detect_faces_batch([frame], apply_orientation_logic=apply_orientation_logic)
        boxes, _, rotation_code = batch_results[0]

        if boxes is None or len(boxes) == 0:
            return None, None, frame

        if rotation_code is not None:
            frame = cv2.rotate(frame, rotation_code)

        best_face_box = boxes[0]
        try:
            x, y, w, h = best_face_box
            face_roi = frame[y:y + h, x:x + w]
            landmarks = landmark_detector.detect_landmarks(frame, best_face_box, face_roi, debug_output_dir)
            if landmarks is not None:
                return landmarks, best_face_box, frame
        except Exception as e:
            logger.error(f"Landmark detection failed on a found face: {e}", exc_info=True)

        return None, None, frame


    def _detect_yolo_batch(self, rgb_frames: List[np.ndarray]) -> List[Tuple[np.ndarray, float]]:
        results = self.detector(rgb_frames, verbose=False, stream=False)
        batch_results = []
        for result in results:
            boxes = result.boxes
            face_boxes = []
            max_confidence = 0.0
            for box in boxes:
                if box.conf[0] > Thresholds.FACE_DETECTION_CONFIDENCE:
                    x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                    w, h = x2 - x1, y2 - y1
                    face_boxes.append([x1, y1, w, h])
                    if box.conf[0] > max_confidence:
                        max_confidence = float(box.conf[0])
            batch_results.append((np.array(face_boxes), max_confidence))
        return batch_results

    def _detect_yolo(self, rgb_frames: np.ndarray) -> Tuple[np.ndarray, float]:
        results = self.detector(rgb_frames, verbose=False)
        boxes = results[0].boxes
        face_boxes = []
        max_confidence = 0.0
        for box in boxes:
            if box.conf[0] > Thresholds.FACE_DETECTION_CONFIDENCE:
                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                w, h = x2 - x1, y2 - y1
                face_boxes.append([x1, y1, w, h])
                if box.conf[0] > max_confidence:
                    max_confidence = float(box.conf[0])
        return np.array(face_boxes), max_confidence

    def _detect_mtcnn_batch(self, rgb_frames: np.ndarray[np.ndarray]) -> List[Tuple[np.ndarray, float]]:
        """
            Detects faces in batch. Requires RGB format input.
        """

        try:
            batch_boxes, batch_probs = self.detector.detect(rgb_frames, landmarks=False)
        except Exception as e:
            logger.error(f"[MTCNN] Batch detection error: {e}", exc_info=True)
            return [(np.array([]), 0.0)] * len(rgb_frames)

        batch_results = []
        for boxes, probs in zip(batch_boxes, batch_probs):
            if boxes is None:
                batch_results.append((np.array([]), 0.0))
                continue

            face_boxes = []
            max_confidence = 0.0
            for box, prob in zip(boxes, probs):
                if prob > Thresholds.FACE_DETECTION_CONFIDENCE:
                    x1, y1, x2, y2 = [int(i) for i in box]
                    w, h = x2 - x1, y2 - y1
                    face_boxes.append([x1, y1, w, h])
                    if prob > max_confidence:
                        max_confidence = float(prob)
            batch_results.append((np.array(face_boxes), max_confidence))
        return batch_results

    def _detect_mtcnn(self, rgb_frame: np.ndarray) -> Tuple[np.ndarray, float]:

        try:
            boxes, probs = self.detector.detect(rgb_frame, landmarks=False)
        except Exception as e:
            logger.error(f"Error in MTCNN detection: {e}", exc_info=True)
            return np.array([]), 0.0

        if boxes is None:
            return np.array([]), 0.0

        face_boxes = []
        max_confidence = 0.0
        for box, prob in zip(boxes, probs):
            if prob > Thresholds.FACE_DETECTION_CONFIDENCE:
                x1, y1, x2, y2 = [int(i) for i in box]
                w, h = x2 - x1, y2 - y1
                face_boxes.append([x1, y1, w, h])
                if prob > max_confidence:
                    max_confidence = float(prob)
        return np.array(face_boxes), max_confidence

    def _detect_mediapipe(self, rgb_frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detects faces in the frame using the MediaPipe model.
        """
        import mediapipe as mp

        h, w, _ = rgb_frame.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = self.detector.process(mp_image)

        if not results.detections:
            return np.array([]), 0.0

        face_boxes = []
        max_confidence = 0.0
        for detection in results.detections:
            confidence = detection.score[0]
            if confidence > Thresholds.FACE_DETECTION_CONFIDENCE:
                bbox_relative = detection.location_data.relative_bounding_box

                x1 = int(bbox_relative.xmin * w)
                y1 = int(bbox_relative.ymin * h)
                box_w = int(bbox_relative.width * w)
                box_h = int(bbox_relative.height * h)

                face_boxes.append([x1, y1, box_w, box_h])

                if confidence > max_confidence:
                    max_confidence = float(confidence)

        return np.array(face_boxes), max_confidence

    def _detect_yunet(self, bgr_frame: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w, _ = bgr_frame.shape

        self.detector.setInputSize((w, h))
        self.detector: cv2.FaceDetectorYN = self.detector
        _, faces = self.detector.detect(bgr_frame)
        if faces is None:
            return np.array([]), 0.0

        face_boxes = []
        max_confidence = 0.0
        for face in faces:
            box = [int(i) for i in face[:4]]
            confidence = float(face[-1])
            face_boxes.append(box)
            if confidence > max_confidence:
                max_confidence = confidence
        return np.array(face_boxes), max_confidence

    def _detect_opencv_cascade(self, gray_frame: np.ndarray) -> Tuple[np.ndarray, float]:
        faces = self.detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        return faces, 1.0 if len(faces) > 0 else 0.0

    def _detect_dlib(self, gray_frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
             Detects faces using dlib's HOG + Linear SVM.
         """

        detections = self.detector(gray_frame, 1)
        face_boxes = []
        for d in detections:
            x, y, w, h = d.left(), d.top(), d.width(), d.height()
            face_boxes.append([x, y, w, h])
        return np.array(face_boxes), 1.0 if len(face_boxes) > 0 else 0.0

    # TODO: Out of memory errors. Possible due to current thread based processing (large load for each thread).
    #  Remove or fix.
    def _detect_dlib_cnn(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detects faces using dlib's MMOD CNN model.
        """

        # The second argument is the number of times to upsample the image.
        # 1 is a good default for decent-sized images.
        detections = self.detector(frame, 1)

        face_boxes = []
        max_confidence = 0.0
        for d in detections:
            confidence = d.confidence
            if confidence > Thresholds.FACE_DETECTION_CONFIDENCE:
                rect = d.rect
                x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
                face_boxes.append([x, y, w, h])
                if confidence > max_confidence:
                    max_confidence = float(confidence)

        return np.array(face_boxes), max_confidence

    def _detect_faces_batch(
            self,
            frames: List[np.ndarray],
            force_sequential: bool = False,
    ) -> List[Tuple[np.ndarray, float, Optional[int]]]:
        """
        Performs a face detection on a batch of frames.
        """
        if self.method in [DetectionMethod.YOLO, DetectionMethod.MTCNN] and not force_sequential:
            batch_results = self._detect_yolo_batch(frames) if self.method == DetectionMethod.YOLO \
                else self._detect_mtcnn_batch(frames)
        else:
            batch_results = [self._perform_detection(frame) for frame in frames]

        final_results = []
        for boxes, confidence in batch_results:
            if len(boxes) > 0:
                final_results.append((boxes, confidence, None))
            else:
                final_results.append((np.array([]), 0.0, None))

        return final_results

    def _perform_detection(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Performs the raw face detection using the selected backend, without tracking metrics.
        """
        if self.method == DetectionMethod.YOLO:
            return self._detect_yolo(frame)
        elif self.method == DetectionMethod.MTCNN:
            return self._detect_mtcnn(frame)
        elif self.method == DetectionMethod.MEDIAPIPE:
            return self._detect_mediapipe(frame)
        elif self.method == DetectionMethod.YUNET:
            return self._detect_yunet(frame)
        elif self.method == DetectionMethod.OPENCV:
            return self._detect_opencv_cascade(frame)
        elif self.method == DetectionMethod.DLIB:
            return self._detect_dlib(frame)
        elif self.method == DetectionMethod.DLIB_CNN:
            return self._detect_dlib_cnn(frame)
        else:
            logger.error(f"Unknown detection method '{self.method}'.")
            return np.array([]), 0.0

    # TODO: remove, deprecated
    def _detect_faces_batch_with_orientation(
            self,
            frames: List[np.ndarray],
            force_sequential: bool = False
    ) -> List[Tuple[np.ndarray, float, Optional[int]]]:
        """
        Robust detection that maximizes utilization by processing the full batch
        for each of the four cardinal rotations and selecting the best result per frame.
        """
        num_frames = len(frames)

        best_results_per_frame = [
            {'score': -1.0, 'result': (np.array([]), 0.0, None)}
            for _ in range(num_frames)
        ]

        rotations = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]

        for rotation_code in rotations:
            if rotation_code is None:
                # For the first pass, use the original frames directly
                current_batch = frames
            else:
                current_batch = [cv2.rotate(frame, rotation_code) for frame in frames]

            if self.method in [DetectionMethod.YOLO, DetectionMethod.MTCNN] and not force_sequential:
                batch_results = self._detect_yolo_batch(current_batch) if self.method == DetectionMethod.YOLO \
                    else self._detect_mtcnn_batch(current_batch)
            else:
                batch_results = [self._perform_detection(f) for f in current_batch]

            for i, (boxes, confidence) in enumerate(batch_results):
                for box in boxes:
                    if not self._is_aspect_ratio_valid(box):
                        continue
                    score = self._calculate_detection_score(box, confidence)
                    if score > best_results_per_frame[i]['score']:
                        best_results_per_frame[i] = {
                            'score': score,
                            'result': (np.array([box]), confidence, rotation_code)
                        }

        final_results = [res['result'] for res in best_results_per_frame]

        return final_results


    def _is_aspect_ratio_valid(self, box: np.ndarray, min_ratio=0.4, max_ratio=1.6) -> bool:
        """
        Validates if the aspect ratio of a face bounding box is plausible.
        A loose ratio is used here as the scoring function is the primary filter.
        """
        if box is None or len(box) == 0:
            return False
        _, _, w, h = box
        if h == 0 or w == 0:
            return False
        aspect_ratio = w / h
        return min_ratio <= aspect_ratio <= max_ratio

    def _calculate_detection_score(self, box: np.ndarray, confidence: float) -> float:
        """
        Calculates a score for a single face detection.
        The score prioritizes high confidence, large area, and an upright aspect ratio.
        """
        if box is None or len(box) == 0:
            return -1.0

        _, _, w, h = box
        if w == 0 or h == 0:
            return -1.0

        area = w * h

        uprightness_factor = (h / w) ** 0.5 if w > 0 else 1.0

        score = confidence * area * uprightness_factor
        return score
