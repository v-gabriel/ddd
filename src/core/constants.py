from enum import Enum
from pathlib import Path
from typing import Final, Tuple


class AppConstants:
    DEBUG_PANEL_HEIGHT: Final[int] = 240


class LandmarkMethod(Enum):
    """Enumeration for landmark detection methods."""
    LBF = "lbf"
    MEDIAPIPE = "mediapipe"


class LandmarkIndices:
    """Stores the specific landmark indices for different models."""
    # Indices for a 68-point model (LBF)
    LBF_LEFT_EYE = list(range(42, 48))
    LBF_RIGHT_EYE = list(range(36, 42))
    LBF_MOUTH = list(range(48, 68))

    # Indices for MediaPipe's 468-point Face Mesh
    # https://github.com/tensorflow/tfjs-models/blob/master/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
    MEDIAPIPE_LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    MEDIAPIPE_RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    MEDIAPIPE_MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


class DetectionMethod(Enum):
    """Supported face detection methods."""
    # FAST_MTCNN = "fastmtcnn"
    # SSD = "ssd"
    # CENTERFACE = "centerface"
    # RETINAFACE = "retinaface"
    DLIB_CNN = "dlib_cnn" # TODO: fix or remove, out of memory issues
    DLIB = "dlib"
    OPENCV = "opencv"
    MEDIAPIPE = "mediapipe"
    YUNET = "yunet"
    MTCNN = "mtcnn"
    YOLO = "yolo"


class Thresholds:
    """Constants for detection thresholds."""
    FACE_DETECTION_CONFIDENCE: Final[float] = 0.8


# --- Path Definitions ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent # Defines the project root as the parent directory of 'src'
class Paths:
    """Centralized, absolute file paths for the project."""

    MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    MEDIAPIPE_LANDMARKER_MODEL: Final[str] = str(MODELS_DIR / "face_landmarker.task")
    LBF_LANDMARKER_MODEL: Final[str] = str(MODELS_DIR / "lbfmodel.yaml")

    FACE_DETECTOR_HAARCASCADE: Final[str] = str(MODELS_DIR / "haarcascade_frontalface_default.xml")
    FACE_DETECTOR_YUNET: Final[str] = str(MODELS_DIR / "face_detection_yunet_2023mar.onnx")
    FACE_DETECTOR_MEDIAPIPE: Final[str] = str(MODELS_DIR / "blaze_face_short_range.tflite")
    FACE_DETECTOR_DLIB_CNN: Final[str] = str(MODELS_DIR / "mmod_human_face_detector.dat")
    FACE_DETECTOR_YOLO = str(MODELS_DIR / "yolov11n-face.pt")


class UI:
    """UI-related constants."""
    FONT = None
    FONT_SCALE: Final[float] = 0.7
    FONT_THICKNESS: Final[int] = 2
    OUTLINE_THICKNESS: Final[int] = 4
    DEFAULT_COLOR: Final[Tuple[int, int, int]] = (255, 255, 255)
    OUTLINE_COLOR: Final[Tuple[int, int, int]] = (0, 0, 0)
    ALARM_COLOR: Final[Tuple[int, int, int]] = (0, 0, 255)
    BOX_COLOR: Final[Tuple[int, int, int]] = (0, 255, 0)
    BOX_THICKNESS: Final[int] = 2
