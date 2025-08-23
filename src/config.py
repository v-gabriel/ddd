import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from src.core.constants import DetectionMethod, LandmarkMethod

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class ExperimentConfig:

    # --- Experiment Setup Parameters  ---

    SETUP_FACE_DETECTION_METHOD: DetectionMethod = DetectionMethod.YOLO
    SETUP_LANDMARK_DETECTION_METHOD: LandmarkMethod = LandmarkMethod.MEDIAPIPE
    SETUP_BINARY_CLASSIFICATION: bool = True # TODO: support [0,5,10] prediction; ATM only supported [0,1], keep always true

    # If True, enables a ROC-based evaluation workflow on a hold-out test set.
    #
    # 1. Split: The full dataset is partitioned into a training/CV set and a final hold-out test set.
    #    The size of the test set is controlled by `SETUP_HOLDOUT_TEST_SIZE`.
    # 2. Cross-Validate: K-fold cross-validation is performed only on the training/CV set. This step is used
    #    to calculate performance metrics and find the optimal decision threshold for each fold.
    # 3. Train Final Model: After CV, a single, new model is trained on the entire training/CV set.
    # 4. Final Evaluation: This final model is evaluated on the hold-out test set. The decision
    #    threshold used for this evaluation is the mean of the optimal thresholds found across all folds in step 2.
    #
    # If False, the entire dataset will be used for cross-validation without a separate final evaluation step.
    # Keep false, use only if fold thresholds seems close enough that a mean value might give better results.
    SETUP_RUN_FINAL_OPTIMAL: bool = False

    SETUP_INCLUDE_VIDEOS: List[int] = field(default_factory=lambda: [0, 10]) # TODO: check for 5 videos inclusion: more drowsy data (then needs imbalanced flows/tests)
    SETUP_INCLUDE_SUBJECTS: List[int] = field(default_factory=lambda: [
        1,2,3,4,5,6,7,8,9,10,
        11,12,13,14,15,16,17,18,19,20,
        21,22,23,24,25,26,27,28,29,30,
        31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,
        51,52,53,54,55,56,57,58,59,60
    ])
    SETUP_EXCLUDE_SUBJECTS: List[int] = field(default_factory=lambda: [
        # Initial blinking issues, fixed with signal eye closure
        # 16, 25, 27, 29, 32, 33, 34, 39, 42, 48, 50, 51, 53, 54, 56,

        # Subjects with issues; reported over effective ~15 FPS, ~60 seconds window
        49, # label 1: 249 blinks; on video sleeping?
        38, # label 1: 122 blinks; looking down
        18, # label 1: 139 blinks; looking down
        15, # label 1: 153 blinks; looking down
        48, # camera moves too much, more closed eyes in label 0
        53, # blinks seemed to fast, can't seem to capture many frame mid-blink or fully closed, no matter FPS

        # Might want to use for optimizations/recalibration
        # 19 # label 0: 20 blinks / label 1: 13 blinks; seems ok
        # 31, # label 1: 93 blinks; seems ok
        # 42, # label 1: 85 blinks; seems ok, a lot of squinting?

        # Other: movement caused noisy landmark positions, triggering signal based blinks: fixed by smoothing + bigger threshold (better score if included)
        # 58,
        # 32,

        # Other: too many blinks?
        # 3
        # 29
        # 55
    ])

    # If using traditional methods, feature builder aggregates data based on this length; result are _std, _mean values
    SETUP_SEQUENCE_LENGTH: int = 600
    SETUP_CALIBRATION_FRAMES: int = None # 300 fps = 30s @10s; unstable: requires too many frames to register blinks: better None so it is rebuilt manually with all available data
    SETUP_TARGET_FPS: float = 10 # try to cap FPS; will process default FPS if it can't skip whole indices (1,2...)
    SETUP_NUM_FRAMES_PER_VIDEO: int = 600 # num of frames to extract; e.g. if FPS capped to 10, for 60s needs 600 frames
    # SETUP_FIND_COMPATIBLE_CALIBRATION_DATA: bool = False TODO: implement

    SETUP_FEATURES_TO_USE: List[str] = field(default_factory=lambda: [
        'hog',
        'engineered',
        'cnn'
    ])
    SETUP_HOG_FEATURES_TO_USE: List[str] = field(default_factory=lambda: ['hog_eye_left', 'hog_eye_right']) # 'hog_eye_left', 'hog_eye_right', 'hog_mouth'
    SETUP_CNN_FEATURE_TO_USE: str = 'cnn_eye_region' # 'cnn_face', 'cnn_eye_right', 'cnn_eye_left', 'cnn_mouth', 'cnn_eye_region'

    SETUP_ENGINEERED_FEATURES_TO_USE: List[str] = field(default_factory=lambda: [
        'ear',
        # 'mar',
        'ear_norm',
        # 'mar_norm',
        # 'is_eye_closed_threshold',
        # 'is_eye_closed_signal',
        'is_eye_closed',
        'eye_closure_acceleration',
        # 'is_yawn', # not enough data to utilize in UTA-RLDD; TODO: utilize hand over mouth as distractions etc.
        'perclos',
        'blink_freq',
        'avg_blink_duration',
        'eye_closure_velocity',
        # 'is_microsleeping',
        'head_pose_roll',
        'head_pose_pitch',
        'head_pose_yaw',
    ])

    # SETUP_APPLY_PADDING_ON_ROIS: bool = False # pad eye and mouth regions TODO: implement or manually define, currently fixed padding?

    SETUP_FRAME_SCALE: Tuple[int, int] = (640,360) # (640,360) or (854,480) // (1280,720) // (1920,1080)

    SETUP_VALIDATION_STRATEGY: str = 'kfold'  # 'kfold' or 'holdout' or 'loso' or 'holdout_per_frame'
    SETUP_HOLDOUT_TEST_SIZE: float = 0.2
    SETUP_KFOLD_N_SPLITS: int = 5 # https://sites.google.com/view/utarldd/home recommendation

    # apply either SMOTEEN or UNDERSAMPLING!; not both. TODO: test, fix and use?
    SETUP_APPLY_UNDERSAMPLING: bool = False
    SETUP_APPLY_SMOTEEN: bool = False # SMOTEENN: oversamples minority, undersamples extra ENN=Edited Nearest Neighbour; keeps samples count equal

    SETUP_APPLY_PCA_ON_FEATURES: bool = False # PCA=Principal Component Analysis; reduce dimensionality
    SETUP_PCA_N_COMPONENTS: float = None # 1 # retain % of variance; reduce dimensionality

    SETUP_UPPER_K_FEATURES: float = None # 200 # max num of features to keep after variance filtering
    SETUP_SAMPLES_PER_FEATURE: float = None # 1 # min number of samples per feature

    SETUP_MIN_SUBJECT_EXTRACTION_RATE = None # 0.95 # min % of required valid feature frames
    SETUP_FEATURE_VARIANCE_THRESHOLD: float = None # 0.001 # overfitting fix - remove low-variance features (e.g. 0.001, 0.01, 0.03, 0.05)



    # --- Thresholds ---

    SETUP_EAR_NORMALIZED_THRESHOLD: float = 0.20
    SETUP_CLOSURE_EAR_THRESHOLD: float = 0.40 # signal based closure threshold: if under and swing is fast enough, might be closed

    SETUP_PERCLOS_WINDOW_SECONDS: int = 30
    SETUP_BLINK_WINDOW_SECONDS: int = 30


    # --- Model-Specific Parameters ---

    # Whether to use search CV for param optimization: overrides SVM_PARAM_GRID or KNN_PARAM_GRID with best found params
    SETUP_HYPERPARAMETER_TUNING: bool = False

    # SVM/svm_deep
    SVM_TYPE: str = 'linear' # 'linear' or 'rbf'
    SVM_PARAM_GRID: Dict[str, Any] = field(default_factory=lambda:
    {
        'C': [ 0.001, 0.01, 0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1, 1, 10],
        'class_weight': [
            'balanced',
            { 0: 1.0, 1: 1.0 }, { 0: 1.0, 1: 2.0 },
            { 0: 1.0, 1: 4.0 }, { 0: 1.0, 1: 6.0 },
        ],
        # 'kernel': ['rbf', 'linear']
    })
    SVM_GAMMA: str = 'scale'
    SVM_C: float = 0.01
    SVM_CLASS_WEIGHT: Dict[int, float] | str = field(default_factory=lambda: 'balanced')

    # KNN
    KNN_PARAM_GRID: Dict[str, Any] = field(default_factory=lambda:
    {
        'n_neighbors': [7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree'],
        'metric': ['euclidean', 'manhattan']
    })
    # {
    #     'n_neighbors': [11, 15],
    #     'weights': ['distance'],
    #     'algorithm': ['ball_tree']
    # } # top performers for HOG

    KNN_NEIGHBORS: int = 11
    KNN_WEIGHTS: str = field(default_factory=lambda: 'uniform')
    KNN_ALGORITHM: str = field(default_factory=lambda: 'ball_tree')
    KNN_METRIC: str = field(default_factory=lambda: 'euclidean')

    # CNN/svm_deep
    CNN_MODEL_TYPE: str = 'mobilenetv2' # 'mobilenetv2' or 'resnet50v2' or 'efficientnetv2'
    CNN_USE_AUGMENTATION: bool = True
    CNN_EPOCHS: int = 10
    CNN_BATCH_SIZE: int = 16
    CNN_LEARNING_RATE: float = 0.001
    CNN_FINE_TUNE: bool = True
    CNN_FINE_TUNE_LAYERS: int = 10
    CNN_FINE_TUNE_LR: float = 1e-5
    CNN_FRAME_STACK_SIZE: int = 3

    # Decision Tree TODO: not implemented, ignore
    DT_MAX_DEPTH: int = 5
    DT_MIN_SAMPLES_SPLIT: int = 2
    DT_PARAM_GRID: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    })

    # Random Forest  TODO: not implemented, ignore
    RF_N_ESTIMATORS: int = 100
    RF_PARAM_GRID: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    })

    # LSTM
    LSTM_MODEL_TYPE: str = 'simplified'  # 'simplified' or 'hierarchical' # TODO: test hierarchical
    LSTM_SEQUENCE_LENGTH: int = 600


    # --- Image features config ---

    SETUP_FACE_ROI_SIZE_FACE: Tuple[int, int] = (64, 64)
    SETUP_FACE_ROI_SIZE_EYE: Tuple[int, int] = (32, 32)
    SETUP_FACE_ROI_SIZE_EYE_REGION: Tuple[int, int] = (96, 32)
    SETUP_FACE_ROI_SIZE_MOUTH: Tuple[int, int] = (48, 24)

    SETUP_HOG_ORIENTATIONS: int = 6
    SETUP_HOG_PIXELS_PER_CELL: Tuple[int, int] = (16, 16)
    SETUP_HOG_CELLS_PER_BLOCK: Tuple[int, int] = (1, 1)



    # --- Internal ---

    # TODO: issues due to different backends requiring different input formats: too big an overhead, ignore
    SETUP_PREPROCESS_FLAGS: List[str] = None # applied BEFORE face and landmark detection; saves frame with these changes

    # `median`, `grayscale`, `clahe`
    SETUP_POSTPROCESS_FLAGS: List[str] = field(default_factory=lambda: ['grayscale']) # applied upon reading the data (not in cache); recommended for dynamic testing when training etc.

    SUITE_NAME: str = None
    NAME: str = None
    MODELS_TO_RUN: List[str] = field(default_factory=lambda: ['svm', 'knn', 'svm_deep']) # 'lstm' or 'cnn' or 'svm' or 'knn' or 'svm_deep' or 'lstm_deep'

    # Extraction config
    FRAMES_BATCH: int = 300 # num of frames to load per thread (limit if lower memory); set to 1 to read frame by frame
    MAX_WORKERS: int = 1 # max(1, int((((os.cpu_count() or 1) / 2) or 1))) # num of video workers (limit if lower memory)
    FORCE_SEQUENTIAL_DETECTION: bool = False, # forces sequential processing, without using batch logic
    OPTIMIZE_BATCH_SIZE: bool = False # reduces frames batch based on workers count, frame resolution and available resources

    ONLY_EXTRACTION: bool = False # if true, skips execution, logs only extraction

    VISUALIZE_BUILT_FEATURES: bool = False
    # 'all' or 'is_eye_closed' or 'is_microsleeping' or 'is_eye_closed_signal' or 'is_eye_closed_threshold'
    VISUALIZE_BUILT_FEATURES_CONFIG: List[str] =\
        field(default_factory=lambda: ['is_eye_closed', 'is_microsleeping', 'is_eye_closed_signal, is_eye_closed_threshold'])
    VISUALIZE_ONLY_LOG: bool = False # if visualize built features if true, only logs calc. without writing new images
    SAVE_DEBUG_IMAGES_ON_FAILURE: bool = True # save face detection/landmark failure, discrepancies etc.

    RANDOM_STATE: int = 42
    DEBUG_SHOW_FACE_ROTATIONS: bool = False # TODO: remove, deprecated

    DATASET_ROOT_DIR: Path = PROJECT_ROOT / "DATASETS"
    OUTPUT_DIR: Path = PROJECT_ROOT / "RESULTS" / "MODELS"
    LOG_DIR: Path = PROJECT_ROOT / "RESULTS"
    DEBUG_VIZ_DIR: Path = PROJECT_ROOT / "RESULTS" / "DEBUG_VISUALIZATIONS"
    RESULTS_EXCEL_FILE: Path = PROJECT_ROOT / "RESULTS"

    CACHE_FEATURES: bool = True # TODO: currently always caches in temp directory, remove



    def get_cache_filenames(self) -> Tuple[str, str, str]:
        """
        Generates unique, hash-based cache filenames based on data processing and feature extraction parameters.
        This ensures that any change to a relevant parameter correctly invalidates the cache.
        """

        cache_params = {
            # Preprocessing
            "SETUP_PREPROCESS_FLAGS": sorted(list(set(self.SETUP_PREPROCESS_FLAGS))) if self.SETUP_PREPROCESS_FLAGS else [], # TODO: remove, deprecated
            "SETUP_FRAME_SCALE": self.SETUP_FRAME_SCALE,

            # Video/subject selection (sorted for consistency)
            "SETUP_INCLUDE_VIDEOS": sorted(list(set(self.SETUP_INCLUDE_VIDEOS))) if self.SETUP_INCLUDE_VIDEOS else [],

            # Quality defined + landmarks are saved in final cache dump, can't use if indices differ
            "SETUP_LANDMARK_DETECTION_METHOD": self.SETUP_LANDMARK_DETECTION_METHOD.value,
            "SETUP_FACE_DETECTION_METHOD": self.SETUP_FACE_DETECTION_METHOD.value,

            # Other parameters
            "SETUP_FPS_CAP": self.SETUP_TARGET_FPS
        }

        # Unique string representation (JSON with sorted keys) + hash
        sorted_params_str = json.dumps(cache_params, sort_keys=True)
        cache_hash = hashlib.md5(sorted_params_str.encode()).hexdigest()

        base_name = (f"data_cache"
                     f"_f-{self.SETUP_NUM_FRAMES_PER_VIDEO}"
                     f"_fc-{self.SETUP_CALIBRATION_FRAMES}"
                     f"_h-{cache_hash}")

        final_cache = os.path.join(self.OUTPUT_DIR, f"{base_name}.npz")
        progress_cache = os.path.join(self.OUTPUT_DIR, f"{base_name}.progress.npz")

        calibration_cache = os.path.join(self.OUTPUT_DIR, f"{base_name}_calib.npz")

        return final_cache, progress_cache, calibration_cache


