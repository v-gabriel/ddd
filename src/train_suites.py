"""
train_suites.py

This file defines a series of systematic experiment suites to test all aspects
of the drowsiness detection pipeline. Each function represents a group of related
experiments and is designed to be run independently or as part of the full suite.
"""

import argparse
import itertools

from src.config import ExperimentConfig
from src.core.constants import DetectionMethod, LandmarkMethod
from src.train_runner import run_experiment


# --- Main Entry Point ---

def run_full_suite():
    """Runs all defined experiment suites in a logical order."""
    print("--- STARTING FULL EXPERIMENT SUITE ---")

    # Quick sanity: test if pipeline/evaluation can be run; and if visualizations make sense
    # run_quick_sanity_check_suite()
    # run_quick_debug_viz_suite()

    # Find the best raw data extraction method. Modify the 'winners' in global config.py or manually in each test.
    # run_face_extraction_best_config()

    # Determine the minimum viable video quality (FPS & Resolution).
    # run_input_quality_tests()
    # run_input_fps_tests()

    # Find the best features and their optimal hyperparameters.
    # run_trad_lstm_feature_selection_tests('lstm')
    # run_trad_lstm_feature_selection_tests('svm')
    # run_roi_feature_hyperparameter_tests()

    # Determine optimal temporal windows for features and model sequences.
    # run_traditional_temporal_window_tests()
    # run_lstm_sequence_length_interaction_tests()

    # Fine-tune the hyperparameters (SVM/KNN)
    # run_model_hyperparameter_tests()

    # Test label 0 vs label 5 (alert vs vigilant)
    # run_video_inclusion_tests()

    # Compare model architectures on the now-optimized data and feature set.
    # run_final_optimal_w_data_quantity_tests()



# --- Debug Entry Point ---

def run_quick_debug_viz_suite():
    """
    Runs a very fast, lightweight visualization debug build of available features, logs, statistics etc.
    """

    suite_name = "Quick_Debug_Visualization_Check"
    print(f"\n--- SUITE: {suite_name} (Fast run for visualization validation) ---")

    base_config = {
        'FRAMES_BATCH': 1,
        'FORCE_SEQUENTIAL_DETECTION': True,
        'SETUP_EXCLUDE_SUBJECTS': [],
        'SETUP_SEQUENCE_LENGTH': 600,
        'SETUP_TARGET_FPS': 12.5,
        'SETUP_NUM_FRAMES_PER_VIDEO': 150,
        'SETUP_POSTPROCESS_FLAGS': ['grayscale'],
        'SETUP_INCLUDE_SUBJECTS': [1],
        'VISUALIZE_BUILT_FEATURES_CONFIG': ['all'],
        # 'SETUP_POSTPROCESS_FLAGS': [],

        'ONLY_EXTRACTION': True,
        'VISUALIZE_BUILT_FEATURES': True,
        'VISUALIZE_ONLY_LOG': False,

        'SETUP_HOG_ORIENTATIONS': 6,
        'SETUP_HOG_PIXELS_PER_CELL': (16, 16),
        'SETUP_PERCLOS_WINDOW_SECONDS': 60,
        'SETUP_BLINK_WINDOW_SECONDS': 60,

        'SETUP_FACE_ROI_SIZE_FACE': (64, 64),
        'OPTIMIZE_BATCH_SIZE': False,
        'MAX_WORKERS': 1,
    }

    # run_experiment(ExperimentConfig(
    #     **base_config,
    #     SUITE_NAME=suite_name,
    #     NAME="quick_viz",
    #     SETUP_FACE_DETECTION_METHOD=DetectionMethod.YOLO,
    #     SETUP_LANDMARK_DETECTION_METHOD=LandmarkMethod.LBF
    # ))

    run_experiment(ExperimentConfig(
        **base_config,
        SUITE_NAME=suite_name,
        NAME="quick_viz",
        SETUP_FACE_DETECTION_METHOD=DetectionMethod.YOLO,
        SETUP_LANDMARK_DETECTION_METHOD=LandmarkMethod.MEDIAPIPE,
    ))

def run_quick_sanity_check_suite():
    """
    Runs a very fast, lightweight test of the entire pipeline.
    This suite is designed for quick validation, not for generating
    publishable performance metrics.
    """
    suite_name = "Quick_Sanity_Check"
    print(f"\n--- SUITE: {suite_name} (Fast run for pipeline validation) ---")

    # Define a base configuration with parameters optimized for speed
    quick_base_config = {
        'SUITE_NAME': suite_name,

        'SETUP_TARGET_FPS': 10,
        'SETUP_NUM_FRAMES_PER_VIDEO': 600,

        'LSTM_SEQUENCE_LENGTH': 300,
        'SETUP_SEQUENCE_LENGTH': 600,
        'SETUP_PERCLOS_WINDOW_SECONDS': 60,
        'SETUP_BLINK_WINDOW_SECONDS': 60,

        # ! To visualize only (see how features are extracted)
        'ONLY_EXTRACTION': False,
        'VISUALIZE_BUILT_FEATURES': True,
        'VISUALIZE_BUILT_FEATURES_CONFIG': ['all', 'is_eye_closed', 'is_eye_closed_threshold', 'is_microsleeping', 'is_eye_closed_signal'],
        'VISUALIZE_ONLY_LOG': False,
        # 'SETUP_INCLUDE_SUBJECTS': [1],
        # 'OPTIMIZE_BATCH_SIZE': False,
        # 'MAX_WORKERS': 1,
        # 'FRAMES_BATCH': 300,
        #

        # --- "Quick" Feature Parameters ---
        'SETUP_HOG_ORIENTATIONS': 6, # Fewer orientations for faster HOG
        'SETUP_HOG_PIXELS_PER_CELL': (16, 16), # Larger cell size means fewer cells to compute
        'SETUP_HOG_CELLS_PER_BLOCK': (1, 1),

        # --- "Quick" Model Parameters ---
        'CNN_EPOCHS': 10, # 3
        'CNN_MODEL_TYPE': 'mobilenetv2',

        'SETUP_FACE_ROI_SIZE_FACE': (64, 64),
        'SETUP_FACE_ROI_SIZE_EYE': (32, 32),
        'SETUP_FACE_ROI_SIZE_EYE_REGION': (96, 32),

        'LSTM_MODEL_TYPE': 'simplified',
    }

    # Quick SVM with "quick" engineered features
    run_experiment(ExperimentConfig(
        **quick_base_config,
        NAME="quick_svm_engineered_ear",
        MODELS_TO_RUN=['svm'],
        SETUP_FEATURES_TO_USE=['engineered'],
        SETUP_ENGINEERED_FEATURES_TO_USE=['ear']
    ))

    # Quick SVM with "quick" HOG features
    # run_experiment(ExperimentConfig(
    #     **quick_base_config,
    #     NAME="quick_svm_hog",
    #     MODELS_TO_RUN=['svm'],
    #     SETUP_FEATURES_TO_USE=['hog'],
    #     SETUP_HOG_FEATURES_TO_USE=['hog_eye_left', 'hog_eye_right'],
    # ))

    # 3. Quick CNN
    # run_experiment(ExperimentConfig(
    #     **quick_base_config,
    #     NAME="quick_cnn_face",
    #     MODELS_TO_RUN=['cnn'],
    #     SETUP_CNN_FEATURE_TO_USE='cnn_eye_right'
    # ))

    # Quick SVM_deep
    # run_experiment(ExperimentConfig(
    #     **quick_base_config,
    #     NAME="quick_cnn_face",
    #     MODELS_TO_RUN=['svm_deep'],
    #     SETUP_CNN_FEATURE_TO_USE='cnn_eye_right'
    # ))

    # Quick LSTM
    # run_experiment(ExperimentConfig(
    #     **quick_base_config,
    #     NAME="quick_lstm",
    #     MODELS_TO_RUN=['lstm'],
    #     SETUP_FEATURES_TO_USE=['engineered'],
    # ))



# --- Tests Entry Point


# --- Data/Processing Suites ---

def run_input_quality_tests():
    """
    Systematically tests the impact of video input quality on the performance (ONLY EXTRACTION: check speed & data size).
    """
    suite_name = "Input_Quality_Tests"
    print(f"\n--- SUITE: {suite_name} ---")

    resolution_options = [
        (640, 360),
        (854, 480),
        (1280, 720),
        None
    ]

    base_config = {
        'SUITE_NAME': suite_name,
    }

    for resolution in resolution_options:
        scaling = f"{resolution[0]}x{resolution[1]}" if resolution else None
        run_name = f"quality_fps_res-{scaling}"
        print(f"\n--- Running Test: {run_name} ---")
        config = ExperimentConfig(
            **base_config,
            NAME=run_name,
            ONLY_EXTRACTION=True,
            SETUP_NUM_FRAMES_PER_VIDEO=int(10 * 10), # 10s
            SETUP_TARGET_FPS=10,
            SETUP_INCLUDE_SUBJECTS=[
                1,2,3,4,5,6,7,8,9,10,
                11,12,13,14,15,16,17,18,19,20
            ],
            CACHE_FEATURES=True,
            SETUP_FRAME_SCALE=resolution,
            OPTIMIZE_BATCH_SIZE=False,
            MAX_WORKERS=1,
            FRAMES_BATCH=100,
        )
        run_experiment(config)

    print(f"\n--- SUITE: {suite_name} Complete ---")

def run_input_fps_tests():
    """
    Systematically tests the impact of FPS on the performance on eye closure (are frames too apart to detect blinks).
    """
    suite_name = "Input_FPS_Tests"
    print(f"\n--- SUITE: {suite_name} ---")

    target_fps_options = [30, 12.5, 10]

    base_config = {
        'SUITE_NAME': suite_name,
    }

    for fps in target_fps_options:
        run_name = f"quality_fps-{fps}"
        print(f"\n--- Running Test: {run_name} ---")
        config = ExperimentConfig(
            **base_config,
            NAME=run_name,
            SETUP_NUM_FRAMES_PER_VIDEO=int(fps * 20), # 20s
            ONLY_EXTRACTION=True,
            SETUP_TARGET_FPS=fps,
            MAX_WORKERS=2,
            SETUP_FRAME_SCALE=(640, 360),

            # !
            VISUALIZE_BUILT_FEATURES=True,
            VISUALIZE_ONLY_LOG=True
        )
        run_experiment(config)

    print(f"\n--- SUITE: {suite_name} Complete ---")

def run_face_extraction_best_config():
    """
    Systematically tests different combinations of face and landmark detectors
    to evaluate their performance (speed and detection rate).
    """
    suite_name = "Face_Extraction_Performance"
    print(f"\n--- SUITE: {suite_name} ---")

    base_config = {
        'SUITE_NAME': suite_name,
        'ONLY_EXTRACTION': True,
        'SETUP_TARGET_FPS': 10,
        'SETUP_NUM_FRAMES_PER_VIDEO': 50, # 5s
        'SETUP_FRAME_SCALE': (1280, 720),
        'FRAMES_BATCH': 1, # Can be really slow if above 1 (memory overhead for some combos, may be OK depending on a good machine or if utilizing only 1 thread)
        'FORCE_SEQUENTIAL_DETECTION': True,
        'SETUP_EXCLUDE_SUBJECTS': [],
        'CACHE_FEATURES': False,
        'OPTIMIZE_BATCH_SIZE': False,
        'MAX_WORKERS': 1,
    }

    face_detectors_to_test = [
        DetectionMethod.DLIB,
        # # DetectionMethod.DLIB_CNN, # memory error
        DetectionMethod.OPENCV,
        DetectionMethod.MTCNN,
        # # DetectionMethod.MEDIAPIPE, # ERROR - Worker: Failed to load needed deps: GPU Delegate is not yet supported for Windows
        DetectionMethod.YOLO,
        DetectionMethod.YUNET,
    ]

    landmark_detectors_to_test = [
        LandmarkMethod.LBF,
        # LandmarkMethod.MEDIAPIPE
    ]

    for landmark_method in landmark_detectors_to_test:
        for face_method in face_detectors_to_test:
            run_name = f"face_{face_method.value}_landmark_{landmark_method.value}"

            print(f"\n--- Running Benchmark: {run_name} ---")

            config = ExperimentConfig(
                **base_config,
                NAME=run_name,
                SETUP_FACE_DETECTION_METHOD=face_method,
                SETUP_LANDMARK_DETECTION_METHOD=landmark_method
            )
            run_experiment(config)

    config = ExperimentConfig(
        **base_config,
        NAME='face_yolo_landmark_mediapipe',
        SETUP_FACE_DETECTION_METHOD=DetectionMethod.YOLO,
        SETUP_LANDMARK_DETECTION_METHOD=LandmarkMethod.MEDIAPIPE,
    )
    run_experiment(config)

    print(f"\n--- SUITE: {suite_name} Complete ---")


# --- Individual Experiment Suites ---

def run_video_inclusion_tests():
    """
        Tests model performance on alert vs vigilant classification.
    """
    suite_name = "Video_Inclusion"
    print(f"\n--- SUITE: {suite_name} ---")

    base_config = {
        'SUITE_NAME': suite_name,
        'SETUP_FEATURES_TO_USE': ['engineered'],
        'SETUP_NUM_FRAMES_PER_VIDEO': 600,
        'LSTM_SEQUENCE_LENGTH': 600,
        'SETUP_PERCLOS_WINDOW_SECONDS': 60,
        'SETUP_BLINK_WINDOW_SECONDS': 60,
        'SETUP_TARGET_FPS': 10,
    }
    feats = [
        ['is_eye_closed'],
        ['perclos'],
        ['is_eye_closed', 'perclos'],
        [   # full
            'ear',
            'ear_norm',
            'is_eye_closed',
            'eye_closure_acceleration',
            'perclos',
            'blink_freq',
            'avg_blink_duration',
            'eye_closure_velocity',
            'head_pose_roll',
            'head_pose_pitch',
            'head_pose_yaw',
        ]
    ]

    video_sets = [
        {'SETUP_INCLUDE_VIDEOS': [0, 5], 'NAME': 'inclusion_alert_vs_mild'},
        # {'SETUP_INCLUDE_VIDEOS': [0, 10], 'NAME': 'inclusion_alert_vs_severe'},
    ]

    for v_set in video_sets:
        for feats in feats:
            # run_experiment(ExperimentConfig(**base_config, **v_set, MODELS_TO_RUN=['svm'], SETUP_ENGINEERED_FEATURES_TO_USE=feats))
            run_experiment(ExperimentConfig(**base_config, **v_set, MODELS_TO_RUN=['lstm'], SETUP_ENGINEERED_FEATURES_TO_USE=feats))

def run_trad_lstm_feature_selection_tests(model: str = 'lstm'):
    """Tests the predictive power of different engineered feature combinations."""
    suite_name = f"Feature_Selection_{model}"
    print(f"\n--- SUITE: {suite_name} ---")
    base_config = {
        'SUITE_NAME': suite_name,
        'SETUP_TARGET_FPS': 10,
        'SETUP_NUM_FRAMES_PER_VIDEO': 300,

        'LSTM_SEQUENCE_LENGTH': 300,
        'SETUP_SEQUENCE_LENGTH': 300,
        'SETUP_PERCLOS_WINDOW_SECONDS': 30,
        'SETUP_BLINK_WINDOW_SECONDS': 30
    }

    run_experiment(ExperimentConfig(**base_config, MODELS_TO_RUN=[model], NAME="features_engineered_only", SETUP_FEATURES_TO_USE=['engineered']))

    # [ 'ear', 'ear_norm', 'is_eye_closed_threshold', 'is_eye_closed_signal', 'is_eye_closed', 'eye_closure_acceleration', 'perclos', 'blink_freq', 'avg_blink_duration', 'eye_closure_velocity', 'is_microsleeping', 'head_pose_roll', 'head_pose_pitch', 'head_pose_yaw']
    for model in [model]:
        run_experiment(ExperimentConfig(
            **base_config,
            MODELS_TO_RUN=[model],
            NAME="features_eye_centric_ear",
            SETUP_FEATURES_TO_USE=['engineered'],
            SETUP_ENGINEERED_FEATURES_TO_USE=['ear']
        ))
        run_experiment(ExperimentConfig(
            **base_config,
            MODELS_TO_RUN=[model],
            NAME="features_eye_centric_ear-norm",
            SETUP_FEATURES_TO_USE=['engineered'],
            SETUP_ENGINEERED_FEATURES_TO_USE=['ear_norm']
        ))
        run_experiment(ExperimentConfig(
            **base_config,
            MODELS_TO_RUN=[model],
            NAME="features_eye_centric_ear_ear-norm",
            SETUP_FEATURES_TO_USE=['engineered'],
            SETUP_ENGINEERED_FEATURES_TO_USE=['ear_norm', 'ear']
        ))
        run_experiment(ExperimentConfig(
            **base_config,
            MODELS_TO_RUN=[model],
            NAME="features_eye_centric_is_eye_closed",
            SETUP_FEATURES_TO_USE = ['engineered'],
            SETUP_ENGINEERED_FEATURES_TO_USE = ['is_eye_closed']
        ))
        run_experiment(ExperimentConfig(
            **base_config,
            MODELS_TO_RUN=[model],
            NAME="features_eye_centric_metrics",
            SETUP_FEATURES_TO_USE=['engineered'],
            SETUP_ENGINEERED_FEATURES_TO_USE=[
                'eye_closure_acceleration', 'blink_freq', 'avg_blink_duration', 'eye_closure_velocity'
            ]
        ))
        run_experiment(ExperimentConfig(
            **base_config,
            MODELS_TO_RUN=[model],
            NAME="features_head_centric",
            SETUP_FEATURES_TO_USE=['engineered'],
            SETUP_ENGINEERED_FEATURES_TO_USE=[
                'head_pose_roll', 'head_pose_pitch', 'head_pose_yaw'
            ]
        ))
        run_experiment(ExperimentConfig(
            **base_config,
            MODELS_TO_RUN=[model],
            NAME="features_eye_perclos",
            SETUP_FEATURES_TO_USE=['engineered'],
            SETUP_ENGINEERED_FEATURES_TO_USE=[
                'perclos'
            ]
        ))

        # Best for Linear SVM
        run_experiment(ExperimentConfig(
            **base_config,
            MODELS_TO_RUN=[model],
            NAME="features_eye_perclos_is_eye_closed",
            SETUP_FEATURES_TO_USE=['engineered'],
            SETUP_ENGINEERED_FEATURES_TO_USE=[
                'is_eye_closed', 'perclos'
            ]
        ))

        if model == 'lstm':
            # Best for LSTM if using ROC OptimalAccuracy
            run_experiment(ExperimentConfig(
                **base_config,
                MODELS_TO_RUN=[model],
                NAME="features_is_eye_closed_opt_acc",
                SETUP_FEATURES_TO_USE=['engineered'],
                SETUP_RUN_FINAL_OPTIMAL=True,
                SETUP_ENGINEERED_FEATURES_TO_USE=[
                    'is_eye_closed'
                ]
            ))
            # run_experiment(ExperimentConfig(
            #     **base_config,
            #     MODELS_TO_RUN=[model],
            #     NAME="features_perclos_acc",
            #     SETUP_FEATURES_TO_USE=['engineered'],
            #     SETUP_RUN_FINAL_OPTIMAL=True,
            #     SETUP_ENGINEERED_FEATURES_TO_USE=[
            #         'perclos'
            #     ]
            # ))
            # run_experiment(ExperimentConfig(
            #     **base_config,
            #     MODELS_TO_RUN=[model],
            #     NAME="features_perclos_acc",
            #     SETUP_FEATURES_TO_USE=['engineered'],
            #     SETUP_RUN_FINAL_OPTIMAL=True,
            #     SETUP_ENGINEERED_FEATURES_TO_USE=[
            #         'perclos', 'is_eye_closed'
            #     ]
            # ))

def run_final_optimal_w_data_quantity_tests():
    """
        Tests the final performance using optimal features & hyperparameters.
    """
    suite_name = "Final_Data_Quantity"
    print(f"\n--- SUITE: {suite_name} ---")

    base_config = {
        'SUITE_NAME': suite_name,
        'SETUP_FEATURES_TO_USE': ['engineered'],
        'SETUP_TARGET_FPS': 10
    }

    for window_size in [150, 300, 450, 600]:
        run_name = f"quantity_{window_size}f_rbf_svm"
        run_experiment(ExperimentConfig(
            **base_config,
            NAME=run_name,

            SETUP_ENGINEERED_FEATURES_TO_USE=[
                'perclos',
                'is_eye_closed'
            ],

            MODELS_TO_RUN=['svm'],
            SVM_TYPE='rbf',
            SVM_C=0.001,
            SVM_CLASS_WEIGHT='balanced',
            SVM_GAMMA='scale',

            SETUP_NUM_FRAMES_PER_VIDEO=window_size,
            SETUP_SEQUENCE_LENGTH=window_size,

            SETUP_PERCLOS_WINDOW_SECONDS=20,
            SETUP_BLINK_WINDOW_SECONDS=20
        ))

        run_name = f"quantity_{window_size}f_linear_svm"
        run_experiment(ExperimentConfig(
            **base_config,
            NAME=run_name,

            SETUP_ENGINEERED_FEATURES_TO_USE=[
                'perclos',
                'is_eye_closed'
            ],

            MODELS_TO_RUN=['svm'],
            SVM_TYPE='linear',
            SVM_C=0.01,
            SVM_CLASS_WEIGHT='balanced',

            SETUP_NUM_FRAMES_PER_VIDEO=window_size,
            SETUP_SEQUENCE_LENGTH=window_size,

            SETUP_PERCLOS_WINDOW_SECONDS=20,
            SETUP_BLINK_WINDOW_SECONDS=20
        ))

        run_name = f"quantity_{window_size}f_knn"
        run_experiment(ExperimentConfig(
            **base_config,
            NAME=run_name,

            SETUP_ENGINEERED_FEATURES_TO_USE=[
                'perclos',
                'is_eye_closed'
            ],

            MODELS_TO_RUN=['knn'],
            KNN_NEIGHBORS=9,
            KNN_WEIGHTS='uniform',
            KNN_ALGORITHM='ball_tree',
            KNN_METRIC='manhattan',

            SETUP_NUM_FRAMES_PER_VIDEO=window_size,
            SETUP_SEQUENCE_LENGTH=window_size,

            SETUP_PERCLOS_WINDOW_SECONDS=20,
            SETUP_BLINK_WINDOW_SECONDS=20
        ))

        run_name = f"quantity_{window_size}f_lstm"
        run_experiment(ExperimentConfig(
            **base_config,
            NAME=run_name,

            # use all
            SETUP_ENGINEERED_FEATURES_TO_USE=[
                'ear',
                'ear_norm',
                'is_eye_closed',
                'eye_closure_acceleration',
                'perclos',
                'blink_freq',
                'avg_blink_duration',
                'eye_closure_velocity',
                'head_pose_roll',
                'head_pose_pitch',
                'head_pose_yaw',
            ],

            MODELS_TO_RUN=['lstm'],

            LSTM_SEQUENCE_LENGTH=window_size,
            SETUP_NUM_FRAMES_PER_VIDEO=window_size,
            SETUP_SEQUENCE_LENGTH=window_size,

            SETUP_PERCLOS_WINDOW_SECONDS=10,
            SETUP_BLINK_WINDOW_SECONDS=10
        ))

def run_roi_feature_hyperparameter_tests():
    """
        Systematically tests all combinations of HOG parameters and
        other key feature extraction hyperparameters.
    """
    suite_name = "Feature_Hyperparameters"
    print(f"\n--- SUITE: {suite_name} ---")
    base_config = {
        'SUITE_NAME': suite_name,
        'SETUP_NUM_FRAMES_PER_VIDEO': 300,
        'SETUP_TARGET_FPS': 10,
        'SETUP_POSTPROCESS_FLAGS': ['grayscale'],
        'SETUP_HOG_FEATURES_TO_USE': ['hog_eye_right', 'hog_eye_left']
    }

    # --- HOG Parameter Sweep ---
    # Define the parameter ranges to test
    orientations = [6, 9, 12] # [6, 9, 12]
    pixels_per_cell = [(16,16)] # [(8, 8), (16, 16)]
    cells_per_block = [(2,2)] # [(1, 1), (2, 2)]

    # itertools.product to test every combination
    for orient, ppc, cpb in itertools.product(orientations, pixels_per_cell, cells_per_block):
        name = f"sweep_hog_o{orient}_ppc{ppc[0]}_cpb{cpb[0]}"
        config = ExperimentConfig(
            **base_config,
            NAME=name,
            MODELS_TO_RUN=['svm'],
            SETUP_FEATURES_TO_USE=['hog'],
            SETUP_HOG_ORIENTATIONS=orient,
            SETUP_HOG_PIXELS_PER_CELL=ppc,
            SETUP_HOG_CELLS_PER_BLOCK=cpb,
            SETUP_FACE_ROI_SIZE_EYE=(32, 32),
            SETUP_FACE_ROI_SIZE_EYE_REGION=(64, 32),
            SETUP_FACE_ROI_SIZE_FACE=(32, 32),
            SETUP_FACE_ROI_SIZE_MOUTH=(32, 32)
        )
        run_experiment(config)


    # --- ROI Resize Dimension Sweep for HOG ---
    print("\n--- Testing HOG performance with different ROI sizes ---")

    eye_sizes = [(32, 32), (48, 48), (64, 64)]
    for eye_size in eye_sizes:
        name = f"sweep_roi-size_eye{eye_size[0]}"
        run_experiment(ExperimentConfig(
            **base_config,
            NAME=name,
            MODELS_TO_RUN=['svm'],
            SETUP_FEATURES_TO_USE=['hog'],
            SETUP_FACE_ROI_SIZE_EYE=eye_size
        ))

    # --- CNN ROI Size Sweep ---
    print("\n--- Testing CNN ROI Sizes ---")

    for eye_size in [(32, 32), (48, 48), (64, 64)]:
        name = f"sweep_cnn-roi_eye_right{eye_size[0]}"
        run_experiment(ExperimentConfig(
            **base_config,
            MODELS_TO_RUN=['svm_deep'],
            NAME=name,
            SETUP_CNN_FEATURE_TO_USE='cnn_eye_right',
            SETUP_FACE_ROI_SIZE_EYE=eye_size
        ))

    for eye_region in [(32, 94), (48, 144)]:
        run_experiment(ExperimentConfig(
            **base_config,
            MODELS_TO_RUN=['svm_deep'],
            NAME=f"sweep_cnn-roi_eye_region{eye_region[0]}x{eye_region[1]}",
            SETUP_CNN_FEATURE_TO_USE='cnn_eye_region',
            SETUP_FACE_ROI_SIZE_EYE_REGION=eye_region
        ))

def run_model_hyperparameter_tests():
    """
        Tests key model hyperparameters using the built-in SearchCV sweep.
    """
    suite_name = "Model_Hyperparameter_Tuning_v2"
    print(f"\n--- SUITE: {suite_name} ---")

    base_config = {
        'SUITE_NAME': suite_name,
        'SETUP_FEATURES_TO_USE': ['engineered'],
        'SETUP_HYPERPARAMETER_TUNING': True,
        'SETUP_NUM_FRAMES_PER_VIDEO': 600,
        'SETUP_SEQUENCE_LENGTH': 600,
        'SETUP_TARGET_FPS': 10,
        'SETUP_PERCLOS_WINDOW_SECONDS': 60,
        'SETUP_BLINK_WINDOW_SECONDS': 60
    }

    run_experiment(ExperimentConfig(
        **base_config,
        MODELS_TO_RUN=['svm'],
        SVM_TYPE='linear',

        SVM_C=0.01,
        SVM_CLASS_WEIGHT='balanced',

        NAME="svm_hyperparam_tuning",
    ))

    run_experiment(ExperimentConfig(
        **base_config,
        MODELS_TO_RUN=['svm'],
        SVM_TYPE='rbf',

        SVM_C=0.001,
        SVM_CLASS_WEIGHT='balanced',
        SVM_GAMMA='scale',

        NAME="svm_hyperparam_tuning",
    ))

    run_experiment(ExperimentConfig(
        **base_config,
        MODELS_TO_RUN=['knn'],

        KNN_NEIGHBORS=9,
        KNN_WEIGHTS='uniform',
        KNN_ALGORITHM='ball_tree',
        KNN_METRIC='manhattan',

        NAME="knn_hyperparam_tuning",
    ))

def run_traditional_temporal_window_tests():
    """
    Systematically tests different window sizes for temporal features
    like PERCLOS and blink frequency to find the optimal balance for traditional models (SVM, KNN).
    """
    suite_name = "Temporal_Window_Tests"
    print(f"\n--- SUITE: {suite_name} ---")

    base_config = {
        'SUITE_NAME': suite_name,
        'MODELS_TO_RUN': ['svm'],
        'SETUP_NUM_FRAMES_PER_VIDEO': 600,
        'SETUP_TARGET_FPS': 10,
        'SETUP_FEATURES_TO_USE': ['engineered'],
        'SETUP_ENGINEERED_FEATURES_TO_USE': [
            'perclos',
            'blink_freq',
            'avg_blink_duration',
        ]
    }

    perclos_windows_sec = [5, 10, 15, 20, 30, 60]
    blink_windows_sec = [5, 10, 15, 20, 30, 60]

    # Use itertools.product to test all combinations
    for p_win, b_win in itertools.product(perclos_windows_sec, blink_windows_sec):
        run_name = f"temporal_p-win{p_win}_b-win{b_win}"
        config = ExperimentConfig(
            **base_config,
            NAME=run_name,
            SETUP_PERCLOS_WINDOW_SECONDS=p_win,
            SETUP_BLINK_WINDOW_SECONDS=b_win,
        )
        run_experiment(config)

def run_lstm_sequence_length_interaction_tests():
    """
     Systematically tests different window sizes for temporal features
     like PERCLOS and blink frequency to find the optimal balance for sequence models (LSTM).
     """
    suite_name = "Sequence_Length_Interaction"
    print(f"\n--- SUITE: {suite_name} ---")

    TARGET_FPS = 10
    TARGET_FRAMES = TARGET_FPS * 60

    base_config = {
        'SUITE_NAME': suite_name,
        'MODELS_TO_RUN': ['lstm'],
        'LSTM_MODEL_TYPE': 'simplified',
        'SETUP_FEATURES_TO_USE': ['engineered'],
        'SETUP_ENGINEERED_FEATURES_TO_USE': [
            'perclos',
            'blink_freq',
            'avg_blink_duration',
        ]
    }

    feature_window_seconds = [5, 10, 15, 20, 25, 30, 60]

    # Use itertools.product to create all combinations
    for model_win, feat_win in itertools.product([TARGET_FRAMES], feature_window_seconds):

        # --- Avoid illogical combinations ---
        # A feature window cannot be larger than the model's view (feature relies on data the model cannot see, which is data leakage)
        if (feat_win * TARGET_FPS) > model_win:
            print(f"--> SKIPPING illogical combo: Feature window ({feat_win}) > Model window ({model_win})")
            continue

        run_name = f"model_win_{model_win}_feat_win_{feat_win}"
        print(f"\n--- Running Test: {run_name} ---")

        config = ExperimentConfig(
            **base_config,
            NAME=run_name,
            SETUP_TARGET_FPS=TARGET_FPS,

            SETUP_NUM_FRAMES_PER_VIDEO=model_win,
            SETUP_SEQUENCE_LENGTH=model_win,
            LSTM_SEQUENCE_LENGTH=model_win,

            SETUP_PERCLOS_WINDOW_SECONDS=feat_win,
            SETUP_BLINK_WINDOW_SECONDS=feat_win
        )
        run_experiment(config)

    print(f"\n--- SUITE: {suite_name} Complete ---")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run drowsiness detection experiment suites.")
    parser.add_argument(
        'suite', nargs='?', default='full',
        choices=[
            'full',
            'features',
            'quantity',
            'feature_params',
            'model_params',
            'videos',
            'quick',
            'sequence',
            'quality'
        ],
        help="Name of the suite to run."
    )
    args = parser.parse_args()
    suite_map = {
        'full': run_full_suite,
        'features': run_trad_lstm_feature_selection_tests,
        'quantity': run_final_optimal_w_data_quantity_tests,
        'feature_params': run_roi_feature_hyperparameter_tests,
        'model_params': run_model_hyperparameter_tests,
        'videos': run_video_inclusion_tests,
        'quick': run_quick_sanity_check_suite,
        'sequence': run_lstm_sequence_length_interaction_tests,
        'quality': run_input_quality_tests,
    }
    suite_map[args.suite]()
