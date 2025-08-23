import concurrent.futures
import gc
import logging
import os
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from keras import applications
from keras.applications.densenet import layers
from tensorflow.python.estimator import keras
from tensorflow.python.keras import models
from tensorflow import keras
from src.config import ExperimentConfig
from src.core.constants import LandmarkMethod
from src.data.ear_normalizer import EarNormalizer
from src.data.feature_builder_utility import FeatureBuilderUtility
from src.data.metric_extractor import MetricExtractor
from src.data.z_score_normalizer import ZScoreNormalizer
from src.visualization.visualizations import visualize_features

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
        Calculates features from raw time-series arrays.
    """

    def __init__(self,
                 landmark_method: LandmarkMethod,
                 config: ExperimentConfig):

        self.config = config
        self.metric_extractor = MetricExtractor(landmark_method)

        self.fb_utility = FeatureBuilderUtility(
            landmark_method=landmark_method,
            config=config
        )

    def build_per_frame_features(self,
                                 main_data: Dict[str, np.ndarray],
                                 calibration_data: Dict[str, list] = None,
                                 build_all: Optional[bool] = False) -> pd.DataFrame:
        logger.info("Step 1/2: Starting comprehensive per-frame feature generation.")

        # --- DataFrame Creation and Initial Metrics ---
        scalar_keys = [k for k, v in main_data.items() if np.array(v).ndim == 1]
        df = pd.DataFrame({k: main_data[k] for k in scalar_keys})
        for key in [k for k, v in main_data.items() if np.array(v).ndim > 1]:
            df[key] = list(main_data[key])

        time_diffs = df['timestamp'].diff().dropna()
        median_interval_sec = time_diffs.median()

        # TODO: try to tweak for general optimal; currently tested for 10/12.5 FPS
        smoothing_time_sec = 0.2 # smaller = less smoothing, larger = more noise reduction; for 12.5/10 FPS: 0.2 seems ok
        window_frames = max(1, round(smoothing_time_sec / median_interval_sec))

        if 'ear' not in df.columns:
            df['ear'] = df['landmarks'].apply(self.metric_extractor.extract_ear)
            df['ear'] = df.groupby(['groups', 'labels'])['ear'].transform(
                lambda x: x.rolling(window=window_frames, min_periods=1, center=True).mean()
            )
        if 'mar' not in df.columns:
            df['mar'] = df['landmarks'].apply(self.metric_extractor.extract_mar)
            df['mar'] = df.groupby(['groups', 'labels'])['mar'].transform(
                lambda x: x.rolling(window=window_frames, min_periods=1, center=True).mean()
            )

        if not calibration_data:
            calibration_data = self.build_calib_data(df)

        # --- Normalization ---
        def normalize_group(group_df: pd.DataFrame) -> pd.DataFrame:
            group_id = group_df.name

            calib = calibration_data.get(group_id, {})

            ear_norm_model = EarNormalizer()
            ear_norm_model.calibrate(calib.get('ear', pd.Series(dtype=float)), info_str=group_id)
            mar_norm_model = ZScoreNormalizer()
            mar_norm_model.calibrate(calib.get('mar', pd.Series(dtype=float)), info_str=group_id)

            group_df['ear_norm'] = ear_norm_model.normalize(group_df['ear'])
            group_df['mar_norm'] = mar_norm_model.normalize(group_df['mar'])
            return group_df

        logger.info("Applying subject-specific normalization...")
        # Apply normalization and then immediately reset the index to resolve ambiguity.
        #  This makes 'groups' a regular column again.
        df = df.groupby('groups', group_keys=False).apply(normalize_group)

        # --- Image-Based Features ---
        features_to_use = set(self.config.SETUP_FEATURES_TO_USE)
        image_features_needed = any(f in features_to_use for f in ['hog', 'cnn']) or \
                                any(f.startswith('head_pose') for f in self.config.SETUP_ENGINEERED_FEATURES_TO_USE)

        if image_features_needed or build_all:
            logger.info("Calculating image-based features (ROIs, HOG, Head Pose, CNN patches)...")
            image_features = df.apply(
                lambda r: self._generate_image_features(r, build_all=build_all),
                axis=1
            )
            df = df.join(image_features)

        # --- Windowed Feature Calculation ---
        logger.info("Calculating windowed features (PERCLOS, Blinks)...")
        df['is_eye_closed_threshold'] = (df['ear_norm'] < self.config.SETUP_EAR_NORMALIZED_THRESHOLD).astype(int)

        def calculate_windowed_for_group(group_df: pd.DataFrame) -> pd.DataFrame:
            group_df['eye_closure_velocity'] = self.fb_utility.calculate_eye_closure_velocity(
                group_df['ear_norm'].to_numpy(),
                group_df['timestamp'].to_numpy()
            )

            velocity_for_calc = group_df['eye_closure_velocity'].to_numpy()

            acceleration = self.fb_utility.calculate_eye_closure_acceleration(
                velocity=velocity_for_calc,
                timestamp=group_df['timestamp'].to_numpy()
            )
            group_df['eye_closure_acceleration'] = acceleration

            open_mask = group_df['is_eye_closed_threshold'] == 0
            calib_velocities = group_df['eye_closure_velocity'][open_mask].to_numpy()

            # TODO: Filter for stable opens (low velocity) to improve quality?
            # if len(calib_velocities) > 0:
            #     stable_open_mask = np.abs(calib_velocities) < 0.1  # Tune this threshold
            #     calib_velocities = calib_velocities[stable_open_mask]
            if len(calib_velocities) < 10:
                logger.warning("[calculate_windowed_for_group] "
                               "Less than 10 frames for calib velocities. Defaulting to np.array[0.0]")
                calib_velocities = np.array([0.0])

            eye_closed_signal, is_closing, is_opening, stable_closed, stable_open  = self.fb_utility.calculate_eye_closed_signal(
                ear_norm=group_df['ear_norm'].to_numpy(),
                velocity=velocity_for_calc,
                acceleration=group_df['eye_closure_acceleration'].to_numpy(),
                timestamp=group_df['timestamp'].to_numpy(),
                calib_velocities=calib_velocities,
            )
            group_df['is_eye_closed_signal'] = eye_closed_signal

            # group_df['is_closing'] = is_closing
            # group_df['is_opening'] = is_opening
            # group_df['stable_closed'] = stable_closed
            # group_df['stable_open'] = stable_open

            group_df['is_eye_closed'] = (group_df['is_eye_closed_signal'] | group_df['is_eye_closed_threshold']).astype(int)
            # group_df['is_eye_closed'] = group_df['is_eye_closed_threshold']

            group_df['perclos'] = self.fb_utility.calculate_perclos_from_array(
                group_df['is_eye_closed'].to_numpy(),
                group_df['timestamp'].to_numpy()
            )

            group_df['is_microsleeping'] = self.fb_utility.calculate_microsleeps_from_array(
                group_df['is_eye_closed'].to_numpy(), group_df['timestamp'].to_numpy())

            freq, dur = self.fb_utility.analyze_blinks_from_array(
                group_df['is_eye_closed'].to_numpy(),
                group_df['timestamp'].to_numpy()
            )
            group_df['blink_freq'] = freq
            group_df['avg_blink_duration'] = dur

            return group_df

        # Because the index was reset above, 'groups' and 'labels' are now unambiguously columns.
        #  The `group_keys=False` prevents the grouping keys from being added back to the index of the result.
        df = df.groupby(['groups', 'labels'], group_keys=False).apply(calculate_windowed_for_group)

        logger.info("Per-frame feature generation complete.")
        return df

    def _generate_image_features(self, row: pd.Series, build_all: Optional[bool] = False) -> pd.Series:
        frame_features = {}

        if ('hog' in self.config.SETUP_FEATURES_TO_USE
                or 'cnn' in self.config.SETUP_FEATURES_TO_USE
                or build_all
            ):
            rois = self.fb_utility.build_rois_from_crop(
                row['face_crop'], row['landmarks'], row['face_coords']
            ).get('rois', {})

            if 'hog' in self.config.SETUP_FEATURES_TO_USE or build_all:
                hog_data, _ = self.fb_utility.process_hog_features(rois, visualize=False)
                if hog_data:
                    frame_features.update(hog_data)

            if 'cnn' in self.config.SETUP_FEATURES_TO_USE or build_all:
                cnn_patches = self.fb_utility.process_cnn_features(rois, row.get('face_crop'))
                if cnn_patches:
                    frame_features.update(cnn_patches)

        if any(f.startswith('head_pose') for f in self.config.SETUP_ENGINEERED_FEATURES_TO_USE) or build_all:
            head_pose, _ = self.fb_utility.process_head_pose_features(row, visualize=False)
            if head_pose is not None:
                frame_features.update({
                    'head_pose_roll': head_pose[0],
                    'head_pose_pitch': head_pose[1],
                    'head_pose_yaw': head_pose[2]
                })

        return pd.Series(frame_features)

    def create_sequences(self, data: np.ndarray, labels: np.ndarray, groups: np.ndarray, seq_length: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates sequences of a fixed length from per-frame data.

        - For video sessions LONGER than seq_length, it creates multiple, overlapping sequences.
        - For video sessions SHORTER than seq_length, it creates a single sequence and pads it with
          zeros to meet the required length.
        """
        X_seq, y_seq, groups_seq = [], [], []

        # Iterate through each unique subject
        for group_id in np.unique(groups):
            subject_mask = (groups == group_id)

            # Iterate through each unique video for that subject
            for label in np.unique(labels[subject_mask]):
                session_mask = (groups == group_id) & (labels == label)
                session_data = data[session_mask]

                num_frames = len(session_data)
                if num_frames == 0:
                    continue

                # Case 1: The video is SHORTER than the required sequence length
                if num_frames < seq_length:
                    # Create one padded sequence
                    padding_size = seq_length - num_frames
                    # Create a block of zeros for padding. Shape is (padding_size, num_features)
                    pad_block = np.zeros((padding_size, session_data.shape[1]))
                    # Append the padding to the end of the session data
                    padded_data = np.vstack([session_data, pad_block])

                    X_seq.append(padded_data)
                    y_seq.append(label)
                    groups_seq.append(group_id)

                # Case 2: The video is LONG ENOUGH for overlapping sequences
                else:
                    num_overlapping_sequences = num_frames - seq_length + 1
                    for i in range(num_overlapping_sequences):
                        sequence = session_data[i:i + seq_length]
                        X_seq.append(sequence)
                        y_seq.append(label)
                        groups_seq.append(group_id)

        if not X_seq:
            return np.array([]), np.array([]), np.array([])

        return np.array(X_seq), np.array(y_seq), np.array(groups_seq)

    def prepare_traditional_or_sequence_data(self, per_frame_df: pd.DataFrame, strategy: str, feature_keys: List[str],
                                             seq_length: int) -> Dict:
        logger.info(f"Step 2/2: Preparing features for model using '{strategy}' strategy.")

        available_cols = [col for col in feature_keys if col in per_frame_df.columns]
        df_subset = per_frame_df[available_cols + ['labels', 'groups']].copy()
        logger.info(f"Data preparation started with {len(df_subset)} rows.")

        nan_report = df_subset[available_cols].isna().sum()
        nan_cols = nan_report[nan_report > 0]
        if not nan_cols.empty:
            logger.warning("--- NaN Report: Found NaN values in the following feature columns ---")
            for col, count in nan_cols.items():
                logger.warning(f" - Column '{col}': {count} NaN values ({count / len(df_subset):.2%})")
            logger.warning("----------------------------------------------------------------------")
            logger.warning(
                "These rows will be dropped. If this is unexpected, check the feature calculation logic for these columns.")

        df = per_frame_df[available_cols + ['labels', 'groups']].dropna(subset=available_cols).copy()

        logger.info("Flattening object-type columns efficiently...")
        new_cols_list = []
        cols_to_drop = []
        original_cols = [col for col in available_cols if col in df.columns]

        for col in original_cols:
            if df[col].dtype == 'object' and df[col].notna().any() and isinstance(df[col].dropna().iloc[0], np.ndarray):
                cols_to_drop.append(col)

                flattened_data = pd.DataFrame(df[col].tolist(), index=df.index)
                flattened_data.columns = [f"{col}_{i}" for i in range(flattened_data.shape[1])]
                new_cols_list.append(flattened_data)

        if new_cols_list:
            df = df.drop(columns=cols_to_drop)
            df = pd.concat([df] + new_cols_list, axis=1)

        available_cols = [c for c in df.columns if c not in ['labels', 'groups']]

        df = df.copy()
        logger.info("Flattening complete. DataFrame is now de-fragmented.")

        if strategy == 'sliding_window':
            X = df[available_cols].to_numpy()
            y = df['labels'].to_numpy()
            g = df['groups'].to_numpy()
            X_seq, y_seq, groups_seq = self.create_sequences(X, y, g, seq_length)
            logger.info(f"Created {len(X_seq)} overlapping sequences.")
            return {
                'features': X_seq,
                'labels': y_seq,
                'groups': groups_seq
            }

        elif strategy == 'aggregate':
            df['seq_idx'] = df.groupby(['groups', 'labels']).cumcount() // seq_length

            # Custom std to handle groups with size 1 (return 0 instead of NaN)
            custom_std = lambda x: np.std(x, ddof=0) if len(x) > 1 else 0.0

            # All columns are scalar, so apply mean and custom std to all
            agg_dict = {col: ['mean', custom_std] for col in available_cols}

            aggregated_df = df.groupby(['groups', 'labels', 'seq_idx']).agg(agg_dict)

            aggregated_df.columns = ['_'.join(map(str, col)).strip('_').replace('<lambda>', '') for col in
                                     aggregated_df.columns.values]
            # aggregated_df.columns = ['_'.join(map(str, col)).strip('_') for col in
            #                          aggregated_df.columns.values]  # e.g., 'ear_mean', 'ear_lambda'
            aggregated_df = aggregated_df.reset_index()

            # Fill any remaining NaNs (though custom_std should prevent most)
            feature_cols = [col for col in aggregated_df.columns if col not in ['groups', 'labels', 'seq_idx']]
            aggregated_df[feature_cols] = aggregated_df[feature_cols].fillna(0)

            # Flatten to feature matrix
            final_features_list = aggregated_df[feature_cols].values.tolist()  # Directly to list of lists
            final_feature_matrix = np.array(final_features_list)

            logger.info(
                f"Created {len(aggregated_df)} aggregated, non-overlapping samples with {final_feature_matrix.shape[1]} final features.")

            return {
                'features': final_feature_matrix,
                'labels': aggregated_df['labels'].to_numpy(),
                'groups': aggregated_df['groups'].to_numpy()
            }
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Must be 'sliding_window' or 'aggregate'.")

    def prepare_deep_model_data(self, per_frame_df: pd.DataFrame) -> Dict:
        logger.info("Preparing data for DEEP models (CNN/svm_deep)...")
        features = {}
        cnn_patch_key = self.config.SETUP_CNN_FEATURE_TO_USE
        stack_size = self.config.CNN_FRAME_STACK_SIZE

        if cnn_patch_key not in per_frame_df.columns:
            raise ValueError(f"Required CNN feature '{cnn_patch_key}' not found in the per-frame DataFrame.")

        valid_patches_df = per_frame_df.dropna(subset=[cnn_patch_key])
        cnn_patches_array = np.stack(valid_patches_df[cnn_patch_key].values)

        stacked_patches, stacked_labels, stacked_groups = (
            self._stack_frames_by_group(
                cnn_patches_array,
                valid_patches_df['labels'].to_numpy(),
                valid_patches_df['groups'].to_numpy(),
                stack_size
            )
        )

        features['cnn_stacked'] = {'X': stacked_patches, 'y': stacked_labels, 'groups': stacked_groups}
        logger.info(f"Created group-aware stacked CNN data with shape: {stacked_patches.shape}")

        if 'svm_deep' in self.config.MODELS_TO_RUN:
            logger.info("Extracting deep features from STACKED frames for svm_deep...")
            extracted_features = self.extract_deep_features(stacked_patches)
            features['deep'] = {
                'X': extracted_features,
                'y': stacked_labels,
                'groups': stacked_groups
            }

        return features


    def _stack_frames_by_group(
            self,
            cnn_patches: np.ndarray,
            labels: np.ndarray,
            groups: np.ndarray,
            stack_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Stacks consecutive frames within each group to create temporal context for a 2D CNN,
        ensuring that stacks never cross subject boundaries.

        Args:
            cnn_patches (np.ndarray): The full array of image patches for all subjects.
            labels (np.ndarray): The corresponding labels for each patch.
            groups (np.ndarray): The corresponding group/subject ID for each patch.
            stack_size (int): The number of consecutive frames to stack.

        Returns:
            A tuple containing:
                - The array of stacked frame data.
                - The array of corresponding labels.
                - The array of corresponding group IDs.
        """
        all_stacked_patches, all_labels, all_groups = [], [], []
        unique_groups = np.unique(groups)

        logger.info(f"Starting group-aware frame stacking for {len(unique_groups)} unique subjects.")

        if len(cnn_patches.shape) == 3:  # (N, H, W) -> Add channel dim
            logger.info("Detected 3D patches (grayscale without channel). Adding channel dimension.")
            cnn_patches = np.expand_dims(cnn_patches, axis=-1)  # (N, H, W, 1)

        for group_id in unique_groups:

            group_mask = (groups == group_id)
            subject_patches = cnn_patches[group_mask]
            subject_labels = labels[group_mask]

            num_subject_frames = len(subject_patches)
            if num_subject_frames < stack_size:
                logger.warning(
                    f"Skipping subject {group_id}: has only {num_subject_frames} frames, needs {stack_size} for stacking.")
                continue

            sequences = [range(i, i + stack_size) for i in range(num_subject_frames - stack_size + 1)]

            stacked_for_subject = np.array([
                np.concatenate(subject_patches[seq], axis=-1) for seq in sequences  # Concat along channels
            ])

            labels_for_subject = subject_labels[stack_size - 1:]

            all_stacked_patches.append(stacked_for_subject)
            all_labels.append(labels_for_subject)

            all_groups.append(np.full(len(stacked_for_subject), group_id))

        if not all_stacked_patches:
            logger.error("Frame stacking resulted in zero samples. Check data and stack size.")
            return np.array([]), np.array([]), np.array([])

        final_patches = np.concatenate(all_stacked_patches)
        final_labels = np.concatenate(all_labels)
        final_groups = np.concatenate(all_groups)

        return final_patches, final_labels, final_groups


    def extract_deep_features(self, X_cnn: np.ndarray) -> np.ndarray:
        logger.info(f"Extracting deep features using {self.config.CNN_MODEL_TYPE}...")

        model_map = {
            'mobilenetv2': applications.MobileNetV2,
            'efficientnetv2': applications.EfficientNetV2B0,
            'resnet50v2': applications.ResNet50V2
        }

        if self.config.CNN_MODEL_TYPE not in model_map:
            raise ValueError(f"Feature extraction not supported for {self.config.CNN_MODEL_TYPE}")

        base_model_class = model_map[self.config.CNN_MODEL_TYPE]
        try:
            input_shape = X_cnn.shape[1:]

            # input layer
            inputs = layers.Input(shape=input_shape)

            if input_shape[-1] == 1:
                # Case 1: Grayscale input. Replicate channel to create a 3-channel image.
                logger.info("Adapting 1-channel input to 3 channels for the base model.")
                x = layers.Concatenate(axis=-1)([inputs, inputs, inputs])
                base_model_input_shape = (*input_shape[:2], 3)
            elif input_shape[-1] == 9:
                # Case 2: 9-channel input. Use an adapter Conv2D layer to project to 3 channels.
                logger.info("Adapting 9-channel input to 3 channels using a Conv2D adapter layer.")
                x = layers.Conv2D(3, (1, 1), padding='same', name='input_adapter')(inputs)
                base_model_input_shape = (*input_shape[:2], 3)
            elif input_shape[-1] == 3:
                # Case 3: Standard 3-channel input. No adaptation needed.
                x = inputs
                base_model_input_shape = input_shape
            else:
                raise ValueError(f"Unsupported number of input channels: {input_shape[-1]}. Expected 1, 3, or 9.")

            base_model = base_model_class(
                input_shape=base_model_input_shape, # color image format (x,y,colour)
                include_top=False,
                weights='imagenet'
            )

            # freeze
            base_model.trainable = False

            # `training=False` ensures layers like BatchNormalization run in inference mode
            x = base_model(x, training=False)

            # pooling layer gives back a fixed-size feature vector per image
            x = layers.GlobalAveragePooling2D()(x)

            # feature extractor model
            feature_model = models.Model(inputs=inputs, outputs=x, name="feature_extractor")

            logger.info(f"Extracting features using {base_model.name}...")
            features = feature_model.predict(X_cnn, batch_size=self.config.CNN_BATCH_SIZE, verbose=1)

            del feature_model, base_model, inputs, x
            gc.collect()
            keras.backend.clear_session()

            logger.info("Feature extraction successful.")
            return features
        except Exception as e:
            logger.error(f"Failed to extract deep features: {e}")
            raise

    def build_calib_data(self, df: pd.DataFrame) -> Dict[int, Dict[str, pd.Series]]:
        """
        Builds a unified calibration dataset for each subject using a DataFrame.

        Returns a dictionary mapping each group_id to their relevant calibration data

        Mocks the initial setup period/first recordings of a new user.
        Requires both opened and closed eyes.

        Args:
            df (pd.DataFrame): The main DataFrame containing all feature data,
                               including pre-computed 'ear' and 'mar' columns.

        Returns:
            Dict[int, Dict[str, pd.Series]]: A dictionary where each key is a subject's
            group_id and the value is a dictionary containing the 'ear' and 'mar'
            pandas Series for their combined calibration frames.
            Example: {'0': {'ear': pd.Series(...), 'mar': pd.Series(...)}, ...}
        """
        logger.info("Building unified calibration data from labels 0 and 1 for each subject.")

        # Filter for calibration frames (labels 0 or 1)
        calib_df = df[df['labels'].isin([0, 1])].copy()

        calibration_data = {}
        feature_keys = ['ear', 'mar']

        # Group by subject to collect calibration data
        for group_id, group_df in calib_df.groupby('groups'):
            if group_df.empty:
                logger.warning(
                    f"No frames with calibration labels (0 or 1) found for group {group_id}. "
                    "Calibration will be empty for this subject."
                )
                # Create empty series to prevent downstream errors
                calibration_data[group_id] = {key: pd.Series(dtype=float) for key in feature_keys}
                continue

            calibration_data[group_id] = {key: group_df[key] for key in feature_keys}

            logger.info(
                f"Built unified calibration set for group {group_id} with {len(group_df)} frames."
            )

        return calibration_data

    def build_visualizations(
            self,
            main_data: dict,
            calibration_data: dict = None,
            save_as_file=False,
            only_log=False,
    ):
        """
        Generates comprehensive visualizations from raw data.

        This method is self-contained: it calls the main `build_per_frame_features`
        internally to ensure all features are available for visualization,
        regardless of the main experiment's configuration.
        """

        logger.info("Starting self-contained visualization pipeline.")

        logger.info("Calling internal feature builder to generate a comprehensive feature set...")
        per_frame_df = self.build_per_frame_features(main_data, calibration_data, build_all=True)
        logger.info("Comprehensive feature set created. Proceeding with visualization.")

        closed_eyes_per_subject_label = defaultdict(dict)
        closed_eye_frames_per_subject_label = defaultdict(dict)
        microsleeps_per_subject_label = defaultdict(dict)
        durations_per_subject_label = defaultdict(dict)

        total_closed_eyes_count = 0

        numerical_feature_keys = [
            'timestamp',
            'ear',
            'mar',
            'ear_norm',
            'mar_norm',
            'is_eye_closed_threshold',
            'is_eye_closed_signal',
            'is_eye_closed',
            'eye_closure_acceleration',
            'eye_closure_velocity',
            'blink_freq',
            'avg_blink_duration',
            'perclos',
            'is_microsleeping',
            'head_pose_roll',
            'head_pose_pitch',
            'head_pose_yaw',
        ]
        available_numerical_keys = [key for key in numerical_feature_keys if key in per_frame_df.columns]

        for group_id, subject_df in per_frame_df.groupby('groups'):
            for label, video_df in subject_df.groupby('labels'):

                closed_eye_frames = video_df[video_df['is_eye_closed'] == 1]
                total_closed_eyes_curr = len(closed_eye_frames)
                closed_eye_indices = closed_eye_frames['frame_idx'].tolist()

                closed_eyes_per_subject_label[group_id][label] = total_closed_eyes_curr
                closed_eye_frames_per_subject_label[group_id][label] = closed_eye_indices
                total_closed_eyes_count += total_closed_eyes_curr

                microsleep_frames = video_df[video_df['is_microsleeping'] == 1]
                microsleeps_per_subject_label[group_id][label] = len(microsleep_frames)

                duration_min = self._compute_label_duration_minutes(video_df)
                durations_per_subject_label[group_id][label] = duration_min

                if only_log:
                    continue

                tasks = [
                    (row, label, group_id, available_numerical_keys, save_as_file)
                    for _, row in video_df.iterrows()
                ]

                logger.info(f"Saving {len(tasks)} debug visualizations for subject '{group_id}', label '{label}'...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                    list(executor.map(self._process_and_visualize_frame, tasks))

                # TODO: remove, parallel seems faster; test again
                # for i, frame_row in video_df.iterrows():
                #     numerical_features = frame_row[available_numerical_keys].to_dict()
                #
                #     rois = self.build_rois_from_crop(
                #         frame_row['face_crop'], frame_row['landmarks'], frame_row['face_coords']
                #     ).get('rois', {})
                #
                #     _, head_pose_viz = self.process_head_pose_features(frame_row, visualize=True)
                #     _, hog_visuals = self.process_hog_features(rois, visualize=True)
                #     cnn_data = {
                #         'cnn_mouth': frame_row.get('cnn_mouth'),
                #         'cnn_eye_right': frame_row.get('cnn_eye_right'),
                #         'cnn_eye_left': frame_row.get('cnn_eye_left'),
                #         'cnn_eye_region': frame_row.get('cnn_eye_region'),
                #         'cnn_face': frame_row.get('cnn_face'),
                #     }
                #
                #     image_visualizations = {
                #         'face_crop': frame_row.get('face_crop'),
                #         'head_pose': head_pose_viz,
                #         **hog_visuals,
                #         **cnn_data,
                #         **rois
                #     }
                #     image_visualizations = {k: v for k, v in image_visualizations.items() if v is not None}
                #
                #     visualize_features(
                #         label=int(str(label)),
                #         frame_idx=frame_row['frame_idx'],
                #         subject_id=f'{group_id}',
                #         config=self.config,
                #         original_frame=None,
                #         landmarks=frame_row.get('landmarks'),
                #         frame=None,
                #         numerical_features=numerical_features,
                #         image_visualizations=image_visualizations,
                #         visualize_features_config=self.config.VISUALIZE_BUILT_FEATURES_CONFIG,
                #         are_calibration_frames=False,
                #         save_as_file=save_as_file
                #     )

        # Logging per label (video) and subject
        for subject_id, labels_dict in closed_eyes_per_subject_label.items():
            for label, count in labels_dict.items():
                logger.info(f"Subject {subject_id}, label {label}: total closed eyes frames: {count}")
                logger.info(f"Frame indices: {closed_eye_frames_per_subject_label[subject_id][label]}")
        for subject_id, labels_dict in microsleeps_per_subject_label.items():
            for label, count in labels_dict.items():
                logger.info(f"Subject {subject_id}, label {label}: total microsleep frames: {count}")

        logger.info(f"Overall total closed eyes frames across all subjects: {total_closed_eyes_count}")

        logger.info("-" * 20 + " AGGREGATE BLINK STATS " + "-" * 20)

        subjects_with_0_blinks_per_label = defaultdict(list)
        subjects_with_1_blink_per_label = defaultdict(list)

        for subject_id, labels_dict in closed_eyes_per_subject_label.items():
            for label, count in labels_dict.items():
                if count == 0:
                    subjects_with_0_blinks_per_label[label].append(subject_id)
                elif count == 1:
                    subjects_with_1_blink_per_label[label].append(subject_id)

        # Log for labels with 0 blinks
        for label, subjects in sorted(subjects_with_0_blinks_per_label.items()):
            if subjects:
                subject_list_str = ", ".join(sorted(subjects))
                logger.info(f"Label '{label}' has 0 blinks in subjects: {subject_list_str}")

        # Log for labels with 1 blink
        for label, subjects in sorted(subjects_with_1_blink_per_label.items()):
            if subjects:
                subject_list_str = ", ".join(sorted(subjects))
                logger.info(f"Label '{label}' has 1 blink in subjects: {subject_list_str}")

        normal_blinks_per_minute = 15  # typical resting rate; adjust to 20 for active scenarios
        threshold_multiplier = 1.25  # buffer for variation

        # Log for labels suspicious blinks
        for subject_id, labels_dict in closed_eyes_per_subject_label.items():
            for label, count in labels_dict.items():
                duration_min = durations_per_subject_label[subject_id].get(label, 1.0)
                if duration_min <= 0:
                    continue

                expected_blinks = normal_blinks_per_minute * duration_min
                suspicious_threshold = expected_blinks * threshold_multiplier

                if count > suspicious_threshold:
                    logger.info(
                        f"Suspiciously high blinks for subject '{subject_id}', label '{label}': "
                        f"{count} blinks (expected ~{int(expected_blinks)}, threshold {suspicious_threshold:.1f}) - possible detection issue"
                    )

    def _compute_label_duration_minutes(self, df: pd.DataFrame) -> float:
        """
        Compute total duration of the video segment (label) in minutes from timestamps.

        Assumes df has a 'timestamp' column in seconds.
        """
        timestamps = df['timestamp'].to_numpy()
        if len(timestamps) < 2:
            return 0.0

        frame_intervals = np.diff(timestamps)
        avg_frame_interval = np.mean(frame_intervals) if len(frame_intervals) > 0 else 0

        duration_sec = (timestamps[-1] - timestamps[0]) + avg_frame_interval
        duration_min = duration_sec / 60.0
        return duration_min

    def _process_and_visualize_frame(self, frame_row_tuple):
        """
        Helper function to process and visualize a single frame.
        Designed to be called by a ThreadPoolExecutor.
        """
        frame_row, label, group_id, available_numerical_keys, save_as_file = frame_row_tuple

        try:
            numerical_features = frame_row[available_numerical_keys].to_dict()

            rois = self.fb_utility.build_rois_from_crop(
                frame_row['face_crop'], frame_row['landmarks'], frame_row['face_coords']
            ).get('rois', {})

            _, head_pose_viz = self.fb_utility.process_head_pose_features(frame_row, visualize=True)
            _, hog_visuals = self.fb_utility.process_hog_features(rois, visualize=True)
            cnn_data = {
                'cnn_mouth': frame_row.get('cnn_mouth'),
                'cnn_eye_right': frame_row.get('cnn_eye_right'),
                'cnn_eye_left': frame_row.get('cnn_eye_left'),
                'cnn_eye_region': frame_row.get('cnn_eye_region'),
                'cnn_face': frame_row.get('cnn_face'),
            }

            image_visualizations = {
                'face_crop': frame_row.get('face_crop'),
                'head_pose': head_pose_viz,
                **hog_visuals,
                **cnn_data,
                **rois
            }
            image_visualizations = {k: v for k, v in image_visualizations.items() if v is not None}

            visualize_features(
                label=int(str(label)),
                frame_idx=frame_row['frame_idx'],
                subject_id=f'{group_id}',
                config=self.config,
                original_frame=None,
                landmarks=frame_row.get('landmarks'),
                face_coords=frame_row['face_coords'],
                frame=None,
                numerical_features=numerical_features,
                visual_features=image_visualizations,
                visualize_features_config=self.config.VISUALIZE_BUILT_FEATURES_CONFIG,
                are_calibration_frames=False,
                save_as_file=save_as_file
            )
        except Exception as e:
            logger.error(f"Failed to process frame {frame_row.get('frame_idx')} for subject {group_id}: {e}", exc_info=True)
