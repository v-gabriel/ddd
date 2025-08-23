import gc
import logging
import os
import random
import re
import time
from dataclasses import asdict
from typing import Dict

import numpy as np
import tensorflow as tf
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, \
    GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import parallel_backend

from src.config import ExperimentConfig
from src.data.feature_builder import FeatureBuilder
from src.evaluation.train_evaluator import evaluate_fold, aggregate_cv_results, \
    evaluate_fold_with_threshold_optimization, get_final_performance_report
from src.evaluation.train_results_logger import ResultsLogger
from src.evaluation.train_validator_split_viewer import ValidationSplitViewer
from src.train_pipeline import load_data
from src.trainer.model_trainer import get_model_trainer
from src.trainer.model_trainer_utility import get_pred_proba, get_param_grid, get_validation_indices

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def setup_logging(config: ExperimentConfig):
    safe_suite_name = re.sub(r'[\\/*?:\[\]]', '_', config.SUITE_NAME)

    suite_log_dir = os.path.join(config.LOG_DIR, safe_suite_name)
    os.makedirs(suite_log_dir, exist_ok=True)

    log_filename = ResultsLogger.generate_safe_filename_from_config(config, 'log')
    log_filepath = os.path.join(suite_log_dir, log_filename)

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_filepath, mode='a'), logging.StreamHandler()])

    logging.info(f"Logging initialized. Log file will be saved to: {log_filepath}")


def _validate_config(config: ExperimentConfig):
    if not config.MODELS_TO_RUN:
        raise ValueError("No models specified in MODELS_TO_RUN")
    if not config.SETUP_FEATURES_TO_USE:
        raise ValueError("No features specified in SETUP_FEATURES_TO_USE")
    if config.SETUP_VALIDATION_STRATEGY not in ['holdout', 'kfold', 'loso']:
        raise ValueError(f"Invalid validation strategy: {config.SETUP_VALIDATION_STRATEGY}")
    if config.SETUP_APPLY_SMOTEEN and config.SETUP_APPLY_UNDERSAMPLING:
        logging.getLogger(__name__).warning(
            "Both SMOTEENN and UNDERSAMPLING are enabled. "
            "This may lead to unexpected resampling effects. Proceed with caution.")


def get_preprocessing_pipeline(
        model_type: str,
        x_train_raw: np.ndarray,
        config: ExperimentConfig,
        logger: logging.Logger
) -> Pipeline:
    """
    Constructs and returns the appropriate scikit-learn preprocessing pipeline
    for a given model type based on the experiment configuration.
    """
    # Models that do not need traditional feature preprocessing
    if model_type in ['cnn', 'lstm']:
        return Pipeline([('passthrough', 'passthrough')])  # Return an identity pipeline

    # --- Build the pipeline for traditional models (svm, knn, svm_deep, ensemble) ---
    pipeline_steps = []

    # Scaling
    pipeline_steps.append(('scaler', StandardScaler()))

    # Variance Threshold Feature Selection
    if config.SETUP_FEATURE_VARIANCE_THRESHOLD is not None:
        pipeline_steps.append(('variance_filter', VarianceThreshold(threshold=config.SETUP_FEATURE_VARIANCE_THRESHOLD)))

    # K-Best Feature Selection
    if config.SETUP_SAMPLES_PER_FEATURE is not None and config.SETUP_UPPER_K_FEATURES is not None:
        max_features = x_train_raw.shape[1]
        optimal_k = min(
            config.SETUP_UPPER_K_FEATURES,
            x_train_raw.shape[0] // config.SETUP_SAMPLES_PER_FEATURE,
            max_features
        )
        if optimal_k == 0:
            logger.warning(f"Calculated optimal_k is 0. This may cause errors. Skipping KBest selection.")
        elif 0 < optimal_k < max_features and max_features > 1:
            pipeline_steps.append(('k_best', SelectKBest(f_classif, k=optimal_k)))
            logger.info(f"Pipeline will use SelectKBest with k={optimal_k}")
        else:
            logger.info("Skipping SelectKBest as optimal_k is not within a useful range.")

    # PCA Dimension Reduction
    if config.SETUP_APPLY_PCA_ON_FEATURES:
        pipeline_steps.append(('pca', PCA(n_components=config.SETUP_PCA_N_COMPONENTS)))

    # Resampling
    resampling_steps = []
    if config.SETUP_APPLY_UNDERSAMPLING:
        resampling_steps.append(('undersampler', RandomUnderSampler(random_state=config.RANDOM_STATE)))

    if config.SETUP_APPLY_SMOTEEN:
        resampling_steps.append(('smoteenn', SMOTEENN(random_state=config.RANDOM_STATE)))

    if resampling_steps:
        feature_pipeline = Pipeline(pipeline_steps)
        final_pipeline_steps = [('features', feature_pipeline)] + resampling_steps
        return ImblearnPipeline(final_pipeline_steps)
    else:
        return Pipeline(pipeline_steps)


def get_model_data(
        data_dict: Dict[str, np.ndarray],
        train_idx: np.ndarray,
        test_idx: np.ndarray
):
    """
    Slices a given data dictionary into train and test sets using the provided indices.
    This function expects a dictionary containing the feature matrix and labels.
    """
    if 'features' in data_dict:
        X = data_dict['features']
        y = data_dict['labels']
    else:
        X = data_dict['X']
        y = data_dict['y']

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test


def run_experiment(config: ExperimentConfig):
    if config.ONLY_EXTRACTION:
        config.SUITE_NAME = f"{config.SUITE_NAME}__Data_Extraction_Only"

    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info(f"--- STARTING EXPERIMENT: {config.NAME or 'Untitled'} (Suite: {config.SUITE_NAME}) ---")
    config_dict = asdict(config)
    config_log_str = '\n'.join([f"  {key}: {value}" for key, value in config_dict.items()])
    logger.info(f"Full Experiment Configuration:\n{config_log_str}")
    logger.info("-" * 60)
    main_start_time = time.time()

    try:
        if config.ONLY_EXTRACTION:
            logger.info("ONLY_EXTRACTION mode is enabled. Running data extraction only.")

        data, calibration_data = load_data(config)

        feature_builder = FeatureBuilder(
            config.SETUP_LANDMARK_DETECTION_METHOD,
            config
        )

        if config.VISUALIZE_BUILT_FEATURES:
            feature_builder.build_visualizations(data, calibration_data, True, config.VISUALIZE_ONLY_LOG)
            return

        if config.ONLY_EXTRACTION:
            logger.info("Data extraction complete. Cache file created/validated.")
            return

        _validate_config(config)

        def set_seed(seed=42):
            np.random.seed(seed)
            random.seed(seed)
            if tf is not None:
                tf.random.set_seed(seed)
            # tf.config.experimental.enable_op_determinism() # Throws, do not use.
            # tf.compat.v1.set_random_seed(seed)

        set_seed(config.RANDOM_STATE)

        logger.info("Step 1/2: Building comprehensive per-frame feature set...")
        per_frame_df = feature_builder.build_per_frame_features(data, calibration_data)

        models_to_run = set(config.MODELS_TO_RUN)

        all_prepared_features = {}
        X_split, y_split, groups_split = None, None, None

        if any(m in ['svm', 'knn', 'ensemble'] for m in models_to_run):
            logger.info("Step 2/2: Preparing data for TRADITIONAL models ('aggregate' strategy)...")
            feature_keys = config.SETUP_ENGINEERED_FEATURES_TO_USE + config.SETUP_HOG_FEATURES_TO_USE
            model_data = feature_builder.prepare_traditional_or_sequence_data(
                per_frame_df,
                'aggregate',
                feature_keys,
                config.SETUP_SEQUENCE_LENGTH
            )
            all_prepared_features['traditional'] = model_data
        elif any(m in ['cnn', 'svm_deep'] for m in models_to_run):
            logger.info("Step 2/2: Preparing data for DEEP models (CNN/svm_deep)...")
            model_data = feature_builder.prepare_deep_model_data(per_frame_df)
            all_prepared_features.update(model_data)
        elif 'lstm' in models_to_run:
            logger.info("Step 2/2: Preparing data for SEQUENCE models (LSTM, 'sliding_window' strategy)...")
            model_data = feature_builder.prepare_traditional_or_sequence_data(
                per_frame_df,
                'sliding_window',
                config.SETUP_ENGINEERED_FEATURES_TO_USE,
                config.LSTM_SEQUENCE_LENGTH
            )
            all_prepared_features['sequences'] = model_data
        elif 'lstm_deep' in models_to_run:
            logger.info("Step 2/2: Preparing data for DEEP SEQUENCE models (CNN-LSTM)...")
            deep_features_per_frame = feature_builder.extract_deep_features(
                np.stack(per_frame_df[config.SETUP_CNN_FEATURE_TO_USE].values)
            )
            X_seq, y_seq, groups_seq = feature_builder.create_sequences(
                deep_features_per_frame,
                per_frame_df['labels'].to_numpy(),
                per_frame_df['groups'].to_numpy(),
                config.LSTM_SEQUENCE_LENGTH
            )
            all_prepared_features['sequences'] = {'X': X_seq, 'y': y_seq, 'groups': groups_seq}
        else:
            raise ValueError("No valid model family could be determined from MODELS_TO_RUN.")

        data_dict = {}
        if 'deep' in all_prepared_features:
            data_dict = all_prepared_features['deep']
            X_split, y_split, groups_split = data_dict['X'], data_dict['y'], data_dict['groups']
        elif 'cnn_stacked' in all_prepared_features:
            data_dict = all_prepared_features['cnn_stacked']
            X_split, y_split, groups_split = data_dict['X'], data_dict['y'], data_dict['groups']
        elif 'sequences' in all_prepared_features:
            data_dict = all_prepared_features['sequences']
            X_split, y_split, groups_split = data_dict['features'], data_dict['labels'], data_dict['groups']
        elif 'traditional' in all_prepared_features:
            data_dict = all_prepared_features['traditional']
            X_split, y_split, groups_split = data_dict['features'], data_dict['labels'], data_dict['groups']

        if config.SETUP_BINARY_CLASSIFICATION and 10 in config.SETUP_INCLUDE_VIDEOS:
            y_split = np.where(y_split > 0, 1, 0)

        if config.SETUP_RUN_FINAL_OPTIMAL:
            gss = GroupShuffleSplit(n_splits=1, test_size=config.SETUP_HOLDOUT_TEST_SIZE,
                                    random_state=config.RANDOM_STATE)

            train_cv_idx, test_idx = next(gss.split(X_split, y_split, groups=groups_split))

            X_train_cv, X_test = X_split[train_cv_idx], X_split[test_idx]
            y_train_cv, y_test = y_split[train_cv_idx], y_split[test_idx]
            groups_train_cv, _ = groups_split[train_cv_idx], groups_split[test_idx]

            logger.info(
                f"Data split by groups: {len(X_train_cv)} samples for training/CV, {len(X_test)} for final hold-out test."
            )
            logger.info(
                f"Unique groups in training set: {len(np.unique(groups_train_cv))}, in test set: {len(np.unique(groups_split[test_idx]))}"
            )
        else:
            logger.info("Skipping hold-out set creation. Using all data for cross-validation.")
            X_train_cv, y_train_cv, groups_train_cv = X_split, y_split, groups_split
            X_test, y_test = None, None

        all_fold_results = []
        validation_indices = get_validation_indices(config, X_train_cv, y_train_cv, groups_train_cv)

        validation_data = ValidationSplitViewer.get_split_data(config, groups_train_cv, y_train_cv)

        for model_type in config.MODELS_TO_RUN:
            if config.SETUP_HYPERPARAMETER_TUNING and model_type in ['svm', 'knn']:
                pipeline = get_preprocessing_pipeline(model_type, X_train_cv, config, logger)
                trainer = get_model_trainer(model_type, config)

                logger.info(f"Pipeline for {model_type}: {pipeline}")

                search_pipeline = ImblearnPipeline(pipeline.steps + [('model', trainer.model)])
                param_grid = get_param_grid(trainer, config.SVM_PARAM_GRID if model_type in
                              ['svm', 'svm_deep'] else config.KNN_PARAM_GRID)
                prefixed_param_grid = {}
                for key, values in param_grid.items():
                    prefixed_param_grid[f'model__{key}'] = values  # e.g., 'model__C', 'model__class_weight'

                logger.info(f"Running GridSearchCV for {model_type.upper()}...")

                inner_cv_splitter = GroupKFold(n_splits=config.SETUP_KFOLD_N_SPLITS)

                hyperparam_search = GridSearchCV(
                    search_pipeline,
                    prefixed_param_grid,
                    cv=inner_cv_splitter,
                    scoring='f1_weighted',
                    n_jobs=-1
                )

                with parallel_backend('threading'):
                    hyperparam_search.fit(X_train_cv, y_train_cv, groups=groups_train_cv)
                # hyperparam_search.fit(X_train_raw, y_train_raw)

                logger.info(f"Best parameters: {hyperparam_search.best_params_}")

                best_params = hyperparam_search.best_params_

                unprefixed_best_params = {key.replace('model__', ''): value for key, value in
                                          best_params.items()}

                if model_type in ['svm', 'svm_deep']:
                    config.SVM_CLASS_WEIGHT = unprefixed_best_params.get('class_weight', None)
                    config.SVM_GAMMA = unprefixed_best_params.get('gamma', None)
                    config.SVM_C = unprefixed_best_params.get('C', None)
                elif model_type in ['knn']:
                    config.KNN_NEIGHBORS = unprefixed_best_params.get('n_neighbors', None)
                    config.KNN_WEIGHTS = unprefixed_best_params.get('weights', None)
                    config.KNN_ALGORITHM = unprefixed_best_params.get('algorithm', None)
                    config.KNN_METRIC = unprefixed_best_params.get('metric', None)

        for fold_num, (train_idx, test_idx) in enumerate(validation_indices, 1):
            logger.info(f"--- Starting Fold {fold_num} ---")
            unique_train, counts_train = np.unique(y_split[train_idx], return_counts=True)
            logger.info(f"Fold {fold_num} training class distribution: {dict(zip(unique_train, counts_train))}")
            if len(unique_train) < 2:
                logger.warning("Fold has only one class in training set! Skipping fold.")
                continue

            for model_type in config.MODELS_TO_RUN:
                try:
                    logger.info(f"--- Processing Model: {model_type.upper()} in Fold {fold_num} ---")

                    # Get Raw Data
                    model_data = get_model_data(data_dict, train_idx, test_idx)
                    if model_data is None:
                        raise ValueError(
                            f"No data extracted for {model_type}! Check get_model_data or feature building.")
                    X_train_raw, X_test_raw, y_train_raw, y_test_raw = model_data

                    pipeline = get_preprocessing_pipeline(model_type, X_train_raw, config, logger)
                    logger.info(f"Pipeline for {model_type}: {pipeline}")

                    # TODO: Test + fix? Unused in train_suites ATM.
                    is_resampling_pipeline = isinstance(pipeline, ImblearnPipeline)
                    if is_resampling_pipeline:
                        # If using SMOTE/undersampling, use fit_resample to get both X and y
                        logger.info("Applying pipeline with resampling...")
                        X_train_final, y_train_final = pipeline.fit_resample(X_train_raw, y_train_raw)
                    else:
                        # Otherwise, just fit_transform X, and y is unchanged
                        logger.info("Applying pipeline without resampling...")
                        X_train_final = pipeline.fit_transform(X_train_raw, y_train_raw)
                        y_train_final = y_train_raw

                    # The test set is always transformed, never resampled
                    X_test_final = pipeline.transform(X_test_raw)

                    unique_labels, counts = np.unique(y_train_final, return_counts=True)
                    logger.info(f"Final training distribution before fit: {dict(zip(unique_labels, counts))}")
                    logger.info(
                        f"Data shapes after pipeline: X_train={X_train_final.shape}, X_test={X_test_final.shape}")

                    if len(unique_labels) < 2:
                        logger.error(
                            f"Cannot train model {model_type}, only one class ({unique_labels[0]}) "
                            f"is present in the final training data. Skipping this model for the fold."
                        )
                        continue

                    trainer = get_model_trainer(model_type, config)

                    if hasattr(trainer, 'fit'):
                        trainer.fit(X_train_final, y_train_final, X_test_final, y_test_raw)
                    else:
                        trainer.model.fit(X_train_final, y_train_final)

                    if model_type in ['cnn', 'lstm']:
                        y_pred_raw_probs = trainer.model.predict(X_test_final)
                        y_pred_proba = y_pred_raw_probs if y_pred_raw_probs.ndim == 1 else y_pred_raw_probs[:, -1]
                        y_pred = np.argmax(y_pred_raw_probs, axis=1) if y_pred_raw_probs.ndim > 1 \
                            else (y_pred_raw_probs > 0.5).astype(int)
                    else:
                        y_pred_proba = get_pred_proba(trainer.model, X_test_final)
                        y_pred = trainer.model.predict(X_test_final)

                    # Log Results
                    model_name = trainer.model_name if config.SETUP_VALIDATION_STRATEGY == 'holdout' else f"{trainer.model_name}_Fold_{fold_num}"
                    if y_pred_proba is not None:
                        fold_results = evaluate_fold_with_threshold_optimization(y_test_raw, y_pred_proba,
                                                                                 data['label_encoder'], model_name,
                                                                                 config)
                        all_fold_results.extend(fold_results)

                    all_fold_results.append(evaluate_fold(y_test_raw, y_pred, data['label_encoder'], model_name))

                except Exception as e:
                    logger.error(f"Model {model_type} in fold {fold_num} failed: {e}", exc_info=True)
                finally:
                    gc.collect()

        if not all_fold_results:
            logger.error("No results were generated. Ending experiment.")
            return

        results_to_log = list(all_fold_results)
        if config.SETUP_VALIDATION_STRATEGY != 'holdout' and len(all_fold_results) > 1:
            summary_stats = aggregate_cv_results(all_fold_results)
            results_to_log.extend(summary_stats)

        if config.SETUP_RUN_FINAL_OPTIMAL and X_test is not None:
            logger.info("--- Starting Final Evaluation on Hold-Out Test Set ---")

            for model_type in config.MODELS_TO_RUN:
                logger.info(f"Training final '{model_type}' model on the full training dataset...")

                # 1. Train the final model on the ENTIRE training set (X_train_cv)
                final_trainer = get_model_trainer(model_type, config)

                final_pipeline = get_preprocessing_pipeline(model_type, X_train_cv, config, logger)

                if isinstance(final_pipeline, ImblearnPipeline):
                    X_train_cv_final, y_train_cv_final = final_pipeline.fit_resample(X_train_cv, y_train_cv)
                else:
                    X_train_cv_final = final_pipeline.fit_transform(X_train_cv)
                    y_train_cv_final = y_train_cv

                final_trainer.fit(X_train_cv_final, y_train_cv_final)  # No validation data here

                # Apply the FITTED pipeline to the hold-out test set
                X_test_final = final_pipeline.transform(X_test)

                # Evaluate the final model on the hold-out test set
                logger.info(f"Evaluating final '{model_type}' model on hold-out data...")

                # Filter CV results to get thresholds for the current model only
                cv_results_for_this_model = [r for r in all_fold_results if r['Model'].startswith(model_type.upper())]

                optimization_strategies = ['OptimalAccuracy', 'OptimalF1', 'Balanced']
                for strategy in optimization_strategies:
                    final_report = get_final_performance_report(
                        all_cv_results=cv_results_for_this_model,
                        final_model=final_trainer.model,
                        X_test=X_test_final,
                        y_test=y_test,
                        model_name=model_type,  # Use the specific model type
                        optimization_strategy=strategy
                    )
                    if final_report:
                        results_to_log.append(final_report)

        ResultsLogger.log_results_to_excel(results_to_log, config, validation_data)

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
    finally:
        logger.info(f"EXPERIMENT FINISHED. Total time: {time.time() - main_start_time:.2f}s")
