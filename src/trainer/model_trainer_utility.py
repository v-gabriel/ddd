import logging
from typing import Any, Iterator, Tuple, List, Dict

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, LeaveOneGroupOut

from src.config import ExperimentConfig

logger = logging.getLogger(__name__)

def get_pred_proba(model, X_test):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        return model.decision_function(X_test)
    return None

def get_param_grid(trainer, config_dict:  dict[str, Any]):
    """Get parameter grid based on actual SVM model type (LinearSVC or SVC)"""

    model = trainer.model

    valid_model_keys = model.get_params().keys()

    filtered_grid = {
        key: value
        for key, value in config_dict.items()
        if key in valid_model_keys
    }

    model_type_name = type(model).__name__
    logger.info(f"Using filtered parameter grid for {model_type_name}: {list(filtered_grid.keys())}")

    return filtered_grid


def get_validation_indices(config: ExperimentConfig, X, y, groups) -> Iterator[Tuple[np.ndarray, np.ndarray]]:

    if config.SETUP_VALIDATION_STRATEGY == 'holdout_per_frame':
        logger.warning("Using holdout per frame: validation will be unrealistic due to subject data leak.")

    if config.SETUP_VALIDATION_STRATEGY == 'holdout':
        gss = GroupShuffleSplit(1, test_size=config.SETUP_HOLDOUT_TEST_SIZE, random_state=config.RANDOM_STATE)
        yield next(gss.split(X, y, groups))
    elif config.SETUP_VALIDATION_STRATEGY == 'kfold':
        gkf = GroupKFold(config.SETUP_KFOLD_N_SPLITS)
        yield from gkf.split(X, y, groups)
    elif config.SETUP_VALIDATION_STRATEGY == 'loso':
        logo = LeaveOneGroupOut()
        yield from logo.split(X, y, groups)


def view_cross_validation_folds(config: ExperimentConfig, groups: np.ndarray, y: np.ndarray, logger) -> List[Dict]:
    """
    Analyzes cross-validation splits and RETURNS a list of dictionaries
    containing the fold distribution data, ready for logging.

    Returns:
        List[Dict]: A list where each dict represents a fold's subject distribution.
    """
    n_splits = config.SETUP_KFOLD_N_SPLITS
    if groups is None or y is None or len(groups) == 0:
        logger.error("Cannot view folds: groups or y array is not available.")
        return []

    logger.info(f"\n--- Analyzing Cross-Validation Folds (n_splits={n_splits}) ---")

    X_dummy = np.zeros(len(groups))
    gkf = GroupKFold(n_splits=n_splits)

    folds_data = []

    logger.info(f"{'Fold':<5} | {'# Train Subjects':<20} | {'# Test Subjects':<20}")
    logger.info("-" * 50)

    for i, (train_idx, test_idx) in enumerate(gkf.split(X_dummy, y, groups), 1):
        train_subjects = np.unique(groups[train_idx])
        test_subjects = np.unique(groups[test_idx])
        logger.info(f"{i:<5} | {len(train_subjects):<20} | {len(test_subjects):<20}")

        folds_data.append({
            'Fold': i,
            'Train Subjects': ', '.join(map(str, sorted(train_subjects))),
            'Test Subjects': ', '.join(map(str, sorted(test_subjects))),
        })
        logger.info(f"  Train Subjects: {train_subjects}")
        logger.info(f"  Test Subjects:  {test_subjects}\n")

    return folds_data