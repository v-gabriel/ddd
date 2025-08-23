import logging

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut
from typing import Dict, Any

from src.config import ExperimentConfig

logger = logging.getLogger(__name__)

class ValidationSplitViewer:
    """A utility to inspect and format different validation split strategies."""

    @staticmethod
    def _format_split_data_for_logging(split_data: Dict[str, Any]) -> str:
        """Takes the structured split data and formats it into a readable multi-line string."""
        log_lines = []
        strategy = split_data.get('strategy', 'Unknown')
        log_lines.append(f"Strategy: {strategy}")

        # Handle K-Fold and LOSO (list of folds)
        if 'Folds' in split_data:
            for fold in split_data['Folds']:
                train_subjects = fold['Train Subjects'].split(', ')
                test_subjects = fold['Test Subjects'].split(', ')
                log_lines.append(f"  - Fold {fold['Fold']}:")
                log_lines.append(f"    Train ({len(train_subjects)} subjects): {fold['Train Subjects']}")
                log_lines.append(f"    Test  ({len(test_subjects)} subjects): {fold['Test Subjects']}")

        # Handle Holdout (single split)
        elif 'Train Subjects' in split_data:
            train_subjects = split_data['Train Subjects'].split(', ')
            test_subjects = split_data['Test Subjects'].split(', ')
            log_lines.append(f"  Train ({len(train_subjects)} subjects): {split_data['Train Subjects']}")
            log_lines.append(f"  Test  ({len(test_subjects)} subjects): {split_data['Test Subjects']}")

        return "\n".join(log_lines)


    @staticmethod
    def get_split_data(config: ExperimentConfig, groups: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Master function that calls the correct method, logs the formatted output,
        and returns the structured data for further use.
        """
        strategy = config.SETUP_VALIDATION_STRATEGY
        logger.info(f"\n--- Analyzing Validation Strategy: {strategy} ---")

        if strategy == 'kfold':
            split_data = ValidationSplitViewer._get_kfold_data(config, groups, y)
        elif strategy == 'holdout':
            split_data = ValidationSplitViewer._get_holdout_data(config, groups, y)
        elif strategy == 'loso':
            split_data = ValidationSplitViewer._get_loso_data(config, groups, y)
        else:
            logger.warning("Unsupported validation strategy for logging.")
            return {}

        formatted_log_string = ValidationSplitViewer._format_split_data_for_logging(split_data)
        logger.info(f"Validation Split Details:\n{formatted_log_string}")

        return split_data


    @staticmethod
    def _get_kfold_data(config: ExperimentConfig, groups, y) -> Dict[str, Any]:
        gkf = GroupKFold(n_splits=config.SETUP_KFOLD_N_SPLITS)
        X_dummy = np.zeros(len(groups))

        folds_list = []
        for i, (train_idx, test_idx) in enumerate(gkf.split(X_dummy, y, groups), 1):
            train_subjects = np.unique(groups[train_idx])
            test_subjects = np.unique(groups[test_idx])
            folds_list.append({
                'Fold': i,
                'Train Subjects': ', '.join(map(str, sorted(train_subjects))),
                'Test Subjects': ', '.join(map(str, sorted(test_subjects))),
            })
        return {'strategy': 'GroupKFold', 'Folds': folds_list}

    @staticmethod
    def _get_holdout_data(config: ExperimentConfig, groups, y) -> Dict[str, Any]:
        gss = GroupShuffleSplit(n_splits=1, test_size=config.SETUP_HOLDOUT_TEST_SIZE, random_state=config.RANDOM_STATE)
        X_dummy = np.zeros(len(groups))

        train_idx, test_idx = next(gss.split(X_dummy, y, groups))
        train_subjects = np.unique(groups[train_idx])
        test_subjects = np.unique(groups[test_idx])

        return {
            'strategy': 'Holdout',
            'Train Subjects': ', '.join(map(str, sorted(train_subjects))),
            'Test Subjects': ', '.join(map(str, sorted(test_subjects))),
        }

    @staticmethod
    def _get_loso_data(config: ExperimentConfig, groups, y) -> Dict[str, Any]:
        logo = LeaveOneGroupOut()
        X_dummy = np.zeros(len(groups))

        folds_list = []
        for i, (train_idx, test_idx) in enumerate(logo.split(X_dummy, y, groups), 1):
            train_subjects = np.unique(groups[train_idx])
            test_subjects = np.unique(groups[test_idx])  # Will be a single subject
            folds_list.append({
                'Fold': i,
                'Train Subjects': ', '.join(map(str, sorted(train_subjects))),
                'Test Subjects': str(test_subjects[0]),
            })
        return {'strategy': 'LeaveOneSubjectOut', 'Folds': folds_list}
