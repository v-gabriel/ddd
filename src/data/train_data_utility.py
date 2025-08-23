import logging

import numpy as np

from src.config import ExperimentConfig

logger = logging.getLogger(__name__)

def validate_feature_consistency(data_lists: dict) -> bool:
    """
        Validates that all feature arrays have consistent sample counts.
    """
    if not data_lists:
        return True

    sample_counts = {}
    for feature_name, feature_list in data_lists.items():
        if feature_name in ['labels', 'groups']:
            sample_counts[feature_name] = len(feature_list)
        elif feature_list:
            sample_counts[feature_name] = len(feature_list)

    if not sample_counts:
        return True

    expected_count = list(sample_counts.values())[0]
    inconsistent_features = [name for name, count in sample_counts.items() if count != expected_count]

    if inconsistent_features:
        logger.error(f"Feature count mismatch! Expected: {expected_count}")
        for name, count in sample_counts.items():
            logger.error(f"  {name}: {count} samples")
        return False

    return True


def validate_frame_features(frame_features: dict, info: str) -> bool:
    """
        Validates a dictionary of extracted features (NaN, empty, None).

        Args:
            frame_features: The dictionary of features to validate.
            info: A string (e.g., "Frame 123 in video.mov") for logging.

        Returns:
            True if all features are valid, False otherwise.
    """
    for feat_name, feat_data in frame_features.items():
        if feat_data is None:
            logger.warning(f"{info}: Feature '{feat_name}' is None.")
            return False

        # This check is crucial for features that might be empty arrays (e.g., from a failed ROI crop)
        if isinstance(feat_data, np.ndarray) and feat_data.size == 0:
            logger.warning(f"{info}: Feature '{feat_name}' is an empty array.")
            return False

        if isinstance(feat_data, (np.ndarray, float, int)) and np.any(np.isnan(feat_data)):
            logger.warning(f"{info}: Feature '{feat_name}' contains NaN values.")
            return False

    return True

# TODO: test, implement
def filter_subjects_by_extraction_rate(data: dict, config: ExperimentConfig) -> dict:
    """
        Filters out subjects with low extraction rates from all per-sample arrays in the data dictionary.
    """
    if 'extraction_rates' not in data or 'groups' not in data:
        logger.warning("Missing 'extraction_rates' or 'groups' key. Skipping filtering.")
        return data

    min_rate = config.SETUP_MIN_SUBJECT_EXTRACTION_RATE
    groups = data['groups']
    original_sample_count = len(groups)
    extraction_rates_dict = data['extraction_rates']

    excluded_subjects = {
        subject_id for subject_id, rate in extraction_rates_dict.items() if rate < min_rate
    }

    if not excluded_subjects:
        logger.info(f"No subjects excluded. All meet the minimum extraction rate of {min_rate:.1%}.")
        return data

    logger.info(f"EXCLUDING {len(excluded_subjects)} subjects with extraction rate < {min_rate:.1%}:")
    for subject_id in sorted(list(excluded_subjects)):
        rate = extraction_rates_dict.get(subject_id, "N/A")
        sample_count = np.sum(groups == subject_id)
        logger.info(f"  Subject '{subject_id}': {rate:.1%} extraction rate ({sample_count} samples)")

    valid_mask = ~np.isin(groups, list(excluded_subjects))
    new_sample_count = np.sum(valid_mask)
    logger.info(f"Filtering data: {original_sample_count} -> {new_sample_count} samples")

    meta_keys = {'label_encoder', 'extraction_rates'}

    for key, value in list(data.items()):
        if key in meta_keys:
            continue  # Skip metadata

        # All other keys contain per-sample data (features, labels, groups, etc.)
        # Filter arrays of the correct original length.
        if hasattr(value, '__len__') and len(value) == original_sample_count:
            if isinstance(value, np.ndarray):
                data[key] = value[valid_mask]
            else:
                logger.warning(f"Key '{key}' is not a NumPy array. Filtering may be slow or incorrect.")
                # Fallback for lists
                data[key] = [item for i, item in enumerate(value) if valid_mask[i]]
        elif value is not None:
            logger.warning(f"Key '{key}' is not a per-sample array or has mismatched length. Skipping filtering.")

    data['extraction_rates'] = {
        sid: rate for sid, rate in extraction_rates_dict.items()
        if sid not in excluded_subjects
    }

    return data