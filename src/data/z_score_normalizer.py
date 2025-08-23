import logging
from typing import Union, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class ZScoreNormalizer:
    """
    Handles Z-score normalization (standardization) for features like MAR.
    This method makes features person-independent by scaling them based on their
    own mean and standard deviation.
    """

    def __init__(self):
        self.mean = None
        self.std_dev = None
        self.threshold = None
        self.is_calibrated = False

        self.FAILURE_THRESHOLD = 1e-6
        self.SUSPICIOUS_THRESHOLD = 1e-3

    def calibrate(self, baseline_values: pd.Series, info_str: Optional[str] = None):
        """Calculates mean and standard deviation from a baseline period."""

        if baseline_values.empty or len(baseline_values) < 2:
            logger.warning("Z-Score calibration failed: Not enough baseline values.")
            return

        log_context = f"({info_str})" if info_str else ""

        self.mean = np.mean(baseline_values)
        self.std_dev = np.std(baseline_values)

        # Avoid division by zero if all baseline values are the same.
        if self.std_dev < self.FAILURE_THRESHOLD:
            logger.warning(f"Z-Score calibration failed: Standard deviation is near zero. {log_context}")
            self.is_calibrated = False
        elif self.std_dev < self.SUSPICIOUS_THRESHOLD:
            logger.warning(f"Z-Score calibration SUSPICIOUS: Standard deviation is very low ({self.std_dev:.6f}). "
                           f"Baseline may be unnaturally stable. {log_context}")
            self.is_calibrated = True
            self.threshold = self.mean + (2 * self.std_dev)
            logger.info(
                f"Mean={self.mean:.4f}, StdDev={self.std_dev:.4f}. Dynamic threshold={self.threshold:.4f} {log_context}")
        else:
            self.is_calibrated = True
            self.threshold = self.mean + (2 * self.std_dev)
            self.is_calibrated = True
            logger.info(f"Z-Score calibration successful: Mean={self.mean:.4f}, StdDev={self.std_dev:.4f}. {log_context}")
            logger.info(f"Dynamic threshold set at: {self.threshold:.4f}")

    def normalize(self, current_value: float) -> float:
        """Applies the Z-score formula: (X - μ) / σ"""
        if not self.is_calibrated:
            return 0.0  # Return a neutral value if not calibrated

        return (current_value - self.mean) / self.std_dev

    def is_above_threshold(self, current_value: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """
        Checks if a given value or an array of values exceeds the calculated threshold.
        This method is vectorized to work with NumPy arrays.
        """
        if not self.is_calibrated:
            # If not calibrated, we must return the correct type.
            if isinstance(current_value, np.ndarray):
                # Return a boolean array of the same shape, filled with False.
                return np.full(current_value.shape, False)
            else:
                # Return a single boolean False for a single float.
                return False

        # This comparison works for both single floats and NumPy arrays due to vectorization.
        return current_value > self.threshold