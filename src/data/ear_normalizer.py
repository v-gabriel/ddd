from typing import Optional, Union

import numpy as np
import logging

import pandas as pd

logger = logging.getLogger(__name__)

class EarNormalizer:
    """
    Handles the calibration and normalization of Eye Aspect Ratio (EAR) values
    for a specific individual. Uses Min-Max scaling.
    """
    def __init__(self):
        self.ear_min = None
        self.ear_max = None
        self.is_calibrated = False

        self.FAILURE_THRESHOLD = 1e-6
        self.SUSPICIOUS_THRESHOLD = 0.125  # A typical blink creates a range > 0.1

    def calibrate(self, ear_values: pd.Series, info_str: Optional[str] = None):
        """
        Calculates and stores the min and max EAR from a list of values.
        Uses Min-Max scaling.
        """
        if ear_values is None or ear_values.empty:
            logger.error("Calibration failed: No EAR values provided. "
                         + (f"({info_str})" if info_str is not None else ''))
            return

        log_context = f"({info_str})" if info_str else ""

        self.ear_min = min(ear_values)
        self.ear_max = max(ear_values)
        ear_range = self.ear_max - self.ear_min

        # avoid division by zero
        if (self.ear_max - self.ear_min) < self.FAILURE_THRESHOLD:
            logger.error(f"Calibration failed: Min and Max EAR are too close. User might not have blinked. {log_context}")
            self.is_calibrated = False
        elif ear_range < self.SUSPICIOUS_THRESHOLD:
            logger.warning(f"Calibration SUSPICIOUS: EAR range is very small ({ear_range:.4f}). "
                           f"Subject may not have blinked, or more frames may be needed. {log_context}")
            self.is_calibrated = True  # Calibrated, but with a warning
            logger.info(f"EAR_min={self.ear_min:.4f}, EAR_max={self.ear_max:.4f} {log_context}")
        else:
            self.is_calibrated = True
            logger.info(f"Calibration successful: "
                        f"EAR_min={self.ear_min:.4f}, "
                        f"EAR_max={self.ear_max:.4f}. {log_context}"    )

    def normalize(self, current_ear: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Normalizes a single EAR value or a pandas Series of EAR values.

        Returns:
            The normalized EAR value(s). Returns original value(s) if not calibrated.
        """
        if not self.is_calibrated:
            return current_ear

        ear_range = self.ear_max - self.ear_min
        if ear_range < self.FAILURE_THRESHOLD:
            # Return 1.0 for a single float, or a Series of 1.0s for a Series input
            return np.ones_like(current_ear)

        # Works on both single floats and entire pandas Series objects.
        normalized = (current_ear - self.ear_min) / ear_range

        # np.clip is also vectorized and works on both scalars and arrays/Series.
        return np.clip(normalized, 0.0, 1.0)
