import math
import psutil
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

BYTES_PER_PIXEL = 3

DEFAULT_MEMORY_FOOTPRINT_MULTIPLIER = 6
RESOLUTION_SCALE_FACTOR = 10
WORKER_FOOTPRINT_MULTIPLIER = 2

class TrainBatchUtility:
    """
    A thread-safe calculator for determining optimal batch size.

    This class encapsulates the batch size calculation logic and protects it with a threading.Lock to prevent race conditions
    (ensures that memory checks and the subsequent batch size calculation are performed as a single, atomic operation).

    Empirically tested to work on dev machine (CPU: AMD Ryzen 5 6600H, RAM: 32 GB, GPU: RTX 3060 6GB VRAM – laptop).
    May not behave as expected on different environments.

    Usage:
        Create a single instance of this class and share it among all threads that need to calculate batch sizes.
    """

    def __init__(self):
        """Initializes the calculator and its synchronization lock."""
        self.lock = threading.Lock()
        logger.info("ThreadSafeBatchCalculator initialized.")


    def calculate_optimal_batch_size(
            self,
            frame_width: int,
            frame_height: int,
            num_workers: int,
            ram_allocation_limit: float = 0.75,
            max_batch_size_cap: Optional[int] = None,
            memory_footprint_multiplier: int = DEFAULT_MEMORY_FOOTPRINT_MULTIPLIER,
            fallback_batch_size: int = 50,
            logger_override: Optional[logging.Logger] = None
    ) -> int:
        """
        Calculates an optimal, practical batch size by balancing RAM, CPU, and a hard cap.

        This function prevents excessively large batch sizes by adding a user-defined
        upper limit, making it robust for complex, memory-intensive pipelines.

        The logic is as follows:
        1.  **Estimate Memory Per Item**: Calculates memory needed per pipeline item, using a
            multiplier to account for complex processing (not just the raw frame).
        2.  **Calculate RAM-Based Max**: Determines the theoretical max batch size that fits
            within the allocated portion of *available* system RAM.
        3.  **Apply Hard Cap**: The batch size is capped at `max_batch_size_cap` (if provided).
            This is the key step to prevent unreasonable sizes (e.g., for 1920x1080).
        4.  **Align with CPU Workers**: The capped batch size is adjusted downwards to the nearest
            multiple of `num_workers` for efficient, stall-free processing.

        Args:
            frame_width: Width of the video frames.
            frame_height: Height of the video frames.
            num_workers: Number of parallel data loading workers.
            ram_allocation_limit: Fraction of *available* RAM to use (e.g., 0.75 for 75%).
            max_batch_size_cap: An optional integer to set a hard upper limit on the
                                batch size. Recommended for high-res video.
            memory_footprint_multiplier: A factor to estimate total memory per item, beyond
                                         just the raw frame size.
            fallback_batch_size: The batch size to use if a calculation is not possible.
            logger_override: An optional logger instance.

        Returns:
            The calculated optimal integer batch size, respecting all constraints.
        """

        with self.lock:
            effective_logger = logger_override or logger

            # --- 1. Input Validation ---
            if not (frame_width > 0 and frame_height > 0):
                raise ValueError("Frame width and height must be positive integers.")
            if num_workers < 0:
                raise ValueError("Number of workers cannot be negative.")

            # --- 2. Memory Constraint Calculation ---
            mem_info = psutil.virtual_memory()
            available_ram = mem_info.available
            ram_to_allocate = available_ram * ram_allocation_limit

            # Estimate the full memory footprint of one item in the pipeline.
            memory_per_item = ((frame_width * frame_height) * RESOLUTION_SCALE_FACTOR) * (BYTES_PER_PIXEL) * (memory_footprint_multiplier)

            if memory_per_item == 0:
                effective_logger.info(f"Cannot determine memory per item; returning fallback: {fallback_batch_size}")
                return fallback_batch_size

            max_batch_by_ram = math.floor(ram_to_allocate / memory_per_item)

            if max_batch_by_ram == 0:
                effective_logger.warning(
                    "Available RAM is too low for a single item. Defaulting to batch size 1."
                )
                return 1

            log_msg = (
                f"Dynamic Batch Size: Res={frame_width}x{frame_height}, Workers={num_workers} | "
                f"RAM Limit Batch={max_batch_by_ram}"
            )

            # --- 3. Apply Hard Cap ---
            if max_batch_size_cap is not None and max_batch_size_cap > 0:
                effective_batch_limit = min(max_batch_by_ram, max_batch_size_cap)
                log_msg += f" | Capped at={max_batch_size_cap} -> Effective Limit={effective_batch_limit}"
            else:
                effective_batch_limit = max_batch_by_ram
                log_msg += " | No Cap"

            # --- 4. CPU Efficiency & Final Batch Size ---
            if num_workers > 0:
                # Adjust the (potentially capped) batch size to be a multiple of the worker count.
                optimal_batch_size = math.floor(effective_batch_limit / (num_workers * WORKER_FOOTPRINT_MULTIPLIER))
            else:
                optimal_batch_size = effective_batch_limit

            final_batch_size = max(1, optimal_batch_size)

            log_msg += f" -> Final Batch Size (multiple of workers): {final_batch_size}"
            effective_logger.info(log_msg)

            return final_batch_size


