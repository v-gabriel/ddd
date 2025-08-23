import concurrent.futures
import logging
import mimetypes
import os
import pickle
import re
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple, Any

import numpy as np
from decord import VideoReader, cpu, DECORDError
from sklearn.preprocessing import LabelEncoder

from src.config import ExperimentConfig
from src.core.image_processing import scale_frame_by_reference, \
    process_frame
from src.core.train_batch_utility import TrainBatchUtility
from src.data.metric_extractor import MetricExtractor
from src.detector.detector_utility import prepare_images_for_detector
from src.detector.face_detector import FaceDetector
from src.detector.landmark_detector import LandmarkDetector
from src.visualization.visualizations import save_face_detection_debug_image_on_failure, \
    save_landmark_failure_debug_image

logger = logging.getLogger(__name__)

thread_lock = threading.Lock()
thread_local_storage = threading.local()

train_batch_utility = TrainBatchUtility()

lock = threading.Lock()

@dataclass
class FrameData:
    """
    A unified container to hold all data and states for a single frame
    as it moves through the processing pipeline.
    """
    frame_idx: int
    is_calib: bool

    # Populated during processing
    rgb_frame: np.ndarray = field(default=None, repr=False)
    face_box: np.ndarray = field(default=None, repr=False)
    landmarks: np.ndarray = field(default=None, repr=False)


def init_worker(config_dict: dict):
    """
        Initializes each worker with required dependencies (landmark extractor, metric extractor, face detector).
    """
    config = ExperimentConfig(**config_dict)

    try:
        thread_local_storage.landmark_detector = LandmarkDetector(method=config.SETUP_LANDMARK_DETECTION_METHOD)

        thread_local_storage.metric_extractor = MetricExtractor(
            landmark_method=thread_local_storage.landmark_detector.method
        )

        thread_local_storage.face_detector = FaceDetector(
            method=config.SETUP_FACE_DETECTION_METHOD
        )
    except Exception as e:
        logging.error(f"Worker: Failed to load needed deps: {e}")
        thread_local_storage.face_detector = None
        thread_local_storage.landmark_detector = None
        thread_local_storage.metric_extractor = None


def get_video_properties(video_path: str, config: 'ExperimentConfig') -> Tuple['VideoReader', float, int, int, int, int, int]:
    """
    Safely opens a video file, reads its properties, and returns the reader and metadata.

    Args:
        video_path: The full path to the video file.
        config: Your experiment configuration object.

    Returns:
        A tuple containing:
        - vr (VideoReader): The initialized Decord video reader object.
        - original_fps (float): The average frames per second of the video.
        - total_frames (int): The total number of frames in the video.
        - frame_width (int): The width of the video frames.
        - frame_height (int): The height of the video frames.
        - scaled_frame_width (int): The scaled width of the video frames.
        - scaled_frame_height (int): The scaled height of the video frames.

    Raises:
        ValueError: If the video cannot be opened, is empty, or is invalid.
    """
    try:
        vr = VideoReader(video_path)

        original_fps = vr.get_avg_fps()
        total_frames = len(vr)

        if total_frames == 0:
            logger.warning(f"[{video_path}] Video file is empty. Skipping.")
            raise ValueError(f"No frames found in video: {video_path}")

        first_frame_ndarray = vr.get_batch([0])
        first_frame_numpy = first_frame_ndarray.asnumpy()[0]

        if first_frame_numpy.ndim != 3:
            raise ValueError(
                f"[{video_path}] Expected a 3D frame, but got shape {first_frame_numpy.shape}."
            )

        frame_height, frame_width, _ = first_frame_numpy.shape
        original_height = frame_height
        original_width = frame_width
        logger.info(
            f"[{video_path}] Video loaded successfully: {frame_width}x{frame_height} at {original_fps:.2f} FPS."
        )

        if config.SETUP_FRAME_SCALE is not None:
            first_frame_resized = scale_frame_by_reference(first_frame_numpy, config.SETUP_FRAME_SCALE)
            new_height, new_width = first_frame_resized.shape[:2]

            vr = VideoReader(video_path, width=new_width, height=new_height)

            first_frame_ndarray = vr.get_batch([0])
            first_frame_numpy = first_frame_ndarray.asnumpy()[0]

            frame_height, frame_width, _ = first_frame_numpy.shape
            logger.info(
                f"[{video_path}] Video resized successfully: {frame_width}x{frame_height} "
                f"(ref. {config.SETUP_FRAME_SCALE}) at {original_fps:.2f} FPS."
            )

        return vr, original_fps, total_frames, original_height, original_width, frame_width, frame_height

    except DECORDError as e:
        logger.error(f"[{video_path}] Decord failed to open or read video file: {e}")
        # Re-raise as a ValueError so the calling function knows to skip this video.
        raise ValueError(f"Could not process video file: {video_path}") from e

    except Exception as e:
        logger.error(f"[{video_path}] An unexpected error occurred in get_video_properties: {e}", exc_info=True)
        raise ValueError(f"Unexpected failure processing video: {video_path}") from e

def extract_features_from_video(video_path: str, label: int, config_dict: dict, calibration_file_required: bool = True):
    config = ExperimentConfig(**config_dict)

    # --- Component Initialization ---
    face_detector: 'FaceDetector' = getattr(thread_local_storage, 'face_detector')
    landmark_detector: 'LandmarkDetector' = getattr(thread_local_storage, 'landmark_detector')
    metric_extractor: 'MetricExtractor' = getattr(thread_local_storage, 'metric_extractor')

    if not all([face_detector, landmark_detector, metric_extractor]):
        logger.error(f"Required components not initialized for process on video {video_path}")
        return None, None, None, None, None

    try:
        vr, original_fps, total_frames, original_height, original_width, frame_width, frame_height \
            = get_video_properties(video_path, config)
    except Exception as e:
        logger.error(f"Failed to open video {video_path} to read properties: {e}")
        return [], [], 0.0, {}, {}

    batch_size = train_batch_utility.calculate_optimal_batch_size(
        frame_width=frame_width,
        frame_height=frame_height,
        num_workers=config.MAX_WORKERS,
        fallback_batch_size=config.FRAMES_BATCH
    ) if config.OPTIMIZE_BATCH_SIZE else config.FRAMES_BATCH

    effective_fps = original_fps
    skip_interval = max(1, int(round(original_fps / config.SETUP_TARGET_FPS))) \
        if config.SETUP_TARGET_FPS and 0 < config.SETUP_TARGET_FPS < original_fps \
        else 1
    if skip_interval > 1:
        effective_fps = original_fps / skip_interval
    logger.info(
        f"{video_path} Video FPS: {original_fps:.2f} | Target: {config.SETUP_TARGET_FPS} FPS. "
        f"-> Processing every {skip_interval} frames (Effective FPS: ~{effective_fps:.2f})"
    )

    video_name = os.path.basename(video_path)
    logger.info(f"--- Processing Video: {video_name} ---")
    logger.info(
        f"  - Resolution (W x H): {original_width}x{original_height} -> {frame_width}x{frame_height} (Target Scale: {config.SETUP_FRAME_SCALE})")
    logger.info(
        f"  - Frame Rate (FPS):   {original_fps:.2f} -> {effective_fps:.2f} (Target FPS: {config.SETUP_TARGET_FPS})")
    logger.info(f"  - Frame Sampling:     Processing 1 of every {skip_interval} frame(s).")
    logger.info("-" * (25 + len(video_name)))

    frame_indices = np.arange(0, total_frames, skip_interval)
    if config.SETUP_NUM_FRAMES_PER_VIDEO:
        frame_indices = frame_indices[:config.SETUP_NUM_FRAMES_PER_VIDEO]

    calib_indices = np.array([], dtype=int)
    if label == 0 and config.SETUP_CALIBRATION_FRAMES and calibration_file_required:
        calib_n_frames_to_sample = config.SETUP_CALIBRATION_FRAMES
        time_padding_seconds = 35
        start_calib = int(time_padding_seconds * config.SETUP_TARGET_FPS) if config.SETUP_TARGET_FPS else 0
        calib_period_duration = calib_n_frames_to_sample * skip_interval
        end_of_calib_period = min(start_calib + calib_period_duration, total_frames)
        calib_indices = np.arange(start_calib, end_of_calib_period, skip_interval)[:calib_n_frames_to_sample]

    all_indices_to_extract = sorted(list(set(frame_indices) | set(calib_indices)))
    calib_indices_set = set(calib_indices)
    logger.info(f"[{video_path}] Identified {len(all_indices_to_extract)} unique frames to extract.")

    # --- Batch Processing Loop ---
    main_features, calib_features = [], []
    timing_summary = defaultdict(float)

    start_time = time.perf_counter()
    for i in range(0, len(all_indices_to_extract), batch_size):
        batch_indices = all_indices_to_extract[i:i + batch_size]
        if not batch_indices:
            continue
        # --- Read frames and create a list of FrameData objects ---
        try:
            frame_tensor = vr.get_batch(batch_indices)
            if frame_tensor.ctx.device_type == 'gpu': # TODO: load frames in GPU? may need overhaul; quick test showed too big an overhead due to memory <-> VRAM swaps
                frame_tensor = frame_tensor.as_in_context(cpu(0))

            numpy_frames_bgr = frame_tensor.asnumpy()
            numpy_frames_rgb = numpy_frames_bgr[:, :, :, ::-1] # Important: standardize to RGB!

            frames_data_batch = [
                FrameData(
                    frame_idx=idx,
                    is_calib=idx in calib_indices_set,
                    rgb_frame=numpy_frames_rgb[k]
                ) for k, idx in enumerate(batch_indices)
            ]
            del numpy_frames_bgr
            del numpy_frames_rgb
            del frame_tensor
        except Exception as e:
            logger.error(f"Failed to read a batch from {video_path}: {e}")
            continue  # Skip to the next batch
        timing_summary['A_frame_loading'] += time.perf_counter() - start_time

        # --- Preprocess frames for face detection ---
        start_time = time.perf_counter()
        tmp_preprocessed_frames = prepare_images_for_detector([fd.rgb_frame for fd in frames_data_batch], config.SETUP_FACE_DETECTION_METHOD.value)
        timing_summary['C0_face_detection_prepro'] += time.perf_counter() - start_time

        # --- Detect Faces ---
        start_time = time.perf_counter()
        batch_detection_results = face_detector.detect_faces_batch(
            frames=tmp_preprocessed_frames,
            force_sequential=config.FORCE_SEQUENTIAL_DETECTION
        )
        timing_summary['C_face_detection'] += time.perf_counter() - start_time
        del tmp_preprocessed_frames

        # --- Populate FrameData with results  ---
        frames_with_faces = []
        for i, (face_boxes, confidence, _) in enumerate(batch_detection_results):
            frame_data = frames_data_batch[i]
            if face_boxes is not None and len(face_boxes) > 0:
                frame_data.face_box = face_boxes[0]
                frames_with_faces.append(frame_data)

        landmark_frames_input = []
        landmark_face_boxes_input = []
        landmark_face_rois_input = []

        # --- Detect Landmarks only on frames with successfully detected faces ---
        if frames_with_faces:
            # --- Preprocess frames for landmark detection ---
            start_time = time.perf_counter()
            tmp_preprocessed_frames = prepare_images_for_detector([fd.rgb_frame for fd in frames_with_faces], config.SETUP_LANDMARK_DETECTION_METHOD.value)
            timing_summary['D0_landmark_detection_prepro'] += time.perf_counter() - start_time

            for fd, processed_frame in zip(frames_with_faces, tmp_preprocessed_frames):
                landmark_frames_input.append(processed_frame)
                landmark_face_boxes_input.append(fd.face_box)
                x, y, w, h = fd.face_box
                face_roi = processed_frame[y:y + h, x:x + w]
                landmark_face_rois_input.append(np.ascontiguousarray(face_roi))
            del tmp_preprocessed_frames

            start_time = time.perf_counter()
            batch_landmark_results = landmark_detector.detect_landmarks_batch(
                frames=landmark_frames_input,
                face_rects=landmark_face_boxes_input,
                face_rois=landmark_face_rois_input,
                force_sequential=config.FORCE_SEQUENTIAL_DETECTION
            )
            timing_summary['D_landmark_detection'] += time.perf_counter() - start_time

            for i, landmarks in enumerate(batch_landmark_results):
                frames_with_faces[i].landmarks = landmarks

        del landmark_face_boxes_input
        del landmark_frames_input
        del landmark_face_rois_input

        # --- Final Loop: Feature Extraction, Visualization, and Error Handling ---
        for frame_data in frames_data_batch:
            # Case 1: No face was detected
            if frame_data.face_box is None:
                logger.warning(f"Frame {frame_data.frame_idx} in {video_path}: No face detected.")
                save_face_detection_debug_image_on_failure(
                    frame=frame_data.rgb_frame,
                    frame_idx=frame_data.frame_idx,
                    video_path=video_path,
                    failure_reason="Face Not Found",
                    config=config,
                )
                continue

            # Case 2: Face was detected, but landmark detection failed
            if frame_data.landmarks is None:
                logger.warning(f"Frame {frame_data.frame_idx} in {video_path}: Landmarks not found.")
                save_landmark_failure_debug_image(
                    frame=frame_data.rgb_frame,
                    frame_idx=frame_data.frame_idx,
                    face_rect=frame_data.face_box,
                    video_path=video_path,
                    failure_reason="Landmarks Not Found",
                    config=config
                )
                continue

            timestamp = frame_data.frame_idx / original_fps # Important: standardize to seconds!

            start_time = time.perf_counter()
            frame_features = metric_extractor.map_raw_features(
                frame_data.rgb_frame, frame_data.landmarks, frame_data.face_box, timestamp, frame_data.frame_idx
            )
            timing_summary['F_raw_feature_extraction'] += time.perf_counter() - start_time

            if not frame_features:
                continue

            # Aggregate results
            if frame_data.is_calib:
                calib_features.append(frame_features)
            else:
                main_features.append((frame_features, label))

    # --- 4. Final Logging and Return ---
    successful_extractions = len(calib_features) + len(main_features)
    extraction_rate = successful_extractions / len(all_indices_to_extract) if len(all_indices_to_extract) > 0 else 0.0
    logger.info(
        f"Finished processing for {video_path}: {successful_extractions}/{len(all_indices_to_extract)} frames extracted ({extraction_rate:.1%})")

    total_pipeline_time = sum(timing_summary.values())
    logger.info(f"--- Performance Breakdown for {video_path} ---")
    logger.info(f"Total pipeline time: {total_pipeline_time:.2f} seconds for {len(all_indices_to_extract)} frames.")
    for stage, duration in sorted(timing_summary.items()):
        percentage = (duration / total_pipeline_time) * 100 if total_pipeline_time > 0 else 0
        stage_name = stage.split('_', 1)[1].replace('_', ' ').title()
        logger.info(f"  - {stage_name:<22}: {duration:.2f}s ({percentage:.1f}%)")
    logger.info("-" * 45)

    face_metrics = face_detector.log_final_metrics(True)
    landmark_metrics = landmark_detector.log_final_metrics(True)

    video_processing_metrics = {
        'original_fps': original_fps,
        'effective_fps': effective_fps,
        'target_fps': config.SETUP_TARGET_FPS,
        'original_resolution': (original_width, original_height),
        'effective_resolution': (frame_width, frame_height),
        'target_scale': config.SETUP_FRAME_SCALE,
        'skip_interval': skip_interval
    }

    return main_features, calib_features, extraction_rate, face_metrics, landmark_metrics, video_processing_metrics


def _process_video_wrapper(args):
    """
        Thread processing function.
    """
    video_path, label, subject_id, config_dict, calibration_file_required, temp_dir = args

    temp_filename = os.path.join(temp_dir, f"{subject_id}_{label}_{os.path.basename(video_path)}.pkl")

    if os.path.exists(temp_filename):
        logger.info(f"Reusing existing temporary result for {video_path} from a previous run.")
        return temp_filename

    logger.info(f"Processing video: {video_path}")
    main_results, calib_results, extraction_rate, face_metrics, landmark_metrics, video_metrics = extract_features_from_video(
        video_path, label, config_dict, calibration_file_required
    )

    data_loaded = (main_results is not None and len(main_results) != 0)
    if data_loaded:
        with open(temp_filename, 'wb') as f:
            pickle.dump({
                "main_results": main_results,
                "calib_results": calib_results,
                "extraction_rate": extraction_rate,
                "face_metrics": face_metrics,
                "landmark_metrics": landmark_metrics,
                "video_metrics": video_metrics,
                "video_path": video_path,
                "subject_id": subject_id
            }, f)

    return temp_filename if data_loaded else None


def _save_progress_atomically(path: str, data: dict):
    abs_path = os.path.abspath(path)
    temp_path = f"{abs_path}.tmp.npz"
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    try:
        np.savez_compressed(temp_path, **data)
        os.replace(temp_path, abs_path)
    except Exception as e:
        logging.error(f"Failed during atomic save: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_data(config: ExperimentConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads and processes video data for experiment runs.
    """

    final_subjects = [
        subject for subject in config.SETUP_INCLUDE_SUBJECTS
        if subject not in config.SETUP_EXCLUDE_SUBJECTS
    ]

    main_data_cache, progress_cache, calibration_data_cache = config.get_cache_filenames()
    cache_hash = re.search(r'_h-([a-f0-9]+)', os.path.basename(main_data_cache)).group(1)

    calibration_data = defaultdict(list)
    calibration_file_required = False # TODO: improve calibration logic (check config.py)?

    cache_dir = os.path.dirname(main_data_cache)
    os.makedirs(cache_dir, exist_ok=True)

    logger.info("Starting data extraction/continue process...")

    main_data_lists = defaultdict(list)
    processed_videos = set()
    extraction_rates_dict = defaultdict(list)

    aggregated_face_metrics = {'calls': 0, 'failures': 0, 'time': 0.0}
    aggregated_landmark_metrics = {'calls': 0, 'failures': 0, 'time': 0.0}
    aggregated_video_metrics = []

    processed_subject_labels = set()
    if processed_videos:
        logger.info(f"Building lookup for {len(processed_videos)} processed videos to prevent re-processing...")
        for path in processed_videos:
            try:
                subject = os.path.basename(os.path.dirname(path))
                basename = os.path.splitext(os.path.basename(path))[0]
                label = int(basename.split('_')[0])
                processed_subject_labels.add((subject, label))
            except (ValueError, IndexError):
                logger.warning(f"Could not parse subject/label from processed video path: {path}")
        logger.info(f"Found {len(processed_subject_labels)} unique subject-label pairs in the cache.")

    # --- Identify and schedule videos for processing: takes higher index videos (e.g. 10_2.mov) if multiple exist ---
    best_video_candidates = defaultdict(dict)
    tasks = []
    for subject in sorted(os.listdir(config.DATASET_ROOT_DIR)):
        subject_path = os.path.join(config.DATASET_ROOT_DIR, subject)
        if not os.path.isdir(subject_path): continue

        for filename in os.listdir(subject_path):
            video_path = os.path.join(subject_path, filename)
            mime_type, _ = mimetypes.guess_type(filename)

            # Skip non-video files
            if not (mime_type and mime_type.startswith('video/')):
                continue

            basename = os.path.splitext(filename)[0]
            try:
                parts = basename.split('_')
                label = int(parts[0])

                subject_id = int(subject)
                if (subject, label) in processed_subject_labels or subject_id not in final_subjects:
                    continue

                # Filter out videos with labels not in the inclusion list
                if config.SETUP_INCLUDE_VIDEOS and label not in config.SETUP_INCLUDE_VIDEOS:
                    continue

                # Use a default suffix of -1 for files without one (e.g., "0.mp4")
                suffix = int(parts[1]) if len(parts) > 1 else -1

                if label not in best_video_candidates[subject]:
                    best_video_candidates[subject][label] = (video_path, suffix)
                else:
                    current_best_suffix = best_video_candidates[subject][label][1]
                    if suffix > current_best_suffix:
                        best_video_candidates[subject][label] = (video_path, suffix)

            except (ValueError, IndexError):
                continue

    temp_dir = os.path.join(cache_dir, f"{os.path.basename(main_data_cache)}_progress")
    os.makedirs(temp_dir, exist_ok=True)

    for subject, videos in best_video_candidates.items():
        for label, (video_path, _) in videos.items():
            tasks.append((video_path, label, subject, asdict(config), calibration_file_required, temp_dir))

    if tasks:
        logger.info(f"Starting extraction for {len(tasks)} new videos...")
        for task in tasks:
            # task[0] is the video_path
            logger.info(f"  -> Queued: {task[0]}")
        logger.info("-------------------------------------------------")

    failed_tasks = []
    result_filepaths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS, initializer=init_worker,
                                               initargs=(asdict(config),)) as executor:
        futures = {executor.submit(_process_video_wrapper, task): task for task in tasks}

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            video_path, _, _, _, _, _ = futures[future]

            try:
                result_filepath = future.result()

                if result_filepath is None:
                    logger.warning(f"Worker for video {video_path} returned no result. Marking as failed.")
                    failed_tasks.append(video_path)
                    continue

                result_filepaths.append(result_filepath)

                logger.info(f"Video {video_path} thread succeeded ({result_filepath}).")

            except Exception as e:
                logger.error(f"Task for video {video_path} generated an exception: {e}", exc_info=True)
                failed_tasks.append(video_path)
            # finally:
            #     if result_filepath and os.path.exists(result_filepath):
            #         os.remove(result_filepath)

    logger.info(f"Aggregating data...")
    total_postprocessing_time = 0.0
    for agg_idx, result_filepath in enumerate(sorted(result_filepaths)):
        if result_filepath in processed_videos:
            logger.info(f"Video {result_filepath} already processed. Skipped from aggregating.")
            continue  # Already processed by another thread, can happen with retries

        with open(result_filepath, 'rb') as f:
            result_data = pickle.load(f)

        calib_results = result_data["calib_results"]
        extraction_rate = result_data["extraction_rate"]
        face_metrics = result_data["face_metrics"]
        landmark_metrics = result_data["landmark_metrics"]
        subject_id = result_data["subject_id"]
        video_path = result_data["video_path"]

        main_results = result_data["main_results"]
        for features_dict, lbl in main_results:
            face_crop_original = features_dict.get('face_crop')

            face_crop_processed, elapsed = process_frame(face_crop_original, config.SETUP_POSTPROCESS_FLAGS)
            total_postprocessing_time += elapsed

            features_dict['face_crop'] = face_crop_processed
            for key, value in features_dict.items():
                main_data_lists[key].append(value)

            main_data_lists['labels'].append(lbl)
            main_data_lists['groups'].append(subject_id)

        if calib_results and calibration_file_required:
            calibration_data[subject_id].extend(calib_results)

        if face_metrics:
            aggregated_face_metrics['calls'] += face_metrics['calls']
            aggregated_face_metrics['failures'] += face_metrics['failures']
            aggregated_face_metrics['time'] += face_metrics['time']

        if landmark_metrics:
            aggregated_landmark_metrics['calls'] += landmark_metrics['calls']
            aggregated_landmark_metrics['failures'] += landmark_metrics['failures']
            aggregated_landmark_metrics['time'] += landmark_metrics['time']

        if "video_metrics" in result_data:
            video_metrics = result_data["video_metrics"]
            video_metrics['video_path'] = result_data["video_path"]

            aggregated_video_metrics.append(video_metrics)

        if extraction_rate is not None:
            extraction_rates_dict[subject_id] = extraction_rate

        processed_videos.add(video_path)

    # --- Extraction Summary ---
    logger.info("--- Extraction Summary ---")
    logger.info(f"Tasks Scheduled: {len(tasks)}")
    logger.info(f"Successfully Processed: {len(processed_videos)}")
    logger.info(f"Failed Tasks: {len(failed_tasks)}")
    logger.info(f"Total time spent in 'process_frame' across all frames: {total_postprocessing_time:.2f} seconds.")
    if failed_tasks:
        logger.warning("The following videos failed to process:")
        for path in failed_tasks:
            logger.warning(f"  - {path}")

    if not main_data_lists.get('labels'):
        raise ValueError("No labels found. Corrupted or nothing was extracted.")

    if not processed_videos:
        logger.error("No videos were successfully processed. Halting before saving.")
        raise ValueError("Extraction failed for all videos. Check logs for details.")

    # --- Finalize & Data Summary ---
    logger.info(f"Pickling data...")
    main_data_np = {key: try_convert_to_array(val) for key, val in main_data_lists.items()}
    le = LabelEncoder().fit(main_data_np['labels'])
    main_data_np['labels'] = le.transform(main_data_np['labels'])
    main_data_np['label_encoder'] = le
    main_data_np['extraction_rates'] = extraction_rates_dict

    if aggregated_face_metrics and aggregated_face_metrics['calls'] > 0:
        failure_rate = (aggregated_face_metrics['failures'] / aggregated_face_metrics['calls']) * 100
        avg_time_ms = (aggregated_face_metrics['time'] / aggregated_face_metrics['calls']) * 1000
        logger.info("--- Aggregated Face Detection Performance ---")
        logger.info(f"Total Detection Calls: {aggregated_face_metrics['calls']:,}")
        logger.info(f"Total Failures: {aggregated_face_metrics['failures']:,} ({failure_rate:.2f}%)")
        logger.info(f"Total Processing Time: {aggregated_face_metrics['time']:.2f} seconds")
        logger.info(f"Average Time per Frame: {avg_time_ms:.2f} ms")
        logger.info("-------------------------------------------")

    if aggregated_landmark_metrics and aggregated_landmark_metrics['calls'] > 0:
        failure_rate = (aggregated_landmark_metrics['failures'] / aggregated_landmark_metrics['calls']) * 100
        avg_time_ms = (aggregated_landmark_metrics['time'] / aggregated_landmark_metrics['calls']) * 1000
        logger.info("--- Aggregated Landmark Detection Performance ---")
        logger.info(f"Total Detection Calls: {aggregated_landmark_metrics['calls']:,}")
        logger.info(f"Total Failures: {aggregated_landmark_metrics['failures']:,} ({failure_rate:.2f}%)")
        logger.info(f"Total Processing Time: {aggregated_landmark_metrics['time']:.2f} seconds")
        logger.info(f"Average Time per Frame: {avg_time_ms:.2f} ms")
        logger.info("-------------------------------------------")

    if aggregated_video_metrics:
        logger.info("--- Aggregated Video Processing Metrics ---")
        # Sort by video path for consistent output
        for metrics in sorted(aggregated_video_metrics, key=lambda x: x.get('video_path', '')):
            video_name = os.path.basename(metrics.get('video_path', 'N/A'))
            logger.info(f"  Video: {video_name}")

            orig_res = metrics.get('original_resolution', ('N/A', 'N/A'))
            eff_res = metrics.get('effective_resolution', ('N/A', 'N/A'))
            logger.info(f"    - Resolution (W x H): {orig_res[0]}x{orig_res[1]} -> {eff_res[0]}x{eff_res[1]}")

            orig_fps = metrics.get('original_fps', 0)
            eff_fps = metrics.get('effective_fps', 0)
            target_fps = metrics.get('target_fps', 'N/A')
            logger.info(f"    - Frame Rate (FPS):   {orig_fps:.2f} -> {eff_fps:.2f} (Target: {target_fps})")
        logger.info("-------------------------------------------")

    log_extraction_summary(extraction_rates_dict)

    return main_data_np, calibration_data

def try_convert_to_array(val_list):
    try:
        # Accept LabelEncoder, dict, or empty
        if not val_list or isinstance(val_list[0], (str, int, float, np.generic)):
            return np.array(val_list)
        if isinstance(val_list[0], np.ndarray):
            shapes = [x.shape for x in val_list]
            # If all shapes match, stack as an ndarray
            if all(s == shapes[0] for s in shapes):
                return np.stack(val_list)
            else:
                # Keep as dtype=object
                return np.array(val_list, dtype=object)
        else:
            # Fallback: most likely a list of scalars
            return np.array(val_list, dtype=object)
    except Exception as e:
        logger.error(f"Error while converting pickled data: {e}")
        raise e

def is_ragged(val):
    return (
        isinstance(val, list) and val and isinstance(val[0], np.ndarray) and
        len(set(x.shape for x in val)) > 1
    )

# TODO: incorporate with config.py, skip if below threshold
def log_extraction_summary(extraction_rates_dict: Dict[int, float]):
    """Prints a summary of feature extraction success rates."""
    logger.info("\n" + "=" * 50)
    logger.info("--- Feature Extraction Rate Summary ---")
    if not extraction_rates_dict:
        logger.info("No extraction rates to report.")
        logger.info("=" * 50 + "\n")
        return

    successful_subjects = 0
    total_subjects = len(extraction_rates_dict)

    for subject_id, rate in extraction_rates_dict.items():
        # You could define a success threshold in your config
        status = "SUCCESS" if rate >= 0.90 else "LOW RATE"
        logger.info(f"  - Subject {subject_id}: {rate:.2%} [{status}]")
        if rate >= 0.90:
            successful_subjects += 1

    overall_success_rate = (successful_subjects / total_subjects) * 100
    logger.info("---")
    logger.info(
        f"Overall Success: {successful_subjects} / {total_subjects} subjects met threshold ({overall_success_rate:.1f}%).")
    logger.info("=" * 50 + "\n")