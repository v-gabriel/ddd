from typing import Optional

import ffmpeg
import logging
import cv2

logger = logging.getLogger(__name__)

def correct_frame_rotation(frame, rotation_angle, label: Optional[str] = None):
    """
        Rotates an OpenCV frame based on the rotation angle,
    """
    if rotation_angle != 0 and label is not None:
        logger.info(f"Applying {rotation_angle}-degree rotation for {label}")

    if rotation_angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_angle == 270 or rotation_angle == -90: # Handle both conventions
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return frame

def get_video_rotation(video_path: str) -> int:
    """
    Uses ffprobe to get the rotation metadata from a video file.
    Checks both the new 'side_data' and old 'tags' for rotation info.
    Returns the rotation in degrees (e.g., 90, 180, -90/270).
    Returns 0 if no rotation tag is found or an error occurs.
    """
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)

        if video_stream is None:
            logger.warning(f"No video stream found for {video_path}")
            return 0

        # metadata
        if 'side_data_list' in video_stream:
            for side_data in video_stream['side_data_list']:
                if side_data.get('side_data_type') == 'Display Matrix':
                    if 'rotation' in side_data:
                        rotation = int(side_data['rotation'])
                        logger.debug(f"Found side_data rotation '{rotation}' for video {video_path}")
                        return rotation

        # older files
        if 'tags' in video_stream and 'rotate' in video_stream['tags']:
            rotation = int(video_stream['tags']['rotate'])
            logger.debug(f"Found tags rotation '{rotation}' for video {video_path}")
            return rotation

    except ffmpeg.Error as e:
        logger.error(f"Could not probe video {video_path}: {e.stderr.decode('utf-8')}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while probing {video_path}: {e}")

    logger.debug(f"No rotation metadata found for {video_path}. Assuming 0 degrees.")
    return 0

