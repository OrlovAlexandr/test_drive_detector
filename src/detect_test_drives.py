import logging
import uuid
from pathlib import Path

import cv2

from src.utils.detection import detect_objects
from src.utils.detection import get_video_time_creation
from src.utils.get_clips_from_events import get_clips_list
from src.utils.get_clips_from_events import get_test_drives
from src.utils.get_clips_from_events import save_clips_to_videos
from src.utils.get_test_drive_events import get_test_drive_events
from src.utils.recalibrate_spaces import recalibrate_spaces


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_test_drives(
        video_paths: list[str],
        parking_polygon: list[tuple[float, float]],
        parking_spaces: dict[int, dict[str, float]],
        lot_id: uuid.UUID,
        output_video_dir: Path,
        clip_duration: int = 30,  # noqa: ARG001
        time_offset: int = 10,  # noqa: ARG001
):
    # Convert parking polygon and parking spaces to pixels
    cap = cv2.VideoCapture(video_paths[0])
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    parking_polygon_pixels = [(int(x * video_width), int(y * video_height)) for x, y in parking_polygon]
    parking_spaces_pixels = {int(space_id): {'cx': space['cx'] * video_width,
                                             'cy': space['cy'] * video_height,
                                             'radius': space['radius'] * video_height}
                             for space_id, space in parking_spaces.items()}
    logger.info('Parking polygon pixels: %s', parking_polygon_pixels)
    logger.info('Parking spaces pixels: %s', parking_spaces_pixels)

    video_time = get_video_time_creation(video_paths)
    detections = detect_objects(video_time, parking_polygon_pixels)
    recalibrated_spaces = recalibrate_spaces(detections, parking_spaces_pixels)
    lots_states = get_test_drive_events(detections, recalibrated_spaces)
    test_drives = get_test_drives(lots_states, lot_id)
    clips = get_clips_list(test_drives, video_time)
    save_clips_to_videos(
        clips,
        output_video_dir,
        parking_spaces_pixels,
        lots_states,
        compress=True,
    )

    return test_drives
