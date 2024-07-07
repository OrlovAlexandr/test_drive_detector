import uuid
from pathlib import Path

import cv2

from src.utils.detection import get_video_time_creation, detect_objects
from src.utils.get_test_drive_events import get_test_drive_events
from src.utils.get_video_clips import get_video_clips
from src.utils.recalibrate_spaces import recalibrate_spaces


def detect_test_drives(
        video_paths: list[str],
        parking_polygon: list[tuple[float, float]],
        parking_spaces: dict[int, dict[str, float]],
        lot_id: uuid.UUID,
        output_video_dir: Path,
        time_offset: int = 15
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
    print('parking_polygon_pixels', parking_polygon_pixels)
    print('parking_spaces_pixels', parking_spaces_pixels)

    video_time = get_video_time_creation(video_paths)
    detections = detect_objects(video_time, parking_polygon_pixels)
    recalibrated_spaces = recalibrate_spaces(detections, parking_spaces_pixels)
    lots_states = get_test_drive_events(detections, recalibrated_spaces)
    output_clips_info = get_video_clips(video_time,
                                        recalibrated_spaces,
                                        lots_states,
                                        output_video_dir,
                                        lot_id=lot_id,
                                        offset=time_offset)
    return output_clips_info
