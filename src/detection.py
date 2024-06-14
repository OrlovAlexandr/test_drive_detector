from src.tools.detect_objects import detect_objects
from src.tools.get_test_drive_events import get_test_drive_events
from src.tools.get_video_clips import get_video_clips
from src.tools.recalibrate_spaces import recalibrate_spaces


def detect_test_drives(
        video_paths: list[str],
        parking_polygon: list[tuple[int, int]],
        parking_spaces: dict[int, dict[str, int | float]],
        output_video_dir: str,
):
    detections = detect_objects(video_paths, parking_polygon)
    reacalibrated_spaces = recalibrate_spaces(detections, parking_spaces)
    lots_states = get_test_drive_events(detections, reacalibrated_spaces)
    output_clips_info = get_video_clips(video_paths, lots_states, output_video_dir)
    return output_clips_info
