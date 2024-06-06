import json
import os.path

import cv2

from utils.process_video_clips import process_video_clips


def get_video_clips(video_data: dict[str, float],
                    events_path: str,
                    output_info_path: str,
                    offset: int = 15,
                    clip_duration: int = 30) -> list[dict[str, str]]:
    """
    Get video clips for each event.

    Args:
        video_data (dict): Dictionary with video paths and timestamps.
        events_path (str): Path to the events file.
        output_info_path (str): Save path to the output info file.
        offset (int, optional): Offset in seconds (N seconds before and 30-N seconds after). Defaults to 15.
        clip_duration (int, optional): Duration of the output clip in seconds. Defaults to 30.

    Returns:
        None
    """
    print('Get video clips for each event...')
    # Load events from the events file
    with open(events_path, 'r') as f:
        lots_states = json.load(f)

    # Get frame range and fps for each video
    video_range = {}
    for video_path, timestamp in video_data.items():
        # Read video with OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get fps
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()
        video_start = timestamp
        video_end = timestamp + duration
        print('fps:', fps, 'duration:', duration, 'video_start:', video_start, 'video_end:', video_end)
        video_range[video_path] = (video_start, video_end, fps)

    print('video_range:', video_range)
    print()

    # Create directory for videos if it doesn't exist
    video_dir = os.path.join(os.path.dirname(output_info_path), 'videos')
    if not os.path.exists(video_dir):
        os.makedirs(video_dir, exist_ok=True)

    # Get video clips for each lot
    output_clips_info = process_video_clips(lots_states,
                                            video_range,
                                            video_dir,
                                            timestamp_offset=offset,
                                            clip_duration=clip_duration)

    # Save clips info to json file
    with open(output_info_path, 'w') as f:
        json.dump(output_clips_info, f)
    print('Video clips info saved to', output_info_path)
    print()
    return output_clips_info
