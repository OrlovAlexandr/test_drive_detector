import os.path
import uuid
from datetime import datetime, timedelta
import time
import cv2
from src.test_drive import TestDrive
from utils.process_video_clips import process_video_clips

EVENT_DATE = '2024-06-15'


def get_video_clips(video_data: dict[str, float],
                    lots_states: dict[int, list[tuple[str, int]]],
                    first_video_time: tuple[int, int, int],
                    offset: int = 15,
                    clip_duration: int = 30) -> list[dict[str, str]]:
    """
    Get video clips for each event.
    """
    print('Get video clips of test drives...')

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
        video_range[video_path] = (video_start, video_end, fps)

    # # Create directory for videos if it doesn't exist
    # if not os.path.exists(output_video_dir):
    #     os.makedirs(output_video_dir, exist_ok=True)

    # Get video clips for each lot
    output_clips_info = process_video_clips(lots_states,
                                            video_range,
                                            timestamp_offset=offset,
                                            clip_duration=clip_duration)
    print('=' * 40)


    sorted_data = sorted(output_clips_info, key=lambda x: x['event_timestamp'])
    print(sorted_data)

    # Function to convert timestamp to datetime
    def timestamp_to_datetime(timestamp):
        return datetime.utcfromtimestamp(timestamp)

    # Dictionary to store TestDrive instances
    test_drives = {}

    # Iterate over sorted data and create/update TestDrive instances
    for entry in sorted_data:
        parking_space = entry['parking_space']
        test_drive_number = entry['test_drive_number']
        event_timestamp = entry['event_timestamp']
        state = entry['state']

        clip_path = entry['clip_path']

        print('parking_space:', parking_space)
        print('test_drive_number:', test_drive_number)
        print('event_timestamp:', event_timestamp)
        print('state:', state)
        print('clip_path:', clip_path)
        print('=' * 40)


    # Convert dictionary to list for output
    test_drive_list = list(test_drives.values())

    # Print out all test drives
    for test_drive in test_drive_list:
        print(test_drive)

    return output_clips_info


