import time
from datetime import datetime, timedelta

import cv2

# EVENT_DATE = '2024-06-15'
EVENT_DATE = '2024-06-15'


def get_video_time_creation(video_paths: list[str],
                            first_video_time: tuple[int, int, int]):
    event_date = EVENT_DATE
    hours, minutes, seconds = first_video_time
    event_date_dt = datetime.strptime(event_date, '%Y-%m-%d')
    add_time = event_date_dt + timedelta(hours=hours, minutes=minutes, seconds=seconds)
    video_time_creation = time.mktime(add_time.timetuple())
    video_time_creation = 0
    video_data = {}
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        video_data[video_path] = video_time_creation
        video_time_creation += duration

    return video_data
