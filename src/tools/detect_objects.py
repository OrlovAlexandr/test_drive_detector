import time

import cv2
import os

import pandas as pd
from datetime import datetime, timedelta

from utils.detection import detect_with_timestamps
from utils.prepare_detections import prepare_detections
from utils.process_parking_lot import calc_crop_from_vertices



def detect_objects(video_data: dict[str, float],
                   parking_polygon: list[tuple[int, int]],
                   first_video_time: tuple[int, int, int]) -> pd.DataFrame:
    """
    Detect objects in the video using timestamps
    """
    print(f'Detecting objects...')
    # Get video creation time in unix timestamp

    video_paths = list(video_data.keys())

    crop = calc_crop_from_vertices(parking_polygon, padding=50, video_path=video_paths[0])
    detections = detect_with_timestamps(video_data, crop)

    # Prepare detections to get centers
    detections = prepare_detections(detections, parking_polygon)

    return detections
