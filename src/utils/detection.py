import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

from src.utils.prepare_detections import prepare_detections
from src.utils.process_parking_lot import calc_crop_from_vertices


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_video_time_creation(video_paths: list[str]) -> dict[str, float]:
    video_time_creation = 0
    video_time = {}
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        video_time[video_path] = video_time_creation
        video_time_creation += duration

    return video_time


def detect(video_path: str,
           frame_range: tuple[int, int],
           crop=None,
           every_n_frame: int = 1) -> np.ndarray:
    """Detect objects in a video."""
    if crop is None:
        crop = []

    # Load model
    model = YOLO("./models/yolov8m.pt")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    # Load video
    cap = cv2.VideoCapture(video_path)

    # Get video settings
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_in_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    logger.info(f"FPS: {fps}, resolution: {width}x{height}")

    if crop:
        crop_x1 = crop[0]
        crop_y1 = crop[1]
        crop_x2 = crop[2]
        crop_y2 = crop[3]
    else:
        crop_x1 = 0
        crop_y1 = 0
        crop_x2 = width
        crop_y2 = height

    if frame_range[0] >= duration_in_frames:
        return np.empty((0, 7), float)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_range[0])

    detections = np.empty((0, 7), float)  # [frame, class, conf, x1, y1, x2, y2]
    duration_of_range = int(duration_in_frames - frame_range[0])
    progress_bar = iter(tqdm(range(duration_of_range)))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if frame_number > frame_range[1]:
            break
        if frame_number % every_n_frame == 0:
            # Crop frame
            frame_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            # Detect objects
            results = model.predict(frame_crop, classes=[2, 5, 6, 7], verbose=False)

            frame_result = None
            for result in results:
                # Get boxes torch tensors
                result_boxes = result.boxes
                cls = result_boxes.cls.unsqueeze(1)
                conf = result_boxes.conf.unsqueeze(1)
                xyxy = result_boxes.xyxy
                fill_frame_number = torch.full_like(cls, frame_number)
                # Concatenate in specific order
                frame_result = torch.cat((fill_frame_number, cls, conf, xyxy), dim=1).cpu().numpy()
                # Return to original coordinates after cropping
                frame_result[:, [3, 4, 5, 6]] += [crop_x1, crop_y1, crop_x1, crop_y1]
            detections = np.vstack((detections, frame_result))
        next(progress_bar)

    cap.release()
    cv2.destroyAllWindows()

    return detections


def detect_in_seconds(videos_dict: dict[str, float],
                      crop=None,
                      every_n_frame: int = 1,
                      frame_range: tuple[int, int] = (1, 10 ** 18)) -> np.ndarray:
    """Detect and save detections with timestamps."""
    if crop is None:
        crop = []

    all_detections = np.empty((0, 8), float)
    for video_path, created_time in videos_dict.items():
        # Detect objects
        logger.info('Video file: %s', Path(video_path).name)
        detections = detect(video_path, crop=crop, every_n_frame=every_n_frame,
                            frame_range=frame_range)

        # Add timestamp to detections
        cap = cv2.VideoCapture(video_path)  # Read video with OpenCV
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get fps
        cap.release()
        time_in_seconds = detections[:, 0] * (1 / fps) + created_time  # Calculate timestamp based on created time

        detections_in_seconds = np.column_stack((detections, time_in_seconds))

        # Stack detections
        all_detections = np.vstack((all_detections, detections_in_seconds))

    return all_detections


def detect_objects(video_time: dict[str, float],
                   parking_polygon: list[tuple[int, int]]) -> pd.DataFrame:
    """Detect objects in the video using timestamps."""
    logger.info('Detecting objects...')
    video_paths = list(video_time.keys())

    crop = calc_crop_from_vertices(vertices=parking_polygon, padding=50, video_path=video_paths[0])
    detections = detect_in_seconds(video_time, crop)

    # Prepare detections to get centers
    return prepare_detections(detections, parking_polygon)
