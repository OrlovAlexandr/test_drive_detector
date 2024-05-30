import cv2
import numpy as np
import torch
from ultralytics import YOLO


def detect(video_path: str,
           crop_xyxy=[],
           n_frame: int = 1,
           frame_range: tuple = (0, np.inf)) -> np.ndarray:
    """
    Detect objects in a video.

    Args:
        video_path: Path to the video.
        crop_xyxy: XYXY coordinates of the frame crop.
        n_frame: Number of frames to process.
        frame_range: Range of frames to process.

    Returns:
        Numpy array of boxes.
    """

    # Load model
    model = YOLO("./source/models/yolov8m.pt")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load video
    cap = cv2.VideoCapture(video_path)

    # Get video settings
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(fps, width, height)

    if crop_xyxy:
        # Crop frame
        crop_x1 = crop_xyxy[0]
        crop_y1 = crop_xyxy[1]
        crop_x2 = crop_xyxy[2]
        crop_y2 = crop_xyxy[3]
    else:
        crop_x1 = 0
        crop_y1 = 0
        crop_x2 = width
        crop_y2 = height

    boxes = np.empty((0, 7), float)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)

        if frame_number > frame_range[1]:
            break

        if frame_number % 500 == 0:
            print(frame_number)

        if frame_number % n_frame == 0 and frame_number >= frame_range[0]:
            # Crop frame
            frame_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            # Detect objects
            results = model.predict(frame_crop, classes=[2, 5, 6, 7], verbose=False)

            for result in results:
                result_boxes = result.boxes
                cls = result_boxes.cls.unsqueeze(1)
                conf = result_boxes.conf.unsqueeze(1)
                xyxy = result_boxes.xyxy
                fill_frame_number = torch.full_like(cls, frame_number)

                frame_result = torch.cat((fill_frame_number, cls, conf, xyxy), dim=1).cpu().numpy()
                # Return to original coordinates
                frame_result[:, [3, 4, 5, 6]] += [crop_x1, crop_y1, crop_x1, crop_y1]

            boxes = np.vstack((boxes, frame_result))

    cap.release()
    cv2.destroyAllWindows()

    return boxes


def detect_with_timestamps(videos_dict: dict, crop: list = [],
                           n_frame: int = 1,
                           frame_range: tuple = (0, np.inf)):
    """
    Detect and save detections with timestamps
    Args:
        videos_dict: dict
            {video_path: created_time}
        crop: list
            [x1, y1, x2, y2]
        n_frame: int
            Read every N frame
    Returns:
        Numpy array of detections with timestamps
    """
    all_detections = np.empty((0, 8), float)
    for video_path, created_time in videos_dict.items():
        # Detect objects
        detections = detect(video_path, crop_xyxy=crop, n_frame=n_frame,
                            frame_range=frame_range)

        # Add timestamp to detections
        cap = cv2.VideoCapture(video_path)  # Read video with OpenCV
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get fps
        cap.release()

        timestamp = detections[:, 0] * (1 / fps) + created_time  # Calculate timestamp based on created time

        detections_ts = np.column_stack((detections, timestamp))

        # Stack all detections from all videos
        all_detections = np.vstack((all_detections, detections_ts))

    return all_detections
