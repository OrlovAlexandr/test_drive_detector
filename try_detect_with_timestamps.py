import cv2
import numpy as np
import pandas as pd
import yaml

from utils.detection import detect, detect_with_timestamps

if __name__ == "__main__":
    # # Videos should be selected based on the workday range
    # # The dictionary should be compiled from the selected videos and their created timestamps
    save_path = f"./results/detections/all_detections.npy"

    video_dict = {'./source/videos/parking_lot_ii_001.mp4': 1716791179.0,
                  './source/videos/parking_lot_ii_002.mp4': 1716791323.88}

    # The cropped area should be the size of a parking lot with some padding.
    crop = [460, 570, 970, 1080]

    all_detections = detect_with_timestamps(video_dict, crop)
    with open(save_path, 'wb') as f:
        np.save(f, all_detections)

    with open(save_path, 'rb') as f:
        all_detections = np.load(f)

    # Create dataframe from detections
    df = pd.DataFrame(all_detections, columns=['frame', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2', 'timestamp'])
    df.to_parquet('./results/detections/all_detections.parquet')

    video_path = 'source/videos/parking_lot_ii_001.mp4'

    # Read config file
    with open('./config/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    parking_spaces = pd.DataFrame(config['parking_spaces'])
    print(parking_spaces)

    cap = cv2.VideoCapture(video_path)  # Read video with OpenCV
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        for space in parking_spaces.values:
            space = [int(x) for x in space]
            cv2.circle(frame, (space[0], space[1]), 5, (0, 0, 255), -1)
            cv2.circle(frame, (space[0], space[1]), space[2], (0, 0, 255), 1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
