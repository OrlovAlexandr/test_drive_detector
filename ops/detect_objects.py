import yaml

from utils.detection import detect_with_timestamps
from utils.prepare_detections import prepare_detections


def detect_objects(video_data: dict[str, float],
                   lot_path: str,
                   detections_path: str) -> None:
    """
    Detect objects in the video using timestamps
    Args:
        video_data: dict - dictionary with video paths and timestamps
        lot_path: str - path to the parking lot file
        detections_path: str - save path to the detections file
    Returns:
        None
    """
    print('Detect objects...')
    # The cropped area should be the size of a parking lot with some padding.
    with open(lot_path, 'r') as f:
        config = yaml.safe_load(f)
        crop = config['crop_xyxy']
        parking_polygon = list(config['parking_polygon'].values())
    print('crop coordinates:', crop)

    detections = detect_with_timestamps(video_data, crop)

    # Prepare detections to get centers
    df = prepare_detections(detections, parking_polygon)

    # Save detections to parquet file
    df.to_parquet(detections_path)
    print('Detections saved to', detections_path)
    print()
    return None