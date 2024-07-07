import cv2

from src.parking_lot import ParkingSpace
from src.utils.detection import detect_in_seconds
from src.utils.prepare_detections import prepare_detections
from src.utils.process_parking_lot import calc_crop_from_vertices, get_parking_spaces, get_order, apply_order


def detect_parking_spaces(video_path: str,
                          parking_polygon: list[tuple[float, float]],
                          space_size: float = 0.5) -> list[ParkingSpace]:
    """
    Get parking spaces with each radius based on the bbox size and selected video fragment
    """
    print('Get parking spaces...')

    # Convert parking polygon to pixels
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    parking_polygon = [(int(x * frame_width), int(y * frame_height)) for x, y in parking_polygon]

    # Get crop from vertices
    video_data = {video_path: 0}
    crop = calc_crop_from_vertices(parking_polygon, padding=50, video_path=video_path)

    # Detect cars with timestamps for the range of frames
    detections = detect_in_seconds(video_data, crop=crop, frame_range=(1, 61))

    # Prepare detections for parking spaces
    fragment_df = prepare_detections(detections, parking_polygon)

    # Get parking spaces with each radius based on the bbox size and selected video fragment
    parking_spaces = get_parking_spaces(fragment_df, eps=15, threshold=0.9, space_size=space_size)

    # Initialize order of parking spaces, only the first semi-manual calibration
    order_left_right = get_order(parking_spaces)

    # Sort parking spaces based on the ratio of width to height, using calculated order
    parking_spaces_sorted = apply_order(parking_spaces, order_left_right)
    parking_spaces_sorted = parking_spaces_sorted.transpose().to_dict()
    parking_spaces_class = [ParkingSpace(number=int(k + 1),
                                         position=(v['cx'] / frame_width,
                                                   v['cy'] / frame_height),
                                         radius=v['radius'] / frame_height)
                            for k, v in parking_spaces_sorted.items()]

    return parking_spaces_class
