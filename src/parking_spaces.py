from src.parking_lot import ParkingSpace
from src.utils.detection import detect_with_timestamps
from src.utils.prepare_detections import prepare_detections
from src.utils.process_parking_lot import apply_order
from src.utils.process_parking_lot import calc_crop_from_vertices
from src.utils.process_parking_lot import get_order
from src.utils.process_parking_lot import get_parking_spaces


def detect_parking_spaces(
        video_path: str,
        parking_polygon: list[tuple[int, int]],
        space_size: float = 0.5,
) -> list[ParkingSpace]:

    # Get crop from vertices
    video_data = {video_path: 0}
    crop = calc_crop_from_vertices(parking_polygon, padding=50, video_path=video_path)

    # Detect cars with timestamps for the range of frames
    detections = detect_with_timestamps(video_data, crop=crop)

    # Prepare detections for parking spaces
    fragment_df = prepare_detections(detections, parking_polygon)

    # Get parking spaces with each radius based on the bbox size and selected video fragment
    parking_spaces = get_parking_spaces(fragment_df, eps=15, threshold=0.9, space_size=space_size)

    # Initialize order of parking spaces, only the first semi-manual calibration
    order_left_right = get_order(parking_spaces)

    # Sort parking spaces based on the ratio of width to height, using calculated order
    parking_spaces_sorted = apply_order(parking_spaces, order_left_right)
    parking_spaces_sorted = parking_spaces_sorted.transpose().to_dict()
    parking_spaces_class = [ParkingSpace(number=int(k + 1), position=(int(v['cx']), int(v['cy'])), radius=v['radius'])
                            for k, v in parking_spaces_sorted.items()]

    return parking_spaces_class
