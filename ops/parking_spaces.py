import yaml

from utils.detection import detect_with_timestamps
from utils.prepare_detections import prepare_detections
from utils.process_parking_lot import calc_crop_from_vertices, get_parking_spaces, get_order, apply_order


def process_parking_spaces(video_data: dict[str, float],
                           parking_polygon: list[tuple[int, int]],
                           frame_range: tuple[int, int],
                           lot_path: str) -> None:
    """
    Get parking spaces with each radius based on the bbox size and selected video fragment
    Args:
        video_data: dict - dictionary with video paths and timestamps
        parking_polygon: list - list of vertices of the parking polygon
        frame_range: tuple - range of frames to be processed
        lot_path: str - path to the parking lot file

    Returns:
        None
    """
    print('Get parking spaces...')
    with open(lot_path, 'w') as f:
        parking_poly_dict = {idx: [i[0], i[1]]
                             for idx, i in enumerate(parking_polygon)}  # Convert vertices to dict for yaml
        yaml.dump({'parking_polygon': parking_poly_dict}, f)

    # Get crop from vertices
    video_path = list(video_data.keys())[0]
    crop = calc_crop_from_vertices(parking_polygon, padding=50, video_path=video_path)

    # Detect cars with timestamps for the range of frames
    detections = detect_with_timestamps(video_data, frame_range=frame_range, crop=crop)

    # Prepare detections for parking spaces
    fragment_df = prepare_detections(detections, parking_polygon)

    # Get parking spaces with each radius based on the bbox size and selected video fragment
    parking_spaces = get_parking_spaces(fragment_df, eps=15, threshold=0.9)

    # Initialize order of parking spaces, only the first semi-manual calibration
    order_left_right = get_order(parking_spaces)

    # Sort parking spaces based on the ratio of width to height, using calculated order
    parking_spaces_sorted = apply_order(parking_spaces, order_left_right)

    # Add values to yaml file
    with open(lot_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(lot_path, 'w') as f:
        config['order_left_right'] = order_left_right
        config['crop_xyxy'] = crop
        config['spaces_number'] = parking_spaces_sorted.shape[0]

        parking_spaces_dict = parking_spaces.to_dict()
        config['parking_spaces'] = parking_spaces_dict

        yaml.dump(config, f)
    print('Parking spaces saved to', lot_path)
    print()
    return None


