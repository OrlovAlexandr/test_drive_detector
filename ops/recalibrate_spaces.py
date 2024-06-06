import pandas as pd
import yaml

from utils.recalibration import full_parking_lot_ranges, calibrate_ranges


def recalibrate_spaces(detections_path: str,
                       lot_path: str,
                       recalibration_path: str) -> None:
    """
    Recalibrate parking spaces to get updated coordinates
    Args:
        detections_path: str - path to the detections file
        lot_path: str - path to the parking lot file
        recalibration_path: str - save path to the recalibration file
    Returns:
        None
    """
    print('Recalibrate spaces...')
    # Read detections
    df = pd.read_parquet(detections_path)

    # Read config file with calibration order
    with open(lot_path, 'r') as f:
        config = yaml.safe_load(f)
        order_left_right = config['order_left_right']
        spaces_number = config['spaces_number']

    # Get parking lot ranges
    ranges = full_parking_lot_ranges(df, spaces_number=spaces_number)

    # Calibrate ranges
    df_spaces = calibrate_ranges(df, ranges, limit_range=10, calibrate_duration=6, order_left_right=order_left_right)

    df_spaces.to_parquet(recalibration_path)
    print('Recalibration saved to', recalibration_path)
    print()
    return None