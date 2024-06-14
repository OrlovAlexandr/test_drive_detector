import pandas as pd

from utils.process_parking_lot import get_order
from utils.recalibration import full_parking_lot_ranges, calibrate_ranges


def recalibrate_spaces(detections: pd.DataFrame,
                       parking_spaces: dict[int, dict[str, float]]) -> pd.DataFrame:
    """
    Recalibrate parking spaces to get updated coordinates
    """
    print('Recalibrating spaces...')
    spaces_number = len(parking_spaces)
    # print(spaces_number)
    # print(parking_spaces)
    parking_spaces = pd.DataFrame(parking_spaces).transpose()
    order_left_right = get_order(parking_spaces)
    # print(parking_spaces, order_left_right)
    # Get parking lot ranges
    ranges = full_parking_lot_ranges(detections, spaces_number=spaces_number)
    print(ranges)

    # Calibrate ranges
    df_spaces = calibrate_ranges(detections, ranges, limit_range=10, calibrate_duration=6, order_left_right=order_left_right)

    return df_spaces