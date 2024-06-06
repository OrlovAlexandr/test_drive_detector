# Find ranges with more than 4 cars with threshold of gaps
import pandas as pd

from utils.process_parking_lot import get_parking_spaces


def get_full_lot_ranges(timestamps_full: list, max_gap: int=3) -> list:
    """
    Find ranges with more than full parking lot
    Args:
        timestamps_full: list of timestamps
        max_gap: maximum gap between two timestamps

    Returns:
        ranges: list of tuples (start_time, end_time)
    """
    ranges = []
    start_time = timestamps_full[0]
    end_time = timestamps_full[0]

    for timestamp in timestamps_full[1:]:
        if timestamp - end_time <= max_gap:
            end_time = timestamp
        else:
            ranges.append((start_time, end_time))
            start_time = timestamp
            end_time = timestamp
    ranges.append((start_time, end_time))

    return ranges

def full_parking_lot_ranges(df: pd.DataFrame, spaces_number: int=4) -> list:
    """
    Args:
        df: dataframe with detections
        spaces_number: number of cars in full parking lot

    Returns:
        ranges: list of tuples (start_time, end_time)
    """
    # Count cars per timestamp
    car_count_per_timestamp = df.groupby('timestamp').size()
    # Get timestamps with more than N cars
    timestamps_full_parking = car_count_per_timestamp[car_count_per_timestamp >= spaces_number].index.values
    # Get ranges
    ranges = get_full_lot_ranges(timestamps_full_parking)
    return ranges

def calibrate_ranges(df: pd.DataFrame,
                     ranges: list,
                     limit_range: int=5,
                     calibrate_duration: int=6,
                     order_left_right: bool = True
                     ) -> list:
    """
    Args:
        df: dataframe with detections
        ranges: list of tuples (start_time, end_time)
        limit_range: limit range for calibration in seconds, to get rid of small ranges
        calibrate_duration: duration of calibration in seconds
        order_left_right: True if order left to right

    Returns:
        calibrate_ranges: list of tuples (start_time, end_time)
    """
    # Initialize dataframe to fill with parking spaces
    df_spaces = pd.DataFrame(columns=['timestamp', 'space', 'cx', 'cy', 'radius', 'timestamp_ord'])

    # Filter ranges with more than 5 seconds
    long_ranges = []
    for i in ranges:
        if i[1] - i[0] > limit_range:
            long_ranges.append(i)
            # print(i)
    print('long ranges:', (long_ranges))
    # Get start and end of each calibration
    calibrate_ranges = []
    for r in long_ranges:
        # Get start and end of calibration based on mean
        mean = (r[0] + r[1]) / 2
        calibrate_start = mean - calibrate_duration / 2
        calibrate_end = mean + calibrate_duration / 2

        # Find start and end of calibration based on real timestamp
        find_calibrate_start = df[df['timestamp'] <= calibrate_start]['timestamp'].max()
        find_calibrate_end = df[df['timestamp'] >= calibrate_end]['timestamp'].min()
        calibrate_ranges.append((find_calibrate_start, find_calibrate_end))

        # Crop detections for calibration
        df_calib = df.copy()
        df_calib = df_calib[df_calib['timestamp'] >= find_calibrate_start]
        df_calib = df_calib[df_calib['timestamp'] <= find_calibrate_end]

        # Get parking spaces with each radius
        df_spaces_new = get_parking_spaces(df_calib, eps=15, threshold=0.9)

        # Add parking spaces order
        if order_left_right:
            df_spaces_new = df_spaces_new.sort_values(by='cx').reset_index(drop=True)
        else:
            df_spaces_new = df_spaces_new.sort_values(by='cy', ascending=False).reset_index(drop=True)

        # Add timestamp
        df_spaces_new = df_spaces_new.reset_index(names='space')
        df_spaces_new['timestamp'] = r[0]
        df_spaces_new['timestamp_ord'] = df[df['timestamp'] == r[0]]['timestamp_ord'].min()

        # Append results to dataframe
        df_spaces = pd.concat([df_spaces, df_spaces_new], ignore_index=True)

    return df_spaces
