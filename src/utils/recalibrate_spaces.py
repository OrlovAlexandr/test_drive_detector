import pandas as pd

from src.utils.process_parking_lot import get_order, get_parking_spaces


def get_full_lot_ranges(timestamps: list[int], max_gap: int = 3) -> list[tuple[int, int]]:
    """
    Find ranges with more than full parking lot
    """
    ranges = []
    start_time = timestamps[0]
    end_time = timestamps[0]

    for timestamp in timestamps[1:]:
        if timestamp - end_time <= max_gap:
            end_time = timestamp
        else:
            ranges.append((start_time, end_time))
            start_time = timestamp
            end_time = timestamp
    ranges.append((start_time, end_time))

    return ranges


def full_parking_lot_ranges(df: pd.DataFrame, spaces_number: int = 4) -> list[tuple[int, int]]:
    """
    Get ranges with more than N cars
    """
    # Count cars per timestamp
    car_count_per_timestamp = df.groupby('timestamp').size()
    # Get timestamps with more than N cars
    timestamps_full_parking = car_count_per_timestamp[car_count_per_timestamp >= spaces_number].index.values
    # Get ranges
    ranges = get_full_lot_ranges(timestamps_full_parking)
    return ranges


def calibrate_ranges(df: pd.DataFrame,
                     ranges: list[tuple[int, int]],
                     parking_spaces_df: pd.DataFrame,
                     limit_range: int = 5,
                     calibrate_duration: int = 6,
                     order_left_right: bool = True
                     ) -> pd.DataFrame:
    """
    Calibrate ranges to get updated coordinates
    """
    # Initialize dataframe to fill with parking spaces
    df_spaces = pd.DataFrame(columns=['timestamp', 'space', 'cx', 'cy', 'radius', 'timestamp_ord'])

    # Filter ranges with more than 5 seconds
    long_ranges = []
    for range in ranges:
        if range[1] - range[0] > limit_range:
            long_ranges.append(range)

    # Get start and end of each calibration
    calibrate_ranges = []
    for long_range in long_ranges:
        # Get start and end of calibration based on mean
        long_range_mean = (long_range[0] + long_range[1]) / 2
        calibrate_start = long_range_mean - calibrate_duration / 2
        calibrate_end = long_range_mean + calibrate_duration / 2

        # Find start and end of calibration based on real timestamp
        find_calibration_start = df[df['timestamp'] <= calibrate_start]['timestamp'].max()
        find_calibration_end = df[df['timestamp'] >= calibrate_end]['timestamp'].min()
        calibrate_ranges.append((find_calibration_start, find_calibration_end))

        # Choose detections range for calibration
        df_calibration = df.copy()
        df_calibration = df_calibration[df_calibration['timestamp'] >= find_calibration_start]
        df_calibration = df_calibration[df_calibration['timestamp'] <= find_calibration_end]

        # Get parking spaces with each radius
        df_spaces_new = get_parking_spaces(df_calibration, eps=15, threshold=0.9)

        # Add parking spaces order
        if order_left_right:
            df_spaces_new = df_spaces_new.sort_values(by='cx').reset_index(drop=True)
        else:
            df_spaces_new = df_spaces_new.sort_values(by='cy', ascending=False).reset_index(drop=True)

        # Add timestamp
        df_spaces_new = df_spaces_new.reset_index(names='space')
        df_spaces_new['timestamp'] = long_range[0]
        df_spaces_new['timestamp_ord'] = df[df['timestamp'] == long_range[0]]['timestamp_ord'].min()

        # Append results to dataframe
        df_spaces = pd.concat([df_spaces, df_spaces_new], ignore_index=True)

    if df_spaces.shape[0] == 0:
        parking_spaces_df = parking_spaces_df.reset_index(names='space').convert_dtypes()
        df_spaces = parking_spaces_df
        df_spaces['timestamp'] = 0
        df_spaces['timestamp_ord'] = 1
        # Move columns
        df_spaces = df_spaces[['timestamp', 'space', 'cx', 'cy', 'radius', 'timestamp_ord']]
        df_spaces['space'] = df_spaces['space'].astype(int)
    return df_spaces


def recalibrate_spaces(detections: pd.DataFrame,
                       parking_spaces: dict[int, dict[str, float]]) -> pd.DataFrame:
    """
    Recalibrate parking spaces to get updated coordinates
    """
    print('Recalibrating spaces...')
    spaces_number = len(parking_spaces)
    parking_spaces_df = pd.DataFrame(parking_spaces).transpose()
    order_left_right = get_order(parking_spaces_df)
    # Get parking lot ranges
    ranges = full_parking_lot_ranges(detections, spaces_number=spaces_number)

    # Calibrate ranges
    df_spaces = calibrate_ranges(detections, ranges, parking_spaces_df, limit_range=10, calibrate_duration=6,
                                 order_left_right=order_left_right)

    return df_spaces
