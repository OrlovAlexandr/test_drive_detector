import json

import pandas as pd

from utils.test_drive_events import detect_events


def get_test_drive_events(detections_path: str,
                          recalibration_path: str,
                          events_path: str) -> None:
    """
    Get test drive events with timestamps
    Args:
        detections_path: str - path to the detections file
        recalibration_path: str - path to the recalibration file
        events_path: str - save path to the events file
    Returns:
        None
    """
    print('Get test drive events...')
    # Get list of center coordinates with timestamp in frames
    df = pd.read_parquet(detections_path)
    coord_list = df.apply(
        lambda row: ((row['cx'].astype(float), row['cy'].astype(float)), row['timestamp_ord'].astype(int)), axis=1).tolist()

    # Get lot list with coordinates and radius in frames
    df_lot_list = pd.read_parquet(recalibration_path)
    lot_list = df_lot_list.apply(lambda row: (row['timestamp_ord'], row['space'], (row['cx'], row['cy']), row['radius']),
                                 axis=1).tolist()

    # Get events list from coordinates and lot list
    lots_states = detect_events(coord_list, lot_list)  # need to fix

    # Replace frames with timestamps
    lots_states = {space: [(df[df['timestamp_ord'] == timestamp]['timestamp'].values[0], state)
                           for timestamp, state in states]
                   for space, states in lots_states.items()}

    # Save to json with keys as integers
    with open(events_path, 'w') as f:
        json.dump(lots_states, f)
    print('Test drive events saved to', events_path)
    print()
    return None