import pandas as pd

from utils.test_drive_events import detect_events


def get_test_drive_events(detections: pd.DataFrame,
                          reacalibrated_spaces: pd.DataFrame) -> dict[int, list[tuple[str, int]]]:
    """
    Get test drive events with timestamps
    """
    print('Get test drive events...')
    coord_list = detections.apply(
        lambda row: ((row['cx'].astype(float), row['cy'].astype(float)), row['timestamp_ord'].astype(int)), axis=1).tolist()

    # Get lot list with coordinates and radius in frames
    lot_list = reacalibrated_spaces.apply(lambda row: (row['timestamp_ord'], row['space'], (row['cx'], row['cy']), row['radius']),
                                 axis=1).tolist()

    # Get events list from coordinates and lot list
    lots_states = detect_events(coord_list, lot_list)  # need to fix

    # Replace frames with timestamps
    lots_states = {space: [(detections[detections['timestamp_ord'] == timestamp]['timestamp'].values[0], state)
                           for timestamp, state in states]
                   for space, states in lots_states.items()}

    return lots_states