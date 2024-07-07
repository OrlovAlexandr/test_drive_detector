from copy import copy
from typing import TypedDict

import numpy as np
import pandas as pd

from src.image_ops import distance


class Coordinate(TypedDict):
    x: float
    y: float
    timestamp: int


def get_coordinates(coordinates: list[tuple[tuple[float, float], int]]) -> list[Coordinate]:
    # Get coordinates from detections with timestamp
    coordinate_list = []
    for coord, timestamp in coordinates:
        coordinate_list.append(Coordinate(x=coord[0], y=coord[1], timestamp=timestamp))
    return coordinate_list


class ParkingLot(TypedDict):
    timestamp: int
    position: int
    center_xy: tuple[float, float]
    radius: float


def get_parking_lots(parking_lot_areas: list[tuple[int, int, tuple[float, float], float]]) -> list[ParkingLot]:
    # Get parking lot list from parking lot areas
    parking_lot_list = []
    for timestamp, position, center, radius in parking_lot_areas:
        parking_lot_list.append(ParkingLot(timestamp=timestamp, position=position, center_xy=center, radius=radius))
    return parking_lot_list


def detect_events(coordinates: list[tuple[tuple[float, float], int]],
                  parking_lot_areas: list[
                      tuple[int, int, tuple[float, float], float]
                  ]) -> dict[int, list[tuple[int, bool]]]:
    """
    Detect events from coordinates and parking lot areas
    """
    all_lots_states = {}
    slide_window_in_frames = 100
    probability_threshold = 0.9
    coords = get_coordinates(coordinates)
    lots = get_parking_lots(parking_lot_areas)
    lots_coords: dict[int, set[int]] = {
        lot['position']: set()
        for lot in lots
    }
    lots_timestamps = np.array(list(set([lot['timestamp'] for lot in lots])))
    for coord in coords:
        for lot in lots:
            previous_lot_timestamp = lots_timestamps[coord['timestamp'] >= lots_timestamps].max()
            if lot['timestamp'] == previous_lot_timestamp:
                if distance(lot['center_xy'], (coord['x'], coord['y'])) <= lot['radius']:
                    lots_coords[lot['position']].add(coord['timestamp'])
                    # break
    lots_state: dict[int, bool] = {
        lot['position']: True
        for lot in lots
    }
    for lot_position, lot_timestamps_busy in lots_coords.items():
        lots_states = []
        for timestamp in range(1, max(lot_timestamps_busy) + 1):
            hits = 0
            for frame in range(max(timestamp - slide_window_in_frames + 1, 0), timestamp + 1):
                if frame in lot_timestamps_busy:
                    hits += 1
                if frame < slide_window_in_frames:
                    hits = slide_window_in_frames
            if lots_state[lot_position] is True:
                hits = slide_window_in_frames - hits
            probability = hits / slide_window_in_frames
            if probability >= probability_threshold:
                lots_state[lot_position] = not lots_state[lot_position]
                lots_states.append((timestamp - slide_window_in_frames, copy(lots_state[lot_position])))
        all_lots_states[lot_position] = lots_states
    return all_lots_states


def get_test_drive_events(detections: pd.DataFrame,
                          reacalibrated_spaces: pd.DataFrame) -> dict[int, list[tuple[int, bool]]]:
    """
    Get test drive events with timestamps
    """
    print('Get test drive events...')
    coord_list = detections.apply(
        lambda row: ((row['cx'].astype(float), row['cy'].astype(float)), row['timestamp_ord'].astype(int)),
        axis=1).tolist()

    # Get lot list with coordinates and radius in frames
    lot_list = reacalibrated_spaces.apply(
        lambda row: (row['timestamp_ord'], row['space'], (row['cx'], row['cy']), row['radius']),
        axis=1).tolist()

    # Get events list from coordinates and lot list
    lots_states = detect_events(coord_list, lot_list)  # need to fix

    # Replace frames with timestamps
    lots_states = {space: [(detections[detections['timestamp_ord'] == timestamp]['timestamp'].values[0], state)
                           for timestamp, state in states]
                   for space, states in lots_states.items()}

    return lots_states
