from copy import copy
from typing import TypedDict

from utils.geometry_utils import distance

coordinates = [
    ((500, 500), 1),
    ((540, 500), 1),
    ((500, 500), 2),
    ((540, 500), 2),
    ((500, 500), 3),
    ((540, 500), 3),
    ((500, 500), 4),
    ((540, 500), 4),
    ((540, 500), 5),
    ((500, 1000), 5),
    ((500, 1000), 6),
    ((540, 1000), 6),
    ((500, 1000), 7),
    ((500, 1000), 8),
    ((500, 500), 9),
    ((500, 500), 10),
    ((500, 500), 11),
    ((500, 500), 12),
    ((500, 500), 13),
    ((500, 500), 14),
]

parking_lot_areas = [
    (1, (500, 500), 10),
    (2, (550, 500), 20),
]


class Coordinate(TypedDict):
    x: float
    y: float
    timestamp: float


def get_coordinates(coordinates: list[tuple[tuple[float, float], float]]) -> list[Coordinate]:
    coordinate_list = []
    for coord, timestamp in coordinates:
        coordinate_list.append(Coordinate(x=coord[0], y=coord[1], timestamp=timestamp))
    return coordinate_list


class ParkingLot(TypedDict):
    position: int
    center_xy: tuple[float, float]
    radius: float


def get_parking_lots(parking_lot_areas: list[tuple[float, tuple[float, float], float]]) -> list[ParkingLot]:
    parking_lot_list = []
    for position, center, radius in parking_lot_areas:
        parking_lot_list.append(ParkingLot(position=position, center_xy=center, radius=radius))
    return parking_lot_list


def detect_events(coordinates: list[tuple[tuple[float, float], int]],
                  parking_lot_areas: list[
                      tuple[int, tuple[float, float], float]
                  ]) -> None:
    lots_states = []
    slide_window_in_frames = 10
    probability_threshold = 0.9
    coords = get_coordinates(coordinates)
    lots = get_parking_lots(parking_lot_areas)
    lots_coords: dict[int, set[int]] = {
        lot['position']: set()
        for lot in lots
    }
    for coord in coords:
        for lot in lots:
            if distance(lot['center_xy'], (coord['x'], coord['y'])) <= lot['radius']:
                lots_coords[lot['position']].add(coord['timestamp'])
    lots_state: dict[int, bool] = {
        lot['position']: True
        for lot in lots
    }
    for lot_position, lot_timestamps_busy in lots_coords.items():
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
                lots_states.append((timestamp, copy(lots_state)))

    return lots_states


if __name__ == '__main__':
    lots_states = detect_events(coordinates, parking_lot_areas)
    print(lots_states)
