import enum

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def calc_crop_from_vertices(
        video_path: str,
        vertices: list,
        padding: int = 50,
) -> list[int]:
    """Calculate the crop region from a list of vertices."""
    if not vertices:
        return []

    x = np.array(vertices)[:, 0]
    y = np.array(vertices)[:, 1]
    crop = [x.min() - padding, y.min() - padding, x.max() + padding, y.max() + padding]

    cap = cv2.VideoCapture(video_path)  # Read video with OpenCV
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return np.clip(crop, 0, [width, height, width, height]).tolist()


def get_parking_spaces(
        df: pd.DataFrame,
        eps: float = 15,
        threshold: float = 0.9,
        space_size: float = 0.5,
) -> pd.DataFrame:
    """Get parking spaces with each radius based on the bbox size."""
    # Point coordinates
    coords = df[['cx', 'cy']].values

    # Clustering with DBSCAN
    db = DBSCAN(eps=eps, min_samples=2).fit(coords)

    # Adding cluster labels to DataFrame
    df['cluster'] = db.labels_

    # Number of frames where each cluster appears
    total_frames = df['timestamp'].nunique()

    # Setting threshold
    threshold = total_frames * threshold

    # Calculate number of frames for each cluster
    cluster_counts = df.groupby('cluster')['timestamp'].nunique()

    # Filter clusters with less than the threshold
    valid_clusters = cluster_counts[cluster_counts >= threshold].index

    # Filter DataFrame with valid clusters
    df_filtered = df[df['cluster'].isin(valid_clusters)]
    df_filtered = df_filtered.copy()
    df_filtered['width'] = df_filtered['x2'] - df_filtered['x1']
    df_filtered['height'] = df_filtered['y2'] - df_filtered['y1']

    # Calculate median coordinates
    median_centers = df_filtered.groupby('cluster')[['cx', 'cy']].median().reset_index()
    median_width = df_filtered.groupby('cluster')['width'].median().reset_index()
    median_height = df_filtered.groupby('cluster')['height'].median().reset_index()
    # Get min side of the bbox and calculate radius
    sides = pd.merge(median_width, median_height, on='cluster')
    sides['min_side'] = sides.apply(lambda x: x['width'] if x['width'] < x['height'] else x['height'], axis=1)
    sides['radius'] = sides['min_side'] * (space_size / 2)
    # Concatenate median coordinates with radius
    median_coords = pd.concat([median_centers, sides['radius']], axis=1)
    return median_coords[['cx', 'cy', 'radius']].reset_index(drop=True)


class ParkingOrientation(enum.Enum):
    LEFT_TO_RIGHT = enum.auto()
    BOTTOM_TO_TOP = enum.auto()


def get_parking_orientation(parking_spaces: pd.DataFrame) -> ParkingOrientation:
    """Calculate the order of the parking spaces based on the coordinates."""
    spaces_width = parking_spaces['cx'].max() - parking_spaces['cx'].min()
    spaces_height = parking_spaces['cy'].max() - parking_spaces['cy'].min()
    return ParkingOrientation.LEFT_TO_RIGHT if spaces_width > spaces_height else ParkingOrientation.BOTTOM_TO_TOP


def apply_order(
        parking_spaces: pd.DataFrame,
        parking_orientation: ParkingOrientation,
) -> pd.DataFrame:
    """Apply the order of the parking spaces based on the coordinates."""
    if parking_orientation is ParkingOrientation.LEFT_TO_RIGHT:
        parking_spaces = parking_spaces.sort_values(by='cx').reset_index(drop=True)
    else:
        parking_spaces = parking_spaces.sort_values(by='cy', ascending=False).reset_index(drop=True)
    return parking_spaces.reset_index(names='space')
