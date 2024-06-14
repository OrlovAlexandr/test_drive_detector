from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN

from utils.geometry_utils import get_center_points, vertices_in_polygon


# Filter close points in one frame function
def process_frame(timestamp: float, df: pd.DataFrame, eps: float = 10, min_samples: int = 1) -> pd.DataFrame:
    """Filter close points in one frame."""
    timestamp_data = df[df['timestamp'] == timestamp].copy()
    coords = timestamp_data[['cx', 'cy']].values

    # Get clusters with DBSCAN to get one point from each cluster
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    timestamp_data.loc[:, 'cluster'] = db.labels_

    # Save one point from each cluster with maximum confidence
    clustered_data = timestamp_data.groupby('cluster').apply(lambda x: x.loc[x['confidence'].idxmax()])
    return clustered_data


# Function for parallel processing
def filter_close_points_parallel(df: pd.DataFrame, eps: float = 10, min_samples: int = 1,
                                 n_jobs: int = -1) -> pd.DataFrame:
    """Parallel processing of process_frame function.
    Args:
        df (pd.DataFrame): dataframe with detections
        eps (float, optional): eps value for DBSCAN. Defaults to 10.
        min_samples (int, optional): min_samples value for DBSCAN. Defaults to 1.
        n_jobs (int, optional): number of parallel jobs. Defaults to -1.

    Returns:
        pd.DataFrame: dataframe with filtered points
    """

    # Get unique timestamps from dataframe
    timestamps = df['timestamp'].unique()

    # Parallel processing of process_frame function
    results = (Parallel(n_jobs=n_jobs)
               (delayed(process_frame)(timestamp, df, eps, min_samples)
                for timestamp in timestamps))

    # Concatenate results into one dataframe
    filtered_df = pd.concat(results).reset_index(drop=True)
    return filtered_df


def prepare_detections(detections: np.ndarray, parking_vertices: List[Tuple[int, int]]) -> pd.DataFrame:
    """Prepare detections. Change bbox coordinates to center coordinates and filter close points.
    Args:
        detections (np.ndarray): array with detections
        parking_vertices (List[Tuple[int, int]]): list with parking vertices

    Returns:
        pd.DataFrame: dataframe with filtered points
    """
    # Get the center points from the bbox coordinates
    centers = get_center_points(detections[:, 3:7])

    # Extend bbox coordinates with center coordinates
    detection_centers = np.append(detections[:, [7, 0, 2]], centers, axis=1)
    detection_coords = np.append(detection_centers, detections[:, 3:7], axis=1)

    # Get DataFrame from array
    columns = ['timestamp', 'frame', 'confidence', 'cx', 'cy', 'x1', 'y1', 'x2', 'y2']
    df_coords = pd.DataFrame(detection_coords, columns=columns)

    # Filter close points
    df_filtered = filter_close_points_parallel(df_coords, eps=10, min_samples=1)

    # Check if points are inside the polygon
    points = df_filtered[['cx', 'cy']].values
    df_filtered['inside_polygon'] = vertices_in_polygon(points, parking_vertices)

    # Filter points that are inside the polygon
    df_filtered = df_filtered[df_filtered['inside_polygon'] == True][['timestamp', 'frame',
                                                             'confidence', 'cx', 'cy',
                                                             'x1', 'y1', 'x2', 'y2']].reset_index(drop=True)

    df_filtered['timestamp_ord'] = pd.factorize(df_filtered['timestamp'])[0] + 1

    return df_filtered
