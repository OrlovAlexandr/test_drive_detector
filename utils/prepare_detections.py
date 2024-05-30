import os
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

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    timestamp_data.loc[:, 'cluster'] = db.labels_

    # Save one point from each cluster with maximum confidence
    clustered_data = timestamp_data.groupby('cluster').apply(lambda x: x.loc[x['confidence'].idxmax()])
    return clustered_data


# Function for parallel processing
def filter_close_points_parallel(df: pd.DataFrame, eps: float = 10, min_samples: int = 1,
                                 n_jobs: int = -1) -> pd.DataFrame:
    """Parallel processing of process_frame function."""
    timestamps = df['timestamp'].unique()
    results = (Parallel(n_jobs=n_jobs)
               (delayed(process_frame)(timestamp, df, eps, min_samples) for timestamp in timestamps))

    filtered_df = pd.concat(results).reset_index(drop=True)
    return filtered_df


def prepare_detections(detections: np.ndarray, parking_vertices: List[Tuple[int, int]]) -> pd.DataFrame:
    """Prepare detections. Change bbox coordinates to center coordinates and filter close points."""
    # Get the center points
    centers = get_center_points(detections[:, 3:7])

    # Replace bbox coordinates with center coordinates
    detection_centers = np.append(detections[:, [7, 0, 2]], centers, axis=1)
    detection_coords = np.append(detection_centers, detections[:, 3:7], axis=1)

    # Get DataFrame from array
    columns = ['timestamp', 'frame', 'confidence', 'cx', 'cy', 'x1', 'y1', 'x2', 'y2']
    df_coords = pd.DataFrame(detection_coords, columns=columns)
    # print(df_centers)

    # Filter close points
    df_clean = filter_close_points_parallel(df_coords, eps=10, min_samples=1)

    # df_clean_path = './source/detections/temp/df_centers_clean.parquet'
    # os.makedirs(os.path.dirname(df_clean_path), exist_ok=True)
    # if os.path.exists(df_clean_path):
    #     df_clean = pd.read_parquet(df_clean_path)
    # else:
    #     print('filter close points')
    #     df_clean = filter_close_points_parallel(df_centers, eps=10, min_samples=1)
    #     df_clean.to_parquet(df_clean_path)

    # Check if points are inside the polygon
    # points = np.array(df_clean[['cx', 'cy']]
    points = df_clean[['cx', 'cy']].values
    df_clean['inside_polygon'] = vertices_in_polygon(points, parking_vertices)
    # print(df_clean.head(50))

    # Filter points that are inside the polygon
    df_clean = df_clean[df_clean['inside_polygon'] == True][['timestamp', 'frame',
                                                             'confidence', 'cx', 'cy',
                                                             'x1', 'y1', 'x2', 'y2']].reset_index(drop=True)
    # print(df_clean.columns)

    return df_clean
