import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from sklearn.cluster import DBSCAN

from src.image_ops import get_center_points
from src.image_ops import vertices_in_polygon


# Filter close points in one frame function
def process_frame(timestamp: float, df: pd.DataFrame, eps: float = 10, min_samples: int = 1) -> pd.DataFrame:
    """Filter close points in one frame."""
    timestamp_data = df[df['timestamp'] == timestamp].copy()
    coords = timestamp_data[['cx', 'cy']].values

    # Get clusters with DBSCAN to get one point from each cluster
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    timestamp_data.loc[:, 'cluster'] = db.labels_

    # Save one point from each cluster with maximum confidence
    return timestamp_data.groupby('cluster').apply(lambda x: x.loc[x['confidence'].idxmax()])


# Function for parallel processing
def filter_close_points_parallel(df: pd.DataFrame, eps: float = 10, min_samples: int = 1,
                                 n_jobs: int = -1) -> pd.DataFrame:
    """Parallel processing of process_frame function."""
    # Get unique timestamps from dataframe
    timestamps = df['timestamp'].unique()

    # Parallel processing of process_frame function
    results = (Parallel(n_jobs=n_jobs)
               (delayed(process_frame)(timestamp, df, eps, min_samples)
                for timestamp in timestamps))

    # Concatenate results into one dataframe
    return pd.concat(results).reset_index(drop=True)


def prepare_detections(detections: np.ndarray, parking_vertices: list[tuple[int, int]]) -> pd.DataFrame:
    """Prepare detections. Change bbox coordinates to center coordinates and filter close points."""
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
    df_filtered = df_filtered[df_filtered['inside_polygon'] == True][['timestamp', 'frame',  # noqa: E712
                                                                      'confidence', 'cx', 'cy',
                                                                      'x1', 'y1', 'x2', 'y2']].reset_index(drop=True)
    df_filtered['timestamp_ord'] = pd.factorize(df_filtered['timestamp'])[0] + 1
    return df_filtered
