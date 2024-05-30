from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def get_parking_spaces(df: pd.DataFrame, eps: float = 15, threshold: float = 0.9) -> tuple[Any, bool]:
    """
    Get parking spaces with each radius based on the bbox size

    Args:
        df (pd.DataFrame): DataFrame with detections
        eps (float, optional): Maximum distance between two samples for DBSCAN clustering.
        threshold (float, optional): Threshold for filtering clusters.

    Returns:
        median_coords (pd.DataFrame): DataFrame with median coordinates of each parking space
    """

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
    # print(df_filtered.head(50))
    df_filtered['width'] = df_filtered['x2'] - df_filtered['x1']
    df_filtered['height'] = df_filtered['y2'] - df_filtered['y1']

    # Calculate median coordinates
    median_centers = df_filtered.groupby('cluster')[['cx', 'cy']].median().reset_index()
    median_width = df_filtered.groupby('cluster')['width'].median().reset_index()
    median_height = df_filtered.groupby('cluster')['height'].median().reset_index()
    # Get min side of the bbox and calculate radius
    sides = pd.merge(median_width, median_height, on='cluster')
    # print(sides)
    sides['min_side'] = sides.apply(lambda x: x['width'] if x['width'] < x['height'] else x['height'], axis=1)
    sides['radius'] = sides['min_side'] / 4
    median_coords = pd.concat([median_centers, sides['radius']], axis=1)

    median_coords = median_coords[['cx', 'cy', 'radius']].reset_index(drop=True)

    return median_coords


def get_crop_from_vertices(vertices=None, padding: int = 50, video_path: str = '') -> list:
    """
    Calculate the crop region from a list of vertices.

    Args:
        vertices (list): A list of vertices of the polygon.
        padding (int): The padding to be applied around the vertices to create the crop region.
            Default is 50.
        video_path (str): The path to the video file. Default is an empty string.

    Returns:
        crop (list): A list representing the crop region in XYXY format.
    """
    if vertices is not None:
        x = np.array(vertices)[:, 0]
        y = np.array(vertices)[:, 1]
        crop = [x.min() - padding, y.min() - padding, x.max() + padding, y.max() + padding]

        cap = cv2.VideoCapture(video_path)  # Read video with OpenCV
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        crop = np.clip(crop, 0, [width, height, width, height]).tolist()
    else:
        crop = []

    return crop
