from typing import Any

import pandas as pd
from sklearn.cluster import DBSCAN


def get_parking_spaces(df: pd.DataFrame, radius: float = 15, threshold: float = 0.9) -> tuple[Any, bool]:
    # Point coordinates
    coords = df[['cx', 'cy']].values

    # Clustering with DBSCAN
    db = DBSCAN(eps=radius, min_samples=2).fit(coords)

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
    sides = pd.merge(median_width, median_height, on='cluster')
    # print(sides)
    sides['min_side'] = sides.apply(lambda x: x['width'] if x['width'] < x['height'] else x['height'], axis=1)
    sides['radius'] = sides['min_side'] / 4
    median_coords = pd.concat([median_centers, sides['radius']], axis=1)

    median_coords = median_coords[['cx', 'cy', 'radius']].reset_index(drop=True)

    return median_coords
