from typing import List, Tuple

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

RADIUS = 10
LINE_THICKNESS = 4
COLOR_YELLOW = (0, 255, 255)
COLOR_DARK_YELLOW = (0, 180, 100)


def draw_polygon(image: np.ndarray,
                 vertices: List[Tuple[int, int]],
                 radius: int = RADIUS,
                 line_thickness: int = LINE_THICKNESS) -> np.ndarray:
    """Draws a polygon on the given image."""
    if len(vertices) > 2:
        draw_closed_path(image, vertices, line_thickness)
    if len(vertices) > 0:
        for idx, vertex in enumerate(vertices):
            draw_circle(image, vertex, radius)
            if idx > 0:
                draw_line(image, vertices[idx - 1], vertex, line_thickness)
    return image


def draw_circle(image: np.ndarray, center: Tuple[int, int], radius: int = RADIUS) -> None:
    """Draws a circle on the given image."""
    cv2.circle(image, center, radius, COLOR_YELLOW, -1)


def draw_line(image: np.ndarray, start_point: Tuple[int, int], end_point: Tuple[int, int],
              line_thickness: int = LINE_THICKNESS) -> None:
    """Draws a line on the given image."""
    cv2.line(image, start_point, end_point, COLOR_YELLOW, line_thickness)


def draw_closed_path(image: np.ndarray, vertices: List[Tuple[int, int]], line_thickness: int = LINE_THICKNESS) -> None:
    """Draws a closed path on the given image with an overlay."""
    cv2.line(image, vertices[0], vertices[-1], COLOR_DARK_YELLOW, line_thickness)


def is_vertex_inside_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Check if a vertex is inside a polygon."""
    polygon = Polygon(polygon)
    point = Point(point)
    is_inside = polygon.contains(point)

    return is_inside


def vertices_in_polygon(points: np.ndarray, polygon: List[Tuple[float, float]]) -> np.ndarray:
    """Check if points are inside a polygon."""
    polygon = Polygon(polygon)  # Shapely Polygon
    shapely_points = [Point(point) for point in points]  # Shapely Points
    # Check if points are inside a polygon vectorized
    return np.array([polygon.contains(point) for point in shapely_points])


def get_center_points(vertices: np.ndarray) -> np.ndarray:
    """Get the center points of a xyxy numpy array."""
    center_xy = np.array([(vertices[:, 0] + vertices[:, 2]) / 2,
                          (vertices[:, 1] + vertices[:, 3]) / 2]).transpose()

    return center_xy


def distance(coord_a: Tuple[float, float], coord_b: Tuple[float, float]) -> float:
    """Get the distance between two coordinates."""
    x, y = coord_a[0] - coord_b[0], coord_a[1] - coord_b[1]
    return (x ** 2 + y ** 2) ** 0.5
