import base64
from collections.abc import Iterable

import cv2
import numpy as np
from shapely import Point
from shapely import Polygon

from src.parking_lot import ParkingSpace


def draw_polygon(image: np.ndarray, vertices: list[tuple[float, float]]) -> np.ndarray:
    """Draws a polygon on the given image."""
    image_width = image.shape[1]
    image_height = image.shape[0]
    # convert vertices to pixels
    vertices = [(int(x * image_width), int(y * image_height)) for x, y in vertices]
    if len(vertices) > 2:
        draw_closed_path(image, vertices)
    if len(vertices) > 0:
        for idx, vertex in enumerate(vertices):
            draw_circle(image, vertex)
            if idx > 0:
                draw_line(image, vertices[idx - 1], vertex)
    return image


def draw_circle(
        image: np.ndarray,
        center: tuple[int, int],
        radius: int = 5,
        color: tuple[int, int, int] = (0, 255, 255),

) -> None:
    """Draws a circle on the given image."""
    cv2.circle(image, center, radius, color, -1)


def draw_line(
        image: np.ndarray,
        start_point: tuple[int, int],
        end_point: tuple[int, int],
        color: tuple[int, int, int] = (0, 255, 255),
        line_thickness: int = 2,
) -> None:
    """Draws a line on the given image."""
    cv2.line(image, start_point, end_point, color, line_thickness)


def draw_closed_path(
        image: np.ndarray,
        vertices: list[tuple[int, int]],
        color: tuple[int, int, int] = (0, 180, 100),
        line_thickness: int = 2,
) -> None:
    """Draws a closed path on the given image."""
    cv2.line(image, vertices[0], vertices[-1], color, line_thickness)


def draw_polygon_on_image_bytes(image_bytes: bytes, vertices: list[tuple[int, int]]) -> bytes:
    """Draws a polygon on the image (given as bytes) and returns the modified image as bytes."""
    # Decode image bytes to numpy array
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Draw the polygon on the image
    image = draw_polygon(image, vertices)

    # Encode the modified image to bytes
    _, image_bytes = cv2.imencode('.png', image)
    return image_bytes.tobytes()


def draw_parking_lot_map(
        image_data: bytes,
        points: list[tuple[float, float]],
        spaces: Iterable[ParkingSpace] | None = None,
) -> bytes:
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    image = draw_polygon(image, points)
    if spaces is not None:
        image_width = image.shape[1]
        image_height = image.shape[0]
        for space in spaces:
            space_x = int(space.position[0] * image_width)
            space_y = int(space.position[1] * image_height)
            space_radius = int(space.radius * image_height)
            cv2.circle(
                image,
                (space_x, space_y),
                space_radius,
                (255, 0, 0),
                1,
            )
            cv2.circle(
                image,
                (space_x, space_y),
                3,
                (0, 255, 255),
                -1,
            )
            cv2.putText(
                image,
                str(space.number),
                (space_x + 3, space_y + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

    _, image_bytes = cv2.imencode('.png', image)
    return image_bytes.tobytes()


def distance(coord_a: tuple[float, float], coord_b: tuple[float, float]) -> float:
    """Get the distance between two coordinates."""
    x, y = coord_a[0] - coord_b[0], coord_a[1] - coord_b[1]
    return (x ** 2 + y ** 2) ** 0.5


def vertices_in_polygon(points: np.ndarray, polygon: list[tuple[float, float]]) -> np.ndarray:
    """Check if points are inside a polygon."""
    polygon = Polygon(polygon)  # Shapely Polygon
    shapely_points = [Point(point) for point in points]  # Shapely Points
    # Check if points are inside a polygon vectorized
    return np.array([polygon.contains(point) for point in shapely_points])


def get_center_points(vertices: np.ndarray) -> np.ndarray:
    """Get the center points of a xyxy numpy array."""
    return np.array([(vertices[:, 0] + vertices[:, 2]) / 2,
                     (vertices[:, 1] + vertices[:, 3]) / 2]).transpose()


def bytes_to_b64_embedded_image(image_bytes: bytes) -> str:
    return 'data:image/png;base64,' + base64.b64encode(image_bytes).decode('utf-8')
