import base64
from collections.abc import Iterable

import cv2
import numpy as np

from src.parking_lot import ParkingSpace


def draw_polygon(image: np.ndarray, vertices: list[tuple[int, int]]) -> np.ndarray:
    """Draws a polygon on the given image."""
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
        points: list[tuple[int, int]],
        spaces: Iterable[ParkingSpace] | None = None,
) -> bytes:
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    image = draw_polygon(image, points)
    if spaces is not None:
        for space in spaces:
            cv2.circle(
                image,
                (int(space.position[0] / 3), int(space.position[1] / 3)),
                int(space.radius / 3),
                (87, 212, 42),
                2,
            )

    _, image_bytes = cv2.imencode('.png', image)
    return image_bytes.tobytes()


def bytes_to_b64_embedded_image(image_bytes: bytes) -> str:
    return 'data:image/png;base64,' + base64.b64encode(image_bytes).decode('utf-8')
