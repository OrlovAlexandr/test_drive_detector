import os

import cv2


class VideoFile:
    def __init__(self, file_path: str | os.PathLike, width, height) -> None:
        self._file_path = str(file_path)
        self._width = width
        self._height = height

    def get_first_frame(self) -> bytes | None:
        try:
            cap = cv2.VideoCapture(self._file_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
            ret, frame = cap.read()
            frame = cv2.resize(frame, (self._width, self._height))
            _, image_bytes = cv2.imencode('.png', frame)
            return image_bytes.tobytes()
        except cv2.error:
            return None
