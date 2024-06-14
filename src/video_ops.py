import os

import cv2


class VideoFile:
    def __init__(self, file_path: str | os.PathLike) -> None:
        self._file_path = str(file_path)

    def get_first_frame(self) -> bytes:
        try:
            cap = cv2.VideoCapture(self._file_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 360))
            _, image_bytes = cv2.imencode('.png', frame)
        except cv2.error:
            print(self._file_path)
        return image_bytes.tobytes()
