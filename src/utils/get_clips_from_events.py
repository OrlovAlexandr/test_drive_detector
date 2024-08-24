import logging
import uuid
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm

from src.test_drive import TestDrive


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVENT_DATE = '2024-06-15 8:00:00'

CLIP_DURATION = 30
CLIP_TIME_EVENT_OFFSET = 10


@dataclass
class TimeRange:
    start: float
    end: float


@dataclass
class FrameRange:
    start_frame: int
    end_frame: int


@dataclass
class ClipSegment:
    time_range: TimeRange
    frame_range: FrameRange


class TimeConvert:
    @staticmethod
    def to_datetime(time_in_sec, day_start_time):
        event_datetime = datetime.strptime(day_start_time, '%Y-%m-%d %H:%M:%S')
        return event_datetime + timedelta(seconds=time_in_sec)

    @staticmethod
    def from_datetime(time_in_dt, day_start_time):
        event_datetime = datetime.strptime(day_start_time, '%Y-%m-%d %H:%M:%S')
        return (time_in_dt - event_datetime).total_seconds()


@dataclass(frozen=True, kw_only=True)
class Clip:
    space_number: int
    clip_range: TimeRange
    video_time_ranges: dict[Path, TimeRange]
    clip: dict[str, ClipSegment] = field(init=False)
    output_file_name: str

    def __post_init__(self):
        object.__setattr__(self, 'clip', self._calculate_clip_ranges())

    def _calculate_clip_ranges(self) -> dict[Path, ClipSegment]:
        clip_segments = {}
        fps_dict = self.get_video_fps()

        for video, time_range in self.video_time_ranges.items():
            fps = fps_dict[video]

            if time_range.end < self.clip_range.start or time_range.start > self.clip_range.end:
                continue  # No overlap

            overlap_start = max(self.clip_range.start, time_range.start)
            overlap_end = min(self.clip_range.end, time_range.end)

            # Convert to frames
            start_frame = int((overlap_start - time_range.start) * fps + 1)
            end_frame = int((overlap_end - time_range.start) * fps)

            clip_segments[video] = ClipSegment(
                time_range=TimeRange(overlap_start, overlap_end),
                frame_range=FrameRange(start_frame, end_frame),
            )

        return clip_segments

    def get_video_fps(self) -> dict[Path, int]:
        """Get video fps for each video file."""
        video_fps = {}
        for video_file in self.video_time_ranges:
            if video_file.is_file():
                cap = cv2.VideoCapture(str(video_file))
                fps = cap.get(cv2.CAP_PROP_FPS)
                video_fps[video_file] = int(fps)
            else:
                logger.error('Video file not found: %s', video_file)
        return video_fps

    def get_clip_params(
            self,
            sorted_clip: dict[str, ClipSegment],
    ) -> dict[str, dict[str, int | cv2.VideoCapture]] | None:
        video_params = {}
        width1, height1, fps1 = 0, 0, 0
        for idx, (video, _clip_segment) in enumerate(sorted_clip.items()):
            cap = cv2.VideoCapture(str(video))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if idx > 0 and width != width1 and height != height1 and fps != fps1:
                logger.error('Video width, height and fps not equal')
                return None
            video_params[video] = {'cap': cap, 'fps': fps, 'width': width, 'height': height}
            width1, height1, fps1 = width, height, fps
        return video_params

    def save_to_video(
            self,
            output_dir: Path,
            parking_spaces: dict[int, dict[str, float]],
            lots_states: dict[int, list[tuple[float, bool]]],
            compress: bool) -> None:
        """Save clip to video file."""
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)

        sorted_clip = dict(sorted(self.clip.items(), key=lambda item: item[1].time_range.start))
        video_params = self.get_clip_params(sorted_clip)
        if not video_params:
            return

        width = video_params[next(iter(video_params))]['width']
        height = video_params[next(iter(video_params))]['height']
        fps = video_params[next(iter(video_params))]['fps']
        output_path = output_dir / Path(self.output_file_name)
        output = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for video, _clip_segment in sorted_clip.items():
            cap = video_params[video]['cap']
            start_frame = sorted_clip[video].frame_range.start_frame
            end_frame = sorted_clip[video].frame_range.end_frame

            clip_start_time = _clip_segment.time_range.start
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for frame_num in tqdm(range(start_frame, end_frame)):
                current_time = clip_start_time + (frame_num - start_frame) / fps
                ret, frame = cap.read()
                if not ret:
                    break
                for space_id, space_data in parking_spaces.items():
                    color = (0, 0, 0)
                    text = ''
                    cx, cy, _radius = int(space_data['cx']), int(space_data['cy']), int(space_data['radius'])

                    x1y1 = cx + 10, cy - 18
                    x2y2 = 0, 0
                    for timestamp, state in lots_states[space_id]:
                        if timestamp > current_time:
                            break
                        if state:
                            color = (0, 255, 255)
                            text = f'{int(space_id)} occupied'
                            x2y2 = cx + 140, cy + 5
                        else:
                            color = (0, 255, 0)
                            text = f'{int(space_id)} free'
                            x2y2 = cx + 90, cy + 5
                    if space_id != self.space_number:
                        color = (255, 255, 255)

                    cv2.circle(frame, (cx, cy), 5, color, thickness=-1)
                    cv2.rectangle(frame, x1y1, x2y2, color, -1)
                    cv2.putText(frame, text, (cx + 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 0), 2)
                current_time_dt = TimeConvert.to_datetime(current_time, EVENT_DATE)
                current_time_dt = current_time_dt.strftime('%Y-%m-%d %H:%M:%S')
                cv2.rectangle(frame, (8, 8), (310, 35), (0, 255, 255), -1)
                cv2.putText(frame, str(current_time_dt), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                frame_hd = cv2.resize(frame, (1280, 720))
                cv2.imshow('frame', frame_hd)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                output.write(frame)
            cap.release()
        output.release()
        cv2.destroyAllWindows()

        # Compress video if compress parameter is True
        if compress:
            temp_path = output_dir / Path('temp.mp4')
            output_path.rename(temp_path)
            compress_clip = VideoFileClip(str(temp_path))
            compress_clip.write_videofile(str(output_path), verbose=False, logger=None, bitrate='1M')
            temp_path.unlink()
        return


def get_video_ranges(
        video_start_time: dict[str, float],
        # video_input_directory: Path,
) -> dict[Path, TimeRange]:
    """Get video ranges for each video file."""
    _video_ranges = {}
    for video_file, start_time in video_start_time.items():
        video_file_path = Path(video_file)
        if video_file_path.is_file():
            cap = cv2.VideoCapture(str(video_file_path))
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            _video_ranges[video_file_path] = TimeRange(start=start_time, end=round(start_time + duration, 4))
        else:
            logger.error('Video file not found: %s', video_file)
    return _video_ranges


def get_test_drives(
        lot_states: dict[int, list[tuple[float, bool]]],
        lot_id: uuid.UUID,
) -> list[TestDrive]:
    test_drives = []
    departure = None
    for space, states in lot_states.items():
        for timestamp, state in states:
            if state is False:
                departure = timestamp
                continue
            if departure is not None and state is True:
                test_drives.append(TestDrive(
                    id=uuid.uuid4(),
                    lot_id=lot_id,
                    space_number=space,
                    start_time=TimeConvert.to_datetime(departure, EVENT_DATE),
                    end_time=TimeConvert.to_datetime(timestamp, EVENT_DATE),

                ))
                departure = None
    return test_drives


def get_clips_list(
        test_drive_list: list[TestDrive],
        # video_input_directory: Path,
        videos_start_time: dict[str, float],
) -> list[Clip]:
    clips = []

    for test_drive in test_drive_list:
        clip1 = get_clip(test_drive,
                         videos_start_time, departure_relative=True)
        clip2 = get_clip(test_drive,
                         videos_start_time, departure_relative=False)
        clips.extend([clip1, clip2])

    return clips


def get_clip(
        test_drive: TestDrive,
        # video_input_directory: Path,
        videos_start_time: dict[str, float],
        departure_relative: bool,
) -> Clip:
    _video_ranges = get_video_ranges(videos_start_time)
    if departure_relative:
        clip_start = TimeConvert.from_datetime(
            test_drive.start_time, EVENT_DATE,
        ) - CLIP_TIME_EVENT_OFFSET
        output_file_name = f'{test_drive.id}_departure.mp4'
    else:
        clip_start = TimeConvert.from_datetime(
            test_drive.end_time, EVENT_DATE,
        ) + CLIP_TIME_EVENT_OFFSET - CLIP_DURATION
        output_file_name = f'{test_drive.id}_arrival.mp4'
    clip_start_rounded = round(clip_start, 4)
    clip_end_rounded = round(clip_start_rounded + CLIP_DURATION, 4)
    return Clip(
        space_number=test_drive.space_number,
        clip_range=TimeRange(clip_start_rounded, clip_end_rounded),
        video_time_ranges=_video_ranges,
        output_file_name=output_file_name,
    )


def save_clips_to_videos(
        list_of_clips: list[Clip],
        video_output_dir: Path,
        parking_spaces: dict[int, dict[str, float]],
        lots_states: dict[int, list[tuple[float, bool]]],
        compress: bool = False,
) -> None:
    lots_states_with_zero = {
        key: [(0.0, not value[0][1]), *value] if len(value) > 0 else [(0.0, True)]
        for key, value in lots_states.items()
    }
    sorted_lots_states = {key: sorted(value, key=lambda x: x[0]) for key, value in lots_states_with_zero.items()}
    for one_clip in list_of_clips:
        one_clip.save_to_video(video_output_dir, parking_spaces,
                               sorted_lots_states, compress)
