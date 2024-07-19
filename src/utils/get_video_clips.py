import logging
import uuid
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import cv2
import pandas as pd
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip

from src.test_drive import TestDrive


EVENT_DATE = '2024-06-15 8:00:00'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def record_video(
        video_time: dict[str, float],
        recalibrated_spaces: pd.DataFrame,
        lots_states_sorted: dict[int, list[tuple[float, bool]]],
        video_input: str,
        video_output: str,
        start: int | None = None,
        end: int | None = None,
) -> None:
    logger.info(f'Recording video... {video_input} -> {video_output}')
    video_start_time = video_time[video_input]

    lots_states_from_start = lots_states_sorted.copy()
    for space_id in lots_states_from_start:
        if len(lots_states_from_start[space_id]) > 0:
            first_space_state = lots_states_from_start[space_id][0][1]
            if first_space_state:
                lots_states_from_start[space_id] = [(0.0, False)] + lots_states_from_start[space_id]
            else:
                lots_states_from_start[space_id] = [(0.0, True)] + lots_states_from_start[space_id]
        else:
            lots_states_from_start[space_id] = [(0.0, True)]

    cap = cv2.VideoCapture(video_input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Get smallest timestamp_ord
    smallest_timestamp_ord = min(recalibrated_spaces['timestamp_ord'])
    # Dict of parking spaces parameters
    df_spaces = recalibrated_spaces[recalibrated_spaces['timestamp_ord'] == smallest_timestamp_ord].iloc[:, 1:5]

    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    while cap.isOpened():

        ret, frame = cap.read()
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_time = video_start_time + (frame_num / fps)

        if not ret or (end and frame_num > end):
            break
        # Draw parking spaces and lots states
        for space_id in lots_states_from_start:

            current_state = None
            for timestamp_, state in lots_states_from_start[space_id]:
                if timestamp_ > frame_time:
                    break
                current_state = state

            row = df_spaces[df_spaces['space'] == space_id].iloc[0]
            if current_state:
                color = (0, 255, 255)
                text = f'{int(space_id)} occupied'
                x1y1 = (int(row.cx + 10), int(row.cy - 18))
                x2y2 = (int(row.cx + 140), int(row.cy + 5))
            else:
                color = (0, 255, 0)
                text = f'{int(space_id)} free'
                x1y1 = (int(row.cx + 10), int(row.cy - 18))
                x2y2 = (int(row.cx + 90), int(row.cy + 5))
            cv2.circle(frame, (int(row.cx), int(row.cy)), radius=5, color=color, thickness=-1)
            cv2.rectangle(frame, x1y1, x2y2, color, -1)
            cv2.putText(
                frame,
                text,
                (int(row.cx + 15), int(row.cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

        output.write(frame)

    cap.release()
    output.release()
    cv2.destroyAllWindows()


def concatenate_clips(clip_paths: list[Path], output_path: Path):
    clips = [VideoFileClip(str(path)) for path in clip_paths]
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(str(output_path), verbose=False, logger=None)
    for path in clip_paths:
        path.unlink()


def get_clip_params(timestamp: float, clip_duration: int, timestamp_offset: int) -> tuple[float, float]:
    start_timestamp = timestamp - (clip_duration - timestamp_offset)
    end_timestamp = timestamp + timestamp_offset
    return start_timestamp, end_timestamp


def find_clips(
        clip_time_params: dict,
        start_timestamp: float,
        end_timestamp: float,
) -> tuple[tuple[str, int], tuple[str, int]]:
    clip_start = clip_end = None
    for video_path, (video_start, video_end, video_fps) in clip_time_params.items():
        if video_start <= start_timestamp <= video_end:
            clip_start = (video_path, int((start_timestamp - video_start) * video_fps))
        if video_start <= end_timestamp <= video_end:
            clip_end = (video_path, int((end_timestamp - video_start) * video_fps))
    return clip_start, clip_end


def process_video_clips(
        video_time: dict[str, float],
        lots_states: dict[int, list[tuple[float, bool]]],
        recalibrated_spaces: pd.DataFrame,
        clip_time_params: dict[str, tuple[float, float, float]],
        output_video_dir: Path,
        lot_id: uuid.UUID,
        timestamp_offset: int = 15,
        clip_duration: int = 30,
) -> list[TestDrive]:
    output_video_dir.mkdir(parents=True, exist_ok=True)

    test_drive_start = {}
    test_drives = []
    test_drive_id = None

    lots_states_sorted = {key: sorted(value, key=lambda x: x[0]) for key, value in lots_states.items()}

    for space, states in lots_states_sorted.items():
        space_int = int(float(space))
        test_drive_number = 1

        for timestamp, state in states:
            state_name = 'arrival' if state else 'departure'
            start_timestamp, end_timestamp = get_clip_params(
                timestamp,
                clip_duration,
                timestamp_offset,
            )
            clip_start, clip_end = find_clips(
                clip_time_params,
                start_timestamp,
                end_timestamp,
            )
            if clip_end and clip_end[1] <= 1:
                clip_end = None

            if not clip_start and not clip_end:
                continue

            if not test_drive_start:
                test_drive_id = uuid.uuid4()

            output_clip_path = output_video_dir / f'{test_drive_id}_{state_name}.mp4'
            logger.info('\nSpace: %s, State: %s, Test drive id: %s', space_int, state_name, test_drive_id)
            logger.info('Clip start: %s', clip_start)
            logger.info('Clip end: %s', clip_end)
            logger.info('Output clip path: %s', output_clip_path)
            if clip_start and clip_end and clip_start[0] == clip_end[0]:
                temp_path = output_video_dir / 'temp.mp4'
                record_video(
                    video_time,
                    recalibrated_spaces,
                    lots_states_sorted,
                    clip_start[0],
                    str(temp_path),
                    clip_start[1],
                    clip_end[1],
                )
                if temp_path.is_file():
                    video = VideoFileClip(str(temp_path))
                    video.write_videofile(str(output_clip_path), verbose=False, logger=None)
                    temp_path.unlink()
                else:
                    logger.error('No clips created')
            else:
                temp_paths = []
                temp_path_start, temp_path_end = Path(), Path()
                if clip_start:
                    temp_path_start = output_video_dir / 'temp_start.mp4'
                    record_video(
                        video_time,
                        recalibrated_spaces,
                        lots_states_sorted,
                        clip_start[0],
                        str(temp_path_start),
                        clip_start[1],
                        None,
                    )
                    temp_paths.append(temp_path_start)
                if clip_end:
                    temp_path_end = output_video_dir / 'temp_end.mp4'
                    record_video(
                        video_time,
                        recalibrated_spaces,
                        lots_states_sorted,
                        clip_end[0],
                        str(temp_path_end),
                        1,
                        clip_end[1],
                    )
                    temp_paths.append(temp_path_end)
                # if os.path.isfile(str(temp_path_start)) and os.path.isfile(str(temp_path_end)):
                if temp_path_start.is_file() and temp_path_end.is_file():
                    concatenate_clips(temp_paths, output_clip_path)
                elif bool(temp_path_start.is_file()) != bool(temp_path_end.is_file()):
                    video = VideoFileClip(str(temp_paths[0]))
                    video.write_videofile(str(output_clip_path), verbose=False, logger=None)
                    for path in temp_paths:
                        path.unlink()
                else:
                    logger.error('No clips created')

            timestamp_datetime = datetime.strptime(EVENT_DATE, '%Y-%m-%d %H:%M:%S') + timedelta(seconds=timestamp)

            if state_name == 'departure':
                test_drive_start['event_timestamp'] = timestamp_datetime
                test_drive_start['clip_path'] = output_clip_path
            elif state_name == 'arrival' and test_drive_start:
                test_drives.append(TestDrive(
                    id=test_drive_id,
                    lot_id=lot_id,
                    space_number=int(space_int),
                    start_time=test_drive_start['event_timestamp'],
                    end_time=timestamp_datetime,
                ))
                test_drive_start = {}
                test_drive_number += 1

    return test_drives


def get_video_clips(
        video_time: dict[str, float],
        recalibrated_spaces: pd.DataFrame,
        lots_states: dict[int, list[tuple[float, bool]]],
        output_video_dir: Path,
        lot_id: uuid.UUID,
        offset: int = 15,
        clip_duration: int = 30,
) -> list[TestDrive]:
    logger.info('Get video clips of test drives...')

    clip_time_params = {}
    for video_path, timestamp in video_time.items():
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()
        video_start = timestamp
        video_end = timestamp + duration
        clip_time_params[video_path] = (video_start, video_end, fps)

    test_drives = process_video_clips(
        video_time,
        lots_states,
        recalibrated_spaces,
        clip_time_params,
        output_video_dir,
        lot_id,
        timestamp_offset=offset,
        clip_duration=clip_duration,
    )
    logger.info('Number of test drives: %s', len(test_drives))
    logger.info('Test drives detected!\n')
    return test_drives
