import os
import tempfile
from datetime import datetime
from typing import TypedDict, List

import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm
from src.test_drive import TestDrive


def record_video(video_input: str, video_output: str, start: int = None, end: int = None):
    """
    Record video from defined start and end frame
    Args:
        video_input: str - path to input video
        video_output: str - path to output video
        start: int - start frame
        end: int - end frame
    Returns:
        None
    """
    cap = cv2.VideoCapture(video_input)  # Read video with OpenCV
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if not ret:
            break

        if end:
            if frame_num > end:
                break

        if frame_num >= start:
            output.write(frame)

    cap.release()
    output.release()
    cv2.destroyAllWindows()

    return None


class Clip(TypedDict):

    parking_space: int
    state: str
    test_drive_number: int
    event_timestamp: float
    path: str


def process_video_clips(lots_states: dict,
                        video_meta: dict,
                        timestamp_offset: int = 15,
                        clip_duration: int = 30) -> List[Clip]:
    """
    Get video clips of start and end of each test drive.

    Args:
        lots_states (dict): Lots states for each parking space.
        video_meta (dict): Video range and fps for each video path (clip start, clip end, fps).
        timestamp_offset (int): Timestamp offset (N seconds before and 30-N seconds after).
        clip_duration (int): Clip duration (30 seconds by default).

    Returns:
        output_clips_info (list): List of video clips paths.
    """
    output_clips_info = []
    video_dir = './inbox'
    if not os.path.exists(video_dir):
        os.makedirs(video_dir, exist_ok=True)

    for space, states in lots_states.items():
        space = int(float(space))
        test_drive_number = 1  # count test drives for each parking space

        for idx, (timestamp, state) in enumerate(states):
            # Get name of state
            state_name = 'end' if state else 'start'

            # Get real datetime
            real_datetime = datetime.fromtimestamp(timestamp).strftime('%d, %B, %Y %H:%M:%S')
            print(
                f"Parking space {space + 1} - {state_name.title()} of Test drive {test_drive_number} at {real_datetime}")

            # Get start and end timestamps
            if state_name == 'end':
                start_timestamp = timestamp - (clip_duration - timestamp_offset)
                end_timestamp = timestamp + timestamp_offset
            else:
                start_timestamp = timestamp - timestamp_offset
                end_timestamp = timestamp + (clip_duration - timestamp_offset)

            # Get clip info: video path, start frame, end frame
            clip_start = None
            clip_end = None

            for video_path, (video_start, video_end, video_fps) in video_meta.items():
                if video_start <= start_timestamp <= video_end:
                    clip_start_frame = (start_timestamp - video_start) * video_fps
                    clip_start = (video_path, int(clip_start_frame))
                if video_start <= end_timestamp <= video_end:
                    clip_end_frame = (end_timestamp - video_start) * video_fps
                    clip_end = (video_path, int(clip_end_frame))

            # Skip if no clip found
            if clip_start is None and clip_end is None:
                continue

            # Get clip range and check if clip is in different videos
            video_path_start, video_path_end, frame_start, frame_end = None, None, None, None
            if clip_start is None:
                video_path_start = os.path.abspath(clip_end[0])
                video_path_end = os.path.abspath(clip_end[0])
                frame_start = 1
                frame_end = clip_end[1]
            elif clip_end is None:
                video_path_start = os.path.abspath(clip_start[0])
                video_path_end = os.path.abspath(clip_start[0])
                frame_start = clip_start[1]
                frame_end = None
            else:
                video_path_start = os.path.abspath(clip_start[0])
                video_path_end = os.path.abspath(clip_end[0])
                frame_start = clip_start[1]
                frame_end = clip_end[1]

            # print(video_path_start, 'Start frame:', frame_start)
            # print(video_path_end, '  End frame:', frame_end)

            # output_clip_path = os.path.abspath(
            #     f'{video_dir}/space{space + 1}_testdrive{test_drive_number:02}_{state_name}.mp4')
            # print('Output path:', output_clip_path)
            output_clip_path = tempfile.mktemp(suffix='.mp4', dir='./inbox')
            print('Output path:', output_clip_path)


            # if video_path_start == video_path_end:
            #     temp_path = os.path.abspath(f'{video_dir}/temp.mp4')
            #
            #     record_video(video_path_start, temp_path, frame_start, frame_end)
            #     video = VideoFileClip(temp_path)
            #     video.write_videofile(output_clip_path)
            #
            #     if os.path.isfile(temp_path):
            #         os.remove(temp_path)
            # else:
            #     temp_path1 = os.path.abspath(f'{video_dir}/temp_start.mp4')
            #     temp_path2 = os.path.abspath(f'{video_dir}/temp_end.mp4')
            #
            #     record_video(video_path_start, temp_path1, frame_start, None)
            #     record_video(video_path_end, temp_path2, 1, frame_end)  #
            #
            #     # Concatenate videos
            #     merged_videos = concatenate_videoclips([VideoFileClip(temp_path1), VideoFileClip(temp_path2)])
            #     merged_videos.write_videofile(output_clip_path)
            #
            #     # Delete temp videos
            #     if os.path.isfile(temp_path1):
            #         os.remove(temp_path1)
            #     if os.path.isfile(temp_path2):
            #         os.remove(temp_path2)


            print(Clip(parking_space=space + 1,
                      state=state_name,
                      test_drive_number=test_drive_number,
                      event_timestamp=timestamp,
                      clip_path=output_clip_path))
            output_clips_info.append(Clip(parking_space=space + 1,
                                          state=state_name,
                                          test_drive_number=test_drive_number,
                                          event_timestamp=timestamp,
                                          clip_path=output_clip_path))

            if state_name == 'end':
                test_drive_number += 1

    # print('output_clips_info:\n', output_clips_info)
    return output_clips_info
