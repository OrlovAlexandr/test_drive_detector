import os
from datetime import datetime

import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips


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
    print(f"Starting {start} and ending {end}. Videopath: {video_input}, output: {video_output}")
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
            if frame_num >= end:
                break

        if start <= frame_num:
            output.write(frame)

    cap.release()
    output.release()
    cv2.destroyAllWindows()

    return None


def get_video_clips(lots_states: dict, video_range: dict, timestamp_offset: int = 15, clip_duration: int = 30):
    """
    Get video clips of start and end of each test drive
    Args:
        lots_states: dict - lots states for each parking space
        video_range: dict - video range for each video path (clip start, clip end, fps)
        timestamp_offset: int - timestamp offset (N seconds before and 30-N seconds after)
        clip_duration: int - clip duration (30 seconds by default)
    Returns:
        None
    """
    output_clips_paths = []
    for space, states in lots_states.items():
        test_drive_number = 1
        for idx, (timestamp, state) in enumerate(states):
            state_name = 'end' if state else 'start'

            real_datetime = datetime.fromtimestamp(timestamp).strftime('%d, %B, %Y %H:%M:%S')
            print(
                f"Parking space {int(float(space)) + 1} - {state_name.title()} of Test drive {test_drive_number} at {real_datetime}")

            if state_name == 'end':
                start_timestamp = timestamp - (clip_duration - timestamp_offset)
                end_timestamp = timestamp + timestamp_offset
            else:
                start_timestamp = timestamp - timestamp_offset
                end_timestamp = timestamp + (clip_duration - timestamp_offset)
            # print('start_timestamp:', start_timestamp, 'end_timestamp:', end_timestamp)

            clip_start = None
            clip_end = None

            for video_path, (video_start, video_end, video_fps) in video_range.items():
                if video_start <= start_timestamp <= video_end:
                    clip_start_frame = (start_timestamp - video_start) * video_fps
                    clip_start = (video_path, int(clip_start_frame))
                if video_start <= end_timestamp <= video_end:
                    clip_end_frame = (end_timestamp - video_start) * video_fps
                    clip_end = (video_path, int(clip_end_frame))
            print(clip_start, clip_end)

            if clip_start is None and clip_end is None:
                continue

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

            print(video_path_start, frame_start)
            print(video_path_end, frame_end)

            output_video_path = os.path.abspath(
                f'results/videos/space{int(float(space)) + 1}_testdrive{test_drive_number:02}_{state_name}.mp4')
            print(output_video_path)
            if video_path_start == video_path_end:
                temp_path = os.path.abspath(f'results/videos/temp.mp4')

                record_video(video_path_start, temp_path, frame_start, frame_end)
                video = VideoFileClip(temp_path)
                video.write_videofile(output_video_path)

                if os.path.isfile(temp_path):
                    os.remove(temp_path)
            else:
                temp_path1 = os.path.abspath(f'results/videos/temp_start.mp4')
                temp_path2 = os.path.abspath(f'results/videos/temp_end.mp4')

                record_video(video_path_start, temp_path1, frame_start, None)
                record_video(video_path_end, temp_path2, 1, frame_end)  #
                # Concatenate videos
                merged_videos = concatenate_videoclips([VideoFileClip(temp_path1), VideoFileClip(temp_path2)])
                merged_videos.write_videofile(output_video_path)

                # Delete temp videos
                if os.path.isfile(temp_path1):
                    os.remove(temp_path1)
                if os.path.isfile(temp_path2):
                    os.remove(temp_path2)

            if state_name == 'end':
                test_drive_number += 1

            output_clips_paths.append(output_video_path)

    return output_clips_paths
