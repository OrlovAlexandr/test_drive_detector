import os
from pprint import pprint

import yaml

from ops.detect_objects import detect_objects
from ops.get_test_drive_events import get_test_drive_events
from ops.get_video_clips import get_video_clips
from ops.parking_spaces import process_parking_spaces
from ops.recalibrate_spaces import recalibrate_spaces

src_folder = 'source/parking_lot_iii'

config_path = os.path.join(src_folder, 'config.yml')

# Read config file to get initial data
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
    video_data = config['video_data']
    parking_polygon = config['parking_polygon']
    frame_range = config['frame_range']

print(video_data)
# Take only video data which path ends with '1.mp4'
selected_video = {k: v for k, v in video_data.items() if k.endswith('1.mp4')}
print(selected_video)

# Convert data to list for processing
parking_polygon = [tuple(i) for i in parking_polygon.values()]
frame_range = (frame_range['from'], frame_range['to'])
# print(parking_polygon)

# Create folder
lot_name = os.path.split(src_folder)[-1]
lot_dir = os.path.join('results', lot_name)
os.makedirs(lot_dir, exist_ok=True)

# Define paths
lot_path = os.path.join(lot_dir, 'parking_lot.yml')
detections_path = os.path.join(lot_dir, 'detections.parquet')
recalibration_path = os.path.join(lot_dir, 'recalibrated_spaces.parquet')
events_path = os.path.join(lot_dir, 'events.json')
output_info_path = os.path.join(lot_dir, 'clips_info.json')

print((selected_video, parking_polygon, frame_range, lot_path))

# RUN STEPS
# Get parking spaces with each radius based on the bbox size and selected video fragment
process_parking_spaces(selected_video, parking_polygon, frame_range, lot_path)

# Get detections with timestamps
detect_objects(video_data, lot_path, detections_path)

# Recalibrate parking spaces to get updated coordinates
recalibrate_spaces(detections_path, lot_path, recalibration_path)

# Get test drive events with timestamps
get_test_drive_events(detections_path, recalibration_path, events_path)

# Get video clips for each event
output_clips_info = get_video_clips(video_data, events_path, output_info_path)
pprint(output_clips_info)
