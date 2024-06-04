import os

import pandas as pd

new_dirs = ['results/detections', 'results/videos', 'results/events', 'results/recalibrate', 'results/parking_lot',
            'source/videos', 'source/models']

for new_dir in new_dirs:
    if not os.path.exists(new_dir):
        os.makedirs(new_dir, exist_ok=True)

