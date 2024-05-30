import os


new_dirs = ['config', 'results/detections', 'results/videos', 'source/videos', 'source/models']

for new_dir in new_dirs:
    if not os.path.exists(new_dir):
        os.makedirs(new_dir, exist_ok=True)
