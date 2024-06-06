import yaml

from utils.detection import detect_with_timestamps
from utils.prepare_detections import prepare_detections

if __name__ == "__main__":
    # INPUTS
    LOT_PATH = 'results/parking_lot_ii/parking_lot.yml'
    # Videos should be selected on the basis of the workday range
    # The dictionary should be compiled from the selected videos and their created timestamps
    VIDEO_DATA = {'./source/parking_lot_ii/videos/parking_lot_ii_001.mp4': 1716791179.0,
                  './source/parking_lot_ii/videos/parking_lot_ii_002.mp4': 1716791323.88}

    # OUTPUTS
    DETECTIONS_DF_PATH = 'results/parking_lot_ii/detections_df.parquet'

    # MAIN PROCESS
    # The cropped area should be the size of a parking lot with some padding.
    with open(LOT_PATH, 'r') as f:
        config = yaml.safe_load(f)
        crop = config['crop_xyxy']
        parking_polygon = list(config['parking_polygon'].values())
    print('crop coordinates:', crop)

    detections = detect_with_timestamps(VIDEO_DATA, crop)

    # Prepare detections to get centers
    df = prepare_detections(detections, parking_polygon)

    # Save detections to parquet file
    df.to_parquet(DETECTIONS_DF_PATH)
    print(df.columns)

    # Create dataframe from detections and save to parquet file
    # df = pd.DataFrame(detections, columns=['frame', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2', 'timestamp'])
    # df.to_parquet(save_df_path)

    # # =======================================================
    # # TEST VISUAL
    # video_path = 'source/parking_lot_ii/videos/parking_lot_ii_001.mp4'
    #
    # config_path = 'config/parking_lot.yml'
    # if os.path.isfile(config_path):
    #     # Read config file
    #     with open('./config/parking_lot.yml', 'r') as f:
    #         config = yaml.safe_load(f)
    #     parking_spaces = pd.DataFrame(config['parking_spaces'])
    #     print(parking_spaces)
    #
    # # Show video to check if the detection is correct
    # cap = cv2.VideoCapture(video_path)  # Read video with OpenCV
    # fps = cap.get(cv2.CAP_PROP_FPS)  # Get fps
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #
    # print(fps, width, height)
    #
    # df['timestamp_ord'] = pd.factorize(df['timestamp'])[0]+1
    # # print(df)
    #
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #
    #     frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
    #     detections = df.loc[df['timestamp_ord'] == frame_num].values
    #     # print(detections)
    #     for detection in detections:
    #         xyxy = [int(x) for x in detection[3:7]]
    #
    #         frame = cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 255), 1)
    #
    #     if os.path.isfile(config_path):
    #         for space in parking_spaces.values:
    #             space = [int(x) for x in space]
    #             cv2.circle(frame, (space[0], space[1]), 5, (0, 0, 255), -1)
    #             cv2.circle(frame, (space[0], space[1]), space[2], (0, 0, 255), 1)
    #
    #     cv2.imshow('frame', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
