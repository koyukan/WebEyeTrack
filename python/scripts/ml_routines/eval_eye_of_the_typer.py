import os
import pathlib
from collections import defaultdict
import argparse

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import seaborn as sns
import pandas as pd
import yaml

import matplotlib
matplotlib.use('TkAgg')

from webeyetrack.data_protocols import TrackingStatus
from webeyetrack import WebEyeTrack, WebEyeTrackConfig
from webeyetrack.constants import GIT_ROOT
# from webeyetrack.kalman_filter import create_kalman_filter

# Set all the seeds
np.random.seed(42)
tf.random.set_seed(42)

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent.parent
OUTPUTS_DIR = CWD / 'outputs'
SKIP_COUNT = 100

with open(FILE_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

EYE_OF_THE_TYPER_DATASET = pathlib.Path(config['datasets']['EyeOfTheTyper']['path'])
assert EYE_OF_THE_TYPER_DATASET.exists(), f"Dataset not found at {EYE_OF_THE_TYPER_DATASET}"
EYE_OF_THE_TYPER_PAR_CHAR = pathlib.Path(config['datasets']['EyeOfTheTyper']['participant_characteristics'])

GENERATED_DATASET_DIR = GIT_ROOT / 'data' / 'generated'
DATA_CORRECTIONS_FILE = GENERATED_DATASET_DIR / 'eye_of_the_typer' / 'tobii_data_corrections.xlsx'
DATA_CORRECTIONS = pd.read_excel(DATA_CORRECTIONS_FILE)
CALIB_PTS_FILE =  GENERATED_DATASET_DIR / 'eye_of_the_typer' / 'calibration_pts.xlsx'
CALIB_PTS = pd.read_excel(CALIB_PTS_FILE)

SECTIONS = [
    'study-dot_test.webm_gazePredictionsDone',
    'study-benefits_of_running_writing.webm_gazePredictionsDone',
    'study-educational_advantages_of_social_networking_sites_writing.webm_gazePredictionsDone',
    'study-where_to_find_morel_mushrooms_writing.webm_gazePredictionsDone',
    'study-tooth_abscess_writing.webm_gazePredictionsDone',
    'study-dot_test_final.webm_gazePredictionsDone'
]
PARTICIPANT_CHARACTERISTICS = pd.read_csv(EYE_OF_THE_TYPER_PAR_CHAR)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
VISUALIZE = True

# A reminder of what the desired field name outputs are.
fieldnames = [
    'participant',
    'frameImageFile',
    'frameTimeEpoch',
    'frameNum',
    'mouseMoveX',
    'mouseMoveY',
    'mouseClickX',
    'mouseClickY',
    'keyPressed',
    'keyPressedX',
    'keyPressedY',
    'tobiiLeftScreenGazeX',
    'tobiiLeftScreenGazeY',
    'tobiiRightScreenGazeX',
    'tobiiRightScreenGazeY',
    'webGazerX',
    'webGazerY',
    'wgError',
    'wgErrorPix',

    # 'fmPos', # 71 2D
    # 'eyeFeatures', # 140 features
    # *['fmPos' + str(i) for i in range(2*71)],
    # *['eyeFeatures' + str(i) for i in range(140)],
]

"""
References:
https://webgazer.cs.brown.edu/data/

Webcam Videos: Resolution 640x480. Their name follows the format ParticipantLogID_VideoID_-study-nameOfTask.mp4 and ParticipantLogID_VideoID_-study-nameOfTask.webm. For each task page that the user visited, there is at least one corresponding webcam video capture. If a user visited the same page multiple times, then a different webcam video would correpond to each individual. The possible task pages in increasing order of visit are:
dot_test_instructions: instruction page for the Dot Test task.
dot_test: Dot Test task.
fitts_law_instructions: instruction page for the Fitts Law task.
fitts_law: Fitts Law task.
serp_instructions: instruction page for the search related tasks.
benefits_of_running_instructions: instruction page for the query benefits of running.
benefits_of_running: benefits of running SERP.
benefits_of_running_writing: Writing portion of benefits of running search task.
educational_advantages_of_social_networking_sites_instructions: instruction page for the query educational advantages of social networking sites.
educational_advantages_of_social_networking_sites: educational advantages of social networking sites SERP.
beducational_advantages_of_social_networking_sites_writing: Writing portion of educational advantages of social networking sites search task.
where_to_find_morel_mushrooms_instructions: instruction page for the query where to find morel mushrooms.
where_to_find_morel_mushrooms: where to find morel mushrooms SERP.
where_to_find_morel_mushrooms_writing: Writing portion of where to find morel mushrooms search task.
tooth_abscess_instructions: instruction page for the query tooth abscess.
tooth_abscess: tooth abscess SERP.
tooth_abscess_writing: Writing portion of tooth abscess sesrch task.
dot_test_final_instructions: instruction page for the Final Dot Test task.
dot_test_final: Final Dot Test task.
thank_you: Questionnaire.
"""

print("FINISHED IMPORTS and SETUP")

def preprocess_csv(pid, csv_path, section) -> pd.DataFrame:
    data = pd.read_csv(csv_path)

    # Drop the columns after 19th column
    data = data.iloc[:, :19]

    # Add the columns to the data
    data.columns = fieldnames

    # Shift all gaze from range [[0,1], [0,1]] to [[-0.5, 0.5], [-0.5, 0.5]], with center of screen being (0, 0)
    columns_to_shift = [
        'mouseMoveX', 'mouseMoveY',
        'mouseClickX', 'mouseClickY',
        'tobiiLeftScreenGazeX', 'tobiiLeftScreenGazeY',
        'tobiiRightScreenGazeX', 'tobiiRightScreenGazeY',
        'webGazerX', 'webGazerY',
    ]
    for col in columns_to_shift:
        # data[col] = data[col].apply(lambda x: (x - 0.5) if pd.notna(x) else x)
        for i, row in data.iterrows():
            item = row[col]
            float_item = None
            try:
                float_item = float(item)
            except ValueError:
                float_item_list = eval(item)
                if len(float_item_list) == 0:
                    float_item = None
                elif len(float_item_list) >= 1:
                    float_item = float_item_list[0]

            if float_item is not None:
                if float_item == -1:
                    data.at[i, col] = None
                else:
                    data.at[i, col] = float_item
            else:
                data.at[i, col] = None

    # Combine the left and right gaze points into a single gaze point
    for i, row in data.iterrows():
        # Compute the normalized gaze point from the Tobii (left and right).
        # If both available, average them.
        # If -1 for one of the eyes, use the other eye.
        # If both are -1, return None
        left_gaze = (row['tobiiLeftScreenGazeX'], row['tobiiLeftScreenGazeY'])
        right_gaze = (row['tobiiRightScreenGazeX'], row['tobiiRightScreenGazeY'])
        gaze = None
        if left_gaze[0] != -1 and right_gaze[0] != -1:
            gaze = ((left_gaze[0] + right_gaze[0]) / 2, (left_gaze[1] + right_gaze[1]) / 2)
        elif left_gaze[0] != -1:
            gaze = left_gaze
        elif right_gaze[0] != -1:
            gaze = right_gaze
        else:
            gaze = None

        data.at[i, 'tobiiGazeX'] = gaze[0] if gaze is not None else None
        data.at[i, 'tobiiGazeY'] = gaze[1] if gaze is not None else None

    # Drop rows where the 'tobiiGazeX' or 'tobiiGazeY' is None
    data = data.dropna(subset=['tobiiGazeX', 'tobiiGazeY'])

    # Apply correction to the gaze points
    # corrections = DATA_CORRECTIONS[(DATA_CORRECTIONS['pid'] == pid) & (DATA_CORRECTIONS['section'] == section)]
    # if not corrections.empty:
    #     corrections = corrections.iloc[0]
    #     data['tobiiGazeX'] = data['tobiiGazeX'] * corrections['scale_x'] + corrections['shift_x']
    #     data['tobiiGazeY'] = data['tobiiGazeY'] * corrections['scale_y'] + corrections['shift_y']

    # If the range of the gaze x and y is not [0, 1], then normalize it to [0, 1]
    gaze_x_min, gaze_x_max = data['tobiiGazeX'].min(), data['tobiiGazeX'].max()
    gaze_y_min, gaze_y_max = data['tobiiGazeY'].min(), data['tobiiGazeY'].max()
    # import pdb; pdb.set_trace()

    data['tobiiGazeX'] = (data['tobiiGazeX'] - gaze_x_min) / (gaze_x_max - gaze_x_min)
    # data['tobiiGazeY'] = (data['tobiiGazeY'] - gaze_y_min) / (gaze_y_max - gaze_y_min)

    # Then shift the gaze points to be in the range [[-0.5, 0.5], [-0.5, 0.5]]
    data['tobiiGazeX'] = data['tobiiGazeX'] - 0.5
    data['tobiiGazeY'] = data['tobiiGazeY'] - 0.5

    return data

def scanpath_video(video_dst_fp, par, calib_pts, returned_calib, csv_data, screen_height_px, screen_width_px, wet) -> plt.Figure:

    # Create a video writer for mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_dst_fp), fourcc, 30.0, (screen_width_px, screen_height_px))

    calib_pts_frames_nums = [pt['frameNum'] for pt in calib_pts]
    screen_img = np.zeros((screen_height_px, screen_width_px, 3), dtype=np.uint8)

    # Create a kalman filter for the gaze
    # tobii_kf = create_kalman_filter(dt=1/120)
    
    for i, row in tqdm(csv_data.iterrows(), total=len(csv_data), desc=f'Visualizing scanpath for {par}'):

        # Draw on the top left of the screen the frame number
        cv2.rectangle(screen_img, (0, 0), (150, 40), (0, 0, 0), -1)
        cv2.putText(screen_img, f'F: {row["frameNum"]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Load the image
        img_path = EYE_OF_THE_TYPER_DATASET / par / "/".join(row['frameImageFile'].split('/')[3:])
        assert img_path.exists(), f"Image not found at {img_path}"
        img = cv2.imread(str(img_path))

        # Apply the Kalman filter to the gaze point
        # if gaze is not None:
        #     # Apply the Kalman filter
        #     tobii_kf.predict()
        #     tobii_kf.update(np.array(gaze_px))
        #     smooth_gaze = (tobii_kf.x[0], tobii_kf.x[1])

        # Obtain the gaze point from the row
        gaze = (row['tobiiGazeX'], row['tobiiGazeY'])
        if gaze[0] is None or gaze[1] is None:
            gaze = None

        # Obtain the webgazer gaze point
        webgazer_gaze = (row['webGazerX'], row['webGazerY'])
        status, gaze_result, _ = wet.process_frame(img)

        # Shifting all gaze points from [[-0.5, 0.5], [-0.5, 0.5]] to [[0,1], [0,1]]
        if gaze is not None:
            gaze = (gaze[0] + 0.5, gaze[1] + 0.5)
        if webgazer_gaze is not None:
            webgazer_gaze = (webgazer_gaze[0] + 0.5, webgazer_gaze[1] + 0.5)

        # Display the gaze point
        if gaze is not None:
            cv2.circle(screen_img, (int(gaze[0]*screen_width_px), int(gaze[1]*screen_height_px)), 5, (0, 255, 0), -1)
            # cv2.circle(screen_img, (int(smooth_gaze[0]*screen_width_px), int(smooth_gaze[1]*screen_height_px)), 5, (255, 0, 0), -1)
        
        if status == TrackingStatus.SUCCESS and gaze_result is not None:
            gaze_point = gaze_result.norm_pog
            gaze_point = (gaze_point[0] + 0.5, gaze_point[1] + 0.5)
            gaze_point = (gaze_point[0] * screen_width_px, gaze_point[1] * screen_height_px)
            cv2.circle(screen_img, (int(gaze_point[0]), int(gaze_point[1])), 5, (0, 0, 255), -1)

            # Draw a gray line between the pred and gt points
            # if gaze is not None:
            #     cv2.line(screen_img, (int(gaze[0]*screen_width_px), int(gaze[1]*screen_height_px)),
            #              (int(gaze_point[0]), int(gaze_point[1])), (128, 128, 128), 1)

        # cv2.circle(screen_img, (int(webgazer_gaze[0]*screen_width_px), int(webgazer_gaze[1]*screen_height_px)), 5, (255, 0, 0), -1)
        # cv2.circle(screen_img, (int(webeyetrack_gaze[0]*screen_width_px), int(webeyetrack_gaze[1]*screen_height_px)), 5, (0, 0, 255), -1)

        # If this is a calibration point, make a big yellow circle
        # if row['frameNum'] in [pt['frameNum'] for pt in calib_pts]:
        frame_idx = calib_pts_frames_nums.index(row['frameNum']) if row['frameNum'] in calib_pts_frames_nums else None
        if frame_idx is not None:
            row = calib_pts[frame_idx]
            x, y = row['tobiiGazeX'], row['tobiiGazeY']
            x, y = (x + 0.5), (y + 0.5)
            cv2.circle(screen_img, (int(x * screen_width_px), int(y * screen_height_px)), 10, (0, 255, 255), -1)

            # Draw the resulting calibration point
            resulting_calib_point = returned_calib[frame_idx]
            x2, y2 = (resulting_calib_point[0] + 0.5), (resulting_calib_point[1] + 0.5)
            cv2.circle(screen_img, (int(x2 * screen_width_px), int(y2 * screen_height_px)), 10, (255, 255, 0), -1)

            # Draw a line between the original calibration point and the resulting calibration point
            cv2.line(screen_img, (int(x * screen_width_px), int(y * screen_height_px)),
                        (int(x2 * screen_width_px), int(y2 * screen_height_px)), (255, 0, 255), 1)

        # Display the image
        if VISUALIZE:
            cv2.imshow('Image', img)
            cv2.imshow('Screen', screen_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Write the frame to the video
        video_writer.write(screen_img)

    # Close the windows
    cv2.destroyAllWindows()

    # Release the video writer
    video_writer.release()


def main():

    print("Starting Eye of the Typer evaluation...")

    timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    RUN_DIR = OUTPUTS_DIR / f'{timestamp}-EyeOfTheTyper'
    os.makedirs(RUN_DIR, exist_ok=True)

    # Iterate over the folders within the dataset
    # p_dirs = [EYE_OF_THE_TYPER_DATASET / 'P_02']
    p_dirs = [p for p in EYE_OF_THE_TYPER_DATASET.iterdir() if p.is_dir()]
    gaze_csvs = [p for p in EYE_OF_THE_TYPER_DATASET.iterdir() if p.is_file() and p.suffix == '.csv']
    options = set(['study' + p.stem.split('-study')[1] for p in gaze_csvs])

    print(f"Found {len(p_dirs)} participants and {len(gaze_csvs)} gaze CSVs.")

    # Sort the gaze into separate containers for each participant
    gaze_by_participant = defaultdict(list)
    for p in p_dirs:
        participant = p.stem
        gaze_csvs = [gaze_csv for gaze_csv in EYE_OF_THE_TYPER_DATASET.iterdir() if gaze_csv.is_file() and gaze_csv.stem.startswith(participant) and gaze_csv.suffix == '.csv']
        gaze_by_participant['par'].append(participant)
        for option in options:
            gaze_csv = [gaze_csv for gaze_csv in gaze_csvs if gaze_csv.stem.endswith(option)]
            if len(gaze_csv) > 0:
                gaze_by_participant[option].append(gaze_csv[0])
            else:
                gaze_by_participant[option].append(None)

    gaze_by_participant = pd.DataFrame(gaze_by_participant)
    print(f"Formed gaze_by_participant dataframe with {len(gaze_by_participant)} rows and {len(gaze_by_participant.columns)} columns.")

    # For each CSV, read the data and display the gaze
    participants_metrics = []
    for par, csvs in tqdm(gaze_by_participant.groupby('par'), total=len(gaze_by_participant), desc=f'Processing participants data'):

        # Create a directory for each participant
        par_output_dir = RUN_DIR / par
        os.makedirs(par_output_dir, exist_ok=True)

        # Obtain the configurations for the participant
        par_config = PARTICIPANT_CHARACTERISTICS[PARTICIPANT_CHARACTERISTICS['Participant ID'] == par]
        screen_width_cm = par_config['Screen Width (cm)'].values[0]
        screen_height_cm = par_config['Screen Height (cm)'].values[0]
        screen_width_px = int(par_config['Display Width (pixels)'].values[0])
        screen_height_px = int(par_config['Display Height (pixels)'].values[0])

        # Create the WebEyeTrack object
        wet = WebEyeTrack(
            WebEyeTrackConfig(
                screen_px_dimensions=(screen_width_px, screen_height_px),
                screen_cm_dimensions=(screen_width_cm, screen_height_cm),
                # verbose=True
            )
        )

        # Perform the calibration using the initial dot test
        calib_csv = csvs[SECTIONS[0]].values[0]
        if calib_csv is None:
            print(f"Calibration CSV not found for participant {par}. Skipping.")
            continue

        # Load the calibration data
        dot_test_data = preprocess_csv(par, calib_csv, SECTIONS[0])
        
        # Use the mouse clicks to calibrate
        mouse_click_data = dot_test_data[dot_test_data['mouseClickX'] != '[]']
        if len(mouse_click_data) == 0:
            print(f"No mouse clicks found for participant {par}. Skipping.")
            continue


        # import pdb; pdb.set_trace()
        calib_pts = []
        if par in CALIB_PTS['pid'].values:
            # Create the calibration points by finding the closest points in the mouse clicks
            pts = eval(CALIB_PTS[CALIB_PTS['pid'] == par].iloc[0].pts)
            for pt in pts:
                row = dot_test_data[dot_test_data['frameNum'] == pt].iloc[0]
                calib_pts.append(row)

        else:
            # Extract the 9-point calibration data (which should be top-left, top-center, top-right, center-left, center-center, center-right, bottom-left, bottom-center, bottom-right)
            # x_coords, y_coords = [0.15, 0.5, 0.85], [0.15, 0.5, 0.85]
            # x_coords, y_coords = [0.15, 0.85], [0.15, 0.85]
            x_coords, y_coords = [-0.45, 0.45], [-0.35, 0.35]
            pts = [(x, y) for x in x_coords for y in y_coords]
            # Top-center, center-left, center-right, bottom-center
            # pts = [
            #     (0, 0.25),  # Top-center
            #     (-0.25, 0),  # Center-left
            #     (0.25, 0),  # Center-right
            #     (0, -0.25),  # Bottom-center
            # ]
            # pts = pts[:2]
            # pts = [pts[3]]
            # print(pts)

            for pt in pts:
                # Find the closest mouse click
                closest_click = None
                closest_click_info = None
                min_dist = float('inf')
                for i, row in dot_test_data.iterrows():
                    # click_x, click_y = eval(row['mouseClickX'])[0], eval(row['mouseClickY'])[0]
                    # click_x, click_y = row['mouseClickX'], row['mouseClickY']
                    x, y = row['tobiiGazeX'], row['tobiiGazeY']
                    if not isinstance(x, float) or not isinstance(y, float):
                        continue

                    dist = np.linalg.norm(np.array([x, y]) - np.array(pt))
                    if dist < min_dist:
                        min_dist = dist
                        closest_click = (x, y)
                        closest_click_info = row
                if closest_click is not None:
                    calib_pts.append(closest_click_info)

        # Create the frame and point lists
        frames = []
        norm_pogs = []
        for i, row in enumerate(calib_pts):
            img_path = EYE_OF_THE_TYPER_DATASET / par / "/".join(row['frameImageFile'].split('/')[3:])
            assert img_path.exists(), f"Image not found at {img_path}"
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Image not found at {img_path}")
                continue
            frames.append(img)
            x, y = row['tobiiGazeX'], row['tobiiGazeY']
            norm_pogs.append(np.array([x, y]))
        
        norm_pogs = np.stack(norm_pogs)
        
        # Perform the calibration
        # returned_calib = []
        returned_calib = wet.adapt_from_frames(frames, norm_pogs)

        # Draw the calibration points on the screen
        screen_img = np.zeros((screen_height_px, screen_width_px, 3), dtype=np.uint8)
        for i, row in enumerate(calib_pts):

            # gt_pt = row['mouseClickX'], row['mouseClickY']
            gt_tb_pt = row['tobiiGazeX'], row['tobiiGazeY']
            pred_pt = returned_calib[i]

            # Draw the points as circles
            x, y = (gt_tb_pt[0] + 0.5), (gt_tb_pt[1] + 0.5)
            cv2.circle(screen_img, (int(x * screen_width_px), int(y * screen_height_px)), 10, (0, 0, 255), -1)
            x2, y2 = (pred_pt[0] + 0.5), (pred_pt[1] + 0.5)
            cv2.circle(screen_img, (int(x2 * screen_width_px), int(y2 * screen_height_px)), 10, (255, 255, 0), -1)

            # Draw a line between the original calibration point and the resulting calibration point
            cv2.line(screen_img, (int(x * screen_width_px), int(y * screen_height_px)),
                        (int(x2 * screen_width_px), int(y2 * screen_height_px)), (255, 0, 255), 1)
            
        # Display the image
        if VISUALIZE:
            cv2.imshow('Calibration Point', screen_img)
            cv2.waitKey(1)

        # Save the calibration image
        calib_img_path = par_output_dir / f'{SECTIONS[0]}_calib.png'
        cv2.imwrite(str(calib_img_path), screen_img)

        # break

        # First, visualize the scanpath of the original dot test that we used for calibration
        # For this section, we want to visualize only the scanpath right after the first click and the last click
        # Get the first and last click
        first_click = mouse_click_data.iloc[0]
        last_click = mouse_click_data.iloc[-1]
        within_dot_test = dot_test_data[(dot_test_data['frameNum'] >= first_click['frameNum']) & (dot_test_data['frameNum'] <= last_click['frameNum'])]

        # Create the figure
        scanpath_video(
            par_output_dir / f'{SECTIONS[0]}.mp4',
            par,
            calib_pts,
            returned_calib,
            within_dot_test,
            screen_height_px,
            screen_width_px,
            wet
        )

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()