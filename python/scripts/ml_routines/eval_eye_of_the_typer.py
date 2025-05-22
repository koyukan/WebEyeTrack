import os
import pathlib
from collections import defaultdict
import argparse

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import seaborn as sns
import pandas as pd
import yaml
import open3d as o3d

import matplotlib
matplotlib.use('TkAgg')

from webeyetrack import WebEyeTrack
from webeyetrack.constants import GIT_ROOT
import webeyetrack.vis as vis
from webeyetrack.model_based import vector_to_pitch_yaw, compute_pog
from webeyetrack.data_protocols import GazeResult
from webeyetrack.kalman_filter import create_kalman_filter
from webeyetrack.utilities import (
    estimate_camera_intrinsics, 
    transform_for_3d_scene,
    transform_3d_to_3d,
    transform_3d_to_2d,
    get_rotation_matrix_from_vector,
    rotation_matrix_to_euler_angles,
    euler_angles_to_rotation_matrix,
    OPEN3D_RT,
    load_3d_axis,
    load_canonical_mesh,
    load_eyeball_model,
    create_transformation_matrix
)

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent.parent
OUTPUTS_DIR = CWD / 'outputs'
SKIP_COUNT = 100

with open(FILE_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

EYE_OF_THE_TYPER_DATASET = pathlib.Path(config['datasets']['EyeOfTheTyper']['path'])
assert EYE_OF_THE_TYPER_DATASET.exists(), f"Dataset not found at {EYE_OF_THE_TYPER_DATASET}"
EYE_OF_THE_TYPER_PAR_CHAR = pathlib.Path(config['datasets']['EyeOfTheTyper']['participant_characteristics'])

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

def preprocess_csv(csv_path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)

    # Drop the columns after 19th column
    data = data.iloc[:, :19]

    # Add the columns to the data
    data.columns = fieldnames

    return data

def visualize_scanpath(par, csv_data, screen_height_px, screen_width_px) -> plt.Figure:

    screen_img = np.zeros((screen_height_px, screen_width_px, 3), dtype=np.uint8)

    # Create a kalman filter for the gaze
    tobii_kf = create_kalman_filter(dt=1/120)
    
    for i, row in tqdm(csv_data.iterrows(), total=len(csv_data), desc=f'Visualizing scanpath for {par}'):
        
        # Load the image
        img_path = EYE_OF_THE_TYPER_DATASET / par / "/".join(row['frameImageFile'].split('/')[3:])
        assert img_path.exists(), f"Image not found at {img_path}"
        img = cv2.imread(str(img_path))

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

        # Apply the Kalman filter to the gaze point
        if gaze is not None:
            # Apply the Kalman filter
            tobii_kf.predict()
            tobii_kf.update(np.array(gaze_px))
            smooth_gaze = (tobii_kf.x[0], tobii_kf.x[1])

        # Obtain the webgazer gaze point
        webgazer_gaze = (row['webGazerX'], row['webGazerY'])

        # Display the gaze point
        if gaze is not None:
            cv2.circle(screen_img, (int(gaze[0]*screen_width_px), int(gaze[1]*screen_height_px)), 5, (0, 255, 0), -1)
            cv2.circle(screen_img, (int(smooth_gaze[0]*screen_width_px), int(smooth_gaze[1]*screen_height_px)), 5, (255, 0, 0), -1)
            
        # cv2.circle(screen_img, (int(webgazer_gaze[0]*screen_width_px), int(webgazer_gaze[1]*screen_height_px)), 5, (255, 0, 0), -1)
        # cv2.circle(screen_img, (int(webeyetrack_gaze[0]*screen_width_px), int(webeyetrack_gaze[1]*screen_height_px)), 5, (0, 0, 255), -1)

        # Display the image
        cv2.imshow('Image', img)
        cv2.imshow('Screen', screen_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close the windows
    cv2.destroyAllWindows()

    # Create the figure from the screen image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(screen_img)
    ax.axis('off')
    ax.set_title(f'Scanpath for {par}')
    plt.tight_layout()
    plt.show()
    return fig


def main():

    print("Starting Eye of the Typer evaluation...")

    timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    RUN_DIR = OUTPUTS_DIR / f'EyeOfTheTyper-{timestamp}'
    os.makedirs(RUN_DIR, exist_ok=True)

    # Iterate over the folders within the dataset
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

        # Create the WebEyeTrack object
        wet = WebEyeTrack()

        # Create a directory for each participant
        par_output_dir = RUN_DIR / par
        os.makedirs(par_output_dir, exist_ok=True)

        # Obtain the configurations for the participant
        par_config = PARTICIPANT_CHARACTERISTICS[PARTICIPANT_CHARACTERISTICS['Participant ID'] == par]
        screen_width_cm = par_config['Screen Width (cm)'].values[0]
        screen_height_cm = par_config['Screen Height (cm)'].values[0]
        screen_width_px = int(par_config['Display Width (pixels)'].values[0])
        screen_height_px = int(par_config['Display Height (pixels)'].values[0])

        # Perform the calibration using the initial dot test
        calib_csv = csvs[SECTIONS[0]].values[0]
        if calib_csv is None:
            print(f"Calibration CSV not found for participant {par}. Skipping.")
            continue

        # Load the calibration data
        dot_test_data = preprocess_csv(calib_csv)
        
        # Use the mouse clicks to calibrate
        mouse_click_data = dot_test_data[dot_test_data['mouseClickX'] != '[]']
        if len(mouse_click_data) == 0:
            print(f"No mouse clicks found for participant {par}. Skipping.")
            continue

        # Extract the 9-point calibration data (which should be top-left, top-center, top-right, center-left, center-center, center-right, bottom-left, bottom-center, bottom-right)
        x_coords, y_coords = [0.025, 0.5, 0.975], [0.15, 0.5, 0.85]
        pts = [(x, y) for x in x_coords for y in y_coords]

        # Create the calibration points by finding the closest points in the mouse clicks
        calib_pts = []
        for pt in pts:
            # Find the closest mouse click
            closest_click = None
            closest_click_info = None
            min_dist = float('inf')
            for i, row in mouse_click_data.iterrows():
                click_x, click_y = eval(row['mouseClickX'])[0], eval(row['mouseClickY'])[0]
                dist = np.linalg.norm(np.array([click_x, click_y]) - np.array(pt))
                if dist < min_dist:
                    min_dist = dist
                    closest_click = (click_x, click_y)
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
            norm_pogs.append((eval(row['mouseClickX'])[0], eval(row['mouseClickY'])[0]))

        # Perform the calibration
        wet.adapt_from_frames(frames, norm_pogs)

        # First, visualize the scanpath of the original dot test that we used for calibration
        # For this section, we want to visualize only the scanpath right after the first click and the last click
        # Get the first and last click
        first_click = mouse_click_data.iloc[0]
        last_click = mouse_click_data.iloc[-1]
        within_dot_test = dot_test_data[(dot_test_data['frameNum'] >= first_click['frameNum']) & (dot_test_data['frameNum'] <= last_click['frameNum'])]

        # Create the figure
        fig = visualize_scanpath(
            par, 
            within_dot_test,
            screen_height_px,
            screen_width_px
        )
        break


if __name__ == '__main__':
    main()