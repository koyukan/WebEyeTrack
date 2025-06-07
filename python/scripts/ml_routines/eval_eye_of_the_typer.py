import os
import pathlib
from collections import defaultdict
import argparse
import time
import json

from sklearn.cluster import KMeans
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
MARGIN = 0.05

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
    # 'study-benefits_of_running_writing.webm_gazePredictionsDone',
    # 'study-educational_advantages_of_social_networking_sites_writing.webm_gazePredictionsDone',
    # 'study-where_to_find_morel_mushrooms_writing.webm_gazePredictionsDone',
    # 'study-tooth_abscess_writing.webm_gazePredictionsDone',
    'study-dot_test_final.webm_gazePredictionsDone'
]
PARTICIPANT_CHARACTERISTICS = pd.read_csv(EYE_OF_THE_TYPER_PAR_CHAR)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

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

def preprocess_csv(config, pid, csv_path, section) -> pd.DataFrame:
    data = pd.read_csv(csv_path)

    # Drop the columns after 19th column
    data = data.iloc[:, :19]

    # Add the columns to the data
    data.columns = fieldnames

    # Preprocess columns to make into proper floats
    columns_to_shift = [
        'mouseMoveX', 'mouseMoveY',
        'mouseClickX', 'mouseClickY',
        'tobiiLeftScreenGazeX', 'tobiiLeftScreenGazeY',
        'tobiiRightScreenGazeX', 'tobiiRightScreenGazeY',
        'webGazerX', 'webGazerY',
    ]
    for col in columns_to_shift:
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

    # If the range of the gaze x and y is not [0, 1], then normalize it to [0, 1]
    gaze_x_min, gaze_x_max = data['tobiiGazeX'].min(), data['tobiiGazeX'].max()
    gaze_y_min, gaze_y_max = data['tobiiGazeY'].min(), data['tobiiGazeY'].max()
    gaze_x_min, gaze_y_min = gaze_x_min - config['preprocess']['margin'], gaze_y_min - config['preprocess']['margin']
    gaze_x_max, gaze_y_max = gaze_x_max + config['preprocess']['margin'], gaze_y_max + config['preprocess']['margin']

    # data['tobiiGazeX'] = (data['tobiiGazeX'] - gaze_x_min) / (gaze_x_max - gaze_x_min)
    # data['tobiiGazeY'] = (data['tobiiGazeY'] - gaze_y_min) / (gaze_y_max - gaze_y_min)
    # data['webGazerX'] = (data['webGazerX'] - gaze_x_min) / (gaze_x_max - gaze_x_min)
    # data['webGazerY'] = (data['webGazerY'] - gaze_y_min) / (gaze_y_max - gaze_y_min)
    # data.iloc[:, 'tobiiGazeX'] = (data['tobiiGazeX'] - gaze_x_min) / (gaze_x_max - gaze_x_min)
    # data.iloc[:, 'tobiiGazeY'] = (data['tobiiGazeY'] - gaze_y_min) / (gaze_y_max - gaze_y_min)
    # data.iloc[:, 'webGazerX'] = (data['webGazerX'] - gaze_x_min) / (gaze_x_max - gaze_x_min)
    # data.iloc[:, 'webGazerY'] = (data['webGazerY'] - gaze_y_min) / (gaze_y_max - gaze_y_min)
    data = (
        data
        .assign(
            tobiiGazeX = lambda d: (d["tobiiGazeX"] - gaze_x_min)/(gaze_x_max - gaze_x_min),
            tobiiGazeY = lambda d: (d["tobiiGazeY"] - gaze_y_min)/(gaze_y_max - gaze_y_min),
            webGazerX  = lambda d: (d["webGazerX"]  - gaze_x_min)/(gaze_x_max - gaze_x_min),
            webGazerY  = lambda d: (d["webGazerY"]  - gaze_y_min)/(gaze_y_max - gaze_y_min),
        )
    )


    # Shift all gaze from range [[0,1], [0,1]] to [[-0.5, 0.5], [-0.5, 0.5]], with center of screen being (0, 0)
    for col in columns_to_shift + ['tobiiGazeX', 'tobiiGazeY']:
        for i, row in data.iterrows():
            item = row[col]
            if item is not None:
                # Shift the gaze point to be in the range [[-0.5, 0.5], [-0.5, 0.5]]
                data.at[i, col] = item - 0.5
            else:
                data.at[i, col] = None

    # Then shift the gaze points to be in the range [[-0.5, 0.5], [-0.5, 0.5]]
    # data['tobiiGazeX'] = data['tobiiGazeX'] - 0.5
    # data['tobiiGazeY'] = data['tobiiGazeY'] - 0.5
    # data['webGazerX'] = data['webGazerX'] - 0.5
    # data['webGazerY'] = data['webGazerY'] - 0.5

    return data

def scanpath_video(
        config, 
        video_dst_fp, 
        par,
        csv_data, 
        screen_height_px, 
        screen_width_px, 
        wet,
        calib_pts=None, 
        returned_calib=None, 
    ) -> plt.Figure:

    # Create a video writer for mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_dst_fp), fourcc, 30.0, (screen_width_px, screen_height_px))

    screen_img = np.zeros((screen_height_px, screen_width_px, 3), dtype=np.uint8)

    # Create a kalman filter for the gaze
    # tobii_kf = create_kalman_filter(dt=1/120)
    csv_data['webEyeTrackX'] = ''
    csv_data['webEyeTrackY'] = ''

    if not (calib_pts is None or returned_calib is None):
        calib_pts_frames_nums = [pt['frameNum'] for pt in calib_pts]
    
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

            # Add the information to the csv_data
            csv_data.at[i, 'webEyeTrackX'] = gaze_result.norm_pog[0]
            csv_data.at[i, 'webEyeTrackY'] = gaze_result.norm_pog[1]

            # Draw a gray line between the pred and gt points
            # if gaze is not None:
            #     cv2.line(screen_img, (int(gaze[0]*screen_width_px), int(gaze[1]*screen_height_px)),
            #              (int(gaze_point[0]), int(gaze_point[1])), (128, 128, 128), 1)

        # cv2.circle(screen_img, (int(webgazer_gaze[0]*screen_width_px), int(webgazer_gaze[1]*screen_height_px)), 5, (255, 0, 0), -1)
        # cv2.circle(screen_img, (int(webeyetrack_gaze[0]*screen_width_px), int(webeyetrack_gaze[1]*screen_height_px)), 5, (0, 0, 255), -1)

        # If this is a calibration point, make a big yellow circle
        # if row['frameNum'] in [pt['frameNum'] for pt in calib_pts]:
        if not (calib_pts is None or returned_calib is None):
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
        if config['visualize']:
            cv2.imshow('Image', img)
            cv2.imshow('Screen', screen_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Write the frame to the video
        video_writer.write(screen_img)

    # Save the final image
    final_img_path = video_dst_fp.parent / f'{video_dst_fp.stem}_final.png'
    cv2.imwrite(final_img_path, screen_img)

    # Close the windows
    cv2.destroyAllWindows()

    # Release the video writer
    video_writer.release()

def compute_metrics(par_output_dir, par, gaze_data, screen_height_cm, screen_width_cm):

    # Compute the L1 error for WebGazer and WebEyeTrack
    webgazer_l1 = []
    webeyetrack_l1 = []
    for i, row in gaze_data.iterrows():
        gt_x, gt_y = row['tobiiGazeX'], row['tobiiGazeY']
        if pd.isna(gt_x) or pd.isna(gt_y):
            continue
        gt_x, gt_y = gt_x * screen_width_cm, gt_y * screen_height_cm

        webgazer_x, webgazer_y = row['webGazerX'], row['webGazerY']
        webeyetrack_x, webeyetrack_y = row['webEyeTrackX'], row['webEyeTrackY']
        if not pd.isna(webgazer_x) and not pd.isna(webgazer_y) and isinstance(webgazer_x, float) and isinstance(webgazer_y, float) and isinstance(webeyetrack_x, float) and isinstance(webeyetrack_y, float):
            webgazer_x, webgazer_y = webgazer_x * screen_width_cm, webgazer_y * screen_height_cm
            webgazer_l1.append(np.linalg.norm(np.array([gt_x, gt_y]) - np.array([webgazer_x, webgazer_y])))
            webeyetrack_x, webeyetrack_y = webeyetrack_x * screen_width_cm, webeyetrack_y * screen_height_cm
            webeyetrack_l1.append(np.linalg.norm(np.array([gt_x, gt_y]) - np.array([webeyetrack_x, webeyetrack_y])))

    # Make a boxplot of the L1 errors
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[webgazer_l1, webeyetrack_l1], palette=["#FF6347", "#4682B4"])
    plt.xticks([0, 1], ['WebGazer', 'WebEyeTrack'])
    plt.ylabel('L1 Error (normalized)')
    plt.title(f'L1 Error for {par}')
    plt.tight_layout()

    # Put a maximum value of 40 for the y-axis
    ylim = min(40, max(max(webgazer_l1, default=0), max(webeyetrack_l1, default=0)))
    plt.ylim(0, ylim)
    
    # Save the boxplot
    boxplot_path = par_output_dir / f'{par}_l1_error_boxplot.png'
    plt.savefig(boxplot_path)
    plt.close()

    # Plot the error as a xy line plot
    plt.figure(figsize=(10, 6))
    x_range = np.arange(len(webgazer_l1))
    plt.plot(x_range, webgazer_l1, label='WebGazer', color='red', marker='o', markersize=3)
    x_range = np.arange(len(webeyetrack_l1))
    plt.plot(x_range, webeyetrack_l1, label='WebEyeTrack', color='blue', marker='o', markersize=3)
    plt.xlabel('Frame Number')
    plt.ylabel('L1 Error (normalized)')
    plt.title(f'L1 Error for {par}')
    plt.legend()
    plt.tight_layout()

    # Save the line plot
    lineplot_path = par_output_dir / f'{par}_l1_error_lineplot.png'
    plt.savefig(lineplot_path)
    plt.close()
    
    # Compute the average L1 error
    webgazer_avg_l1 = np.mean(webgazer_l1)
    webeyetrack_avg_l1 = np.mean(webeyetrack_l1)

    # Create a dataframe with the metrics
    metrics_df = pd.DataFrame({
        'participant': [par],
        'webGazerAvgL1': [webgazer_avg_l1],
        'webEyeTrackAvgL1': [webeyetrack_avg_l1],
    })

    # Save the metrics to a CSV file
    metrics_csv_path = par_output_dir / f'{par}_metrics.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)

    return webgazer_l1, webeyetrack_l1
            
def perform_calib(
    config,
    par,
    gaze_data,
    par_output_dir,
    wet,
    screen_height_px,
    screen_width_px,
):
    # ▶ 1 ─── Build initial guesses ────────────────────────────────────────────────
    # A loose 3×3 grid that roughly spans your data cloud.
    # Feel free to tweak these if your calibration rig isn’t perfectly symmetric.
    calib_pts = []
    x_guess = np.array([-0.45, 0.0, 0.45])
    y_guess = np.array([-0.45, 0.0, 0.45])
    init_centroids = np.array([(x, y) for y in y_guess for x in x_guess])

    # ▶ 2 ─── Run K-means ──────────────────────────────────────────────────────────
    xy = gaze_data[["tobiiGazeX", "tobiiGazeY"]].to_numpy()

    kmeans = KMeans(
        n_clusters=9,
        init=init_centroids,
        n_init=1,          # crucial – keeps scikit-learn from re-initialising
        max_iter=300,
        random_state=42
    )
    labels = kmeans.fit_predict(xy)
    centroids = kmeans.cluster_centers_

    # Attach labels to your original frame if you want them later
    gaze_data["cluster"] = labels

    # ▶ 3 ─── Visualise ────────────────────────────────────────────────────────────
    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(
        gaze_data["tobiiGazeX"],
        gaze_data["tobiiGazeY"],
        c=labels,
        cmap="tab10",
        s=15,
        alpha=0.8,
        edgecolor="none"
    )

    # draw centroids
    ax.scatter(
        centroids[:, 0], centroids[:, 1],
        marker="P", s=200, c="black", label="centroid"
    )
    # label centroids 0-8 for clarity
    for idx, (cx, cy) in enumerate(centroids):
        ax.text(cx, cy, str(idx), ha="center", va="center",
                color="white", weight="bold", fontsize=9)

    # ax.set_xlim(-0.5, 0.5)
    # ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Normalized X Coordinate")
    ax.set_ylabel("Normalized Y Coordinate")
    ax.set_title(f"Calibration clusters for {par}")
    # ax.legend(frameon=False, loc="upper left")

    plt.tight_layout()
    # plt.show()

    # Save the figure
    fig.savefig(par_output_dir / f'{SECTIONS[0]}_kmeans_clusters.png')
    plt.close(fig)

    # Use the centroids to obtain the calibration points
    used_centroids = [centroids[0], centroids[2], centroids[6], centroids[8]]
    for centroid in used_centroids:
        # Find the closest point to the centroid
        closest_point = gaze_data.iloc[
            np.argmin(np.linalg.norm(gaze_data[["tobiiGazeX", "tobiiGazeY"]].to_numpy() - centroid, axis=1))
        ]
        calib_pts.append(closest_point)

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
    if config['calibration']:
        returned_calib = wet.adapt_from_frames(
            frames, 
            norm_pogs,
            affine_transform=config['affine_transform'],
            steps_inner=config['steps_inner'],
            inner_lr=config['inner_lr'],
        )
    else:
        returned_calib = []

    # Draw the calibration points on the screen
    screen_img = np.zeros((screen_height_px, screen_width_px, 3), dtype=np.uint8)
    if config['calibration']:
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
    if config['visualize']:
        cv2.imshow('Calibration Point', screen_img)
        cv2.waitKey(1)

    # Save the calibration image
    calib_img_path = par_output_dir / f'{SECTIONS[0]}_calib.png'
    cv2.imwrite(str(calib_img_path), screen_img)

    return calib_pts, returned_calib

def main(args, config):

    print("Starting Eye of the Typer evaluation...")

    timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    RUN_DIR = OUTPUTS_DIR / f'{timestamp}-EyeOfTheTyper-{args.exp}'
    os.makedirs(RUN_DIR, exist_ok=True)

    # Iterate over the folders within the dataset
    # p_dirs = [EYE_OF_THE_TYPER_DATASET / 'P_01', EYE_OF_THE_TYPER_DATASET / 'P_02']
    p_dirs = [p for p in EYE_OF_THE_TYPER_DATASET.iterdir() if p.is_dir()]
    gaze_csvs = [p for p in EYE_OF_THE_TYPER_DATASET.iterdir() if p.is_file() and p.suffix == '.csv']
    options = set(['study' + p.stem.split('-study')[1] for p in gaze_csvs])

    if config['num_of_participants'] > 0:
        p_dirs = p_dirs[:config['num_of_participants']]

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

    per_participant_dir = RUN_DIR / 'per_participant'
    os.makedirs(per_participant_dir, exist_ok=True)

    # For each CSV, read the data and display the gaze
    # participants_metrics = defaultdict(list)
    participants_metrics = {k: defaultdict(list) for k in SECTIONS}
    for par, csvs in tqdm(gaze_by_participant.groupby('par'), total=len(gaze_by_participant), desc=f'Processing participants data'):

        # Create a directory for each participant
        par_output_dir = per_participant_dir / par
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
            return

        # Load the calibration data
        dot_test_data = preprocess_csv(config, par, calib_csv, SECTIONS[0])
        section_output_dir = par_output_dir / SECTIONS[0]
        os.makedirs(section_output_dir, exist_ok=True)
        
        # Use the mouse clicks to calibrate
        mouse_click_data = dot_test_data[dot_test_data['mouseClickX'] != '[]']
        if len(mouse_click_data) == 0:
            print(f"No mouse clicks found for participant {par}. Skipping.")
            return

        # For this section, we want to visualize only the scanpath right after the first click and the last click
        # Get the first and last click
        first_click = mouse_click_data.iloc[0]
        last_click = mouse_click_data.iloc[-1]
        within_dot_test = dot_test_data[(dot_test_data['frameNum'] >= first_click['frameNum']) & (dot_test_data['frameNum'] <= last_click['frameNum'])]

        # Perform the calibration
        if config['calibration']:
            calib_pts, returned_calib = perform_calib(
                config,
                par,
                within_dot_test,
                section_output_dir,
                wet,
                screen_height_px,
                screen_width_px,
            )
        else:
            calib_pts = []
            returned_calib = []

        # Create the figure
        scanpath_video(
            config,
            section_output_dir / f'{SECTIONS[0]}.mp4',
            par,
            within_dot_test,
            screen_height_px,
            screen_width_px,
            wet,
            calib_pts=calib_pts,
            returned_calib=returned_calib,
        )

        # Compute the metrics
        webgazer_l1, webeyetrack_l1 = compute_metrics(
            section_output_dir,
            par,
            within_dot_test,
            screen_height_cm,
            screen_width_cm,
        )

        # Save the metrics to the participants_metrics
        par_list = [par] * len(webgazer_l1)
        participants_metrics[SECTIONS[0]]['participant'].extend(par_list)
        participants_metrics[SECTIONS[0]]['gaze'].extend(webgazer_l1)
        participants_metrics[SECTIONS[0]]['class'].extend(['webGazer'] * len(webgazer_l1))
        participants_metrics[SECTIONS[0]]['participant'].extend(par_list)
        participants_metrics[SECTIONS[0]]['gaze'].extend(webeyetrack_l1)
        participants_metrics[SECTIONS[0]]['class'].extend(['webEyeTrack'] * len(webeyetrack_l1))

        for section in SECTIONS[1:]:
            csv_path = csvs[section].values[0]
            if csv_path is None:
                print(f"CSV not found for participant {par} in section {section}. Skipping.")
                continue

            section_output_dir = par_output_dir / section
            os.makedirs(section_output_dir, exist_ok=True)

            # Preprocess the CSV data
            gaze_data = preprocess_csv(config, par, csv_path, section)

            # Create the video for the scanpath
            scanpath_video(
                config,
                section_output_dir / f'{section}.mp4',
                par,
                gaze_data,
                screen_height_px,
                screen_width_px,
                wet
            )

            # Compute the metrics
            webgazer_l1, webeyetrack_l1 = compute_metrics(
                section_output_dir,
                par,
                gaze_data,
                screen_height_cm,
                screen_width_cm,
            )

            # Save the metrics to the participants_metrics
            par_list = [par] * len(webgazer_l1)
            participants_metrics[section]['participant'].extend(par_list)
            participants_metrics[section]['gaze'].extend(webgazer_l1)
            participants_metrics[section]['class'].extend(['webGazer'] * len(webgazer_l1))
            participants_metrics[section]['participant'].extend(par_list)
            participants_metrics[section]['gaze'].extend(webeyetrack_l1)
            participants_metrics[section]['class'].extend(['webEyeTrack'] * len(webeyetrack_l1))

    # participants_metrics_df = pd.DataFrame(participants_metrics)
    for k, v in participants_metrics.items():
        participants_metrics[k] = pd.DataFrame(v)

    sections_output_dir = RUN_DIR / 'sections'
    os.makedirs(sections_output_dir, exist_ok=True)

    # For each section, run the analysis separately
    for name, group in participants_metrics.items():
        section_output_dir = sections_output_dir / name
        os.makedirs(section_output_dir, exist_ok=True)

        # Add a boxplot for the participants metrics
        sns.set(style="whitegrid")
        plt.figure(figsize=(15, 6))
        sns.boxplot(group, x="participant", y="gaze", hue="class")
        # plt.xticks([0, 1], ['WebGazer', 'WebEyeTrack'])
        plt.ylabel('L1 Error (normalized)')
        plt.title('L1 Error for all participants')
        plt.tight_layout()

        # Put a limit of 40 for the y-axis
        ylim = min(40, group['gaze'].max())
        plt.ylim(0, ylim)

        # Save the boxplot
        boxplot_path = section_output_dir / 'participants_l1_error_boxplot.png'
        plt.savefig(boxplot_path)
        plt.close()

        # Compress the data into avg and std for each participant and class
        new_data = defaultdict(list)
        for name2, group2 in group.groupby('participant'):
            for name3, group3 in group2.groupby('class'):
                new_data['Avg'].append(group3['gaze'].mean())
                new_data['Std'].append(group3['gaze'].std())
                new_data['participant'].append(name2.split('_')[1])  # Get the participant ID
                new_data['class'].append(name3)
        
        metrics_df = pd.DataFrame(new_data)

        # Make a boxplot of the averages
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=metrics_df, x='class', y='Avg')
        plt.xticks(rotation=45)
        plt.ylabel('Average L1 Error (normalized)')
        plt.title('Average L1 Error for all participants')
        plt.tight_layout()

        # Put a limit of 40 for the y-axis
        ylim = min(40, metrics_df['Avg'].max())
        plt.ylim(0, ylim)

        # Save the boxplot
        boxplot_path = section_output_dir / 'participants_avg_l1_error_boxplot.png'
        plt.savefig(boxplot_path)
        plt.close()
        
        # Save the participants metrics to a XLSX file
        metrics_df.to_excel(section_output_dir / 'participants_metrics.xlsx', index=False)
        cv2.destroyAllWindows()

        # Compress a single avg and std value for the entire dataset
        data = defaultdict(list)
        for name2, group2 in metrics_df.groupby('class'):
            overall_avg = group2['Avg'].mean()
            overall_std = group2['Avg'].std()
            data[f"{name2} Overall Avg L1 Error"].append(overall_avg)
            data[f"{name2} Overall Std L1 Error"].append(overall_std)

        # Create a DataFrame from the data
        overall_metrics_df = pd.DataFrame(data)

        # Save the overall metrics to a CSV file
        overall_metrics_df.to_excel(section_output_dir / 'overall_metrics.xlsx', index=False)

    # Save a copy of the configuration file
    with open(RUN_DIR / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

if __name__ == '__main__':

    # Add arguments to specify the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to the configuration file")
    parser.add_argument("--exp", type=str, required=True, help="Experiment name for logging purposes")
    args = parser.parse_args()

    # Load the configuration file (YAML)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Validate the configuration
    if config['config']['type'] != 'eye_of_the_typer':
        raise ValueError("Only 'eye_of_the_typer' type configuration is allowed.")
    
    # Print the configuration
    print("\n")
    print("#" * 80)
    print("Configuration:")
    print(json.dumps(config, indent=4))
    print("#" * 80)
    print("\n")

    # Ask confirmation for evaluation with a 5 second loading bar
    print("Starting evaluation in 5 seconds...\n")
    for i in tqdm(range(5)):    
        time.sleep(1)

    main(args, config)