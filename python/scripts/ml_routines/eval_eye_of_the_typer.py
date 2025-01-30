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
from webeyetrack.datasets.utils import draw_landmarks_on_image
import webeyetrack.vis as vis
from webeyetrack.model_based import vector_to_pitch_yaw, compute_pog
from webeyetrack.data_protocols import GazeResult
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
    load_eyeball_model
)

CWD = pathlib.Path(__file__).parent
FILE_DIR = pathlib.Path(__file__).parent
OUTPUTS_DIR = CWD / 'outputs'
SKIP_COUNT = 100

with open(FILE_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

EYE_OF_THE_TYPER_DATASET = pathlib.Path(config['datasets']['EyeOfTheTyper']['path'])
assert EYE_OF_THE_TYPER_DATASET.exists(), f"Dataset not found at {EYE_OF_THE_TYPER_DATASET}"

SECTIONS = [
    'study-dot_test.webm_gazePredictionsDone',
    'study-benefits_of_running_writing.webm_gazePredictionsDone',
    'study-educational_advantages_of_social_networking_sites_writing.webm_gazePredictionsDone',
    'study-where_to_find_morel_mushrooms_writing.webm_gazePredictionsDone',
    'study-tooth_abscess_writing.webm_gazePredictionsDone',
    'study-dot_test_final.webm_gazePredictionsDone'
]

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

def main():

    # Create pipeline
    algo = WebEyeTrack(
        str(GIT_ROOT / 'python' / 'weights' / 'face_landmarker_v2_with_blendshapes.task'),
    )
    K = estimate_camera_intrinsics(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3)))

    # Initialize Open3D Visualizer
    visual = o3d.visualization.Visualizer()
    visual.create_window(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    visual.get_render_option().background_color = [0.1, 0.1, 0.1]
    visual.get_render_option().mesh_show_back_face = True
    visual.get_render_option().point_size = 10

    # Change the z far to 1000
    vis = visual.get_view_control()
    vis.set_constant_z_far(1000)
    params = o3d.camera.PinholeCameraParameters()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(intrinsic_matrix=K, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    params.extrinsic = np.eye(4)
    params.intrinsic = intrinsic
    vis.convert_from_pinhole_camera_parameters(parameter=params)

    face_mesh, face_mesh_lines = load_canonical_mesh(visual)
    face_coordinate_axes = load_3d_axis(visual)
    eyeball_meshes, _, eyeball_R = load_eyeball_model(visual)
    
    # Iterate over the folders within the dataset
    p_dirs = [p for p in EYE_OF_THE_TYPER_DATASET.iterdir() if p.is_dir()]
    gaze_csvs = [p for p in EYE_OF_THE_TYPER_DATASET.iterdir() if p.is_file() and p.suffix == '.csv']

    # Sort the gaze into separate containers for each participant
    # Name convention is: P_01_1491423217564_3_-study-dot_test.webm_gazePredictionsDone.csv
    gaze_by_participant = defaultdict(list)
    for gaze_csv in gaze_csvs:
        participant = '_'.join(gaze_csv.stem.split('_')[:2])
        gaze_by_participant[participant].append(gaze_csv)

    # For each CSV, read the data and display the gaze
    for par, csvs in gaze_by_participant.items():

        # Config for the participant's screen and other attributes

        for csv in csvs:
            # data = pd.read_csv(csv, names=fieldnames)
            data = pd.read_csv(csv)

            # Drop the columns after 19th column
            data = data.iloc[:, :19]

            # Add the columns to the data
            data.columns = fieldnames

            # Iterate over the rows of the CSV and display the data
            for i, row in data.iterrows():

                # Load the image
                img_path = EYE_OF_THE_TYPER_DATASET / par / "/".join(row['frameImageFile'].split('/')[3:])
                assert img_path.exists(), f"Image not found at {img_path}"
                img = cv2.imread(str(img_path))

                # Update the last configs
                height, width = img.shape[:2]
                algo.config(
                    frame_height=height,
                    frame_width=width,
                    intrinsics=K
                )

                # Process the sample
                result, detection_results = algo.process_frame(img)
                draw_frame = img.copy()
                draw_frame = draw_landmarks_on_image(draw_frame, detection_results)

                # Compute the face mesh
                face_mesh.vertices = o3d.utility.Vector3dVector(transform_for_3d_scene(result.metric_face))
                new_face_mesh_lines = o3d.geometry.LineSet.create_from_triangle_mesh(face_mesh)
                face_mesh_lines.points = new_face_mesh_lines.points
                visual.update_geometry(face_mesh)
                visual.update_geometry(face_mesh_lines)

                # Draw the canonical face axes by using the final_transform
                canonical_face_axes = np.array([
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ]) * 5
                camera_pts_3d = transform_3d_to_3d(canonical_face_axes, result.metric_transform)
                canonical_face_axes_2d = transform_3d_to_2d(camera_pts_3d, K)
                cv2.line(draw_frame, tuple(canonical_face_axes_2d[0]), tuple(canonical_face_axes_2d[1]), (0, 0, 255), 2)
                cv2.line(draw_frame, tuple(canonical_face_axes_2d[0]), tuple(canonical_face_axes_2d[2]), (0, 255, 0), 2)
                cv2.line(draw_frame, tuple(canonical_face_axes_2d[0]), tuple(canonical_face_axes_2d[3]), (255, 0, 0), 2)

                # Update the 3d axes in the visualizer as well
                face_coordinate_axes.points = o3d.utility.Vector3dVector(transform_for_3d_scene(camera_pts_3d))
                visual.update_geometry(face_coordinate_axes)

                # Compute the 3D eye origin
                for i in ['left', 'right']:
                    eye_result = result.left if i == 'left' else result.right
                    origin = eye_result.origin
                    direction = eye_result.direction

                    # final_position = transform_for_3d_scene(eye_g_o[k].reshape((-1,3))).flatten()
                    final_position = transform_for_3d_scene(origin.reshape((-1,3))).flatten()
                    eyeball_meshes[i].translate(final_position, relative=False)

                    # Rotation
                    current_eye_R = eyeball_R[i]
                    eye_R = get_rotation_matrix_from_vector(direction)
                    pitch, yaw, roll = rotation_matrix_to_euler_angles(eye_R)
                    pitch, yaw, roll = yaw, pitch, roll # Flip the pitch and yaw
                    eye_R = euler_angles_to_rotation_matrix(pitch, yaw, 0)

                    # Apply the scene transformation to the new eye rotation
                    eye_R = np.dot(eye_R, OPEN3D_RT[:3, :3])

                    # Compute the rotation matrix to rotate the current to the target
                    new_eye_R = np.dot(eye_R, current_eye_R.T)
                    eyeball_R[i] = eye_R
                    eyeball_meshes[i].rotate(new_eye_R)
                    visual.update_geometry(eyeball_meshes[i])

                # Update visualizer
                visual.poll_events()
                visual.update_renderer()

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

                # Obtain the webgazer gaze point
                webgazer_gaze = (row['webGazerX'], row['webGazerY'])

                # Construct the screen
                screen_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

                # Display the gaze point
                if gaze is not None:
                    screen_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
                    cv2.circle(screen_img, (int(gaze[0]*1920), int(gaze[1]*1080)), 5, (0, 255, 0), -1)
                cv2.circle(screen_img, (int(webgazer_gaze[0]*1920), int(webgazer_gaze[1]*1080)), 5, (255, 0, 0), -1)

                cv2.imshow('screen', screen_img)
                cv2.imshow('image', draw_frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            break
        break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()