import time
import pathlib
from dataclasses import asdict
from collections import defaultdict
from typing import List, Dict, Union, Tuple, Optional, Literal, Any
import copy
import os
import json
import pickle

import imutils
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import cv2
from PIL import Image
import scipy.io
import yaml
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from webeyetrack.constants import GIT_ROOT
from webeyetrack.vis import draw_gaze_origin, draw_axis, draw_landmarks_on_image
from webeyetrack.data_protocols import Annotations, CalibrationData, Sample
from webeyetrack.model_based import vector_to_pitch_yaw

from utils import data_normalization_entry

CWD = pathlib.Path(__file__).parent

"""
Following the recommendation of: 
Park, S., Zhang, X., Bulling, A., & Hilliges, O. (2018, June 14). Learning to find eye region landmarks for remote gaze estimation in unconstrained settings. Eye Tracking Research and Applications Symposium (ETRA). https://doi.org/10.1145/3204493.3204545

For the EYEDIAP dataset, we evaluate on VGA images with static head pose for
fair comparison with similar evaluations from [Wang and Ji 2017].
Please note that we do not discard the challenging floating target sequences 
from EYEDIAP.

References:
[1] https://www.idiap.ch/en/scientific-research/data/eyediap#publications
[2] https://publications.idiap.ch/attachments/reports/2014/FunesMora_Idiap-RR-08-2014.pdf
"""

# Static head pose sessions
SESSION_INDEX = [0,2,6,8,11,12,16,18,22,24,28,30,34,36,40,42,46,48,52,54,58,60,64,66,70,72,76,78,82,84,88,90]
# SESSION_INDEX = [0,2,4,6,8,10,11,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92]

OVERWRITE_SESSIONS = True
# OVERWRITE_SESSIONS = False

TIMESTAMP = time.strftime("%H%M%S")

# ------------------------  DEFINE THE LIST OF RECORDING SESSIONS -------------------------------------
sessions = []
for P in range(16):
    if P < 11:
        Cs = ['A', ]
    elif P < 13:
        Cs = ['B', ]
    else:
        Cs = ['A', 'B']
    for C in Cs:
        if (P < 11 or P > 12) and C == 'A':
            Ts = ['DS', 'CS', 'FT']
        else:
            Ts = ['FT']
        for T in Ts:
            for H in ['S', 'M']:
                sessions.append((P+1, C, T, H))

def get_session_index(P, C, T, H):
    try:
        idx = sessions.index((P, C, T, H))
        return idx
    except ValueError:
        return None

def get_session_string(session_idx):
    """
    Gets the string identifier for session "session_idx". It's useful for browsing the data
    """
    if session_idx >= 0 and session_idx < len(sessions):
        P, C, T, H = sessions[session_idx]
        return str(P)+'_'+C+'_'+T+'_'+H
    else:
        return ''

# ------------ DEFINE A FEW HELPER FUNCTION TO DRAW THE DIFFERENT ELEMENTS --------------
def draw_point(frame, point, size= 5, thickness=2):
    """
    Draw a point in the given image as a yellow "+"
    """
    point = int(point[0]), int(point[1])
    cv2.line(frame, (point[0]-size, point[1]), (point[0]+size, point[1]), (0, 255, 255), thickness =thickness)
    cv2.line(frame, (point[0]  , point[1]-size), (point[0], point[1]+size), (0, 255, 255), thickness =thickness)

def draw_line(frame, point_1, point_2, size=5, color=(0,255,255), thickness =2):
    cv2.line(frame, (point_1[0], point_1[1]), (point_2[0], point_2[1]), color, thickness =thickness)

def draw_ball(frames, vals, thickness =2):
    """
    Interpret the floating target parameters to draw a point in the corresponding frame
    """
    for i in range(3):
        center = vals[i*2:i*2+2]
        draw_point(frames[i], center, thickness=thickness)

def draw_eyes(frames, vals, thickness =2):
    """
    Interpret the eye tracking parameters to draw a point at each eye in the corresponding frames
    """
    for i in range(3):
        center_left = vals[4*i:4*i+2]
        center_right = vals[4*i+2:4*i+4]
        draw_point(frames[i], center_left, thickness=thickness)
        draw_point(frames[i], center_right, thickness=thickness)

def draw_screen(frame, vals):
    """
    Draws a point with the current screen coordinates for reference
    """
    pos_x, pos_y = int(vals[0]), int(vals[1])
    cv2.circle(frame, (pos_x, pos_y), radius=15, color=(0,0,255), thickness=-1)

def project_points(points, calibration):
    """
    Projects a set of points into the cameras coordinate system
    """
    R = calibration['R']
    T = calibration['T']
    intr = calibration['intrinsics']
    # W.r.t. to the camera coordinate system
    points_c = np.dot(points, R) - np.dot(R.transpose(), T).reshape(1, 3)
    points_2D = np.dot(points_c, intr.transpose())
    points_2D = points_2D[:,:2]/(points_2D[:, 2].reshape(-1,1))
    return points_2D

def draw_head_pose(frames, vals, calibrations):
    """
    Draw a coordinate system describing the current head pose
    """
    size = 0.05
    points = [[0.0, 0.0, 0.0],
              [size, 0.0, 0.0],
              [0.0, size, 0.0],
              [0.0, 0.0, size]]
    points = np.array(points)
    points+= np.array([0.0, 0.0, 0.13]).reshape(1, 3)
    R = np.array(vals[:9]).reshape(3, 3)
    T = np.array(vals[9:12]).reshape(3, 1)
    # Gets the points positions in 3D
    points = np.dot(points, R.transpose())+T.reshape(1, 3)
    for frame, calibration in zip(frames, calibrations):
        points_2D = np.int32(project_points(points, calibration))
        draw_line(frame, points_2D[0, :], points_2D[1, :], color=(255,0,0))
        draw_line(frame, points_2D[0, :], points_2D[2, :], color=(0,255,0))
        draw_line(frame, points_2D[0, :], points_2D[3, :], color=(0,0,255))
    return points_2D[0,:]

def read_calibration_file(calibration_file):
    """
    Reads the calibration parameters
    """
    cal  = {}
    fh = open(calibration_file, 'r')
    # Read the [resolution] section
    fh.readline().strip()
    cal['size'] = [int(val) for val in fh.readline().strip().split(';')]
    cal['size'] = cal['size'][0], cal['size'][1]
    # Read the [intrinsics] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(';')])
    cal['intrinsics'] = np.array(vals).reshape(3,3)
    # Read the [R] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(';')])
    cal['R'] = np.array(vals).reshape(3,3)
    # Read the [T] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(';')])
    cal['T'] = np.array(vals).reshape(3,1)
    fh.close()
    return cal

def extract_next_file_values(file_obj, delimiter=';', typecast=float):
    """
    From a string row in a file creates a list of float values together with the frame index
    """
    line = file_obj.readline().strip()
    if len(line)>0:
        # vals = [float(el) for el in line.split(delimiter)]
        vals = [typecast(el) for el in line.split(delimiter)]
        return int(vals[0]), vals[1:]
    else:
        return None, None

class EyeDiapDataset():

    def __init__(
            self, 
            dataset_dir: Union[pathlib.Path, str], 
            participants: List[int],
            face_size: Tuple[int, int] = None,
            img_size: Tuple[int, int] = None,
            dataset_size: Optional[int] = None,
            per_participant_size: Optional[int] = None,
            frame_skip_rate: int = 30,
            video_type: Literal['vga', 'hd'] = 'hd',
            visualize: bool = False
        ):

        # Process input variables
        if isinstance(dataset_dir, str):
            dataset_dir = pathlib.Path(dataset_dir)
        assert dataset_dir.is_dir(), f"Dataset directory {dataset_dir} does not exist."
        self.dataset_dir = dataset_dir
        
        self.img_size = img_size
        self.face_size = face_size
        self.dataset_size = dataset_size
        self.per_participant_size = per_participant_size
        self.participants = participants
        self.frame_skip_rate = frame_skip_rate
        self.video_type = video_type
        self.visualize = visualize

        # if not self.participants:
        #     raise ValueError("No participants were selected.")

        # if self.video_type == 'hd':
        #     raise RuntimeError("HD video is not supported yet. TODO: Gaze direction does not appear to be correct in HD video.")

        # Setup MediaPipe Face Facial Landmark model
        base_options = python.BaseOptions(model_asset_path=str(GIT_ROOT / 'python' / 'weights' / 'face_landmarker_v2_with_blendshapes.task'))
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        # Saving information
        # self.samples: List[Sample] = []
        self.samples = defaultdict(list)

        num_samples = 0

        # Tracking the number of loaded samples
        for session_id in tqdm(SESSION_INDEX):

            if self.dataset_size is not None and num_samples >= self.dataset_size:
                break

            P, C, T, H = sessions[session_id]

            # Do not process the session if it is not in the list of participants
            if P not in self.participants:
                continue

            # Skip T='FT', as it does not contain PoG 
            if T == 'FT':
                continue    

            session_str = get_session_string(session_id)
            session_fp = self.dataset_dir / f'EYEDIAP{P}' / 'EYEDIAP' / 'Data' / session_str

            # ------------------------- FILE PATHS DEFINITION -------------------------------------
            rgb_vga_file    = str(session_fp / 'rgb_vga.mov')
            depth_file      = str(session_fp / 'depth.mov')
            rgb_hd_file     = str(session_fp / 'rgb_hd.mov')
            head_track_file = str(session_fp / 'head_pose.txt')
            eyes_track_file = str(session_fp / 'eye_tracking.txt')
            ball_track_file = str(session_fp / 'ball_tracking.txt')
            screen_track_file = str(session_fp / 'screen_coordinates.txt')
            rgb_vga_calibration_file = str(session_fp / 'rgb_vga_calibration.txt')
            rgb_hd_calibration_file = str(session_fp / 'rgb_hd_calibration.txt')
            depth_calibration_file = str(session_fp / 'depth_calibration.txt')
            gaze_state = str(self.dataset_dir / 'EYEDIAP_GazeStateAnnotations' / 'EYEDIAP' / 'Annotations' / 'GazeState' / 'GazeStateExport' / 'Data' / session_str / 'gaze_state.txt')

            gaze_state_track = None
            ball_track = None
            screen_track = None
            eyes_track = open(eyes_track_file, 'r')
            header = eyes_track.readline()
            head_track = open(head_track_file)
            header = head_track.readline()
            if T == 'FT':
                ball_track = open(ball_track_file, 'r')
                header = ball_track.readline()
            else:
                screen_track = open(screen_track_file, 'r')
                header = screen_track.readline()
            if T != 'DS':
                gaze_state_track = open(gaze_state, 'r')

            # Read the calibration parameter files
            rgb_vga_calibration = read_calibration_file(rgb_vga_calibration_file)
            depth_calibration   = read_calibration_file(depth_calibration_file)
            rgb_hd_calibration  = read_calibration_file(rgb_hd_calibration_file)

            # Makes a convenient tuple to use later
            calibrations = rgb_vga_calibration, depth_calibration, rgb_hd_calibration

            # ----------------- Create Video Readers ------------------------------------------------
            rgb_vga = cv2.VideoCapture(rgb_vga_file)
            depth   = cv2.VideoCapture(depth_file)
            if os.path.exists(rgb_hd_file):
                rgb_hd  = cv2.VideoCapture(rgb_hd_file)
            else:
                rgb_hd = None

            if rgb_hd is None and self.video_type == 'hd':
                continue

            # Determine the total number of frames
            total_frames = int(rgb_vga.get(cv2.CAP_PROP_FRAME_COUNT))

            # Using the frame skip rate and the gaze state information, determine the list of frames needed
            if self.frame_skip_rate == 0:
                requested_frames = list(range(0, total_frames))
            else:
                requested_frames = list(range(0, total_frames, self.frame_skip_rate))
            if gaze_state_track is not None:
                lines = gaze_state_track.readlines()
                safe_requested_frames = []
                for i in requested_frames:
                    gaze_state = lines[i].strip().split('\t')[1]
                    if gaze_state == 'OK':
                        safe_requested_frames.append(i)
                requested_frames = safe_requested_frames
                gaze_state_track.seek(0)

            # Create progress bar
            pbar = tqdm(total=len(requested_frames))
            per_participant_size = 0

            # Construct the calibration data for the session and video type
            if self.video_type == 'vga':
                calibration_data = CalibrationData(
                    camera_matrix=rgb_vga_calibration['intrinsics'],
                    dist_coeffs=None,
                    camera_retval=None,
                    screen_RT=None,
                    monitor_height_cm=29.5,
                    monitor_height_px=740,
                    monitor_width_cm=53.4,
                    monitor_width_px=1340
                )
            elif self.video_type == 'hd':
                calibration_data = CalibrationData(
                    camera_matrix=rgb_hd_calibration['intrinsics'],
                    dist_coeffs=None,
                    camera_retval=None,
                    screen_RT=None,
                    monitor_height_cm=29.5,
                    monitor_height_px=740,
                    monitor_width_cm=53.4,
                    monitor_width_px=1340
                )
            else:
                raise ValueError("Invalid video type. Must be 'vga' or 'hd'.")

            if rgb_vga.isOpened() and depth.isOpened() and (rgb_hd is None or rgb_hd.isOpened()):
                all_ok = True
                key = 0
                frameIndex = 0
                fps = 30
                ball_vals, screen_vals = None, None
                screen_img = None
                while all_ok and key != ord('q'):

                    # If the dataset size is set, break if the number of samples exceeds the dataset size
                    if self.dataset_size is not None and num_samples >= self.dataset_size:
                        break

                    if self.per_participant_size is not None and per_participant_size >= self.per_participant_size:
                        break

                    if self.frame_skip_rate > 1:
                        if frameIndex not in requested_frames:
                            frameIndex += 1
                            if frameIndex >= total_frames:
                                break
                            continue
                        else:
                            # Update the video captures and the track files
                            rgb_vga.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
                            depth.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
                            if rgb_hd is not None:
                                rgb_hd.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
                            if frameIndex > 0:
                                if ball_track is not None:
                                    while True:
                                        ball_frame_index, ball_vals = extract_next_file_values(ball_track)
                                        if ball_frame_index is None:
                                            break
                                        if ball_frame_index >= frameIndex - 1:
                                            break
                                if screen_track is not None:
                                    while True:
                                        screen_frame_index, screen_vals = extract_next_file_values(screen_track)
                                        if screen_frame_index is None:
                                            break
                                        if screen_frame_index >= frameIndex - 1:
                                            break
                                if gaze_state_track is not None:
                                    while True:
                                        gaze_frame_index, gaze_vals = extract_next_file_values(gaze_state_track, delimiter='\t', typecast=str)
                                        if gaze_frame_index is None:
                                            break
                                        if gaze_frame_index >= frameIndex - 1:
                                            break
                                if eyes_track is not None:
                                    while True:
                                        eyes_frame_index, eyes_vals = extract_next_file_values(eyes_track)
                                        if eyes_frame_index is None:
                                            break
                                        if eyes_frame_index >= frameIndex - 1:
                                            break
                                if head_track is not None:
                                    while True:
                                        head_frame_index, head_vals = extract_next_file_values(head_track)
                                        if head_frame_index is None:
                                            break
                                        if head_frame_index >= frameIndex - 1:
                                            break

                    # If the sample already exists, load it
                    # import pdb; pdb.set_trace()
                    annotation_fp = session_fp / f'samples_{self.video_type}' / f'{frameIndex}.pkl'
                    frame_fp = session_fp / 'frames' / f'{frameIndex}_{self.video_type}.png'
                    if annotation_fp.exists() and not OVERWRITE_SESSIONS:
                        self.samples['id'].append(f"{P}_{frameIndex}_{self.video_type}_{T}")
                        self.samples['participant_id'].append(P)
                        self.samples['image_fp'].append(frame_fp)
                        self.samples['annotation_fp'].append(annotation_fp)

                        pbar.update(1)
                        frameIndex += 1
                        num_samples += 1
                        per_participant_size += 1
                        rgb_vga.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
                        depth.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
                        if rgb_hd is not None:
                            rgb_hd.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
                        if ball_track is not None:
                            ball_track.readline()
                        if screen_track is not None:
                            screen_track.readline()
                        if gaze_state_track is not None:
                            gaze_state_track.readline()
                        continue

                    # Load the data
                    tic = time.time()
                    ok_rgb, frame_rgb     = rgb_vga.read()
                    ok_depth, frame_depth = depth.read()
                    if rgb_hd  is not None:
                        ok_hd, frame_hd       = rgb_hd.read()
                    else:
                        ok_hd, frame_hd = True, None
                    target_vals = None
                    gaze_vals = None
                    ball_frame_index = None
                    screen_frame_index = None
                    gaze_frame_index = None
                    if ball_track is not None:
                        ball_frame_index, ball_vals = extract_next_file_values(ball_track)
                        target_vals = ball_vals
                    if screen_track is not None:
                        screen_frame_index, screen_vals = extract_next_file_values(screen_track)
                        target_vals = screen_vals
                    if gaze_state_track is not None:
                        gaze_frame_index, gaze_vals = extract_next_file_values(gaze_state_track, delimiter='\t', typecast=str)
                    eyes_frame_index, eyes_vals = extract_next_file_values(eyes_track)
                    head_frame_index, head_vals = extract_next_file_values(head_track)
                    toc = time.time()

                    if gaze_vals:
                        if gaze_vals[0] == 'NL':
                            continue

                    # Get the current video frame index
                    if self.video_type == 'vga':
                        backup_frameIndex = int(rgb_vga.get(cv2.CAP_PROP_POS_FRAMES))
                    elif self.video_type == 'hd':
                        backup_frameIndex = int(rgb_hd.get(cv2.CAP_PROP_POS_FRAMES))
                    else:
                        raise ValueError("Invalid video type. Must be 'vga' or 'hd'.")

                    all_ok = ok_rgb and ok_depth and ok_hd and eyes_vals is not None and head_vals is not None and target_vals is not None
                    # print(ball_frame_index, screen_frame_index, gaze_frame_index, eyes_frame_index, head_frame_index, frameIndex, backup_frameIndex, all_ok)

                    if all_ok:

                        # Select the frame to use
                        desired_frame = None
                        desired_calibration = None
                        if self.video_type == 'vga':
                            desired_frame = frame_rgb
                            desired_calibration = rgb_vga_calibration
                        elif self.video_type == 'hd':
                            desired_frame = frame_hd
                            desired_calibration = rgb_hd_calibration
                        else:
                            raise ValueError("Invalid video type. Must be 'vga' or 'hd'.")

                        # Obtain facial landmarks
                        detection_results, face_landmarks_proto = self.obtain_facial_landmarks(desired_frame)
                        if detection_results is None:
                            # print("No face detected.")
                            frameIndex += 1
                            continue

                        # Draw the landmarks and see
                        # Draw the landmarks on the image
                        # image_landmarks = draw_landmarks_on_image(desired_frame, detection_results)
                        # cv2.imshow('image', image_landmarks)
                        # cv2.waitKey(0)

                        i_h, i_w, _ = desired_frame.shape

                        # Face landmakrs
                        face_landmarks_all = np.array([[lm.x, lm.y, lm.z, lm.visibility, lm.presence] for lm in face_landmarks_proto])
                        face_landmarks_rt = detection_results.facial_transformation_matrixes[0]
                        face_blendshapes = detection_results.face_blendshapes[0]
                        face_blendshapes = np.array([x.score for x in face_blendshapes])
                        face_landmarks = np.array([[lm.x * i_w, lm.y * i_h] for lm in face_landmarks_proto])

                        # Compute the bounding box
                        face_bbox = np.array([
                            int(np.min(face_landmarks[:, 1])), 
                            int(np.min(face_landmarks[:, 0])), 
                            int(np.max(face_landmarks[:, 1])), 
                            int(np.max(face_landmarks[:, 0]))
                        ])

                        # Determine the eye information
                        left_eye_origin_3d = np.array(eyes_vals[-6:-3])
                        right_eye_origin_3d = np.array(eyes_vals[-3:])

                        if self.video_type == 'vga':
                            left_eye_origin_2d = np.array(eyes_vals[:2])
                            right_eye_origin_2d = np.array(eyes_vals[2:4])
                        elif self.video_type == 'hd':
                            left_eye_origin_2d = np.array(eyes_vals[8:10])
                            right_eye_origin_2d = np.array(eyes_vals[10:12])

                        # Obtain the target information
                        if ball_track:
                            gaze_target_2d = np.zeros((2))
                            gaze_target_3d = np.array(ball_vals[-3:])
                        elif screen_track:
                            gaze_target_2d = np.array(screen_vals[:2])
                            gaze_target_3d = np.array(screen_vals[2:])
                        
                        # If HD camera, shift all 3D points to the HD coordinate system
                        if self.video_type == 'hd':
                            hd_r, hd_t = rgb_hd_calibration['R'], rgb_hd_calibration['T']
                            # hd_r = np.eye(3)
                            hd_r = np.linalg.inv(hd_r)

                            # Convert rotation to pitch yaw
                            r = Rotation.from_matrix(hd_r)
                            pitch, yaw, roll = r.as_euler('xyz', degrees=True)
                            pitch -= 180
                            # pitch = 180 - pitch

                            # Convert back to rotation matrix
                            r = Rotation.from_euler('xyz', [pitch, yaw, roll], degrees=True)
                            hd_r = r.as_matrix()

                            rt = np.hstack((hd_r, hd_t))
                            rt = np.vstack((rt, np.array([0, 0, 0, 1])))
                            # rt = np.linalg.inv(rt)

                            # Apply the transformation to the 3D points
                            left_eye_origin_3d = np.dot(rt, np.hstack((left_eye_origin_3d, 1)))
                            right_eye_origin_3d = np.dot(rt, np.hstack((right_eye_origin_3d, 1)))
                            gaze_target_3d = np.dot(rt, np.hstack((gaze_target_3d, 1)))

                            # Convert homogeneous coordinates to 3D
                            left_eye_origin_3d = left_eye_origin_3d[:3]
                            right_eye_origin_3d = right_eye_origin_3d[:3]
                            gaze_target_3d = gaze_target_3d[:3]

                        # Compute the gaze angle from the eye origin to the target for both eyes
                        left_gaze_vector = gaze_target_3d - left_eye_origin_3d
                        left_gaze_vector = left_gaze_vector / np.linalg.norm(left_gaze_vector)
                        right_gaze_vector = gaze_target_3d - right_eye_origin_3d
                        right_gaze_vector = right_gaze_vector / np.linalg.norm(right_gaze_vector)

                        # Compute the average gaze vector for the face
                        face_gaze_vector = (left_gaze_vector + right_gaze_vector) / 2
                        face_gaze_vector = face_gaze_vector / np.linalg.norm(face_gaze_vector)

                        # Compute the average gaze origin for the face
                        face_origin_3d = (left_eye_origin_3d + right_eye_origin_3d) / 2
                        face_origin_2d = (left_eye_origin_2d + right_eye_origin_2d) / 2

                        # if gaze_state is 'BK', eyes are closed
                        is_closed = False
                        if gaze_vals:
                            is_closed = gaze_vals[0] == 'BK'

                        # Convert the PoG by using the gaze target 3d
                        pog_px = gaze_target_2d
                        pog_norm = np.array([
                            gaze_target_2d[0] / calibration_data.monitor_width_px,
                            gaze_target_2d[1] / calibration_data.monitor_height_px
                        ])
                        pog_cm = np.array([
                            pog_norm[0] * calibration_data.monitor_width_cm,
                            pog_norm[1] * calibration_data.monitor_height_cm
                        ])

                        annotation = Annotations(
                            original_img_size=np.array(desired_frame.shape),
                            intrinsics=desired_calibration['intrinsics'],
                            # Facial landmarks
                            facial_detection_results=detection_results,
                            facial_landmarks=face_landmarks_all,
                            facial_landmarks_2d=face_landmarks,
                            facial_rt=face_landmarks_rt,
                            face_blendshapes=face_blendshapes,
                            face_bbox=np.array([]), # Not important
                            head_pose_3d=np.array([]),
                            # Face Gaze
                            face_origin_3d=face_origin_3d,
                            face_origin_2d=face_origin_2d,
                            face_gaze_vector=face_gaze_vector,
                            # Eye Gaze
                            left_eye_origin_3d=np.empty((3, 0), dtype=np.float32),
                            left_eye_origin_2d=np.empty((2, 0), dtype=np.float32),
                            left_gaze_vector=np.empty((3, 0), dtype=np.float32),
                            right_eye_origin_3d=np.empty((3, 0), dtype=np.float32),
                            right_eye_origin_2d=np.empty((2, 0), dtype=np.float32),
                            right_gaze_vector=np.empty((3, 0), dtype=np.float32),
                            # Target information
                            gaze_target_3d=gaze_target_3d,
                            gaze_target_2d=np.zeros((2, 0), dtype=np.float32),
                            pog_px=pog_px,
                            pog_norm=pog_norm,
                            pog_cm=pog_cm,
                            # Gaze State Information
                            is_closed=np.array([False])
                        )

                        # Write frame into a folder
                        os.makedirs(session_fp / 'frames', exist_ok=True)
                        if not frame_fp.exists():
                            cv2.imwrite(str(frame_fp), desired_frame)

                        # Save the annotations
                        with open(annotation_fp, 'wb') as f:
                            pickle.dump(annotation, f)

                        # Create a sample
                        # sample = Sample(
                        #     participant_id=P,
                        #     image_fp=frame_fp,
                        #     annotations=annotation
                        # )
                        # self.samples.append(sample)
                        # Create a sample
                        self.samples['id'].append(f"{P}_{frameIndex}_{self.video_type}_{T}")
                        self.samples['participant_id'].append(P)
                        self.samples['image_fp'].append(frame_fp)
                        self.samples['annotation_fp'].append(annotation_fp)
                        num_samples += 1
                        per_participant_size += 1
                        pbar.update(1)

                        # Save the sample within the ``samples`` directory to avoid having to reprocess the data, as a pickle
                        # os.makedirs(session_fp / f'samples_{self.video_type}', exist_ok=True)
                        # with open(session_fp / f'samples_{self.video_type}' / f'{frameIndex}.pkl', 'wb') as f:
                        #     pickle.dump(sample, f)

                        if self.visualize:
                            # Draw the landmarks
                            # desired_frame = draw_landmarks_on_image(desired_frame, detection_results)
                            if self.video_type == 'vga':
                                frame_rgb = desired_frame
                            elif self.video_type == 'hd':
                                frame_hd = desired_frame

                            # Visualize sample
                            frames = (frame_rgb, frame_depth, frame_hd)
                            if ball_vals is not None:
                                draw_ball(frames, ball_vals)
                            if screen_vals is not None:
                                screen_img = np.zeros((1000, 1680, 3), dtype=np.uint8)
                                draw_screen(screen_img, screen_vals)

                            for origin, vector in zip([left_eye_origin_2d, right_eye_origin_2d], [left_gaze_vector, right_gaze_vector]):
                                pitch, yaw = vector_to_pitch_yaw(vector)

                                if self.video_type == 'vga':
                                    frame_rgb = draw_axis(frame_rgb, pitch, yaw, 0, int(origin[0]), int(origin[1]), 100)
                                elif self.video_type == 'hd':
                                    frame_hd = draw_axis(frame_hd, pitch, yaw, 0, int(origin[0]), int(origin[1]), 100)
                            
                            frames = (frame_rgb, frame_depth, frame_hd)

                            # Write the face_gaze_vector onto the hd image
                            if self.video_type == 'hd':
                                cv2.putText(frame_hd, f"Face Gaze Vector: {face_gaze_vector}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 2, cv2.LINE_AA)

                            # Draw the gaze
                            # draw_eyes(frames, eyes_vals)
                            headHD_2D= draw_head_pose(frames, head_vals, calibrations)
                            if screen_img is not None:
                                cv2.imshow('screen', screen_img)
                            if frame_hd is not None:
                                cv2.imshow('rgb_hd' , frame_hd)
                            cv2.imshow('depth'  , frame_depth)
                            cv2.imshow('rgb_vga', frame_rgb)

                            # cv2.imwrite(str(CWD / 'outputs' / f'{TIMESTAMP}_frame_rgb_{frameIndex}.png'), frame_rgb)
                            # if frame_hd is not None:
                            #     cv2.imwrite(str(CWD / 'outputs' / f'{TIMESTAMP}_frame_hd_{frameIndex}.png'), imutils.resize(frame_hd, width=800))

                            if head_frame_index == 0:
                                key = cv2.waitKey(1000)
                            else:
                                key = cv2.waitKey(int(1000/fps))
                            # key = cv2.waitKey(0)
                            # if key != -1:
                            #     key = key & 255
                            #     if key == 113:
                            #         break
                            #     if key == 82:
                            #         fps += 5
                            #         if fps > 30*5:
                            #             fps = 30*5
                            #     if key == 84:
                            #         fps -= 5
                            #         if fps < 1:
                            #             fps = 1

                    frameIndex += 1

            pbar.close() 

            rgb_vga.release()
            depth.release()
            if rgb_hd is not None:
                rgb_hd.release()
            if ball_track is not None:
                ball_track.close()
            if screen_track is not None:
                screen_track.close()

            cv2.destroyAllWindows()

        # Convert the samples to a DataFrame
        self.samples = pd.DataFrame(self.samples)

    def obtain_facial_landmarks(self, frame):

        # Detect the facial landmarks via MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.astype(np.uint8))
        detection_results = self.face_landmarker.detect(mp_image)
        
        # Compute the face bounding box based on the MediaPipe landmarks
        try:
            face_landmarks_proto = detection_results.face_landmarks[0]
        except:
            return None, None

        return detection_results, face_landmarks_proto

    def __len__(self):
        return len(self.samples)

    def get_sample(self, index: int) -> Dict[str, Any]:
        sample = self.samples.iloc[index]

        # Load image
        image = Image.open(sample.image_fp)
        image_np = np.array(image)

        # Get the calibration
        # calibration_data = self.participant_calibration_data[sample.participant_id]

        # Load the annotations
        with open(sample.annotation_fp, 'rb') as f:
            annotations = pickle.load(f)

        item_dict = {
            'person_id': sample.participant_id,
            'image': image_np,
            # 'intrinsics': calibration_data.camera_matrix,
            # 'dist_coeffs': calibration_data.dist_coeffs,
            # 'screen_RT': calibration_data.screen_RT.astype(np.float32),
            'screen_height_cm': 29.5,
            'screen_height_px': 740,
            'screen_width_cm': 53.4,
            'screen_width_px': 1340,
        }
        item_dict.update(asdict(annotations))
        return item_dict
        
    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        return self.get_sample(index)

    def to_df(self):

        data = defaultdict(list)
        for sample in self.samples:
            data['image_fp'].append(sample.image_fp)
            data['participant_id'].append(sample.participant_id)
            for k, v in asdict(sample.annotations).items():
                data[k].append(v)

        return pd.DataFrame(data)
        

if __name__ == '__main__':

    from webeyetrack.constants import DEFAULT_CONFIG
    with open(DEFAULT_CONFIG, 'r') as f:
        config = yaml.safe_load(f)

    dataset = EyeDiapDataset(
        GIT_ROOT / pathlib.Path(config['datasets']['EyeDiap']['path']),
        dataset_size=10,
        participants=[1],
        # per_participant_size=2,
        video_type='hd',
        visualize=False,
        frame_skip_rate=4
    )
    print(len(dataset))
    
    sample = dataset[0]
    # print(json.dumps({k: str(v.dtype) for k, v in sample.items()}, indent=4))
    print(dataset.samples.head())
    print(sample.keys())

    for sample in dataset:
        sample['image'] = cv2.cvtColor(sample['image'], cv2.COLOR_BGR2RGB)
        data_normalization_entry(0, sample)