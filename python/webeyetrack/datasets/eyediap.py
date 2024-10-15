import pathlib
from dataclasses import asdict
from typing import List, Dict, Union, Tuple, Optional
import copy
import os
import json

from scipy.spatial.transform import Rotation
from tqdm import tqdm
import cv2
from PIL import Image
import scipy.io
import yaml
import numpy as np
from torch.utils.data import Dataset
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ..constants import GIT_ROOT
from ..vis import draw_gaze_origin
from ..data_protocols import Annotations, CalibrationData, Sample
from .utils import resize_annotations, resize_intrinsics, draw_landmarks_on_image, compute_uv_texture

CWD = pathlib.Path(__file__).parent

"""
Following the recommendation of: 
Park, S., Zhang, X., Bulling, A., & Hilliges, O. (2018, June 14). Learning to find eye region landmarks for remote gaze estimation in unconstrained settings. Eye Tracking Research and Applications Symposium (ETRA). https://doi.org/10.1145/3204493.3204545

For the EYEDIAP dataset, we evaluate on VGA images with static head pose for
fair comparison with similar evaluations from [Wang and Ji 2017].
Please note that we do not discard the challenging floating target sequences 
from EYEDIAP.
"""

# Static head pose sessions
SESSION_INDEX = [0,2,4,6,8,10,11,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92]

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

def extract_next_file_values(file_obj):
    """
    From a string row in a file creates a list of float values together with the frame index
    """
    line = file_obj.readline().strip()
    if len(line)>0:
        vals = [float(el) for el in line.split(';')]
        return int(vals[0]), vals[1:]
    else:
        return None, None

class EyeDiapDataset(Dataset):

    def __init__(
            self, 
            dataset_dir: Union[pathlib.Path, str], 
            participants: List[int],
            face_size: Tuple[int, int] = None,
            img_size: Tuple[int, int] = None,
            dataset_size: Optional[int] = None,
            per_participant_size: Optional[int] = None
        ):

        # Process input variables
        if isinstance(dataset_dir, str):
            dataset_dir = pathlib.Path(dataset_dir)
        assert dataset_dir.is_dir(), f"Dataset directory {dataset_dir} does not exist."
        self.dataset_dir = dataset_dir
        
        # Only dataset size or per participant size can be set
        assert (dataset_size is None) or (per_participant_size is None), "Only one of dataset_size or per_participant_size can be set."

        self.img_size = img_size
        self.face_size = face_size
        self.dataset_size = dataset_size
        self.per_participant_size = per_participant_size
        self.participants = participants

        if not self.participants:
            raise ValueError("No participants were selected.")

        # Setup MediaPipe Face Facial Landmark model
        base_options = python.BaseOptions(model_asset_path=str(GIT_ROOT / 'python' / 'weights' / 'face_landmarker_v2_with_blendshapes.task'))
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        # Determine the number of samples in the dataset
        # participant_dirs = [self.dataset_dir / f'EYEDIAP{p}' for p in self.participants]

        # Saving information
        self.samples: List[Sample] = []

        num_samples = 0

        # Tracking the number of loaded samples
        # for participant_dir in tqdm(participant_dirs, total=len(participant_dirs), desc='Participants Preprocessing'):
        for session_id in SESSION_INDEX:

            if self.dataset_size is not None and num_samples >= self.dataset_size:
                break

            P, C, T, H = sessions[session_id]
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

            if rgb_vga.isOpened() and depth.isOpened() and (rgb_hd is None or rgb_hd.isOpened()):
                all_ok = True
                key = 0
                frameIndex = 0
                fps = 30
                ball_vals, screen_vals = None, None
                screen_img = None
                while all_ok and key != ord('q'):
                    ok_rgb, frame_rgb     = rgb_vga.read()
                    ok_depth, frame_depth = depth.read()
                    if rgb_hd  is not None:
                        ok_hd, frame_hd       = rgb_hd.read()
                    else:
                        ok_hd, frame_hd = True, None
                    target_vals = None
                    if ball_track is not None:
                        ball_frame_index, ball_vals = extract_next_file_values(ball_track)
                        target_vals = ball_vals
                    if screen_track is not None:
                        screen_frame_index, screen_vals = extract_next_file_values(screen_track)
                        target_vals = screen_vals
                    eyes_frame_index, eyes_vals = extract_next_file_values(eyes_track)
                    head_frame_index, head_vals = extract_next_file_values(head_track)

                    all_ok = ok_rgb and ok_depth and ok_hd and eyes_vals is not None and head_vals is not None and target_vals is not None

                    if all_ok:
                        frames = (frame_rgb, frame_depth, frame_hd)
                        if ball_vals is not None:
                            draw_ball(frames, ball_vals)
                        if screen_vals is not None:
                            screen_img = np.zeros((1000, 1680, 3), dtype=np.uint8)
                            draw_screen(screen_img, screen_vals)
                        draw_eyes(frames, eyes_vals)
                        headHD_2D= draw_head_pose(frames, head_vals, calibrations)
                        if screen_img is not None:
                            cv2.imshow('screen', screen_img)
                        if frame_hd is not None:
                            cv2.imshow('rgb_hd' , frame_hd)
                        cv2.imshow('depth'  , frame_depth)
                        cv2.imshow('rgb_vga', frame_rgb)
                        if head_frame_index == 0:
                            key = cv2.waitKey(1000)
                        else:
                            key = cv2.waitKey(int(1000/fps))
                        if key != -1:
                            key = key & 255
                            if key == 113:
                                break
                            if key == 82:
                                fps += 5
                                if fps > 30*5:
                                    fps = 30*5
                            if key == 84:
                                fps -= 5
                                if fps < 1:
                                    fps = 1
                    frameIndex += 1
            rgb_vga.release()
            depth.release()
            if rgb_hd is not None:
                rgb_hd.release()
            if ball_track is not None:
                ball_track.close()
            if screen_track is not None:
                screen_track.close()

            break

if __name__ == '__main__':

    from ..constants import DEFAULT_CONFIG
    with open(DEFAULT_CONFIG, 'r') as f:
        config = yaml.safe_load(f)

    dataset = EyeDiapDataset(
        GIT_ROOT / pathlib.Path(config['datasets']['EyeDiap']['path']),
        dataset_size=2,
        participants=[1]
    )
    print(len(dataset))
    
    sample = dataset[0]
    print(json.dumps({k: str(v.dtype) for k, v in sample.items()}, indent=4))