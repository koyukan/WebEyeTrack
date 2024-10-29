import pathlib

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import imutils
import math

from webeyetrack.constants import GIT_ROOT
from webeyetrack.datasets.utils import draw_landmarks_on_image
from webeyetrack import vis
from webeyetrack.pipelines.flge import FLGE

EYE_TRACKING_APPROACH = "model-based"
# EYE_TRACKING_APPROACH = "landmark"
# EYE_TRACKING_APPROACH = "blendshape"

if __name__ == '__main__':
    
    # Load the webcam 
    cap = cv2.VideoCapture(0)

    # Pipeline
    pipeline = FLGE(str(GIT_ROOT / 'python'/ 'weights' / 'face_landmarker_v2_with_blendshapes.task'), EYE_TRACKING_APPROACH)

    # Load the frames and draw the landmarks
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Define intrinsics based on the frame
        width, height = frame.shape[:2]
        intrinsics = np.array([[width, 0, width // 2], [0, height, height // 2], [0, 0, 1]])

        result = pipeline.process_frame(frame, intrinsics)

        if result:
            if EYE_TRACKING_APPROACH == "model-based":
                img = vis.model_based_gaze_render(frame, result)
                if type(img) == np.ndarray:
                    cv2.imshow('frame', img) 
            elif EYE_TRACKING_APPROACH == "landmark":
                img = vis.landmark_gaze_render(frame, result)
                if type(img) == np.ndarray:
                    cv2.imshow('frame', img)
            elif EYE_TRACKING_APPROACH == 'blendshape':
                img = vis.blendshape_gaze_render(frame, result)
                if type(img) == np.ndarray:
                    cv2.imshow('frame', img)

        # cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break