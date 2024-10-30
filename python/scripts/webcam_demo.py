import pathlib
from screeninfo import get_monitors

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
# EYE_TRACKING_APPROACH = "landmark2d"
# EYE_TRACKING_APPROACH = "blendshape"

# Screen dimensions
m = get_monitors()[0]

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

        result = pipeline.process_frame(
            frame, 
            intrinsics, 
            smooth=True,
            screen_R=np.deg2rad(np.array([0, -180, 0]).astype(np.float32)),
            screen_t=np.array([0.5*m.width_mm, 0, 0]).astype(np.float32),
            screen_width_mm=m.width_mm,
            screen_height_mm=m.height_mm,
            screen_width_px=m.width,
            screen_height_px=m.height
        )

        if result:
            
            # Render the PoG
            screen = np.zeros((m.height, m.width, 3), dtype=np.uint8)
            result.pog_px[1] = m.height/2
            screen = vis.draw_pog(screen, result.pog_px, size=100)
            cv2.imshow('screen', screen)

            if EYE_TRACKING_APPROACH == "model-based":
                img = vis.model_based_gaze_render(frame, result)
                if type(img) == np.ndarray:
                    cv2.imshow('frame', img) 
            elif EYE_TRACKING_APPROACH == "landmark2d":
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

cv2.destroyAllWindows()