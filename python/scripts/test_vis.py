import cv2
import numpy as np

from webeyetrack import WebEyeTrack
from webeyetrack.constants import *
from webeyetrack.vis import (
    render_3d_gaze,
    render_3d_gaze_with_screen
)
from webeyetrack.utilities import (
    estimate_camera_intrinsics,
    get_screen_attributes
)

CWD = pathlib.Path(__file__).parent
SCREEN_HEIGHT_MM, SCREEN_WIDTH_MM, SCREEN_HEIGHT_PX, SCREEN_WIDTH_PX = get_screen_attributes()

if __name__ == '__main__':
    
    # Get the cap sizes
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K = estimate_camera_intrinsics(np.zeros((height, width, 3)))

    # Pipeline
    pipeline = WebEyeTrack(
        model_asset_path=str(GIT_ROOT / 'python'/ 'weights' / 'face_landmarker_v2_with_blendshapes.task'), 
        frame_height=height,
        frame_width=width,
        intrinsics=K,
        screen_R=np.deg2rad(np.array([0, 0, 0]).astype(np.float32)),
        screen_t=np.array([0, 0, 0]).astype(np.float32),
        screen_width_mm=SCREEN_WIDTH_MM,
        screen_height_mm=SCREEN_HEIGHT_MM,
        screen_width_px=SCREEN_WIDTH_PX,
        screen_height_px=SCREEN_HEIGHT_PX
    )

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        result, detection_results = pipeline.process_frame(frame)
        if not result:
            cv2.imshow("Face Mesh", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Render the gaze in 3D
        # render_3d_gaze(frame, result, CWD/'test.png')

        # Render the gaze with screen
        render_3d_gaze_with_screen(frame, result, CWD/'test_screen.png', SCREEN_WIDTH_MM, SCREEN_HEIGHT_MM)

        draw_frame = frame.copy()
        cv2.imshow("Face Mesh", draw_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break