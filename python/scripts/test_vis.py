import cv2
import numpy as np

from webeyetrack import WebEyeTrack
from webeyetrack.constants import *
from webeyetrack.vis import (
    render_3d_gaze,
    render_3d_gaze_with_frame,
    render_pog,
    render_pog_with_screen
)
from webeyetrack.utilities import (
    estimate_camera_intrinsics,
    get_screen_attributes,
    create_transformation_matrix
)

CWD = pathlib.Path(__file__).parent
SCREEN_HEIGHT_CM, SCREEN_WIDTH_CM, SCREEN_HEIGHT_PX, SCREEN_WIDTH_PX = get_screen_attributes()

if __name__ == '__main__':
    
    # Get the cap sizes
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K = estimate_camera_intrinsics(np.zeros((height, width, 3)))

    # Define a transformation matrix between the camera and the screen
    screen_RT = create_transformation_matrix(
        scale=1,
        translation=np.array([(SCREEN_WIDTH_CM)/2, 0, 0]),
        rotation=np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    )

    # Pipeline
    pipeline = WebEyeTrack(
        model_asset_path=str(GIT_ROOT / 'python'/ 'weights' / 'face_landmarker_v2_with_blendshapes.task'), 
        frame_height=height,
        frame_width=width,
        intrinsics=K,
        screen_RT=screen_RT,
        screen_width_cm=SCREEN_WIDTH_CM,
        screen_height_cm=SCREEN_HEIGHT_CM,
        screen_width_px=SCREEN_WIDTH_PX,
        screen_height_px=SCREEN_HEIGHT_PX
    )

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        result, detection_results = pipeline.process_frame(frame)
        if not result:
            # cv2.imshow("Face Mesh", frame)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break
            continue

        # Render the gaze in 3D
        # render_3d_gaze(frame, result, CWD/'test.png')
        # render_3d_gaze_with_frame(
        #     frame, 
        #     result,
        #     CWD/'test.png'
        # )

        # Render the gaze with screen
        # render_pog(frame, result, CWD/'test_screen.png', screen_RT, SCREEN_WIDTH_CM, SCREEN_HEIGHT_CM)
        render_pog_with_screen(
            frame, 
            result, 
            CWD/'test_screen.png', 
            screen_RT, 
            SCREEN_WIDTH_CM, 
            SCREEN_HEIGHT_CM,
            SCREEN_WIDTH_PX,
            SCREEN_HEIGHT_PX
        )

        # draw_frame = frame.copy()
        # cv2.imshow("Face Mesh", draw_frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
        break