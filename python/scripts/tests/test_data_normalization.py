import cv2
import numpy as np

from webeyetrack import WebEyeTrack
from webeyetrack.constants import *
from webeyetrack.vis import (
    draw_landmarks_on_image
)
from webeyetrack.utilities import (
    estimate_camera_intrinsics,
    get_screen_attributes,
    create_transformation_matrix
)

normalized_camera = {
    'focal_length': 1300,
    'distance': 600,
    'size': (256, 64),
}

IMG_SIZE = 256

norm_camera_matrix = np.array(
    [
        [normalized_camera['focal_length'], 0, 0.5*normalized_camera['size'][0]],  # noqa
        [0, normalized_camera['focal_length'], 0.5*normalized_camera['size'][1]],  # noqa
        [0, 0, 1],
    ],
    dtype=np.float64,
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

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        result, detection_results = pipeline.process_frame(frame)
        if not result:
            # cv2.imshow("Face Mesh", frame)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break
            continue

        draw_frame = frame.copy()

        # Draw the landmarks
        draw_frame = draw_landmarks_on_image(draw_frame, detection_results)

        # Drop the face from the image (old fashioned way)
        facial_landmarks = (result.facial_landmarks[:, :2] * np.array([width, height])).astype(np.int32)
        rightmost = facial_landmarks[LEFTMOST_LANDMARK] # Flipped from the original
        leftmost = facial_landmarks[RIGHTMOST_LANDMARK] # Flipped from the original
        topmost = facial_landmarks[TOPMOST_LANDMARK]
        bottommost = facial_landmarks[BOTTOMMOST_LANDMARK]
        face_crop = frame[topmost[1]:bottommost[1], leftmost[0]:rightmost[0]]

        # Compute the homography matrix (4 pts) from the points to a final flat rectangle
        # Compute the corners
        # lefttop = [leftmost[0], topmost[1]]
        # leftbottom = [leftmost[0], bottommost[1]]
        # righttop = [rightmost[0], topmost[1]]
        # rightbottom = [rightmost[0], bottommost[1]]
        lefttop = facial_landmarks[103]
        leftbottom = facial_landmarks[150]
        righttop = facial_landmarks[332]
        rightbottom = facial_landmarks[379]
        center = facial_landmarks[4]

        src_pts = np.array([
            lefttop,
            leftbottom,
            rightbottom,
            righttop
        ], dtype=np.float32)

        # Add padding to the points, radially away from the center
        src_direction = src_pts - center
        src_pts = src_pts + np.array([0.4, 0.2]) * src_direction

        for src_pt, color in zip(src_pts, [(0,0,0), (100, 100, 100), (200, 200, 200), (255, 255, 255)]):
            cv2.circle(draw_frame, tuple(src_pt.astype(np.int32)), 5, color, -1)

        dst_pts = np.array([
            [0, 0],
            [0, IMG_SIZE],
            [IMG_SIZE, IMG_SIZE],
            [IMG_SIZE, 0],
        ], dtype=np.float32)

        # Compute the homography matrix
        M, _ = cv2.findHomography(src_pts, dst_pts)
        warped_face_crop = cv2.warpPerspective(frame, M, (IMG_SIZE, IMG_SIZE))

        cv2.imshow("Face Mesh", draw_frame)
        cv2.imshow("Face Crop", face_crop)
        cv2.imshow("Warped Face Crop", warped_face_crop)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
