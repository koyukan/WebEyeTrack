import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import math

from .data_protocols import FLGEResult, EyeResult
from .core import vector_to_pitch_yaw, rotation_matrix_to_euler_angles

LEFT_EYE_LANDMARKS = [263, 362, 386, 374, 380]
RIGHT_EYE_LANDMARKS = [33, 133, 159, 145, 153]
LEFT_BLENDSHAPES = [14, 16, 18, 12]
RIGHT_BLENDSHAPES = [13, 15, 17, 11]
REAL_WORLD_IPD = 6.3 # Inter-pupilary distance (cm)
HFOV = 100
VFOV = 90

# Format [leftmost, rightmost, topmost, bottommost]
LEFT_EYEAREA_LANDMARKS = [463, 359, 257, 253]
RIGHT_EYEAREA_LANDMARKS = [130, 243, 27, 23]
LEFT_EYEAREA_TOTAL_LANDMARKS = [463,  341, 256, 252, 253, 254, 339, 255, 359, 467, 260, 259, 257, 258, 286, 414]
RIGHT_EYEAREA_TOTAL_LANDMARKS = [130, 25,  110, 24,  23,  22,  26,  112, 243, 190, 56,  28,  27,  29,  30,  247]

# Format [leftmost, rightmost, topmost, bottommost]
LEFT_EYELID_LANDMARKS = [362, 263, 386, 374]
RIGHT_EYELID_LANDMARKS = [33, 133, 159, 145]
LEFT_EYELID_TOTAL_LANDMARKS = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYELID_TOTAL_LANDMARKS = [33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

RIGHT_IRIS_LANDMARKS = [468, 470, 469, 472, 471] # center, top, right, botton, left
LEFT_IRIS_LANDMARKS = [473, 475, 474, 477, 476] # center, top, right, botton, left

# EYE_PADDING_HEIGHT = 0.1
EYE_PADDING_WIDTH = 0.3
EYE_HEIGHT_RATIO = 1


def model_based_gaze_render(frame: np.ndarray, result: FLGEResult):

    # Extract the information
    facial_landmarks = result.facial_landmarks
    height, width = frame.shape[:2]

    intrinsics = np.array([
        [frame.shape[1], 0, frame.shape[1] / 2],
        [0, frame.shape[0], frame.shape[0] / 2],
        [0, 0, 1]
    ]).astype(np.float32)

    # Compute the bbox by using the edges of the each eyes
    left_2d_eye_px = facial_landmarks[LEFT_EYEAREA_LANDMARKS, :2] * np.array([width, height])
    left_2d_eyelid_px = facial_landmarks[LEFT_EYELID_LANDMARKS, :2] * np.array([width, height])
    left_2d_iris_px = facial_landmarks[LEFT_IRIS_LANDMARKS, :2] * np.array([width, height])
    left_2d_eyearea_total_px = facial_landmarks[LEFT_EYEAREA_TOTAL_LANDMARKS, :2] * np.array([width, height])
    left_2d_eyelid_total_px = facial_landmarks[LEFT_EYELID_TOTAL_LANDMARKS, :2] * np.array([width, height])
    
    right_2d_eye_px = facial_landmarks[RIGHT_EYEAREA_LANDMARKS, :2] * np.array([width, height])
    right_2d_eyelid_px = facial_landmarks[RIGHT_EYELID_LANDMARKS, :2] * np.array([width, height])
    right_2d_iris_px = facial_landmarks[RIGHT_IRIS_LANDMARKS, :2] * np.array([width, height])
    right_2d_eyearea_total_px = facial_landmarks[RIGHT_EYEAREA_TOTAL_LANDMARKS, :2] * np.array([width, height])
    right_2d_eyelid_total_px = facial_landmarks[RIGHT_EYELID_TOTAL_LANDMARKS, :2] * np.array([width, height])

    left_landmarks = [
        left_2d_eye_px,
        left_2d_eyelid_px,
        left_2d_eyearea_total_px,
        left_2d_eyelid_total_px,
    ]

    right_landmarks = [
        right_2d_eye_px,
        right_2d_eyelid_px,
        right_2d_eyearea_total_px,
        right_2d_eyelid_total_px,
    ]

    eye_images = {}
    for i, (eye, eyelid, eyearea, eyelid_total) in {'left': left_landmarks, 'right': right_landmarks}.items():
        centroid = np.mean(eye, axis=0)
        width = np.abs(eye[0,0] - eye[1, 0]) * (1 + EYE_PADDING_WIDTH)
        height = width * EYE_HEIGHT_RATIO

        # Determine if closed by the eyelid
        eyelid_width = np.abs(eyelid[0,0] - eyelid[1, 0])
        eyelid_height = np.abs(eyelid[3,1] - eyelid[2, 1])
        eye_result = result.left if i == 'left' else result.right
        is_closed = eye_result.is_closed

        if width == 0 or height == 0:
            continue

        # Crop the eye
        eye_image = frame[
            int(centroid[1] - height/2):int(centroid[1] + height/2),
            int(centroid[0] - width/2):int(centroid[0] + width/2)
        ]

        eye_image_shape = eye_image.shape[:2]
        if eye_image_shape[0] == 0 or eye_image_shape[1] == 0:
            continue

        # Create eye image
        original_height, original_width = eye_image.shape[:2]
        new_width, new_height = 400, int(400*EYE_HEIGHT_RATIO)
        eye_image = cv2.resize(eye_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        eye_images[i] = eye_image

        # Draw the outline of the eyelid
        shifted_eyelid_px = eyelid_total - np.array([int(centroid[0] - width/2), int(centroid[1] - height/2)])
        prior_px = None
        for px in shifted_eyelid_px:
            resized_px = px * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
            if prior_px is not None:
                cv2.line(eye_image, tuple(prior_px.astype(int)), tuple(resized_px.astype(int)), (255, 0, 0), 1)
            prior_px = resized_px
        
        # Draw the last line to close the loop
        resized_first_px = shifted_eyelid_px[0] * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
        cv2.line(eye_image, tuple(prior_px.astype(int)), tuple(resized_first_px.astype(int)), (255, 0, 0), 1)

        # Shift the IRIS landmarks to the cropped eye
        iris_px = left_2d_iris_px if i == 'left' else right_2d_iris_px
        shifted_iris_px = iris_px - np.array([int(centroid[0] - width/2), int(centroid[1] - height/2)])
        scaled_shifted_iris_px = shifted_iris_px * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
        cv2.circle(eye_image, tuple(scaled_shifted_iris_px[0].astype(int)), 3, (0, 0, 255), -1)
        # for iris_px_pt in shifted_iris_px:
        #     resized_iris_px = iris_px_pt * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
        #     cv2.circle(eye_image, tuple(resized_iris_px.astype(int)), 3, (0, 0, 255), -1)

        # Draw the eyeball
        if 'eyeball_center_2d' in eye_result.meta_data and eye_result.meta_data['eyeball_center_2d'] is not None:
            # Offset the eyeball center to the cropped eye
            eyeball_center_2d = eye_result.meta_data['eyeball_center_2d']
            eyeball_radius_2d = eye_result.meta_data['eyeball_radius_2d']
            shifted_eyeball_center_2d = eyeball_center_2d - np.array([int(centroid[0] - width/2), int(centroid[1] - height/2)])

            # Apply the scaling factor
            resized_eyeball_center_2d = shifted_eyeball_center_2d * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
            resized_eyeball_radius_2d = eyeball_radius_2d * 400/original_width

            # Draw the eyeball
            # import pdb; pdb.set_trace()
            cv2.circle(eye_image, tuple(resized_eyeball_center_2d.astype(int)), int(resized_eyeball_radius_2d), (0, 0, 255), 1)

        # Draw the centroid of the eyeball
        # cv2.circle(eye_image, (int(new_width/2), int(new_height/2)), 3, (255, 0, 0), -1)
        
        # Compute the line between the iris center and the centroid
        # new_shifted_iris_px_center = shifted_iris_px[0] * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
        # cv2.line(eye_image, (int(new_width/2), int(new_height/2)), tuple(new_shifted_iris_px_center.astype(int)), (0, 255, 0), 2)

        if is_closed:
            cv2.putText(eye_image, 'Closed', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            continue

        # Convert 3D to pitch and yaw
        iris_center = iris_px[0]
        pitch, yaw = vector_to_pitch_yaw(eye_result.direction)
        frame = draw_axis(frame, pitch, yaw, 0, int(iris_center[0]), int(iris_center[1]), 100)

    # Draw the eyeballs on the frame itself
    for eye_result in [result.left, result.right]:
        if 'eyeball_center_2d' in eye_result.meta_data and eye_result.meta_data['eyeball_center_2d'] is not None:
            eyeball_center_2d = eye_result.meta_data['eyeball_center_2d']
            eyeball_radius_2d = eye_result.meta_data['eyeball_radius_2d']
            cv2.circle(frame, tuple(eyeball_center_2d.astype(int)), int(eyeball_radius_2d), (0, 0, 255), 1)

    # Draw the FPS on the topright
    fps = 1/result.duration
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw the headpose on the frame
    # headrot = result.face_rt[:3, :3]
    # pitch, yaw, roll = rotation_matrix_to_euler_angles(headrot)
    # pitch, yaw = yaw, pitch
    # face_origin = result.face_origin_2d
    # frame = draw_axis(frame, pitch, yaw, -roll, int(face_origin[0]), int(face_origin[1]), 100)

    # Concatenate the images
    if 'right' not in eye_images:
        right_eye_image = np.zeros((400, 400, 3), dtype=np.uint8)
    else:
        right_eye_image = eye_images['right']

    if 'left' not in eye_images:
        left_eye_image = np.zeros((400, 400, 3), dtype=np.uint8)
    else:
        left_eye_image = eye_images['left']

    # Resize the combined eyes horizontally to match the width of the frame (640 pixels wide)
    eyes_combined = cv2.hconcat([right_eye_image, left_eye_image])
    eyes_combined_resized = cv2.resize(eyes_combined, (frame.shape[1], eyes_combined.shape[0]))

    # Concatenate the combined eyes image vertically with the frame
    return cv2.vconcat([frame, eyes_combined_resized])

def landmark_gaze_render(frame: np.ndarray, result: FLGEResult):

    # Extract the information
    facial_landmarks = result.facial_landmarks
    height, width = frame.shape[:2]

    intrinsics = np.array([
        [frame.shape[1], 0, frame.shape[1] / 2],
        [0, frame.shape[0], frame.shape[0] / 2],
        [0, 0, 1]
    ]).astype(np.float32)

    # Compute the bbox by using the edges of the each eyes
    left_2d_eye_px = facial_landmarks[LEFT_EYEAREA_LANDMARKS, :2] * np.array([width, height])
    left_2d_eyelid_px = facial_landmarks[LEFT_EYELID_LANDMARKS, :2] * np.array([width, height])
    left_2d_iris_px = facial_landmarks[LEFT_IRIS_LANDMARKS, :2] * np.array([width, height])
    left_2d_eyearea_total_px = facial_landmarks[LEFT_EYEAREA_TOTAL_LANDMARKS, :2] * np.array([width, height])
    left_2d_eyelid_total_px = facial_landmarks[LEFT_EYELID_TOTAL_LANDMARKS, :2] * np.array([width, height])
    
    right_2d_eye_px = facial_landmarks[RIGHT_EYEAREA_LANDMARKS, :2] * np.array([width, height])
    right_2d_eyelid_px = facial_landmarks[RIGHT_EYELID_LANDMARKS, :2] * np.array([width, height])
    right_2d_iris_px = facial_landmarks[RIGHT_IRIS_LANDMARKS, :2] * np.array([width, height])
    right_2d_eyearea_total_px = facial_landmarks[RIGHT_EYEAREA_TOTAL_LANDMARKS, :2] * np.array([width, height])
    right_2d_eyelid_total_px = facial_landmarks[RIGHT_EYELID_TOTAL_LANDMARKS, :2] * np.array([width, height])

    left_landmarks = [
        left_2d_eye_px,
        left_2d_eyelid_px,
        left_2d_eyearea_total_px,
        left_2d_eyelid_total_px,
    ]

    right_landmarks = [
        right_2d_eye_px,
        right_2d_eyelid_px,
        right_2d_eyearea_total_px,
        right_2d_eyelid_total_px,
    ]

    eye_images = {}
    for i, (eye, eyelid, eyearea, eyelid_total) in {'left': left_landmarks, 'right': right_landmarks}.items():
        centroid = np.mean(eye, axis=0)
        width = np.abs(eye[0,0] - eye[1, 0]) * (1 + EYE_PADDING_WIDTH)
        height = width * EYE_HEIGHT_RATIO

        # Determine if closed by the eyelid
        eyelid_width = np.abs(eyelid[0,0] - eyelid[1, 0])
        eyelid_height = np.abs(eyelid[3,1] - eyelid[2, 1])
        eye_result = result.left if i == 'left' else result.right
        is_closed = eye_result.is_closed

        # Determine if the eye is closed by the ratio of the height based on the width
        # if eyelid_height / eyelid_width < 0.05:
        #     is_closed = True

        if width == 0 or height == 0:
            continue

        # Crop the eye
        eye_image = frame[
            int(centroid[1] - height/2):int(centroid[1] + height/2),
            int(centroid[0] - width/2):int(centroid[0] + width/2)
        ]

        # Create eye image
        original_height, original_width = eye_image.shape[:2]
        new_width, new_height = 400, int(400*EYE_HEIGHT_RATIO)
        eye_image = cv2.resize(eye_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        eye_images[i] = eye_image

        # Draw the outline of the eyearea
        shifted_eyearea_px = eyearea - np.array([int(centroid[0] - width/2), int(centroid[1] - height/2)])
        prior_px = None
        for px in shifted_eyearea_px:
            resized_px = px * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
            if prior_px is not None:
                cv2.line(eye_image, tuple(prior_px.astype(int)), tuple(resized_px.astype(int)), (0, 255, 0), 1)
            prior_px = resized_px
        # Draw the last line to close the loop
        resized_first_px = shifted_eyearea_px[0] * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
        cv2.line(eye_image, tuple(prior_px.astype(int)), tuple(resized_first_px.astype(int)), (0, 255, 0), 1)

        # Draw the outline of the eyelid
        shifted_eyelid_px = eyelid_total - np.array([int(centroid[0] - width/2), int(centroid[1] - height/2)])
        prior_px = None
        for px in shifted_eyelid_px:
            resized_px = px * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
            if prior_px is not None:
                cv2.line(eye_image, tuple(prior_px.astype(int)), tuple(resized_px.astype(int)), (255, 0, 0), 1)
            prior_px = resized_px
        
        # Draw the last line to close the loop
        resized_first_px = shifted_eyelid_px[0] * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
        cv2.line(eye_image, tuple(prior_px.astype(int)), tuple(resized_first_px.astype(int)), (255, 0, 0), 1)

        # Shift the IRIS landmarks to the cropped eye
        iris_px = left_2d_iris_px if i == 'left' else right_2d_iris_px
        shifted_iris_px = iris_px - np.array([int(centroid[0] - width/2), int(centroid[1] - height/2)])
        for iris_px_pt in shifted_iris_px:
            resized_iris_px = iris_px_pt * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
            cv2.circle(eye_image, tuple(resized_iris_px.astype(int)), 3, (0, 0, 255), -1)

        # Draw the centroid of the eyeball
        cv2.circle(eye_image, (int(new_width/2), int(new_height/2)), 3, (255, 0, 0), -1)
        
        # Compute the line between the iris center and the centroid
        new_shifted_iris_px_center = shifted_iris_px[0] * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
        cv2.line(eye_image, (int(new_width/2), int(new_height/2)), tuple(new_shifted_iris_px_center.astype(int)), (0, 255, 0), 2)

        # If available, visualize the headpose-corrected iris center
        if 'headpose_corrected_eye_center' in eye_result.meta_data:
            headpose_corrected_eye_center = eye_result.meta_data['headpose_corrected_eye_center']
            if headpose_corrected_eye_center is not None:
                new_headpose_corrected_eye_center = headpose_corrected_eye_center * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
                cv2.circle(eye_image, tuple(new_headpose_corrected_eye_center.astype(int)), 3, (0, 255, 255), -1)

                # Draw a line between the headpose corrected iris center and the iris center
                cv2.line(eye_image, tuple(new_headpose_corrected_eye_center.astype(int)), tuple(new_shifted_iris_px_center.astype(int)), (0, 255, 255), 2)
  
        if is_closed:
            cv2.putText(eye_image, 'Closed', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            continue

        # Convert 3D to pitch and yaw
        iris_center = iris_px[0]
        pitch, yaw = vector_to_pitch_yaw(eye_result.direction)
        frame = draw_axis(frame, pitch, yaw, 0, int(iris_center[0]), int(iris_center[1]), 100)

    # Draw the FPS on the topright
    fps = 1/result.duration
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw the headpose on the frame
    # headrot = result.face_rt[:3, :3]
    # pitch, yaw, roll = rotation_matrix_to_euler_angles(headrot)
    # pitch, yaw = yaw, pitch
    # face_origin = result.face_origin_2d
    # frame = draw_axis(frame, pitch, yaw, -roll, int(face_origin[0]), int(face_origin[1]), 100)

    # Concatenate the images
    right_eye_image = eye_images['right']
    left_eye_image = eye_images['left']
    eyes_combined = cv2.hconcat([right_eye_image, left_eye_image])

    # Resize the combined eyes horizontally to match the width of the frame (640 pixels wide)
    eyes_combined_resized = cv2.resize(eyes_combined, (frame.shape[1], eyes_combined.shape[0]))

    # Concatenate the combined eyes image vertically with the frame
    return cv2.vconcat([frame, eyes_combined_resized])

def blendshape_gaze_render(frame: np.ndarray, result: FLGEResult):

    # Extract the information
    facial_landmarks = result.facial_landmarks
    height, width = frame.shape[:2]

    left_2d_eye_px = facial_landmarks[LEFT_EYEAREA_LANDMARKS, :2] * np.array([width, height])
    right_2d_eye_px = facial_landmarks[RIGHT_EYEAREA_LANDMARKS, :2] * np.array([width, height])

    for i, eye in {'left': left_2d_eye_px, 'right': right_2d_eye_px}.items():
        centroid = np.mean(eye, axis=0)
        eye_result = result.left if i == 'left' else result.right 

        # Apply a correct R to the gaze direction
        # R = np.array([
        #     [1, 0, 0],
        #     [0, 1, 0],
        #     [0, 0, -1]
        # ])
        # rotvec = np.array([90, 0, 0], dtype=np.float32) # in degree
        rotvec = np.array([0, 90, 0], dtype=np.float32) # in degree
        rotvec = np.radians(rotvec)
        R = cv2.Rodrigues(rotvec)[0]
        corrected_gaze_direction = np.dot(R, eye_result.direction) * np.array([1, 1, 1])

        # if i == 'right':
        #     pitch, yaw = vector_to_pitch_yaw(eye_result.direction)
        #     frame = draw_axis(frame, pitch, yaw, 0, int(centroid[0]), int(centroid[1]), 100)
        #     cv2.putText(frame, f"{i}", (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # else:
        pitch, yaw = vector_to_pitch_yaw(corrected_gaze_direction)
        frame = draw_axis(frame, -yaw, pitch, 0, int(centroid[0]), int(centroid[1]), 100)
        # cv2.putText(frame, f"{i}", (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw the FPS on the topright
    fps = 1/result.duration
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame

# def draw_axis(img, pitch, yaw, roll=0, tdx=None, tdy=None, size = 100):

#     pitch = -(pitch * np.pi / 180)
#     yaw = (yaw * np.pi / 180)
#     roll = roll * np.pi / 180

#     if tdx != None and tdy != None:
#         tdx = tdx
#         tdy = tdy
#     else:
#         height, width = img.shape[:2]
#         tdx = width / 2
#         tdy = height / 2

#     # X-Axis pointing to right. drawn in red
#     x1 = size * (math.cos(yaw) * math.cos(roll)) + tdx
#     y1 = size * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)) + tdy

#     # Y-Axis | drawn in green
#     #        v
#     x2 = size * (-math.cos(yaw) * math.sin(roll)) + tdx
#     y2 = size * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll)) + tdy

#     # Z-Axis (out of the screen) drawn in blue
#     x3 = size * (math.sin(yaw)) + tdx
#     y3 = size * (-math.cos(yaw) * math.sin(pitch)) + tdy

#     # cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
#     # cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
#     # cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),3)
    
#     # cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(255,0,0),3)
#     # cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
#     # cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(0,0,255),3)
    
#     cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,0),3, tipLength=0.2)
#     cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(100,100,100),3, tipLength=0.2)
#     cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,255,255),3, tipLength=0.2)

#     return img

def draw_axis(img, pitch, yaw, roll=0, tdx=None, tdy=None, size=100):
    """
    Draws the 3D axes based on the given pitch, yaw, and roll. The Z-axis is drawn
    pointing towards the negative Z direction.
    
    Arguments:
    img -- the image to draw the axes on
    pitch -- pitch angle in degrees
    yaw -- yaw angle in degrees
    roll -- roll angle in degrees
    tdx -- x translation (optional)
    tdy -- y translation (optional)
    size -- the length of the axis to be drawn
    """
    pitch = (pitch * np.pi / 180)
    yaw = (yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx is not None and tdy is not None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (math.cos(yaw) * math.cos(roll)) + tdx
    y1 = size * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-math.cos(yaw) * math.sin(roll)) + tdx
    y2 = size * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll)) + tdy

    # Z-Axis (negative Z) drawn in blue
    x3 = size * (math.sin(yaw)) + tdx
    y3 = size * (math.cos(yaw) * math.sin(pitch)) + tdy  # Note the change here for negative Z

    # Draw the axes with appropriate colors
    # cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 0), 3, tipLength=0.2)  # X-axis (Black)
    # cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (100, 100, 100), 3, tipLength=0.2)  # Y-axis (Gray)
    cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 255, 255), 3, tipLength=0.2)  # Z-axis (White)

    return img

def draw_gaze_from_vector(img, face_origin_2d, gaze_direction_3d, scale=100, color=(0, 0, 255)):
    """
    Draws a 2D gaze direction based on a 3D unit gaze vector projected onto a 2D plane.
    
    Arguments:
    img -- the image where the gaze will be drawn
    face_origin_2d -- the 2D position of the face origin (where the gaze starts)
    gaze_direction_3d -- the 3D gaze direction vector
    scale -- the scaling factor for the length of the arrow
    color -- the color of the gaze direction line
    """
    # Normalize the 3D gaze direction vector
    gaze_direction_3d = gaze_direction_3d / np.linalg.norm(gaze_direction_3d)

    # Project the 3D gaze vector onto the 2D plane (ignore Z-axis)
    gaze_target_2d = face_origin_2d + scale * np.array([gaze_direction_3d[0], gaze_direction_3d[1]])

    # Draw the gaze direction as an arrowed line
    cv2.arrowedLine(img, 
                    (int(face_origin_2d[0]), int(face_origin_2d[1])), 
                    (int(gaze_target_2d[0]), int(gaze_target_2d[1])), 
                    color, 3, tipLength=0.3)

    return img

def draw_axis_from_rotation_matrix(img, R, tdx=None, tdy=None, size=100):
    """
    Draws the transformed 3D axes using the provided rotation matrix `R`.
    """
    # Define the 3D axes (X, Y, Z)
    x_axis = np.array([1, 0, 0])  # X-axis (Red)
    y_axis = np.array([0, 1, 0])  # Y-axis (Green)
    z_axis = np.array([0, 0, 1])  # Z-axis (Blue)
    
    # Apply the rotation matrix to the axes
    x_rot = np.dot(R, x_axis)
    y_rot = np.dot(R, y_axis)
    z_rot = np.dot(R, z_axis)
    
    if tdx is None or tdy is None:
        height, width = img.shape[:2]
        tdx = width // 2
        tdy = height // 2

    # Project the transformed axes onto the image (2D)
    x1 = size * x_rot[0] + tdx
    y1 = size * x_rot[1] + tdy
    x2 = size * y_rot[0] + tdx
    y2 = size * y_rot[1] + tdy
    x3 = size * z_rot[0] + tdx
    y3 = size * z_rot[1] + tdy

    # Draw the axes
    cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)  # X-axis (Red)
    cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)  # Y-axis (Green)
    cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 3)  # Z-axis (Blue)

    return img

def draw_gaze_origin_heatmap(image, heatmap, alpha=0.5, cmap='jet'):
    """
    Overlay a semi-transparent heatmap on an image.

    Parameters:
    - image: numpy array of shape (h, w, 3)
    - heatmap: numpy array of shape (h, w)
    - alpha: float, transparency level of the heatmap
    - cmap: str, colormap to use for the heatmap

    Returns:
    - overlay: numpy array of the overlaid image
    """
    # Normalize the heatmap to be between 0 and 1
    heatmap_normalized = Normalize()(heatmap)
    
    # Create a color map
    colormap = plt.get_cmap(cmap)
    
    # Apply the colormap to the heatmap
    heatmap_colored = colormap(heatmap_normalized)
    
    # Remove the alpha channel from the colormap output
    heatmap_colored = heatmap_colored[:, :, :3]
    
    # Overlay the heatmap on the image
    overlay = image * (1 - alpha) + heatmap_colored * alpha
    
    # Clip the values to be in the valid range [0, 1]
    overlay = np.clip(overlay, 0, 1)

    return overlay

def draw_gaze_depth_map(image, depth_map, alpha=0.5, cmap='jet'):
    # Normalize the heatmap to be between 0 and 1
    heatmap_normalized = Normalize()(depth_map)
    
    # Create a color map
    colormap = plt.get_cmap(cmap)
    
    # Apply the colormap to the heatmap
    heatmap_colored = colormap(heatmap_normalized)
    
    # Remove the alpha channel from the colormap output
    heatmap_colored = heatmap_colored[:, :, :3]
    
    # Overlay the heatmap on the image
    overlay = image * (1 - alpha) + heatmap_colored * alpha
    
    # Clip the values to be in the valid range [0, 1]
    overlay = np.clip(overlay, 0, 1)

    return overlay

def draw_gaze_origin(image, gaze_origin, color=(255, 0, 0)):
    # Draw gaze origin
    draw_image = image.copy()
    x, y = gaze_origin
    cv2.circle(draw_image, (int(x), int(y)), 10, color, -1)

    return draw_image

def draw_gaze_direction(image, gaze_origin, gaze_dst, color=(255, 0, 0)):
    # Draw gaze direction
    draw_image = image.copy()
    x, y = gaze_origin
    dx, dy = gaze_dst
    cv2.arrowedLine(draw_image, (int(x), int(y)), (int(dx), int(dy)), color, 2)

    return draw_image

def draw_pog(img, pog, color=(0,0,255), size=10):
    # Draw point of gaze (POG)
    x, y = pog
    cv2.circle(img, (int(x), int(y)), size, color, -1)
    return img