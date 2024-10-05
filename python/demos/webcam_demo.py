import pathlib

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import imutils
import math

from webeyetrack.datasets.utils import draw_landmarks_on_image
from webeyetrack import vis

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
EYE_HEIGHT_RATIO = 0.7

# Average radius of an eyeball in cm
EYEBALL_RADIUS = 1.15

CWD = pathlib.Path(__file__).parent

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = -(pitch * np.pi / 180)
    yaw = (yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
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

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (math.sin(yaw)) + tdx
    y3 = size * (-math.cos(yaw) * math.sin(pitch)) + tdy

    # cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    # cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(0,0,255),3)

    return img

if __name__ == '__main__':
    
    # Load the webcam 
    cap = cv2.VideoCapture(0)

    # Setup MediaPipe Face Facial Landmark model
    base_options = python.BaseOptions(model_asset_path=str(CWD / 'face_landmarker_v2_with_blendshapes.task'))
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    # Load the frames and draw the landmarks
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect the landmarks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.astype(np.uint8))
        detection_results = face_landmarker.detect(mp_image)

        # Compute the face bounding box based on the MediaPipe landmarks
        try:
            face_landmarks_proto = detection_results.face_landmarks[0]
        except:
            # print(f"Participant {participant_id} image {items[0]} does not have a face detected.")
            continue

        # Compute the bbox by using the edges of the each eyes
        face_landmarks_all = np.array([[lm.x, lm.y, lm.z, lm.visibility, lm.presence] for lm in face_landmarks_proto])
        left_2d_eye_px = face_landmarks_all[LEFT_EYEAREA_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])
        left_2d_eyelid_px = face_landmarks_all[LEFT_EYELID_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])
        left_2d_iris_px = face_landmarks_all[LEFT_IRIS_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])
        left_2d_eyearea_total_px = face_landmarks_all[LEFT_EYEAREA_TOTAL_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])
        left_2d_eyelid_total_px = face_landmarks_all[LEFT_EYELID_TOTAL_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])
        
        right_2d_eye_px = face_landmarks_all[RIGHT_EYEAREA_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])
        right_2d_eyelid_px = face_landmarks_all[RIGHT_EYELID_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])
        right_2d_iris_px = face_landmarks_all[RIGHT_IRIS_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])
        right_2d_eyearea_total_px = face_landmarks_all[RIGHT_EYEAREA_TOTAL_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])
        right_2d_eyelid_total_px = face_landmarks_all[RIGHT_EYELID_TOTAL_LANDMARKS, :2] * np.array([frame.shape[1], frame.shape[0]])

        # 3D
        left_eye_fl = face_landmarks_all[LEFT_EYELID_LANDMARKS, :3]
        right_eye_fl = face_landmarks_all[RIGHT_EYELID_LANDMARKS, :3]

        left_landmarks = [
            left_2d_eye_px, 
            left_2d_eyelid_px,
            left_2d_eyearea_total_px,
            left_2d_eyelid_total_px
        ]
        right_landmarks = [
            right_2d_eye_px, 
            right_2d_eyelid_px,
            right_2d_eyearea_total_px,
            right_2d_eyelid_total_px
        ]

        eye_images = {}
        gaze_vectors = {}
        gaze_origins = {}
        for i, (eye, eyelid, eyearea, eyelid_total) in {'left': left_landmarks, 'right': right_landmarks}.items():
            centroid = np.mean(eye, axis=0)
            width = np.abs(eye[0,0] - eye[1, 0]) * (1 + EYE_PADDING_WIDTH)
            height = width * EYE_HEIGHT_RATIO

            # Determine if closed by the eyelid
            eyelid_width = np.abs(eyelid[0,0] - eyelid[1, 0])
            eyelid_height = np.abs(eyelid[3,1] - eyelid[2, 1])
            is_closed = False

            # Determine if the eye is closed by the ratio of the height based on the width
            if eyelid_height / eyelid_width < 0.05:
                is_closed = True

            if width == 0 or height == 0:
                continue

            # Crop the eye
            eye_image = frame[
                int(centroid[1] - height/2):int(centroid[1] + height/2),
                int(centroid[0] - width/2):int(centroid[0] + width/2)
            ]

            if eye_image.shape[0] == 0 or eye_image.shape[1] == 0:
                continue

            # Resize the eye
            # eye_image = imutils.resize(eye_image, width=400)
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

            # Draw if the eye is closed on the top left corner
            if is_closed:
                cv2.putText(eye_image, 'Closed', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                continue

            # Shift the IRIS landmarks to the cropped eye
            iris_px = left_2d_iris_px if i == 'left' else right_2d_iris_px
            shifted_iris_px = iris_px - np.array([int(centroid[0] - width/2), int(centroid[1] - height/2)])
            iris_center = shifted_iris_px[0]
            eye_center = np.array([width/2, height/2])

            # Based on the direction and magnitude of the line, compute the gaze direction
            # Compute 2D vector from eyeball center to iris center
            # gaze_vector_2d = shifted_iris_px[0] - iris_px[0]
            gaze_vector_2d = iris_center - eye_center
            # gaze_vector_2d = np.array([0,0])

            # Estimate the depth (Z) based on the 2D vector length
            # z_depth = EYEBALL_RADIUS / np.linalg.norm(gaze_vector_2d)
            # import pdb; pdb.set_trace()
            z_depth = 2.0

            # Compute yaw (horizontal rotation)
            yaw = np.arctan2(gaze_vector_2d[0] * 0.1, z_depth) * (180 / np.pi)  # Convert from radians to degrees

            # Compute pitch (vertical rotation)
            pitch = np.arctan2(gaze_vector_2d[1] * 0.1, z_depth) * (180 / np.pi)  # Convert from radians to degrees

            # Store pitch and yaw
            gaze_vectors[i] = np.array([pitch, yaw])

            # Compute 3D gaze origin
            eye_fl = left_eye_fl if i == 'left' else right_eye_fl
            gaze_origin = np.mean(eye_fl, axis=0)
            gaze_origins[i] = gaze_origin

            # Draw on the resized eye image
            # Draw the center dot
            for iris_px in shifted_iris_px:
                resized_iris_px = iris_px * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
                cv2.circle(eye_image, tuple(resized_iris_px.astype(int)), 3, (0, 0, 255), -1)

            # Draw the centroid of the eyeball
            cv2.circle(eye_image, (int(new_width/2), int(new_height/2)), 3, (255, 0, 0), -1)

            # Compute the line between the iris center and the centroid
            new_shifted_iris_px_center = shifted_iris_px[0] * np.array([400/original_width, 400*EYE_HEIGHT_RATIO/original_height])
            cv2.line(eye_image, (int(new_width/2), int(new_height/2)), tuple(new_shifted_iris_px_center.astype(int)), (0, 255, 0), 2)

        # Draw the landmarks
        # frame = draw_landmarks_on_image(frame, detection_results)

        # Define intrinsics baesed on the image size
        intrinsics = np.array([
            [frame.shape[1], 0, frame.shape[1] / 2],
            [0, frame.shape[0], frame.shape[0] / 2],
            [0, 0, 1]
        ]).astype(np.float32)

        h, w = frame.shape[:2]

        # Draw the 3D Gaze vector
        for eye in ['left', 'right']:
            if eye not in gaze_origins:
                continue
            gaze_origin = gaze_origins[eye]
            gaze_origin = gaze_origins[eye] * np.array([w, h, 1])
            gaze_pitch_yaw = gaze_vectors[eye]

            # Compute the gaze target
            # gaze_target_3d_semi = gaze_origin + gaze_vector_3d_normalized * 1
            # gaze_target_2d, _ = cv2.projectPoints(
            #     gaze_target_3d_semi, 
            #     np.array([0, 0, 0], dtype=np.float32),
            #     np.array([0, 0, 0], dtype=np.float32),
            #     intrinsics,
            #     np.zeros((5,1), dtype=np.float32)
            # )

            # Draw the gaze vector
            # cv2.line(frame, tuple(gaze_target_2d[0].astype(int).ravel()), tuple(gaze_origin[:2].astype(int)), (0, 0, 255), 2)
            # import pdb; pdb.set_trace() 
            # gaze_origin = gaze_origins[eye] * np.array([w, h, 1])
            # frame = vis.draw_gaze_direction(frame, gaze_origin[:2], gaze_target_2d[0,0], color=(0, 0, 255))
            frame = draw_axis(frame, gaze_pitch_yaw[1], gaze_pitch_yaw[0], 0, tdx=gaze_origin[0], tdy=gaze_origin[1], size=100)

        # Display the eyes
        # for i, eye_image in eye_images.items():
        #     cv2.imshow(f'eye_{i}', eye_image)

        # # Display the frame
        # cv2.imshow('frame', frame)

        # Assuming eye_images contains the left and right eye images (both resized to 400x280)
        if 'left' in eye_images and 'right' in eye_images:
            left_eye_image = eye_images['left']
            right_eye_image = eye_images['right']

            # Concatenate the eye images horizontally
            eyes_combined = cv2.hconcat([right_eye_image, left_eye_image])

            # Resize the combined eyes horizontally to match the width of the frame (640 pixels wide)
            eyes_combined_resized = cv2.resize(eyes_combined, (frame.shape[1], eyes_combined.shape[0]))

            # Concatenate the combined eyes image vertically with the frame
            final_image = cv2.vconcat([frame, eyes_combined_resized])

            # Display the final concatenated image
            cv2.imshow('Gaze Visualization', final_image)

        # cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break