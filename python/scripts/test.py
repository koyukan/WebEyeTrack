import pathlib
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import imutils
import math

from webeyetrack.datasets.utils import draw_landmarks_on_image
from webeyetrack import vis, core

CWD = pathlib.Path(__file__).parent
PYTHON_DIR = CWD.parent

LEFT_EYEAREA_LANDMARKS = [463, 359, 257, 253]
# vertical_fov_degrees = 1
# near = 2
# far = 3

# According to https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/graphs/face_effect/face_effect_gpu.pbtxt#L61-L65
# vertical_fov_degrees = 50
vertical_fov_degrees = 60
# vertical_fov_degrees = 63.0 
# vertical_fov_degrees = 90.0
near = 1.0 # 1cm
far = 10000 # 100m 

origin_point_location = 'BOTTOM_LEFT_CORNER'

def reproject_2d_to_3d(u, v, z, intrinsics):
    """
    Reproject a 2D point (u, v) with a given depth z into 3D space.

    :param u: 2D x-coordinate (pixel)
    :param v: 2D y-coordinate (pixel)
    :param z: Depth value at (u, v)
    :param intrinsics: Camera intrinsic matrix (3x3)
    :return: 3D point in camera coordinates
    """
    # Create the 2D point in homogeneous coordinates
    uv_homogeneous = np.array([u, v, 1.0])

    # Invert the intrinsic matrix to map from image space to normalized camera coordinates
    inv_intrinsics = np.linalg.inv(intrinsics)

    # Reproject to 3D normalized coordinates
    normalized_coords = np.dot(inv_intrinsics, uv_homogeneous)

    # Scale by the depth (z) to get the 3D coordinates in camera space
    X = normalized_coords * z

    return X

def create_perspective_matrix(aspect_ratio):
    k_degrees_to_radians = np.pi / 180.0

    # Initialize a 4x4 matrix filled with zeros
    perspective_matrix = np.zeros((4, 4), dtype=np.float32)

    # Standard perspective projection matrix calculations
    f = 1.0 / np.tan(k_degrees_to_radians * vertical_fov_degrees / 2.0)
    denom = 1.0 / (near - far)

    # Populate the matrix values
    perspective_matrix[0, 0] = f / aspect_ratio
    perspective_matrix[1, 1] = f
    perspective_matrix[2, 2] = (near + far) * denom
    perspective_matrix[2, 3] = -1.0
    perspective_matrix[3, 2] = 2.0 * far * near * denom

    # Flip Y-axis if origin point location is top-left corner
    if origin_point_location == 'TOP_LEFT_CORNER':
        perspective_matrix[1, 1] *= -1.0

    return perspective_matrix

if __name__ == '__main__':

    PRIOR_GAZE = np.array([0, 0, -1])
    
    # Load the webcam 
    cap = cv2.VideoCapture(0)

    # Setup MediaPipe Face Landmark model
    base_options = python.BaseOptions(model_asset_path=str(PYTHON_DIR / 'weights' / 'face_landmarker_v2_with_blendshapes.task'))
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

        # Define intrinsics based on the frame dimensions
        height, width = frame.shape[:2]
        focal_length = width  # Assuming fx = fy = width for simplicity, adjust based on real camera
        # perspective_matrix = np.array([[focal_length, 0, width // 2], 
        #                        [0, focal_length, height // 2], 
        #                        [0, 0, 1]])
        perspective_matrix = create_perspective_matrix(aspect_ratio=width / height)
        inv_perspective_matrix = np.linalg.inv(perspective_matrix)
        # import pdb; pdb.set_trace()

        # Detect the landmarks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.astype(np.uint8))
        detection_results = face_landmarker.detect(mp_image)

        # Ensure there is at least one face detected
        if len(detection_results.face_landmarks) == 0:
            cv2.imshow('frame', imutils.resize(frame, width=1000))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Draw the canonical face origin xyz=[0,0,0] via the facial transformation matrix
        face_rt = detection_results.facial_transformation_matrixes[0]

        # Flip the translation
        face_rt[:3, 3] *= np.array([-1, -1, -1])
     
        # Draw the landmarks
        # frame = draw_landmarks_on_image(frame, detection_results)

        pupil_landmark = detection_results.face_landmarks[0][473]
        pupil = np.array([pupil_landmark.x, pupil_landmark.y, pupil_landmark.z, 1])

        # metric_landmark_homogenous = np.array([detection_results.face_landmarks[0][1].x, detection_results.face_landmarks[0][1].y, detection_results.face_landmarks[0][1].z, 1])
        # metric_landmark_homogenous = np.array([0,0,0,1])
        # metric_landmark_homogenous = np.array([3.2,2.6,2.5,1])
        metric_landmark_homogenous = np.array([-3.2,-3,-2.5,1])
        camera_landmark_homogenous = face_rt @ metric_landmark_homogenous
        screen_landmark_homogenous = perspective_matrix @ camera_landmark_homogenous
        screen_x = screen_landmark_homogenous[0] / screen_landmark_homogenous[2]
        screen_y = screen_landmark_homogenous[1] / screen_landmark_homogenous[2]
        screen_x = (screen_x + 1) * width / 2
        screen_y = (screen_y * -1 + 1) * height / 2
        # print(screen_x, screen_y, camera_landmark_homogenous, detection_results.face_landmarks[0][1].z)
        # print(camera_landmark_homogenous)
        eyeball_radius = 0.85
        eyeball_center_2d = np.array([screen_x, screen_y])
        # import pdb; pdb.set_trace()

        # Scale the eyeball radius based on the depth
        draw_eyeball_radius = eyeball_radius * (500/camera_landmark_homogenous[2])
        cv2.circle(frame, (int(screen_x), int(screen_y)), int(draw_eyeball_radius), (0, 0, 255), 1)
        
        pupil2d = np.array([pupil[0] * width, pupil[1] * height])
        cv2.circle(frame, (int(pupil2d[0]), int(pupil2d[1])), 3, (0, 0, 255), -1)

        # Compute the 3D pupil by using a line-sphere intersection problem
        # Reference: https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        # Convert from 0-1 to -1 to 1
        # ndc_x = (2 * pupil2d[0] / width) - 1
        ndc_y = 1 - (2 * pupil2d[1] / height)
        ndc_x = (2 * pupil2d[0] / width) - 1
        # ndc_y = 0
        sphere_center = camera_landmark_homogenous[:3]

        # Homogeneous 4D point in NDC
        ndc_point = np.array([ndc_x, ndc_y, -1.0, 1.9])
        # ndc_point = np.array([ndc_x, ndc_y, -1.0, 2.1])

        # Invert the perspective matrix to go from NDC to world space
        inv_perspective_matrix = np.linalg.inv(perspective_matrix)

        # Compute the ray in 3D space
        world_point_homogeneous = np.dot(inv_perspective_matrix, ndc_point)

        # Dehomogenize (convert from homogeneous to Cartesian coordinates)
        world_point = world_point_homogeneous[:3] / world_point_homogeneous[3]

        # Ray direction from the camera origin to the dehomogenized world point
        ray_direction = world_point - np.array([0, 0, 0])
        ray_direction /= np.linalg.norm(ray_direction)  # Normalize the direction
        
        # Camera origin
        camera_origin = np.array([0.0, 0.0, 0.0])

        # Define a scale to draw the ray visually (use an appropriate scaling factor)
        scale = 1

        # Compute a point along the ray for visualization
        ray_end_point = camera_origin + scale * ray_direction

        # Convert 3D points to 2D image space (assuming the ray end point is projected onto the image plane)
        ray_end_2d = (int(width / 2 + ray_end_point[0] * width / 2), int(height / 2 - ray_end_point[1] * height / 2))

        # Draw the ray direction on the frame (from the camera origin to the end point of the ray)
        # cv2.line(frame, (int(width / 2), int(height / 2)), ray_end_2d, (0, 255, 0), 2)

        # Calculate intersection with the sphere
        oc = camera_origin - sphere_center
        oc_norm = oc / np.linalg.norm(oc)

        # For testing, make the ray direction the same as the oc
        # ray_direction = oc_norm

        sphere_radius = eyeball_radius
        # a = np.dot(ray_direction, ray_direction)
        # b = 2.0 * np.dot(oc, ray_direction)
        # c = np.dot(oc, oc) - sphere_radius ** 2

        # Solve the quadratic equation ax^2 + bx + c = 0
        # discriminant = b ** 2 - 4 * a * c
        discriminant = np.dot(ray_direction, oc) ** 2 - (np.dot(oc, oc) - sphere_radius ** 2)

        if discriminant < 0:
            # No real intersections
            cv2.imshow('frame', imutils.resize(frame, width=1000))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
            # return None

        # Calculate the two possible intersection points
        # t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        # t2 = (-b + np.sqrt(discriminant)) / (2 * a)
        t1 = np.dot(-ray_direction, oc) - np.sqrt(discriminant)
        t2 = np.dot(-ray_direction, oc) + np.sqrt(discriminant)

        # We are interested in the first intersection that is in front of the camera
        pupil_3d = None
        # if t1 >= 0:
        #     pupil_3d = camera_origin + t1 * ray_direction
        # elif t2 >= 0:
        #     pupil_3d = camera_origin + t2 * ray_direction
        if t1 > t2:
            pupil_3d = camera_origin + t1 * ray_direction
        else:
            pupil_3d = camera_origin + t2 * ray_direction
        # print(pupil_3d)

        # Project back the 3D point to ensure
        pupil_3d_homo = np.array([pupil_3d[0], pupil_3d[1], pupil_3d[2], 1])
        screen_landmark_homogenous = perspective_matrix @ pupil_3d_homo
        screen_x = screen_landmark_homogenous[0] / screen_landmark_homogenous[2]
        screen_y = screen_landmark_homogenous[1] / screen_landmark_homogenous[2]
        screen_x = (screen_x + 1) * width / 2
        screen_y = (screen_y * -1 + 1) * height / 2
        cv2.circle(frame, (int(screen_x), int(screen_y)), 2, (0, 255, 0), -1)

        # Compute the gaze direction based on the eyeball center and 3D pupil
        gaze_direction = pupil_3d - sphere_center
        gaze_direction /= np.linalg.norm(gaze_direction)

        # Runnign average with PRIOR_GAZE
        gaze_direction = gaze_direction + PRIOR_GAZE
        gaze_direction /= np.linalg.norm(gaze_direction)
        PRIOR_GAZE = gaze_direction

        # Convert to pitch, yaw, roll
        pitch, yaw = core.vector_to_pitch_yaw(gaze_direction)
        pitch, yaw = pitch, yaw * -1
        print(pitch, yaw)
        frame = vis.draw_axis(frame, pitch, yaw, tdx=pupil2d[0], tdy=pupil2d[1])

        # Draw the rotation matrix
        # rotation_matrix = face_rt[:3, :3].copy()

        # # Convert to pitch, yaw, roll
        # pitch, yaw, roll = vis.rotation_matrix_to_euler_angles(rotation_matrix)
        # pitch, yaw, roll = yaw, pitch, -roll
        # # pitch, yaw, roll = yaw, pitch, roll
        # # pitch, yaw, roll = 0, 0, 0
        # frame = vis.draw_axis(frame, pitch, yaw, roll, int(x), int(y), 100)
        # # frame = vis.draw_axis_from_rotation_matrix(frame, rotation_matrix, x, y)

        # # Create a new rotatiom matrix
        # new_rotation_matrix = core.euler_angles_to_rotation_matrix(pitch, yaw, roll)
        # new_face_rt = np.eye(4)
        # new_face_rt[:3, :3] = new_rotation_matrix
        # # translation = face_rt[:3, 3]
        # # translation *= np.array([-1, 1, 1])
        # # translation = np.array([0, 0, -5])

        # # Compute the translation via reprojecting the nose as the origin
        # # translation = reproject_2d_to_3d(x, y, face_rt[2, 3], intrinsics)
        # translation = reproject_2d_to_3d(x, y, z, intrinsics)

        # # Compute the average left eye area
        # left_eye_landmarks = face_landmarks[LEFT_EYEAREA_LANDMARKS]
        # left_eye_landmarks = left_eye_landmarks[:, :3]
        # left_eye_center = np.mean(left_eye_landmarks, axis=0) / 75

        # new_face_rt[:3, 3] = translation
        # # face_origin = np.array([2.2, 2.5, 3, 1])  # 3D origin point in canonical face space
        # # face_origin = np.array([1e-3, 0, 0, 1])  # 3D origin point in canonical face space
        # # print(left_eye_center)
        # face_origin = np.array([left_eye_center[0], left_eye_center[1], left_eye_center[2], 1])  # 3D origin point in canonical face space
        
        # # Transform the face origin from canonical to world space
        # face_origin_3d = np.dot(new_face_rt, face_origin)
        # face_origin_3d = face_origin_3d[:3] / face_origin_3d[3]  # Homogeneous to Cartesian coordinates

        # # Project the point from 3D world coordinates to 2D image plane
        # face_origin = np.dot(intrinsics, face_origin_3d)
        # face_origin = face_origin[:2] / face_origin[2]  # Perspective divide to get 2D coordinates

        # # Draw the projected point on the frame
        # cv2.circle(frame, (int(face_origin[0]), int(face_origin[1])), 5, (0, 0, 255), -1)
        # # print(face_origin_3d[2], detection_results.face_landmarks[0][1].z)


        cv2.imshow('frame', imutils.resize(frame, width=1000))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
