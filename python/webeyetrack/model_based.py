import numpy as np
import math
import cv2

from .constants import *

def rotation_matrix_to_euler_angles(R):
    # Ensure the matrix is 3x3
    assert R.shape == (3, 3)
    
    # Extract pitch, yaw, roll from the rotation matrix
    pitch = np.arcsin(-R[2, 0])  # Pitch around X-axis
    yaw = np.arctan2(R[2, 1], R[2, 2])  # Yaw around Y-axis
    roll = np.arctan2(R[1, 0], R[0, 0])  # Roll around Z-axis (optional)
# 
    # Convert radians to degrees if necessary
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    roll = np.degrees(roll)

    return pitch, yaw, roll

def euler_angles_to_rotation_matrix(pitch, yaw, roll):

    # Convert degrees to radians
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    roll = np.radians(roll)

    # Compute rotation matrix from Euler angles
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch)],
                    [0, np.sin(pitch), np.cos(pitch)]])
    
    R_y = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                    [0, 1, 0],
                    [-np.sin(yaw), 0, np.cos(yaw)]])
    
    R_z = np.array([[np.cos(roll), -np.sin(roll), 0],
                    [np.sin(roll), np.cos(roll), 0],
                    [0, 0, 1]])
    
    R = np.dot(R_z, np.dot(R_y, R_x))
    
    return R

# def pitch_yaw_to_gaze_vector(pitch, yaw):
#     """
#     Converts pitch and yaw angles into a 3D gaze direction vector (unit vector),
#     with pitch=0 and yaw=0 corresponding to a gaze direction [0, 0, 1] (forward).

#     Arguments:
#     pitch -- pitch angle in degrees
#     yaw -- yaw angle in degrees

#     Returns:
#     A 3D unit gaze direction vector as a numpy array [x, y, z].
#     """
#     # Convert degrees to radians
#     pitch_rad = np.radians(pitch)
#     yaw_rad = np.radians(yaw)

#     # Calculate the 3D gaze vector using spherical-to-Cartesian transformation
#     z = np.cos(pitch_rad) * np.cos(yaw_rad)  # Z becomes the forward direction
#     x = np.cos(pitch_rad) * np.sin(yaw_rad)  # X is horizontal
#     y = np.sin(pitch_rad)                    # Y is vertical

#     # Return the 3D gaze vector
#     return np.array([x, y, z])

def pitch_yaw_to_gaze_vector(pitch, yaw):
    """
    Converts pitch and yaw angles into a 3D gaze direction vector (unit vector),
    with pitch=0 and yaw=0 corresponding to a gaze direction [0, 0, -1] (forward).

    Arguments:
    pitch -- pitch angle in degrees
    yaw -- yaw angle in degrees

    Returns:
    A 3D unit gaze direction vector as a numpy array [x, y, z].
    """
    # Convert degrees to radians
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)

    # Calculate the 3D gaze vector using spherical-to-Cartesian transformation
    z = -np.cos(pitch_rad) * np.cos(yaw_rad)  # Z becomes the negative forward direction
    x = np.cos(pitch_rad) * np.sin(yaw_rad)   # X is horizontal
    y = np.sin(pitch_rad)                     # Y is vertical

    # Return the 3D gaze vector
    return np.array([x, y, z])

# def vector_to_pitch_yaw(vector):
#     """
#     Converts a 3D gaze direction vector (unit vector) into pitch and yaw angles,
#     assuming [0, 0, 1] corresponds to pitch=0 and yaw=0 (forward direction).

#     Arguments:
#     vector -- 3D unit gaze direction vector as a numpy array [x, y, z].

#     Returns:
#     pitch -- pitch angle in degrees
#     yaw -- yaw angle in degrees
#     """
#     # Ensure the input vector is normalized (unit vector)
#     vector = vector / np.linalg.norm(vector)
    
#     # Extract components
#     x, y, z = vector
    
#     # Yaw (azimuth angle): the angle in the XZ plane from the Z-axis
#     yaw = np.arctan2(x, z)  # In radians, between -π and π
    
#     # Pitch (elevation angle): the angle from the XZ plane
#     pitch = np.arctan2(y, np.sqrt(x**2 + z**2))  # In radians, between -π/2 and π/2

#     # Convert radians to degrees
#     yaw_deg = np.degrees(yaw)
#     pitch_deg = np.degrees(pitch)
    
#     return pitch_deg, yaw_deg

def vector_to_pitch_yaw(vector):
    """
    Converts a 3D gaze direction vector (unit vector) into pitch and yaw angles,
    assuming [0, 0, -1] corresponds to pitch=0 and yaw=0 (forward direction).

    Arguments:
    vector -- 3D unit gaze direction vector as a numpy array [x, y, z].

    Returns:
    pitch -- pitch angle in degrees
    yaw -- yaw angle in degrees
    """
    # Ensure the input vector is normalized (unit vector)
    vector = vector / np.linalg.norm(vector)
    
    # Extract components
    x, y, z = vector
    
    # Yaw (azimuth angle): the angle in the XZ plane from the Z-axis
    yaw = np.arctan2(x, -z)  # In radians, between -π and π, Z is negative now
    
    # Pitch (elevation angle): the angle from the XZ plane
    pitch = np.arctan2(y, np.sqrt(x**2 + z**2))  # In radians, between -π/2 and π/2

    # Convert radians to degrees
    yaw_deg = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)
    
    return pitch_deg, yaw_deg

def get_rotation_matrix_from_vector(vec):
    """
    Generates a rotation matrix that aligns the Z-axis with the input 3D unit vector.
    """
    # Normalize the input vector to ensure it's a unit vector
    vec = vec / np.linalg.norm(vec)
    x, y, z = vec
    
    # Default Z-axis vector
    z_axis = np.array([0, 0, 1])
    
    # Cross product to find the axis of rotation
    axis = np.cross(z_axis, vec)
    axis_len = np.linalg.norm(axis)
    
    if axis_len != 0:
        axis = axis / axis_len  # Normalize the rotation axis
    
    # Angle between the Z-axis and the input vector
    angle = np.arccos(np.dot(z_axis, vec))
    
    # Compute rotation matrix using axis-angle formula (Rodrigues' rotation formula)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    return R


def compute_2d_origin(points):
    (cx, cy), radius = cv2.minEnclosingCircle(points.astype(np.float32))
    center = np.array([cx, cy], dtype=np.int32)
    return center

def create_perspective_matrix(aspect_ratio):
    k_degrees_to_radians = np.pi / 180.0

    # Initialize a 4x4 matrix filled with zeros
    perspective_matrix = np.zeros((4, 4), dtype=np.float32)

    # Standard perspective projection matrix calculations
    f = 1.0 / np.tan(k_degrees_to_radians * VERTICAL_FOV_DEGREES / 2.0)
    denom = 1.0 / (NEAR - FAR)

    # Populate the matrix values
    perspective_matrix[0, 0] = f / aspect_ratio
    perspective_matrix[1, 1] = f
    perspective_matrix[2, 2] = (NEAR + FAR) * denom
    perspective_matrix[2, 3] = -1.0
    perspective_matrix[3, 2] = 2.0 * FAR * NEAR * denom

    # Flip Y-axis if origin point location is top-left corner
    if ORIGIN_POINT_LOCATION == 'TOP_LEFT_CORNER':
        perspective_matrix[1, 1] *= -1.0

    return perspective_matrix

def convert_uv_to_xyz(perspective_matrix, u, v, z_relative):
    # Step 1: Convert normalized (u, v) to Normalized Device Coordinates (NDC)
    ndc_x = 2 * u - 1
    ndc_y = 1 - 2 * v

    # Step 2: Create the NDC point in homogeneous coordinates
    ndc_point = np.array([ndc_x, ndc_y, -1.0, 1.0])

    # Step 3: Invert the perspective matrix to go from NDC to world space
    inv_perspective_matrix = np.linalg.inv(perspective_matrix)

    # Step 4: Compute the point in world space (in homogeneous coordinates)
    world_point_homogeneous = np.dot(inv_perspective_matrix, ndc_point)

    # Step 5: Dehomogenize (convert from homogeneous to Cartesian coordinates)
    x = world_point_homogeneous[0] / world_point_homogeneous[3]
    y = world_point_homogeneous[1] / world_point_homogeneous[3]
    z = world_point_homogeneous[2] / world_point_homogeneous[3]

    # Step 6: Scale using the relative depth
    # Option A
    x_relative = -x #* z_relative
    y_relative = y #* z_relative
    # z_relative = z * z_relative

    # Option B
    # x_relative = x * z_relative
    # y_relative = y * z_relative
    # z_relative = z * z_relative

    return np.array([x_relative, y_relative, z_relative])

def convert_xyz_to_uv(perspective_matrix, x, y, z):
    # Step 1: Convert (x, y, z) to homogeneous coordinates (x, y, z, 1)
    world_point = np.array([x, -y, z, 1.0])
    # world_point = np.array([x, y, z, 1.0])

    # Step 2: Apply the perspective projection matrix
    ndc_point_homogeneous = np.dot(perspective_matrix, world_point)

    # Step 3: Dehomogenize to convert from homogeneous to Cartesian coordinates
    u_ndc = ndc_point_homogeneous[0] / ndc_point_homogeneous[3]
    v_ndc = ndc_point_homogeneous[1] / ndc_point_homogeneous[3]
    z_ndc = ndc_point_homogeneous[2] / ndc_point_homogeneous[3]

    # Step 4: Convert from NDC to normalized coordinates (u, v) in the range [0, 1]
    u = (u_ndc + 1) / 2
    v = (1 - v_ndc) / 2

    return u, v

def convert_xyz_to_uv_with_intrinsic(intrinsic_matrix, x, y, z):
    # Step 1: Create the 3D point in homogeneous coordinates
    point_3d = np.array([-x, -y, z, 1.0])

    # Step 2: Project the 3D point to the image plane using the intrinsic matrix
    # Remove the homogeneous component before applying K
    point_3d_camera = point_3d[:3]  # Only use x, y, z

    # Apply the intrinsic matrix to project to 2D
    projected_point_homogeneous = np.dot(intrinsic_matrix, point_3d_camera)

    # Step 3: Dehomogenize to convert to Cartesian coordinates (u, v)
    u = projected_point_homogeneous[0] / projected_point_homogeneous[2]
    v = projected_point_homogeneous[1] / projected_point_homogeneous[2]

    return np.array([u, v])


def estimate_inter_pupillary_distance_2d(facial_landmarks, height, width):
    data_2d_pairs = {
        'left': facial_landmarks[LEFT_EYE_LANDMARKS][:, :2] * np.array([width, height]),
        'right': facial_landmarks[RIGHT_EYE_LANDMARKS][:, :2] * np.array([width, height])
    }
    data_3d_pairs = {
        'left': facial_landmarks[LEFT_EYE_LANDMARKS][:, :3],
        'right': facial_landmarks[RIGHT_EYE_LANDMARKS][:, :3]
    }

    # Compute the 2D eye origin
    origins_2d = {}
    for k,v in data_2d_pairs.items():
        origins_2d[k] = compute_2d_origin(v)

    # Compute the 3D eye origin
    origins_3d = {}
    for k, v in data_3d_pairs.items():
        origins_3d[k] = np.mean(v, axis=0)

    # Compute the scaling factor between mediapipe canonical & world coordinate
    l, r = origins_3d['left'], origins_3d['right']
    canonical_ipd = np.sqrt(np.power(l[0] - r[0], 2) + np.power(l[1] - r[1],2) + np.power(l[2] - r[2], 2))

    # Compute the distance in 2d 
    l, r = origins_2d['left'], origins_2d['right']
    image_ipd = np.sqrt(np.power(l[0] - r[0], 2) + np.power(l[1] - r[1], 2))

    positions = {
        'eye_origins_2d': {
            'left': l,
            'right': r
        },
        'eye_origins_3d_canonical': {
            'left': origins_3d['left'],
            'right': origins_3d['right']
        }
    }
    distances = {
        'canonical_ipd_3d': canonical_ipd,
        'image_ipd': image_ipd
    }

    return (positions, distances)

def estimate_2d_3d_eye_face_origins(perspective_matrix, facial_landmarks, face_rt, height, width, intrinsics):

    # First, compute the inter-pupillary distance
    positions, distances = estimate_inter_pupillary_distance_2d(
        facial_landmarks, 
        height, 
        width
    )

    # Estimate the scale
    metric_scale = REAL_WORLD_IPD_CM * 10 / distances['canonical_ipd_3d']

    # Convert uvz to xyz
    relative_face_mesh = np.array([convert_uv_to_xyz(perspective_matrix, x[0], x[1], x[2]) for x in facial_landmarks[:, :3]])
    centroid = relative_face_mesh.mean(axis=0)
    demeaned_relative_face_mesh = relative_face_mesh.copy() # - centroid
    
    data_3d_pairs = {
        'left': demeaned_relative_face_mesh[LEFT_EYE_LANDMARKS][:, :3],
        'right': demeaned_relative_face_mesh[RIGHT_EYE_LANDMARKS][:, :3]
    }

    # Compute the 3D eye origin
    origins_3d = {}
    for k, v in data_3d_pairs.items():
        origins_3d[k] = np.mean(v, axis=0)

    # Compute the scaling factor between mediapipe per-world & world mm coordinate
    l, r = origins_3d['left'], origins_3d['right']
    per_frame_ipd = np.sqrt(np.power(l[0] - r[0], 2) + np.power(l[1] - r[1],2) + np.power(l[2] - r[2], 2))
    scale = (10 * REAL_WORLD_IPD_CM) / per_frame_ipd

    # Compute the depth
    theta = np.arctan(face_rt[0, 2] / face_rt[2, 2])
    focal_length_pixels = 1 / np.tan(np.deg2rad(VERTICAL_FOV_DEGREES) / 2) * height / 2
    depth_mm = (focal_length_pixels * REAL_WORLD_IPD_CM * 10 * np.cos(theta)) / distances['image_ipd'] * 2.25

    # Apply the scale
    scaled_demeaned_relative_face_mesh = demeaned_relative_face_mesh * scale

    # Returned to the position
    translation = np.array([0, 0, depth_mm])
    shifted_s_d_relative_face_mesh = scaled_demeaned_relative_face_mesh + translation
    
    # Compute the 3D bounding box dimensions of the shifted_s_d_relative_face_mesh
    min_xyz = np.min(shifted_s_d_relative_face_mesh, axis=0)
    max_xyz = np.max(shifted_s_d_relative_face_mesh, axis=0)
    distances = max_xyz - min_xyz
    # print(f"Distances: {distances}")

    # Estimate intrinsics based on width
    intrinsics = np.array([
        [width*1.5, 0, width / 2],
        [0, height*1.9, height / 2],
        [0, 0, 1]
    ])

    # Convert xyz back to uvz
    re_facial_landmarks = np.array([convert_xyz_to_uv_with_intrinsic(intrinsics, x[0], x[1], x[2]) for x in shifted_s_d_relative_face_mesh])

    # Draw the original facial (DEBUGGING)
    # draw_frame = frame.copy()
    # for (u,v), (nu, nv) in zip(facial_landmarks[:, :2], re_facial_landmarks[:, :2]):
    #     cv2.circle(draw_frame, (int(u * width), int(v * height)), 2, (0, 255, 0), -1)
    #     cv2.circle(draw_frame, (int(nu), int(nv)), 2, (0, 0, 255), -1)
    # cv2.imshow('draw', draw_frame)

    # Compute the average of the 2D eye origins
    face_origin = (positions['eye_origins_2d']['left'] + positions['eye_origins_2d']['right']) / 2
    tf_face_points = shifted_s_d_relative_face_mesh

    # Compute the eye gaze origin in metric space
    eye_g_o = {
        'left': tf_face_points[LEFT_EYE_LANDMARKS],
        'right': tf_face_points[RIGHT_EYE_LANDMARKS]
    }

    # Compute the 3D eye origin
    for k, v in eye_g_o.items():
        eye_g_o[k] = np.mean(v, axis=0)

    # Compute face gaze origin
    face_g_o = (eye_g_o['left'] + eye_g_o['right']) / 2

    return {
        'tf_face_points': tf_face_points,
        'face_origin_3d': face_g_o,
        'face_origin_2d': face_origin,
        'eye_origins_3d': eye_g_o,
        # 'eye_origins_3d': {'left': np.array([0,0,100]), 'right': np.array([0,0,100])},
        'eye_origins_2d': positions['eye_origins_2d']
    }

def estimate_gaze_vector_based_on_model_based(
        eyeball_centers, 
        eyeball_radius, 
        perspective_matrix, 
        inv_perspective_matrix, 
        facial_landmarks, 
        face_rt, 
        height, 
        width, 
        ear_threshold=EAR_THRESHOLD
    ):
    
    # Estimate the eyeball centers
    face_rt_copy = face_rt.copy()
    face_rt_copy[:3, 3] *= np.array([-1, -1, -1])

    # Must for gaze estimation
    gaze_vectors = {}
    eye_closed = {}

    # Visualization for debug
    eyeball_center_2d = {'left': None, 'right': None}
    # eyeball_radius_2d = {'left': None, 'right': None}

    for i, canonical_eyeball in zip(['left', 'right'], eyeball_centers):
        # if i == 'right':
        #     eye_closed[i] = True
        #     continue

        # Convert to homogenous
        eyeball_homogeneous = np.append(canonical_eyeball, 1)

        # Convert from canonical to camera space
        camera_eyeball = face_rt_copy @ eyeball_homogeneous
        sphere_center = camera_eyeball[:3]

        # Obtain the 2D eyeball center and radius
        screen_landmark_homogenous = perspective_matrix @ camera_eyeball
        eyeball_x_2d_n = screen_landmark_homogenous[0] / screen_landmark_homogenous[2]
        eyeball_y_2d_n = screen_landmark_homogenous[1] / screen_landmark_homogenous[2]
        eyeball_x_2d = (eyeball_x_2d_n + 1) * width / 2
        eyeball_y_2d = (eyeball_y_2d_n * -1 + 1) * height / 2
        eyeball_center_2d[i] = np.array([eyeball_x_2d, eyeball_y_2d])
        # eyeball_radius_2d[i] = 0.85 * (500/camera_eyeball[2]) # TODO: This is not correct since the camera's instrinsics are not the same

        # print(f"A - {i}: canonical_eyeball: {canonical_eyeball}, face_rt_copy: {face_rt_copy}, width, height: {width, height}, eyeball_center_2d: {eyeball_center_2d[i]}")

        # Draw the eyeball center and radius
        # cv2.circle(frame, (int(eyeball_x_2d), int(eyeball_y_2d)), 2, (0, 0, 255), -1)
        
        # First, determine if the eye is closed, by computing the EAR
        # EAR = ||p_2 - p_6|| + ||p_3 - p_5|| / (2 * ||p_1 - p_4||)
        EYE_EAR_LANDMARKS = LEFT_EYE_EAR_LANDMARKS if i == 'left' else RIGHT_EYE_EAR_LANDMARKS
        p1 = facial_landmarks[EYE_EAR_LANDMARKS[0], :2]
        p2 = facial_landmarks[EYE_EAR_LANDMARKS[1], :2]
        p3 = facial_landmarks[EYE_EAR_LANDMARKS[2], :2]
        p4 = facial_landmarks[EYE_EAR_LANDMARKS[3], :2]
        p5 = facial_landmarks[EYE_EAR_LANDMARKS[4], :2]
        p6 = facial_landmarks[EYE_EAR_LANDMARKS[5], :2]

        # Draw all the EAR landmarks
        # for j, landmark in enumerate(EYE_EAR_LANDMARKS):
        #     x, y = facial_landmarks[landmark, :2]
        #     cv2.circle(frame, (int(x * width), int(y * height)), 2, (0, 255, 0), -1)
        #     cv2.putText(frame, f"p{j+1}", (int(x * width), int(y * height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4))
        eye_closed[i] = False
        if ear < ear_threshold:
            eye_closed[i] = True
            continue

        # Compute the 3D pupil by using a line-sphere intersection problem
        # Reference: https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        # Convert from 0-1 to -1 to 1
        pupil = facial_landmarks[LEFT_IRIS_LANDMARKS[0], :3] if i == 'left' else facial_landmarks[RIGHT_IRIS_LANDMARKS[0], :3]
        pupil2d = np.array([pupil[0] * width, pupil[1] * height])
        ndc_y = 1 - (2 * pupil2d[1] / height)
        ndc_x = (2 * pupil2d[0] / width) - 1
        ndc_point = np.array([ndc_x, ndc_y, -1.0, 1.9])
        
        # Draw the pupil
        # cv2.circle(frame, (int(pupil2d[0]), int(pupil2d[1])), 2, (0, 255, 0), -1)

        # Compute the ray in 3D space
        world_point_homogeneous = np.dot(inv_perspective_matrix, ndc_point)
        world_point = world_point_homogeneous[:3] / world_point_homogeneous[3]
        ray_direction = world_point - np.array([0, 0, 0])
        ray_direction /= np.linalg.norm(ray_direction)  # Normalize the direction
        
        # Camera origin and Calculate intersection with the sphere
        camera_origin = np.array([0.0, 0.0, 0.0])
        oc = camera_origin - sphere_center

        # Solve the quadratic equation ax^2 + bx + c = 0
        discriminant = np.dot(ray_direction, oc) ** 2 - (np.dot(oc, oc) - eyeball_radius ** 2)

        if discriminant < 0:
            # No real intersections
            # cv2.imshow('frame', imutils.resize(frame, width=1000))

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            continue
            # return None

        # Calculate the two possible intersection points
        t1 = np.dot(-ray_direction, oc) - np.sqrt(discriminant)
        t2 = np.dot(-ray_direction, oc) + np.sqrt(discriminant)

        # We are interested in the first intersection that is in front of the camera
        pupil_3d = None
        if t1 > t2:
            pupil_3d = camera_origin + t1 * ray_direction
        else:
            pupil_3d = camera_origin + t2 * ray_direction

        # Compute the gaze direction based on the eyeball center and 3D pupil
        gaze_vector = pupil_3d - sphere_center
        gaze_vector /= np.linalg.norm(gaze_vector)

        # DEBUG: For debugging purposes, make the gaze vector straight to the z-axis
        # gaze_vector = np.array([0, 0, -1])
        # gaze_vector = np.array([-0.1, 0.0, -0.9])
        # gaze_vector /= np.linalg.norm(gaze_vector)
        # gaze_vector = np.array([-0.04122334, -0.25422794, -0.96626538])

        # Convert gaze vector to pitch and yaw to correct
        pitch, yaw = vector_to_pitch_yaw(gaze_vector)
        pitch, yaw = -pitch, yaw
        gaze_vector = pitch_yaw_to_gaze_vector(pitch, yaw)

        # Store
        gaze_vectors[i] = gaze_vector

    # Compute the average gaze vector
    if 'left' in gaze_vectors and 'right' in gaze_vectors:
        face_gaze_vector = (gaze_vectors['left'] + gaze_vectors['right'])
        face_gaze_vector /= np.linalg.norm(face_gaze_vector)
    elif 'left' in gaze_vectors:
        face_gaze_vector = gaze_vectors['left']
        gaze_vectors['right'] = np.array([0,0,-1])
    elif 'right' in gaze_vectors:
        face_gaze_vector = gaze_vectors['right']
        gaze_vectors['left'] = np.array([0,0,-1])
    else:
        face_gaze_vector = np.array([0,0,-1])
        gaze_vectors['left'] = np.array([0,0,-1])
        gaze_vectors['right'] = np.array([0,0,-1])

    # Debugging purposes
    # cv2.imshow('debug_frame', frame)

    return {
        'face': face_gaze_vector,
        'eyes': {
            'is_closed': eye_closed,
            'vector': gaze_vectors,
            'meta_data': {
                'left': {
                    'eyeball_center_2d': eyeball_center_2d['left'],
                    # 'eyeball_radius_2d': eyeball_radius_2d['left']
                },
                'right': {
                    'eyeball_center_2d': eyeball_center_2d['right'],
                    # 'eyeball_radius_2d': eyeball_radius_2d['right']
                }
            }
        }
    }

def estimate_gaze_vector_based_on_eye_landmarks(facial_landmarks, face_rt, height, width):

    # Compute the bbox by using the edges of the each eyes
    left_2d_eye_px = facial_landmarks[LEFT_EYEAREA_LANDMARKS, :2] * np.array([height, width])
    left_2d_eyelid_px = facial_landmarks[LEFT_EYELID_LANDMARKS, :2] * np.array([height, width])
    left_2d_iris_px = facial_landmarks[LEFT_IRIS_LANDMARKS, :2] * np.array([height, width])
    
    right_2d_eye_px = facial_landmarks[RIGHT_EYEAREA_LANDMARKS, :2] * np.array([height, width])
    right_2d_eyelid_px = facial_landmarks[RIGHT_EYELID_LANDMARKS, :2] * np.array([height, width])
    right_2d_iris_px = facial_landmarks[RIGHT_IRIS_LANDMARKS, :2] * np.array([height, width])

    # Apply face_rt to the EYEBALL_CENTERs to get the 3D position
    # canonical_lefteye_center_homo = np.append(LEFT_EYEBALL_CENTER, 1)
    # canonical_righteye_center_homo = np.append(RIGHT_EYEBALL_CENTER, 1)
    # left_eye_ball_center = np.dot(face_rt[:3, :3], LEFT_EYEBALL_CENTER) + face_rt[:3, 3]
    # right_eye_ball_center = np.dot(face_rt[:3, :3], RIGHT_EYEBALL_CENTER) + face_rt[:3, 3]

    # tf_lefteye_center_homo = face_rt @ canonical_lefteye_center_homo
    # tf_lefteye_center = tf_lefteye_center_homo[:3] / tf_lefteye_center_homo[-1]
    # u_normalized = tf_lefteye_center[0] / width
    # v_normalized = tf_lefteye_center[1] / height
    # z_relative = tf_lefteye_center[2] / tf_lefteye_center[0]
    # actual_UVZ = facial_landmarks[LEFT_EYEAREA_LANDMARKS[0], :3]
    # UVZ = (u_normalized, v_normalized, z_relative)

    # 3D
    left_eye_fl = facial_landmarks[LEFT_EYELID_LANDMARKS, :3]
    right_eye_fl = facial_landmarks[RIGHT_EYELID_LANDMARKS, :3]

    left_landmarks = [
        left_2d_eye_px, 
        left_2d_eyelid_px,
    ]
    right_landmarks = [
        right_2d_eye_px, 
        right_2d_eyelid_px,
    ]

    eye_closed = {}
    eye_images = {}
    gaze_vectors = {}
    gaze_origins_2d = {}
    headpose_corrected_eye_center = {}
    for i, (eye, eyelid) in {'left': left_landmarks, 'right': right_landmarks}.items():
        centroid = np.mean(eye, axis=0)
        actual_width = np.abs(eye[1,0] - eye[0, 0])
        width = actual_width * (1 + EYE_PADDING_WIDTH)
        height = width * EYE_HEIGHT_RATIO

        gaze_origins_2d[i] = centroid

        # Determine if closed by the eyelid
        eyelid_width = np.abs(eyelid[0,0] - eyelid[1, 0])
        eyelid_height = np.abs(eyelid[3,1] - eyelid[2, 1])
        is_closed = False

        # Determine if the eye is closed by the ratio of the height based on the width
        if eyelid_height / eyelid_width < 0.05:
            is_closed = True

        if width == 0 or height == 0:
            continue

        # Draw if the eye is closed on the top left corner
        eye_closed[i] = is_closed
        if is_closed:
            continue

        # Shift the IRIS landmarks to the cropped eye
        iris_px = left_2d_iris_px if i == 'left' else right_2d_iris_px
        shifted_iris_px = iris_px - np.array([int(centroid[0] - width/2), int(centroid[1] - height/2)])
        iris_center = shifted_iris_px[0]
        eye_center = np.array([width/2, height/2])

        # Compute the radius of the iris
        left_iris_radius = np.linalg.norm(iris_center - shifted_iris_px[2])
        right_iris_radius = np.linalg.norm(iris_center - shifted_iris_px[4])
        iris_radius = np.mean([left_iris_radius, right_iris_radius]) # 10

        # Shift the eye center by the headpose
        headrot = face_rt[:3, :3]
        pitch, yaw, roll = rotation_matrix_to_euler_angles(headrot)
        pitch, yaw = yaw, -pitch # Swap the pitch and yaw
        size = actual_width / 4
        pitch = (pitch * np.pi / 180)
        yaw = (yaw * np.pi / 180)
        x3 = size * (math.sin(yaw))
        y3 = size * (-math.cos(yaw) * math.sin(pitch))
        # frame = draw_axis(frame, -pitch, yaw, 0, int(face_origin[0]), int(face_origin[1]), 100)

        old_iris_center = iris_px[0]
        # cv2.circle(frame, (int(old_iris_center[0]), int(old_iris_center[1])), 2, (0, 0, 255), -1)
        shifted_iris_center = old_iris_center + np.array([int(x3), int(y3)])
        # cv2.circle(frame, (int(shifted_iris_center[0]), int(shifted_iris_center[1])), 2, (0, 255, 0), -1)
        # cv2.line(frame, (int(old_iris_center[0]), int(old_iris_center[1])), (int(shifted_iris_center[0]), int(shifted_iris_center[1])), (0, 255, 0), 1)

        # Shifting the eye_center by the headpose
        # print(f"Eye center: {eye_center}, shift: {np.array([x3, y3])}, new: {eye_center + np.array([x3, y3])}")
        eye_center = eye_center + np.array([x3, y3])
        headpose_corrected_eye_center[i] = eye_center

        # Based on the direction and magnitude of the line, compute the gaze direction
        # Compute 2D vector from eyeball center to iris center
        # gaze_vector_2d = shifted_iris_px[0] - iris_px[0]
        gaze_vector_2d = iris_center - eye_center
        # gaze_vector_2d = np.array([0,0])

        # Estimate the depth (Z) based on the 2D vector length
        # z_depth = EYEBALL_RADIUS / np.linalg.norm(gaze_vector_2d)
        z_depth = 2.0
        # Estimate the depth (Z) based on the size of the iris
        # z_depth = EYEBALL_RADIUS

        # Compute yaw (horizontal rotation)
        yaw = np.arctan2(gaze_vector_2d[0] / iris_radius, z_depth) * (180 / np.pi)  # Convert from radians to degrees

        # Compute pitch (vertical rotation)
        pitch = np.arctan2(gaze_vector_2d[1] / iris_radius, z_depth) * (180 / np.pi)  # Convert from radians to degrees

        # Convert the pitch and yaw to a 3D vector
        gaze_vector = pitch_yaw_to_gaze_vector(pitch, yaw)
        gaze_vectors[i] = gaze_vector

        # # Compute 3D gaze origin
        # eye_fl = left_eye_fl if i == 'left' else right_eye_fl
        # gaze_origin = np.mean(eye_fl, axis=0)
        # gaze_origins[i] = gaze_origin

    # Compute average gaze origin 2d
    face_origin_2d = (gaze_origins_2d['left'] + gaze_origins_2d['right']) / 2

    # Draw the headpose on the frame
    # headrot = face_rt[:3, :3]
    # pitch, yaw, roll = rotation_matrix_to_euler_angles(headrot)
    # pitch, yaw = yaw, pitch
    # face_origin = face_origin_2d
    # frame = draw_axis(frame, -pitch, yaw, -roll, int(face_origin[0]), int(face_origin[1]), 100)

    # cv2.imshow('frame', frame)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     exit()

    # Compute the average gaze vector
    if 'left' in gaze_vectors and 'right' in gaze_vectors:
        face_gaze_vector = (gaze_vectors['left'] + gaze_vectors['right'])
        face_gaze_vector /= np.linalg.norm(face_gaze_vector)
    elif 'left' in gaze_vectors:
        face_gaze_vector = gaze_vectors['left']
        gaze_vectors['right'] = np.array([0,0,1])
        headpose_corrected_eye_center['right'] = None
    elif 'right' in gaze_vectors:
        face_gaze_vector = gaze_vectors['right']
        gaze_vectors['left'] = np.array([0,0,1])
        headpose_corrected_eye_center['left'] = None
    else:
        face_gaze_vector = np.array([0,0,1])
        gaze_vectors['left'] = np.array([0,0,1])
        gaze_vectors['right'] = np.array([0,0,1])
        headpose_corrected_eye_center['left'] = None
        headpose_corrected_eye_center['right'] = None

    return {
        'face': face_gaze_vector,
        'eyes': {
            'is_closed': eye_closed,
            'vector': gaze_vectors,
            'meta_data': {
                'left': {
                    'headpose_corrected_eye_center': headpose_corrected_eye_center['left']
                },
                'right': {
                    'headpose_corrected_eye_center': headpose_corrected_eye_center['right']
                }
            }
        }
    }

def estimate_gaze_vector_based_on_eye_blendshapes(face_blendshapes, face_rt):

    # Get the transformation matrix
    # Invert the y and z axis
    transform = face_rt.copy()
    transform = np.diag([-1, 1, 1, 1]) @ transform
    
    # Compute the iris direction
    gaze_directions = {}
    for option, value in {'left': LEFT_BLENDSHAPES, 'right': RIGHT_BLENDSHAPES}.items():
        blendshapes = face_blendshapes
        look_in, look_out, look_up, look_down = ([blendshapes[i] for i in value])
        hfov = np.deg2rad(HFOV)
        vfov = np.deg2rad(VFOV)

        rx = hfov * 0.5 * (look_down - look_up)
        ry = vfov * 0.5 * (look_in - look_out) * (1 if option == 'left' else -1)

        # Create euler angle
        euler_angles = np.array([rx, ry, 0])

        # # Convert to rotation matrix
        rotation_matrix = cv2.Rodrigues(euler_angles)[0]

        # Compute the gaze direction
        gaze_directions[option] = rotation_matrix

    # Apply the rotation to the gaze direction
    for k, v in gaze_directions.items():
        rotation_matrix = transform[:3, :3]
        gaze_directions[k] = v.dot(rotation_matrix)

    # Compute the gaze direction by apply the rotation to a [0,0,-1] vector
    gaze_vectors = {
        'left': np.array([0,0,-1]),
        'right': np.array([0,0,-1])
    }
    for k, v in gaze_directions.items():
        gaze_vectors[k] = v.dot(gaze_vectors[k])

    # Compute the average gaze vector
    face_gaze_vector = (gaze_vectors['left'] + gaze_vectors['right'])
    face_gaze_vector /= np.linalg.norm(face_gaze_vector)

    return {
        'face': face_gaze_vector,
        'eyes': {
            'is_closed': {'left': False, 'right': False},
            'vector': gaze_vectors,
            'meta_data': {
                'left': {},
                'right': {}
            }
        }            
    }

def screen_plane_intersection(o, d, screen_R, screen_t):
    """
    Calculate the intersection of a gaze direction with a screen plane.
    
    Parameters:
    - o: Gaze origin (3D coordinates)
    - d: Gaze direction (3D coordinates)
    - screen_R: Rotation vector (Rodrigues vector) for the screen
    - screen_t: Translation vector for the screen
    
    Returns:
    - pog_mm: 2D point of gaze on the screen in millimeters (x, y)
    """

    # Obtain rotation matrix from the Rodrigues vector
    R_matrix, _ = cv2.Rodrigues(screen_R)  # screen_R should be a 3D vector (Rodrigues rotation)
    inv_R_matrix = np.linalg.inv(R_matrix)  # Inverse of the rotation matrix

    # Transform gaze origin and direction to screen coordinates
    o_s = np.dot(inv_R_matrix, (o - screen_t.T[0]))
    d_s = np.dot(inv_R_matrix, d)

    # Screen plane: z = 0 (assumed to be at origin with a normal vector along z-axis)
    a_s = np.array([0, 0, 0], dtype=np.float32)  # Point on the screen plane
    n_s = np.array([0, 0, 1], dtype=np.float32)  # Normal vector of the screen plane

    # Calculate the distance (lambda) to the screen plane
    lambda_ = np.dot(a_s - o_s, n_s) / np.dot(d_s, n_s)

    # Calculate the intersection point (3D)
    p = o_s + lambda_ * d_s

    # Keep only the x and y coordinates (2D point of gaze on screen)
    pog_mm = p[:2]

    return pog_mm

def screen_plane_intersection_2(o, d):
    """
    Calculate the intersection of a gaze direction with a screen plane.
    
    Parameters:
    - o: Gaze origin (3D coordinates)
    - d: Gaze direction (3D coordinates)
    
    Returns:
    - pog_mm: 2D point of gaze on the screen in millimeters (x, y)
    """

    # Screen plane: z = 0 (assumed to be at origin with a normal vector along z-axis)
    a = np.array([0, 0, 0], dtype=np.float32)  # Point on the screen plane
    n = np.array([0, 0, 1], dtype=np.float32)  # Normal vector of the screen plane

    # Calculate the distance (lambda) to the screen plane
    lambda_ = np.dot(a - o, n) / np.dot(d, n)

    # Calculate the intersection point (3D)
    p = o + lambda_ * d

    # Keep only the x and y coordinates (2D point of gaze on screen)
    pog_mm = p[:2]

    return pog_mm

def compute_pog(gaze_origins, gaze_vectors, screen_R, screen_t, screen_width_mm, screen_height_mm, screen_width_px, screen_height_px):
    
    # Perform intersection with plane using gaze origin and vector
    # c for camera, s for screen
    left_pog_mm_c = screen_plane_intersection_2(
        gaze_origins['eye_origins_3d']['left'],
        gaze_vectors['eyes']['vector']['left'],
    )
    right_pog_mm_c = screen_plane_intersection_2(
        gaze_origins['eye_origins_3d']['right'],
        gaze_vectors['eyes']['vector']['right'],
    )

    # Then convert the PoG to screen coordinates
    # Obtain rotation matrix from the Rodrigues vector
    R_matrix, _ = cv2.Rodrigues(screen_R)  # screen_R should be a 3D vector (Rodrigues rotation)
    inv_R_matrix = np.linalg.inv(R_matrix)  # Inverse of the rotation matrix

    # Pad the points from 2 to 3 dimensions
    pad_left_pog_mm_c = np.append(left_pog_mm_c, 0)
    pad_right_pog_mm_c = np.append(right_pog_mm_c, 0)

    # Transform gaze origin and direction to screen coordinates
    left_pog_mm_s = np.dot(inv_R_matrix, (pad_right_pog_mm_c - screen_t.T[0]))
    right_pog_mm_s = np.dot(inv_R_matrix, (pad_left_pog_mm_c - screen_t.T[0]))

    # Remove the z-axis
    left_pog_mm_s = left_pog_mm_s[:2]
    right_pog_mm_s = right_pog_mm_s[:2]

    # Convert mm to normalized coordinates
    left_pog_norm = np.array([left_pog_mm_s[0] / screen_width_mm, left_pog_mm_s[1] / screen_height_mm])
    right_pog_norm = np.array([right_pog_mm_s[0] / screen_width_mm, right_pog_mm_s[1] / screen_height_mm])

    # Convert normalized coordinates to pixel coordinates
    left_pog_px = np.array([left_pog_norm[0] * screen_width_px, left_pog_norm[1] * screen_height_px])
    right_pog_px = np.array([right_pog_norm[0] * screen_width_px, right_pog_norm[1] * screen_height_mm])

    return {
        'face_pog_mm_c': (left_pog_mm_c + right_pog_mm_c) / 2,
        'face_pog_mm_s': (left_pog_mm_s + right_pog_mm_s) / 2,
        'face_pog_norm': (left_pog_norm + right_pog_norm) / 2,
        'face_pog_px': (left_pog_px + right_pog_px) / 2,
        'eye': {
            'left_pog_mm_c': left_pog_mm_c,
            'left_pog_mm_s': left_pog_mm_s,
            'left_pog_norm': left_pog_norm,
            'left_pog_px': left_pog_px,
            'right_pog_mm_c': right_pog_mm_c,
            'right_pog_mm_s': right_pog_mm_s,
            'right_pog_norm': right_pog_norm,
            'right_pog_px': right_pog_px
        }
    }