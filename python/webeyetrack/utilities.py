from typing import Optional

import numpy as np
import open3d as o3d
import trimesh
import cv2

MAX_STEP_CM = 5
from .constants import *

########################################################################################
# Utilities
########################################################################################

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

def create_rotation_matrix(rotation):
    """
    Creates a rotation matrix with deg vector.
    
    Parameters:
    - rotation (list or np.array): Rotation vector in degrees [pitch, yaw, roll].
    
    Returns:
    - rotation matrix(np.array): 3x3 transformation matrix.
    """
    # Convert rotation from degrees to radians
    pitch, yaw, roll = np.radians(rotation)
    
    # Rotation matrices
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    
    R_y = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    
    R_z = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    
    # Combine rotations into a single matrix (R = Rz * Ry * Rx)
    R = R_z @ R_y @ R_x
    return R

def create_transformation_matrix(scale, translation, rotation):
    """
    Creates a transformation matrix with scaling, translation, and rotation.
    
    Parameters:
    - scale (float): Scaling scalar value.
    - translation (list or np.array): Translation vector in cm [tx, ty, tz].
    - rotation (list or np.array): Rotation vector in degrees [pitch, yaw, roll].
    
    Returns:
    - transformation_matrix (np.array): 4x4 transformation matrix.
    """
    # Convert the rotation vector to matrix
    R = create_rotation_matrix(rotation)
     
    # Apply scaling to the rotation matrix
    R *= scale
    
    # Create the 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R  # Top-left 3x3 part is the rotation (scaled)
    transformation_matrix[:3, 3] = translation  # Last column is the translation vector
    
    return transformation_matrix

OPEN3D_RT = create_transformation_matrix(1, [0,0,50], [0,180,180])

def transform_for_3d_scene(pts):
    """
    Apply a RT transformation to the points to get the desired 3D scene.
    """
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    transformed_pts_h = (OPEN3D_RT @ pts_h.T).T
    return transformed_pts_h[:, :3]

def estimate_camera_intrinsics(frame, fov_x=None):
    """
    Estimate focal length and camera intrinsic parameters.
    
    Parameters:
    - frame: NumPy array representing the image.
    - fov_x: Horizontal field of view of the camera in degrees (optional).
    
    Returns:
    - K: Intrinsics matrix (3x3).
    """
    h, w = frame.shape[:2]
    
    # Assume optical center is at the image center
    c_x = w / 2
    c_y = h / 2
    
    if fov_x is not None:
        # Convert FOV from degrees to radians
        fov_x_rad = np.radians(fov_x)
        # Estimate focal length in pixels
        f_x = w / (2 * np.tan(fov_x_rad / 2))
        f_y = f_x  # Assume square pixels (f_x = f_y)
    else:
        # If no FOV is provided, assume a generic focal length
        f_x = f_y = w  # Rough estimate (assuming 1 pixel ≈ 1 focal length)
    
    # Construct the camera intrinsic matrix
    K = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0,  0,   1]
    ])
    
    return K

def refine_depth_by_radial_magnitude(
    final_projected_pts: np.ndarray,
    detected_2d: np.ndarray,
    old_z: float,
    alpha: float = 0.5,
    frame: Optional[np.ndarray] = None,
) -> float:
    """
    Refines the face depth (Z) by comparing the radial 2D magnitude
    from the center of the face in 'final_projected_pts' versus
    the center of the face in 'detected_2d'.

    Args:
      final_projected_pts: (N, 2) the 2D projection of the canonical mesh
                          AFTER we've aligned X/Y.
      detected_2d:        (N, 2) the detected landmarks in pixel coords.
      old_z:              float, the current guess for Z (negative if forward).
      alpha:              a blending factor [0..1]. 0 -> no update, 1 -> full update.

    Returns:
      new_z: float, the updated depth
    """
    # Make a copy of the frame
    if frame is not None:
        draw_frame = frame.copy()
    else:
        draw_frame = None

    # Compute the centroid of the detected 2D points
    detected_center = detected_2d.mean(axis=0)
    total_distance = 0
    # Draw the center
    # cv2.circle(draw_frame, tuple(detected_center.astype(np.int32)), 10, (0, 255, 255), -1)

    # For each landmark pair, draw the lines between
    for i in range(len(final_projected_pts)):
        p1 = final_projected_pts[i]
        p2 = detected_2d[i]

        # Determine if the line is pointing towards the center
        # of the detected face or away from it.
        # Vector from p1 to p2
        v = p2 - p1
        v_norm = np.linalg.norm(v)

        # Vector from p1 to the detected center
        c = detected_center - p1
        dot_product = np.dot(v, c)
        if dot_product < 0:
            # The line is pointing towards the center
            # Draw the line in red
            color = (0, 0, 255)
            total_distance -= v_norm
        else:
            # The line is pointing away from the center
            # Draw the line in green
            color = (0, 255, 0)
            total_distance += v_norm

        # cv2.line(draw_frame, tuple(p1.astype(np.int32)), tuple(p2.astype(np.int32)), color, 2)

    distance_per_point = total_distance / len(final_projected_pts)
    # print(f"Distance per point: {distance_per_point}")

    # Use the total distance to update the depth
    delta = 1e-1 * distance_per_point
    safe_delta = max(-MAX_STEP_CM, min(MAX_STEP_CM, delta))
    new_z = old_z + safe_delta

    return new_z, draw_frame

def partial_procrustes_translation_2d(canonical_2d, detected_2d):
    # c_center = canonical_2d.mean(axis=0)
    # d_center = detected_2d.mean(axis=0)
    # return d_center - c_center
    c_nose = canonical_2d[4]
    d_nose = detected_2d[4]
    return d_nose - c_nose

def line_sphere_intersection(line_origin, line_direction, sphere_center, sphere_radius):
    # line_origin = np.array([0, 0, 0])
    
    # Camera origin and Calculate intersection with the sphere
    oc = line_origin - sphere_center

    # Solve the quadratic equation ax^2 + bx + c = 0
    discriminant = np.dot(line_direction, oc) ** 2 - (np.dot(oc, oc) - sphere_radius ** 2)

    if discriminant < 0:
        return None

    # Calculate the two possible intersection points
    t1 = np.dot(-line_direction, oc) - np.sqrt(discriminant)
    t2 = np.dot(-line_direction, oc) + np.sqrt(discriminant)

    # We are interested in the first intersection that is in front of the camera
    intersection_pt = None
    if t1 > t2:
    # if abs(t1) < abs(t2):
        intersection_pt = line_origin + t1 * line_direction
    else:
        intersection_pt = line_origin + t2 * line_direction
    return intersection_pt

def visualize_line_sphere_intersection(line_origin, line_direction, sphere_center, sphere_radius, intersection_pt):

    # Visualize the line sphere problem using Trimesh
    scene = trimesh.Scene()
    eyeball_mesh = trimesh.creation.icosphere(subdivisions=3, radius=sphere_radius)
    eyeball_mesh.apply_translation(sphere_center)
    line = np.stack([line_origin, line_direction * 100]).reshape((-1, 2, 3))
    path = trimesh.load_path(line)
    colors = np.array([[255, 0, 255]])
    path.colors = colors
    intersection = trimesh.creation.icosphere(subdivisions=3, radius=sphere_radius * 0.25)
    intersection.apply_translation(intersection_pt)
    intersection.visual.face_colors = [255, 0, 0]

    # Draw the xyz axis with paths
    length = 1
    x_line = np.stack([np.array([0, 0, 0]), np.array([length, 0, 0])]).reshape((-1, 2, 3))
    y_line = np.stack([np.array([0, 0, 0]), np.array([0, length, 0])]).reshape((-1, 2, 3))
    z_line = np.stack([np.array([0, 0, 0]), np.array([0, 0, length])]).reshape((-1, 2, 3))
    x_path = trimesh.load_path(x_line)
    y_path = trimesh.load_path(y_line)
    z_path = trimesh.load_path(z_line)
    x_path.colors = np.array([[255, 0, 0]])
    y_path.colors = np.array([[0, 255, 0]])
    z_path.colors = np.array([[0, 0, 255]])
    scene.add_geometry(x_path)
    scene.add_geometry(y_path)
    scene.add_geometry(z_path)

    scene.add_geometry(eyeball_mesh)
    scene.add_geometry(path)
    scene.add_geometry(intersection)
    scene.show()
    exit(0)

def image_shift_to_3d(shift_2d, depth_z, K):
    fx = K[0, 0]
    fy = K[1, 1]
    dx_3d = shift_2d[0] * (depth_z / fx)
    dy_3d = shift_2d[1] * (depth_z / fy)
    return np.array([dx_3d, dy_3d, 0.0], dtype=np.float32)

def load_canonical_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.triangles))
    return mesh

def transform_3d_to_3d(pts_3d, rt_matrix):
    # same as you had before, with perspective divide...
    num_points = len(pts_3d)
    pts_3d_h= np.hstack([
        pts_3d, 
        np.ones((num_points, 1), dtype=np.float32)
    ])
    transformed_points_h = (rt_matrix @ pts_3d_h.T).T
    transformed_xyz = transformed_points_h[:, :3]
    return transformed_xyz

def transform_3d_to_2d(camera_pts_3d, K):
    camera_space = (K @ camera_pts_3d.T).T

    eps = 1e-6
    zs = np.where(np.abs(camera_space[:, 2]) < eps, eps, camera_space[:, 2])
    camera_space[:, 0] /= zs
    camera_space[:, 1] /= zs

    projected_points = camera_space[:, :2]
    projected_points = projected_points.astype(np.int32)
    return projected_points

########################################################################################
# Open3D Loading
########################################################################################

def load_canonical_mesh(visual=None):
    # 1) Load canonical mesh
    # mesh_path = str(GIT_ROOT / 'python' / 'assets' / 'face_model_with_iris.obj')
    mesh_path = str(GIT_ROOT / 'python' / 'assets' / 'canonical_face_model.obj')
    canonical_mesh = trimesh.load(mesh_path, force='mesh')
    face_width_cm = 14
    canonical_norm_pts_3d = np.asarray(canonical_mesh.vertices, dtype=np.float32) * face_width_cm * np.array([-1, 1, -1])
    face_mesh = o3d.geometry.TriangleMesh()
    face_mesh.vertices = o3d.utility.Vector3dVector(np.array(canonical_mesh.vertices))
    face_mesh.triangles = o3d.utility.Vector3iVector(canonical_mesh.faces)
    face_mesh_lines = o3d.geometry.LineSet.create_from_triangle_mesh(face_mesh)

    # Generate colors for the face mesh, with all being default color except for the iris
    # colors = np.array([[3/256, 161/256, 252/256, 1] for _ in range(len(canonical_mesh.vertices))])
    # for i in IRIS_LANDMARKS:
    #     colors[i] = [0, 0, 0, 0]
    # face_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    if visual is not None:
        visual.add_geometry(face_mesh)
        visual.add_geometry(face_mesh_lines)

    return face_mesh, face_mesh_lines

def load_3d_axis(visual=None):
    
    # Define points for the origin and endpoints of the axes
    points = [
        [0, 0, 0],  # Origin
        [1, 0, 0],  # X-axis end
        [0, 1, 0],  # Y-axis end
        [0, 0, 1],  # Z-axis end
    ]

    # Define lines connecting the origin to each axis endpoint
    lines = [
        [0, 1],  # X-axis
        [0, 2],  # Y-axis
        [0, 3],  # Z-axis
    ]

    # Define colors for the axes: Red for X, Green for Y, Blue for Z
    colors = [
        [1, 0, 0],  # X-axis is red
        [0, 1, 0],  # Y-axis is green
        [0, 0, 1],  # Z-axis is blue
    ]

    # Create the LineSet object
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    if visual is not None:
        visual.add_geometry(line_set)

    return line_set

def load_eyeball_model(visual=None):
    
    # Load eyeball model
    eyeball_mesh_fp = GIT_ROOT / 'python' / 'assets' / 'eyeball' / 'eyeball_model_simplified.obj'
    assert eyeball_mesh_fp.exists()
    eyeball_diameter_cm = 2.5
    eyeball_meshes = {}
    eyeball_R = {}
    iris_3d_pt = {}
    for i in ['left', 'right']:
        eyeball_mesh = o3d.io.read_triangle_mesh(str(eyeball_mesh_fp), True)
        vertices = np.array(eyeball_mesh.vertices)
        min_x, min_y, min_z = vertices.min(axis=0)
        max_x, max_y, max_z = vertices.max(axis=0)
        center = np.array([min_x + max_x, min_y + max_y, min_z + max_z]) / 2
        vertices -= center
        min_x, min_y, min_z = vertices.min(axis=0)
        max_x, max_y, max_z = vertices.max(axis=0)
        range_x, range_y, range_z = max_x - min_x, max_y - min_y, max_z - min_z
        latest_range = max(range_x, range_y, range_z)
        norm_vertices = vertices / range_y

        scaled_vertices = norm_vertices * eyeball_diameter_cm
        eyeball_mesh.vertices = o3d.utility.Vector3dVector(scaled_vertices)
        eyeball_mesh.compute_vertex_normals()
        eyeball_meshes[i] = eyeball_mesh
        eyeball_R[i] = np.eye(3)
        iris_3d_pt[i] = o3d.geometry.PointCloud()

    if visual is not None:
        visual.add_geometry(eyeball_meshes['left'])
        visual.add_geometry(eyeball_meshes['right'])
        visual.add_geometry(iris_3d_pt['left'])
        visual.add_geometry(iris_3d_pt['right'])

    return eyeball_meshes, iris_3d_pt, eyeball_R

def load_camera_frustrum(w_ratio, h_ratio, visual=None):
    
    # Add a camera frustrum of the webcam
    # Frustum parameters
    frustrum_scale = 1.0  # Scale factor for the frustum
    origin = np.array([0, 0, 0])  # Camera location
    near_plane_dist = 0.5 # Distance to the near plane
    far_plane_dist = 1.0  # Distance to the far plane
    frustum_width = w_ratio / frustrum_scale    # Width of the frustum at the far plane
    frustum_height = h_ratio / frustrum_scale   # Height of the frustum at the far plane

    # Define points: camera origin and 4 points at the far plane
    points = np.array([
        origin,
        [frustum_width, frustum_height, far_plane_dist],   # Top-right
        [-frustum_width, frustum_height, far_plane_dist],  # Top-left
        [-frustum_width, -frustum_height, far_plane_dist], # Bottom-left
        [frustum_width, -frustum_height, far_plane_dist]   # Bottom-right
    ])
    transformed_pts = transform_for_3d_scene(points)

    # Define lines to form the frustum
    lines = [
        [0, 1],  # Origin to top-right
        [0, 2],  # Origin to top-left
        [0, 3],  # Origin to bottom-left
        [0, 4],  # Origin to bottom-right
        [1, 2],  # Top edge
        [2, 3],  # Left edge
        [3, 4],  # Bottom edge
        [4, 1]   # Right edge
    ]

    # Set color for each line (optional)
    colors = [[1, 0, 0] for _ in range(len(lines))]  # Green color

    # Create the LineSet object
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(transformed_pts)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector(colors)

    # Add the frustum to the visualizer
    if visual is not None:
        visual.add_geometry(frustum)

    return frustum

def load_screen_rect(visual, screen_width_mm, screen_height_mm):
    
    # Screen Display
    rw, rh = screen_width_mm / 10, screen_height_mm / 10
    rectangle_points = np.array([
        [-rw/2, 0, 0],
        [rw/2, 0, 0],
        [rw/2, rh, 0],
        [-rw/2, rh, 0]
    ]).astype(np.float32)

    # Define triangles using indices to the points (two triangles to form a rectangle)
    triangles = np.array([
        [0, 1, 2],  # Triangle 1
        [0, 2, 3]   # Triangle 2
    ])

    # Apply the Open3D transformation
    transformed_pts = transform_for_3d_scene(rectangle_points)

    # Create the TriangleMesh object
    rectangle_mesh = o3d.geometry.TriangleMesh()
    rectangle_mesh.vertices = o3d.utility.Vector3dVector(transformed_pts)
    rectangle_mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Set the color for each vertex
    rectangle_mesh.paint_uniform_color([0, 0, 0])  # Red color

    visual.add_geometry(rectangle_mesh)

def load_pog_balls(visual):
    
    # PoG
    left_pog = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
    right_pog = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
    left_pog.paint_uniform_color([0, 1, 0])
    right_pog.paint_uniform_color([0, 0, 1])
    visual.add_geometry(left_pog)
    visual.add_geometry(right_pog)

    return left_pog, right_pog

def load_gaze_vectors(visual):
    
    # Initial Setup for Gaze Vectors
    left_gaze_vector = o3d.geometry.LineSet()
    left_gaze_vector.paint_uniform_color([0, 1, 0])  # Green color for left
    right_gaze_vector = o3d.geometry.LineSet()
    right_gaze_vector.paint_uniform_color([0, 0, 1])  # Blue color for right

    # Add the gaze vectors to the visualizer
    visual.add_geometry(left_gaze_vector)
    visual.add_geometry(right_gaze_vector)

    return left_gaze_vector, right_gaze_vector
