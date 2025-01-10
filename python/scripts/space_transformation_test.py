import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import open3d as o3d
import pathlib
import trimesh

from webeyetrack.constants import *
from webeyetrack.model_based import vector_to_pitch_yaw, pitch_yaw_to_gaze_vector, get_rotation_matrix_from_vector
from webeyetrack.datasets.utils import draw_landmarks_on_image

import numpy as np

from create_canonical_face import convert_uv_to_xyz, create_perspective_matrix

MAX_STEP_CM = 5
SCALE = 2e-3

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

RT = create_transformation_matrix(1, [0,0,50], [0,180,180])

def transform_for_3d_scene(pts):
    """
    Apply a RT transformation to the points to get the desired 3D scene.
    """
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    transformed_pts_h = (RT @ pts_h.T).T
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
    frame: np.ndarray,
    final_projected_pts: np.ndarray,
    detected_2d: np.ndarray,
    old_z: float,
    alpha: float = 0.5,
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
    draw_frame = frame.copy()

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

def canonical_to_camera(canonical_points, rt_matrix):
    # same as you had before, with perspective divide...
    num_points = len(canonical_points)
    canonical_points_h = np.hstack([
        canonical_points, 
        np.ones((num_points, 1), dtype=np.float32)
    ])
    transformed_points_h = (rt_matrix @ canonical_points_h.T).T
    transformed_xyz = transformed_points_h[:, :3]
    return transformed_xyz

def camera_to_canonical(camera_points, rt_matrix):
    # same as you had before, with perspective divide...
    num_points = len(camera_points)
    camera_points_h = np.hstack([
        camera_points, 
        np.ones((num_points, 1), dtype=np.float32)
    ])
    inv_rt_matrix = np.linalg.inv(rt_matrix)
    transformed_points_h = (inv_rt_matrix @ camera_points_h.T).T
    transformed_xyz = transformed_points_h[:, :3]
    return transformed_xyz

def transform_canonical_mesh(canonical_points, rt_matrix, K):
    # same as you had before, with perspective divide...
    num_points = len(canonical_points)
    canonical_points_h = np.hstack([
        canonical_points, 
        np.ones((num_points, 1), dtype=np.float32)
    ])
    transformed_points_h = (rt_matrix @ canonical_points_h.T).T
    transformed_xyz = transformed_points_h[:, :3]
    camera_space = (K @ transformed_xyz.T).T

    eps = 1e-6
    zs = np.where(np.abs(camera_space[:, 2]) < eps, eps, camera_space[:, 2])
    camera_space[:, 0] /= zs
    camera_space[:, 1] /= zs

    projected_points = camera_space[:, :2]
    projected_points = projected_points.astype(np.int32)
    return projected_points

def main():
    CWD = pathlib.Path(__file__).parent
    PYTHON_DIR = CWD.parent

    # # 1) Load canonical mesh
    mesh_path = str(GIT_ROOT / 'python' / 'assets' / 'face_model_with_iris.obj')
    canonical_mesh = trimesh.load(mesh_path, force='mesh')
    face_width_cm = 14
    # canonical_pts_3d = np.asarray(canonical_mesh.vertices, dtype=np.float32) * face_width_cm * np.array([-1, 1, -1])
    # facemesh_triangles = np.load(GIT_ROOT / 'python' / 'assets' / 'facemesh_triangles.npy')
    face_mesh = o3d.geometry.TriangleMesh()
    face_mesh.vertices = o3d.utility.Vector3dVector(np.array(canonical_mesh.vertices))
    face_mesh.triangles = o3d.utility.Vector3iVector(canonical_mesh.faces)
    face_mesh_lines = o3d.geometry.LineSet.create_from_triangle_mesh(face_mesh)

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
        # vertices -= vertices.mean(axis=0)
        min_x, min_y, min_z = vertices.min(axis=0)
        max_x, max_y, max_z = vertices.max(axis=0)
        center = np.array([min_x + max_x, min_y + max_y, min_z + max_z]) / 2
        vertices -= center
        min_x, min_y, min_z = vertices.min(axis=0)
        max_x, max_y, max_z = vertices.max(axis=0)
        range_x, range_y, range_z = max_x - min_x, max_y - min_y, max_z - min_z
        latest_range = max(range_x, range_y, range_z)
        # norm_vertices = vertices / latest_range
        norm_vertices = vertices / range_y

        # Debugging
        # trimesh_eyeball = trimesh.creation.icosphere(subdivisions=3, radius=eyeball_diameter_cm / 2)

        scaled_vertices = norm_vertices * eyeball_diameter_cm
        eyeball_mesh.vertices = o3d.utility.Vector3dVector(scaled_vertices)
        # eyeball_mesh.vertices = o3d.utility.Vector3dVector(trimesh_eyeball.vertices)
        # import pdb; pdb.set_trace()
        eyeball_mesh.compute_vertex_normals()
        eyeball_meshes[i] = eyeball_mesh
        eyeball_R[i] = np.eye(3)
        iris_3d_pt[i] = o3d.geometry.PointCloud()

    # 2) Setup MediaPipe
    base_options = python.BaseOptions(
        model_asset_path=str(PYTHON_DIR / 'weights' / 'face_landmarker_v2_with_blendshapes.task')
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    # Initialize Open3D Visualizer
    visual = o3d.visualization.Visualizer()
    visual.create_window(width=1920, height=1080)
    visual.get_render_option().background_color = [0.1, 0.1, 0.1]
    visual.get_render_option().mesh_show_back_face = True
    visual.get_render_option().point_size = 10

    # Get the cap sizes
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K = estimate_camera_intrinsics(np.zeros((height, width, 3)))
    perspective_matrix = create_perspective_matrix(aspect_ratio=width / height)

    # Change the z far to 1000
    vis = visual.get_view_control()
    vis.set_constant_z_far(1000)
    params = o3d.camera.PinholeCameraParameters()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(intrinsic_matrix=K, width=width, height=height)
    params.extrinsic = np.eye(4)
    params.intrinsic = intrinsic
    vis.convert_from_pinhole_camera_parameters(parameter=params)

    # Add the face mesh to the visualizer
    visual.add_geometry(face_mesh)
    visual.add_geometry(face_mesh_lines)
    visual.add_geometry(eyeball_meshes['left'])
    visual.add_geometry(eyeball_meshes['right'])
    visual.add_geometry(iris_3d_pt['left'])
    visual.add_geometry(iris_3d_pt['right'])
 
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.astype(np.uint8))
        detection_results = face_landmarker.detect(mp_image)
        if not detection_results.face_landmarks:
            cv2.imshow("Face Mesh", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Draw the landmarks
        # frame = draw_landmarks_on_image(frame, detection_results)

        # Compute the face bounding box based on the MediaPipe landmarks
        try:
            face_landmarks_proto = detection_results.face_landmarks[0]
        except:
            continue

        # Extract information fro the results
        face_landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility, lm.presence] for lm in face_landmarks_proto])

        # Convert uvz to xyz
        relative_face_mesh = np.array([convert_uv_to_xyz(perspective_matrix, x[0], x[1], x[2]) for x in face_landmarks[:, :3]])
        
        # Center to the nose 
        nose = relative_face_mesh[4]
        relative_face_mesh = relative_face_mesh - nose
        relative_face_mesh *= np.array([-1, -1, 1])
        
        # make the width of the face length=1
        LEFTMOST_LANDMARK = 356
        RIGHTMOST_LANDMARK = 127
        leftmost = relative_face_mesh[LEFTMOST_LANDMARK, 0]
        rightmost = relative_face_mesh[RIGHTMOST_LANDMARK, 0]
        leftmost2 = np.min(relative_face_mesh[:, 0])
        rightmost2 = np.max(relative_face_mesh[:, 0])
        # import pdb; pdb.set_trace()
        # relative_face_mesh[:, :] /= rightmost - leftmost
        relative_face_mesh[:, :] /= leftmost - rightmost
        canonical_pts_3d = relative_face_mesh * face_width_cm
        
        # 3) Extract the face transformation matrix from MediaPipe
        face_rt = detection_results.facial_transformation_matrixes[0]  # shape (4,4)
        face_r = face_rt[:3, :3].copy()
        
        # Derotate the face based face transformation matrix
        # canonical_pts_3d = canonical_pts_3d @ face_r.T

        # Scale is embedded in face_r's columns
        scales = np.linalg.norm(face_r, axis=0)
        face_s = scales.mean()  # average scale
        face_r /= face_s

        # ---------------------------------------------------------------
        # (A) Build an initial 4x4 transform that has R, s, and some guess at Z
        #     For example, -60 in front of the camera
        # ---------------------------------------------------------------
        guess_z = 60.0
        init_transform = np.eye(4, dtype=np.float32)
        # init_transform[:3, :3] = np.linalg.inv(face_r)
        init_transform[:3, 3]  = np.array([0, 0, guess_z], dtype=np.float32)

        # ---------------------------------------------------------------
        # (B) Project canonical mesh using this initial transform
        #     We'll get a set of 2D points in pixel space
        # ---------------------------------------------------------------
        canonical_proj_2d = transform_canonical_mesh(
            canonical_pts_3d, init_transform, K 
        ).astype(np.float32)  # shape (N, 2)

        # ---------------------------------------------------------------
        # (C) Get the DETECTED 2D landmarks from MediaPipe
        #     They are in normalized [0..1], so multiply by width/height
        # ---------------------------------------------------------------
        mp_landmarks = detection_results.face_landmarks[0]
        detected_2d = np.array([
            [lm.x * width, lm.y * height] for lm in mp_landmarks
        ], dtype=np.float32)  # shape (468, 2)

        # ---------------------------------------------------------------
        # (D) Do partial Procrustes in 2D: translation only
        #     shift_2d = (mean(detected) - mean(canonical_proj))
        # ---------------------------------------------------------------
        shift_2d = partial_procrustes_translation_2d(canonical_proj_2d, detected_2d)

        # ---------------------------------------------------------------
        # (E) Convert that 2D shift to a 3D offset at depth guess_z
        #     Then add it to the transform's translation
        # ---------------------------------------------------------------
        # Estimate the fx and fy based on the frame size
        shift_3d = image_shift_to_3d(shift_2d, depth_z=guess_z, K=K)
        final_transform = init_transform.copy()
        final_transform[:3, 3] += shift_3d
        first_final_transform = final_transform.copy()

        new_zs = [guess_z]
        for i in range(10):
            # Now do the final projection
            final_projected_pts = transform_canonical_mesh(
                canonical_pts_3d, final_transform, K
            )
            
            new_z, draw_frame = refine_depth_by_radial_magnitude(
                frame, final_projected_pts, detected_2d, old_z=final_transform[2, 3], alpha=0.5
            )

            # Compute the difference of the Z
            new_zs.append(new_z)
            diff_z = new_z - final_transform[2, 3]
            if np.abs(diff_z) < 0.25:
                break

            # Use similar triangles to compute the new x and y
            prior_x = first_final_transform[0, 3]
            prior_y = first_final_transform[1, 3]
            new_x = prior_x * (new_z / guess_z)
            new_y = prior_y * (new_z / guess_z)

            # Compute the new xy shift
            final_transform[0, 3] = new_x
            final_transform[1, 3] = new_y
            final_transform[2, 3] = new_z

        # print(f"Zs: {new_zs}")

        # 7) Project again with updated Z
        final_projected_pts = transform_canonical_mesh(
            canonical_pts_3d, final_transform, K
        ).astype(np.int32)

        # Draw the transformed face mesh
        # for triangle in canonical_mesh.faces:
        # for triangle in facemesh_triangles:
        for triangle in canonical_mesh.faces:
            if triangle[0] in IRIS_LANDMARKS or triangle[1] in IRIS_LANDMARKS or triangle[2] in IRIS_LANDMARKS:
                color = (0, 0, 255)
                thickness = 2
            else:
                color = (0, 255, 0)
                thickness = 1
            p1 = final_projected_pts[triangle[0]]
            p2 = final_projected_pts[triangle[1]]
            p3 = final_projected_pts[triangle[2]]
            cv2.line(draw_frame, p1, p2, color, thickness)
            cv2.line(draw_frame, p2, p3, color, thickness)
            cv2.line(draw_frame, p3, p1, color, thickness)

        # Draw the transformed face vertices only
        # for pt in final_projected_pts:
        #     cv2.circle(draw_frame, tuple(pt), 1, (0, 255, 0), -1)

        # Draw the depth as text on the top-left corner
        cv2.putText(draw_frame, f"Depth: {final_transform[2,3]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Create a new final transform that also includes the rotation matrix
        new_final_transform = final_transform.copy()
        new_final_transform[:3, :3] = face_r

        # Compute the line-sphere intersection problem
        # Using the projected 2D center pupil landmark, estimate where a line 
        # intersects with the eyeball sphere
        eX, eY, eZ = 3,-2.8,3
        left_eyeball_pt = canonical_to_camera(np.array([-eX, eY, eZ]).reshape((1,3)), new_final_transform).flatten()
        right_eyeball_pt = canonical_to_camera(np.array([eX, eY, eZ]).reshape((1,3)), new_final_transform).flatten()
        transformed_pts = canonical_to_camera(canonical_pts_3d, final_transform)
        camera_iris_pts = {}
        camera_eyeball_center = {}
        gaze_vectors = {}
        for i in ['left', 'right']:
            # Compute the sphere center in 3D
            eye_landmark = LEFT_EYE_LANDMARKS if i == 'left' else RIGHT_EYE_LANDMARKS
            # eye_outline_landmark = [258, 253] if i == 'left' else [28, 23]
            eyeball_center_from_pts = transformed_pts[eye_landmark].mean(axis=0)

            # Using the eye outline, compute the y mean
            # eye_outline_center = transformed_pts[eye_outline_landmark].mean(axis=0)
            # eyeball_center_from_canonical = left_eyeball_pt if i == 'left' else right_eyeball_pt

            # Create an eyeball center using the x and z from the pts and the y from canonical
            eyeball_center = eyeball_center_from_pts.copy()
            # eyeball_center[1] = eye_outline_center[1]
            # eyeball_center[1] = eyeball_center_from_canonical[1]

            # Compute the line direction
            iris_landmarks = LEFT_IRIS_LANDMARKS if i == 'left' else RIGHT_IRIS_LANDMARKS
            center_iris = final_projected_pts[iris_landmarks[0]]
            line_direction = np.dot(np.linalg.inv(K), np.array([center_iris[0], center_iris[1], 1])) * np.array([-1, -1, -1])
            line_direction = line_direction[:3] / np.linalg.norm(line_direction[:3])
            line_direction /= np.linalg.norm(line_direction)
            iris_eyeball_pt = line_sphere_intersection(
                np.array([0, 0, 0]),
                line_direction,
                eyeball_center, 
                eyeball_diameter_cm / 2
            )

            # Check what is the euclidean distance between the iris_eyeball_pt and the eyeball center, ensure it's the sphere radius
            # import pdb; pdb.set_trace()
            # distance = np.linalg.norm(iris_eyeball_pt - eyeball_center)
            # assert np.isclose(distance, eyeball_diameter_cm / 2)

            # camera_iris_pts[i] = iris_eyeball_pt
            camera_eyeball_center[i] = eyeball_center

            # Project the iris 3D point to the image plane and draw it
            # iris_2d = np.dot(K, iris_eyeball_pt)
            # iris_2d = iris_2d[:2] / iris_2d[2]
            # error = np.linalg.norm(iris_2d - center_iris)
            # print(f"Iris 2D points ({i}): {iris_2d} - {center_iris} - Error: {error}")
            # print(f"CAMERA1: Iris 3D points ({i}): {iris_eyeball_pt} - {eyeball_center}")
            # cv2.circle(draw_frame, tuple(iris_2d.astype(np.int32)), 5, (255, 0, 255), -1)

            # Visualize the line-sphere intersection
            # visualize_line_sphere_intersection(
            #     np.array([0, 0, 0]),
            #     line_direction,
            #     eyeball_center,
            #     eyeball_diameter_cm / 2,
            #     iris_eyeball_pt
            # )
            # iris_eyeball_pt = None

            if iris_eyeball_pt is None:
                gaze_vectors[i] = np.array([1e-5, 1e-5, -1.0])
                gaze_vectors[i] /= np.linalg.norm(gaze_vectors[i]) 
            else:
                # Estimate a gaze vector
                gaze_vector = iris_eyeball_pt - eyeball_center
                gaze_vector /= np.linalg.norm(gaze_vector)

                # # Convert gaze vector to pitch and yaw to correct
                pitch, yaw = vector_to_pitch_yaw(gaze_vector)
                # pitch, yaw = -pitch, yaw
                pitch, yaw = pitch, -yaw
                # pitch, yaw = 0, 5
                gaze_vector = pitch_yaw_to_gaze_vector(pitch, yaw)

                # Store
                print(f"Gaze vector ({i}): {gaze_vector}")
                gaze_vectors[i] = gaze_vector

        # Compute the face mesh
        face_mesh.vertices = o3d.utility.Vector3dVector(transform_for_3d_scene(transformed_pts))
        new_face_mesh_lines = o3d.geometry.LineSet.create_from_triangle_mesh(face_mesh)
        face_mesh_lines.points = new_face_mesh_lines.points

        # Compute the eye gaze origin in metric space
        # eye_g_o = {
        #     'left': transformed_pts[LEFT_EYE_LANDMARKS],
        #     'right': transformed_pts[RIGHT_EYE_LANDMARKS]
        # }

        # Compute the 3D eye origin
        # for k, v in eye_g_o.items():
        for i in ['left', 'right']:
            # eye_g_o[k] = np.mean(v, axis=0)

            # final_position = transform_for_3d_scene(eye_g_o[k].reshape((-1,3))).flatten()
            final_position = transform_for_3d_scene(camera_eyeball_center[i].reshape((-1,3))).flatten()
            # final_position -= np.array([0,0,0.25])
            eyeball_meshes[i].translate(final_position, relative=False)

            # Rotation
            current_eye_R = eyeball_R[i]
            eye_R = get_rotation_matrix_from_vector(gaze_vectors[i])

            # Apply the scene transformation to the new eye rotation
            eye_R = np.dot(eye_R, RT[:3, :3])

            # Compute the rotation matrix to rotate the current to the target
            new_eye_R = np.dot(eye_R, current_eye_R.T)
            eyeball_R[i] = eye_R
            eyeball_meshes[i].rotate(new_eye_R)

            # Debug, print out the mean eye gaze origin of the eyeball mesh
            vertices = np.array(eyeball_meshes[i].vertices)
            centroid = vertices.mean(axis=0)
            # print(f"Eye gaze origin ({k}): {centroid}")

        # Update the geometry
        visual.update_geometry(face_mesh)
        visual.update_geometry(face_mesh_lines)
        for i in ['left', 'right']:
            # ...
            visual.update_geometry(eyeball_meshes[i])
            # visual.update_geometry(iris_3d_pt[i])

        # Update visualizer
        visual.poll_events()
        visual.update_renderer()

        cv2.imshow("Face Mesh", draw_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # exit(0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
