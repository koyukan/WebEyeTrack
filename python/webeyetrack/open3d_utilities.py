from pathlib import Path

import open3d as o3d
import trimesh
import numpy as np

from .data_protocols import GazeResult
from .constants import *
from .utilities import (
    estimate_camera_intrinsics, 
    load_canonical_mesh, 
    load_3d_axis, 
    load_eyeball_model,
    load_camera_frustrum,
    load_screen_rect,
    load_pog_balls,
    load_gaze_vectors,
    transform_for_3d_scene,
    get_rotation_matrix_from_vector,
    euler_angles_to_rotation_matrix,
    transform_3d_to_3d,
    create_transformation_matrix,
    transform_3d_to_2d,
    OPEN3D_RT,
    OPEN3D_RT_SCREEN,
)
from .model_based import vector_to_pitch_yaw, rotation_matrix_to_euler_angles

##########################################
# Visualizations
##########################################

def render_pog(
        frame: np.ndarray, 
        result: GazeResult, 
        output_path: pathlib.Path,
        screen_RT: np.ndarray,
        screen_width_cm: float,
        screen_height_cm: float,
    ):

    # Get the frame size 
    height, width = frame.shape[:2]
    max_size = max(height, width)
    w_ratio, h_ratio = width/max_size, height/max_size
    K = estimate_camera_intrinsics(np.zeros((height, width, 3)))

    # Initialize Open3D Visualizer
    visual = o3d.visualization.Visualizer()
    visual.create_window(width=width, height=height)
    visual.get_render_option().background_color = [1, 1, 1]
    visual.get_render_option().mesh_show_back_face = True

    # Change the z far to 1000
    vis = visual.get_view_control()
    vis.set_constant_z_far(1000)
    params = o3d.camera.PinholeCameraParameters()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(intrinsic_matrix=K, width=width, height=height)
    params.extrinsic = np.eye(4)
    params.intrinsic = intrinsic
    vis.convert_from_pinhole_camera_parameters(parameter=params)
    
    # Define intrinsics based on the frame
    intrinsics = np.array([[width, 0, width // 2], [0, height, height // 2], [0, 0, 1]])

    face_mesh, face_mesh_lines = load_canonical_mesh(visual)
    face_coordinate_axes = load_3d_axis(visual)
    eyeball_meshes, _, eyeball_R = load_eyeball_model(visual)
    load_screen_rect(visual, screen_width_cm, screen_height_cm, screen_rt=screen_RT, scene_rt=OPEN3D_RT_SCREEN)
    load_camera_frustrum(w_ratio, h_ratio, visual, rt=OPEN3D_RT_SCREEN)
    left_pog, right_pog = load_pog_balls(visual)
    left_gaze_vector, right_gaze_vector = load_gaze_vectors(visual)
    
    camera_coordinate_axes = load_3d_axis(visual)
    points = [
        [0, 0, 0],  # Origin
        [1, 0, 0],  # X-axis end
        [0, 1, 0],  # Y-axis end
        [0, 0, 1],  # Z-axis end
    ]
    camera_coordinate_axes.points = o3d.utility.Vector3dVector(transform_for_3d_scene(np.array(points) * 5, OPEN3D_RT_SCREEN))
    visual.update_geometry(camera_coordinate_axes)

    # Compute face mesh
    face_mesh.vertices = o3d.utility.Vector3dVector(transform_for_3d_scene(result.metric_face, OPEN3D_RT_SCREEN))
    new_face_mesh_lines = o3d.geometry.LineSet.create_from_triangle_mesh(face_mesh)
    face_mesh_lines.points = new_face_mesh_lines.points
    visual.update_geometry(face_mesh)
    visual.update_geometry(face_mesh_lines)

    # Update the 3d axes in the visualizer as well
    # Draw the canonical face axes by using the final_transform
    canonical_face_axes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]) * 5
    camera_pts_3d = transform_3d_to_3d(canonical_face_axes, result.metric_transform)
    face_coordinate_axes.points = o3d.utility.Vector3dVector(transform_for_3d_scene(camera_pts_3d, OPEN3D_RT_SCREEN))
    visual.update_geometry(face_coordinate_axes)

    # Draw the 3D eyeball and gaze vector
    for i in ['left', 'right']:
        eye_result = result.left if i == 'left' else result.right
        origin = eye_result.origin
        direction = eye_result.direction
        pog_ball = left_pog if i == 'left' else right_pog
        gaze_vector = left_gaze_vector if i == 'left' else right_gaze_vector

        # final_position = transform_for_3d_scene(eye_g_o[k].reshape((-1,3))).flatten()
        final_position = transform_for_3d_scene(origin.reshape((-1,3)), OPEN3D_RT_SCREEN).flatten()
        eyeball_meshes[i].translate(final_position, relative=False)

        # Rotation
        current_eye_R = eyeball_R[i]
        eye_R = get_rotation_matrix_from_vector(direction)
        pitch, yaw, roll = rotation_matrix_to_euler_angles(eye_R)
        pitch, yaw, roll = yaw, pitch, roll # Flip the pitch and yaw
        eye_R = euler_angles_to_rotation_matrix(pitch, yaw, 0)

        # Apply the scene transformation to the new eye rotation
        eye_R = np.dot(eye_R, OPEN3D_RT[:3, :3])

        # Compute the rotation matrix to rotate the current to the target
        new_eye_R = np.dot(eye_R, current_eye_R.T)
        eyeball_R[i] = eye_R
        eyeball_meshes[i].rotate(new_eye_R)
        visual.update_geometry(eyeball_meshes[i])

        # Draw the gaze vectors
        pts = np.array([origin, origin + direction * 100])
        transform_pts = transform_for_3d_scene(pts, OPEN3D_RT_SCREEN)
        gaze_vector.points = o3d.utility.Vector3dVector(transform_pts)
        gaze_vector.lines = o3d.utility.Vector2iVector([[0, 1]])
        if i == 'left':
            gaze_vector.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]))
        else:
            gaze_vector.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0]]))
        visual.update_geometry(gaze_vector)

        # Position the PoG balls
        pog_position = transform_for_3d_scene(eye_result.pog.pog_cm_c.reshape((-1,3)), OPEN3D_RT_SCREEN).flatten()
        pog_ball.translate(pog_position, relative=False)
        visual.update_geometry(pog_ball)

    # Update visualizer
    visual.poll_events()
    visual.update_renderer()

    # Save the image
    visual.capture_screen_image(str(output_path))

    # Cleanup
    visual.destroy_window()
    print(f"3D render saved to {output_path}")

def render_3d_gaze_with_frame(frame: np.ndarray, result: GazeResult, output_path: pathlib.Path):
    render_3d_gaze(frame, result, output_path)
    render_img = cv2.imread(str(output_path))
    combined_frame = np.hstack([frame, render_img])
    return combined_frame

def render_3d_gaze(frame: np.ndarray, result: GazeResult, output_path: pathlib.Path) -> np.ndarray:

    # Get the frame size 
    height, width = frame.shape[:2]
    K = estimate_camera_intrinsics(np.zeros((height, width, 3)))
    
    # Initialize Open3D visual
    visual = o3d.visualization.Visualizer()
    visual.create_window(width=width, height=height)
    visual.get_render_option().background_color = [1, 1, 1]
    visual.get_render_option().mesh_show_back_face = True
    visual.get_render_option().point_size = 10

    # Change the z far to 1000
    control = visual.get_view_control()
    control.set_constant_z_far(1000)
    params = o3d.camera.PinholeCameraParameters()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(intrinsic_matrix=K, width=width, height=height)
    params.extrinsic = np.eye(4)
    params.intrinsic = intrinsic
    control.convert_from_pinhole_camera_parameters(parameter=params)

    face_mesh, face_mesh_lines = load_canonical_mesh(visual)
    face_coordinate_axes = load_3d_axis(visual)
    eyeball_meshes, _, eyeball_R = load_eyeball_model(visual)

    # Compute face mesh
    face_mesh.vertices = o3d.utility.Vector3dVector(transform_for_3d_scene(result.metric_face))
    new_face_mesh_lines = o3d.geometry.LineSet.create_from_triangle_mesh(face_mesh)
    face_mesh_lines.points = new_face_mesh_lines.points
    visual.update_geometry(face_mesh)
    visual.update_geometry(face_mesh_lines)

    # Draw canonical face axes
    canonical_face_axes = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]) * 5
    camera_pts_3d = transform_3d_to_3d(canonical_face_axes, result.metric_transform)
    face_coordinate_axes.points = o3d.utility.Vector3dVector(transform_for_3d_scene(camera_pts_3d))
    visual.update_geometry(face_coordinate_axes)

    # Compute the 3D eye origin
    for i in ['left', 'right']:
        eye_result = result.left if i == 'left' else result.right
        origin = eye_result.origin
        direction = eye_result.direction

        final_position = transform_for_3d_scene(origin.reshape((-1, 3))).flatten()
        eyeball_meshes[i].translate(final_position, relative=False)

        # Rotation
        current_eye_R = eyeball_R[i]
        eye_R = get_rotation_matrix_from_vector(direction)
        pitch, yaw, roll = rotation_matrix_to_euler_angles(eye_R)
        pitch, yaw, roll = yaw, pitch, roll  # Flip pitch and yaw
        # pitch, yaw, roll = -yaw, pitch, roll  # Flip pitch and yaw
        eye_R = euler_angles_to_rotation_matrix(pitch, yaw, 0)
        eye_R = np.dot(eye_R, OPEN3D_RT[:3, :3])

        # Compute the rotation matrix to rotate the current to the target
        new_eye_R = np.dot(eye_R, current_eye_R.T)
        eyeball_R[i] = eye_R
        eyeball_meshes[i].rotate(new_eye_R)
        visual.update_geometry(eyeball_meshes[i])

    # Render the scene
    visual.poll_events()
    visual.update_renderer()

    # Save the image
    visual.capture_screen_image(str(output_path))

    # Cleanup
    visual.close()
    visual.destroy_window()
    print(f"3D render saved to {output_path}")

def model_based_gaze_render(frame: np.ndarray, result: GazeResult):

    # Extract the information
    facial_landmarks = result.facial_landmarks
    height, width = frame.shape[:2]
    draw_frame = frame.copy()

    # Estimate the eyeball centers
    face_rt_copy = result.face_rt.copy()
    face_rt_copy[:3, 3] *= np.array([-1, -1, -1])

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

    # Compute the eyeball locations
    # Draw the eyeballs on the frame itsef
    named_sphere_points = {}
    for i, eye_result in enumerate([result.left, result.right]):
        eyeball_center = result.eyeball_centers[0] if i == 0 else result.eyeball_centers[1]
        name = 'left' if i == 0 else 'right'
        
        # Draw the eyeball as a 3D sphere
        # First, construct the 3D sphere within the canonical coordinate system
        sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=result.eyeball_radius)
        sphere_points = np.array(sphere_mesh.vertices)
        sphere_points += eyeball_center
        sphere_points_homogenous = np.hstack([sphere_points, np.ones((sphere_points.shape[0], 1))])
        sphere_points_transformed = (face_rt_copy @ sphere_points_homogenous.T).T

        # Project the 3D sphere to the 2D image
        height, width = frame.shape[:2]
        screen_sphere_homogenous = (result.perspective_matrix @ sphere_points_transformed.T).T
        screen_sphere_x_2d_n = screen_sphere_homogenous[:, 0] / screen_sphere_homogenous[:, 2]
        screen_sphere_y_2d_n = screen_sphere_homogenous[:, 1] / screen_sphere_homogenous[:, 2]
        screen_sphere_x_2d = (screen_sphere_x_2d_n + 1) * width / 2
        screen_sphere_y_2d = (1 - screen_sphere_y_2d_n) * height / 2

        screen_sphere_points = np.hstack([screen_sphere_x_2d.reshape(-1, 1), screen_sphere_y_2d.reshape(-1, 1)])
        named_sphere_points[name] = screen_sphere_points

        # Draw the 2D sphere
        for j in range(sphere_points.shape[0]):
            # cv2.circle(frame, (int(screen_sphere_x_2d[i]), int(screen_sphere_y_2d[i])), 1, (0, 0, 255), -1)
            prior_value = draw_frame[int(screen_sphere_y_2d[j]), int(screen_sphere_x_2d[j])]
            # Add a bit of grey to the color
            new_value = (prior_value + np.array([150, 150, 150])) / 2
            draw_frame[int(screen_sphere_y_2d[j]), int(screen_sphere_x_2d[j])] = new_value

        # Draw the eyeball center
        if i == 0:
            eyeball_center_2d = result.left.meta_data['eyeball_center_2d']
        else:
            eyeball_center_2d = result.right.meta_data['eyeball_center_2d']
        cv2.circle(draw_frame, tuple(eyeball_center_2d.astype(int)), 3, (0, 0, 255), -1)

        # Draw the iris center
        iris_center = left_2d_iris_px if i == 0 else right_2d_iris_px
        iris_center = iris_center[0]
        cv2.circle(draw_frame, tuple(iris_center.astype(int)), 3, (0, 255, 0), -1)

    eye_images = {}
    original_sizes = {}
    centroids = {}
    for i, (eye, eyelid, eyearea, eyelid_total) in {'left': left_landmarks, 'right': right_landmarks}.items():
        centroid = np.mean(eye, axis=0)
        width = np.abs(eye[0,0] - eye[1, 0]) * (1 + EYE_PADDING_WIDTH)
        height = width * EYE_HEIGHT_RATIO
        centroids[i] = centroid

        # Determine if closed by the eyelid
        eyelid_width = np.abs(eyelid[0,0] - eyelid[1, 0])
        eyelid_height = np.abs(eyelid[3,1] - eyelid[2, 1])
        eye_result = result.left if i == 'left' else result.right
        eyeball_center = result.eyeball_centers[0] if i == 'left' else result.eyeball_centers[1]
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
        original_sizes[i] = (original_width, original_height)

        # new_width, new_height = EYE_IMAGE_WIDTH, int(EYE_IMAGE_WIDTH*EYE_HEIGHT_RATIO)
        new_width, new_height = EYE_IMAGE_WIDTH, (EYE_IMAGE_WIDTH*original_height) // original_width
        eye_image = cv2.resize(eye_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        eye_images[i] = eye_image

        # Compute the shift and the scale
        shift_value = np.array([int(centroid[0] - width/2), int(centroid[1] - height/2)])
        scale_value = np.array([new_width/original_width, new_height/original_height])

        # Draw the outline of the eyelid
        shifted_eyelid_px = eyelid_total - shift_value
        prior_px = None
        for px in shifted_eyelid_px:
            resized_px = px * scale_value
            if prior_px is not None:
                cv2.line(eye_image, tuple(prior_px.astype(int)), tuple(resized_px.astype(int)), (255, 0, 0), 1)
            prior_px = resized_px
        
        # Draw the last line to close the loop
        resized_first_px = shifted_eyelid_px[0] * scale_value
        cv2.line(eye_image, tuple(prior_px.astype(int)), tuple(resized_first_px.astype(int)), (255, 0, 0), 1)

        # Draw the eyeball
        screen_eye_points = named_sphere_points[i]
        for j in range(screen_eye_points.shape[0]):
            shifted_eyeball_pt = screen_eye_points[j] - shift_value
            resized_eyeball_pt = shifted_eyeball_pt * scale_value
            cv2.circle(eye_image, tuple(resized_eyeball_pt.astype(int)), 1, (200, 200, 200), -1)

        if is_closed:
            cv2.putText(eye_image, 'Closed', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            continue

        # Draw the eyeball center
        eyeball_center_2d = eye_result.meta_data['eyeball_center_2d']
        shifted_eyeball_2d = eyeball_center_2d - shift_value
        scaled_eyeball_2d = shifted_eyeball_2d * scale_value
        cv2.circle(eye_image, scaled_eyeball_2d.astype(int), 5, (0, 255, 0), -1)
        
        # Convert 3D to pitch and yaw
        pitch, yaw = vector_to_pitch_yaw(eye_result.direction)
        draw_frame = draw_axis(draw_frame, -pitch, -yaw, 0, int(eyeball_center_2d[0]), int(eyeball_center_2d[1]), 100)
        eye_image = draw_axis(eye_image, -pitch, -yaw, 0, int(scaled_eyeball_2d[0]), int(scaled_eyeball_2d[1]), 100)

        # Shift the IRIS landmarks to the cropped eye
        iris_px = left_2d_iris_px if i == 'left' else right_2d_iris_px
        shifted_iris_px = iris_px - shift_value
        scaled_shifted_iris_px = shifted_iris_px * scale_value
        cv2.circle(eye_image, tuple(scaled_shifted_iris_px[0].astype(int)), 3, (0, 0, 255), -1)

    # Draw the FPS on the topright
    fps = 1/result.duration
    cv2.putText(draw_frame, f'FPS: {fps:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw the headpose on the frame
    # headrot = result.face_rt[:3, :3]
    # pitch, yaw, roll = rotation_matrix_to_euler_angles(headrot)
    # pitch, yaw = yaw, pitch
    # face_origin = result.face_origin_2d
    # frame = draw_axis(frame, pitch, yaw, -roll, int(face_origin[0]), int(face_origin[1]), 100)

    # # Concatenate the images
    # if 'right' not in eye_images:
    #     right_eye_image = np.zeros((EYE_IMAGE_WIDTH, EYE_IMAGE_WIDTH, 3), dtype=np.uint8)
    # else:
    #     right_eye_image = eye_images['right']

    # if 'left' not in eye_images:
    #     left_eye_image = np.zeros((EYE_IMAGE_WIDTH, EYE_IMAGE_WIDTH, 3), dtype=np.uint8)
    # else:
    #     left_eye_image = eye_images['left']

    # Combine the eye images
    eye_combined = pad_and_concat_images(eye_images['right'], eye_images['left'])

    # Resize the combined eyes horizontally to match the width of the frame (640 pixels wide)
    eyes_combined_resized = imutils.resize(eye_combined, width=frame.shape[1])
    
    # Concatenate the combined eyes image vertically with the frame
    total_frame = cv2.vconcat([draw_frame, eyes_combined_resized])

    # Pad the total frame with black at the bottom to avoid gittering when displaying
    total_height = total_frame.shape[0]
    padding_height = max(0, 800 - total_height)
    total_frame = cv2.copyMakeBorder(total_frame, 0, padding_height, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return total_frame

##########################################
# Routines
##########################################

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

def load_canonical_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.triangles))
    return mesh

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

def load_camera_frustrum(w_ratio, h_ratio, visual=None, rt=OPEN3D_RT):
    
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
    transformed_pts = transform_for_3d_scene(points, rt)

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

def load_screen_rect(visual, screen_width_cm, screen_height_cm, screen_rt, scene_rt=OPEN3D_RT):
    
    # Screen Display
    rw, rh = screen_width_cm, screen_height_cm
    rectangle_points = np.array([
        [0,0,0],
        [rw,0,0],
        [rw,rh,0],
        [0,rh,0]
    ]).astype(np.float32)

    # Define triangles using indices to the points (two triangles to form a rectangle)
    triangles = np.array([
        [0, 1, 2],  # Triangle 1
        [0, 2, 3]   # Triangle 2
    ])

    # Transform from screen coordinate to camera coordinate
    transformed_pts = transform_3d_to_3d(rectangle_points, screen_rt)

    # Apply the Open3D transformation
    transformed_pts = transform_for_3d_scene(transformed_pts, scene_rt)

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
