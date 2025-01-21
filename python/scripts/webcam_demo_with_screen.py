import platform

import open3d as o3d
import cv2
import numpy as np

from webeyetrack import WebEyeTrack
from webeyetrack.model_based import get_rotation_matrix_from_vector
from webeyetrack.constants import GIT_ROOT
from webeyetrack.datasets.utils import draw_landmarks_on_image
from webeyetrack.utilities import (
    estimate_camera_intrinsics, 
    transform_for_3d_scene,
    transform_3d_to_3d,
    transform_3d_to_2d,
    get_rotation_matrix_from_vector,
    rotation_matrix_to_euler_angles,
    euler_angles_to_rotation_matrix,
    OPEN3D_RT,
    OPEN3D_RT_SCREEN,
    load_3d_axis, 
    load_canonical_mesh, 
    load_eyeball_model,
    load_camera_frustrum,
    load_pog_balls,
    load_gaze_vectors,
    load_screen_rect,
    get_screen_attributes,
    create_transformation_matrix
)

SCREEN_HEIGHT_CM, SCREEN_WIDTH_CM, SCREEN_HEIGHT_PX, SCREEN_WIDTH_PX = get_screen_attributes()
print(f"Screen Height: {SCREEN_HEIGHT_CM} cm, Screen Width: {SCREEN_WIDTH_CM} cm")

if __name__ == '__main__':

    # Initialize Open3D Visualizer
    visual = o3d.visualization.Visualizer()
    visual.create_window(width=1920, height=1080)
    visual.get_render_option().background_color = [1, 1, 1]
    visual.get_render_option().mesh_show_back_face = True
    
    # Load the webcam 
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_size = max(width, height)
    w_ratio, h_ratio = width/max_size, height/max_size
    K = estimate_camera_intrinsics(np.zeros((height, width, 3)))

    # Define intrinsics based on the frame
    intrinsics = np.array([[width, 0, width // 2], [0, height, height // 2], [0, 0, 1]])

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

    face_mesh, face_mesh_lines = load_canonical_mesh(visual)
    face_coordinate_axes = load_3d_axis(visual)
    eyeball_meshes, _, eyeball_R = load_eyeball_model(visual)
    load_screen_rect(visual, SCREEN_WIDTH_CM, SCREEN_HEIGHT_CM, screen_rt=screen_RT,scene_rt=OPEN3D_RT_SCREEN)
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
    
    # Pipeline
    pipeline = WebEyeTrack(
        model_asset_path=str(GIT_ROOT / 'python'/ 'weights' / 'face_landmarker_v2_with_blendshapes.task'), 
        frame_height=height,
        frame_width=width,
        intrinsics=intrinsics,
        # screen_R=np.deg2rad(np.array([0, 0, 0]).astype(np.float32)),
        # screen_t=np.array([0, 0, 0]).astype(np.float32),
        screen_RT=screen_RT,
        screen_width_cm=SCREEN_WIDTH_CM,
        screen_height_cm=SCREEN_HEIGHT_CM,
        screen_width_px=SCREEN_WIDTH_PX,
        screen_height_px=SCREEN_HEIGHT_PX
    )

    # Load the frames and draw the landmarks
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        result, detection_results = pipeline.process_frame(frame)
        if not result:
            cv2.imshow("Face Mesh", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        draw_frame = frame.copy()

        # Draw the landmarks
        draw_frame = draw_landmarks_on_image(draw_frame, detection_results)

        # Compute the face mesh
        face_mesh.vertices = o3d.utility.Vector3dVector(transform_for_3d_scene(result.metric_face, OPEN3D_RT_SCREEN))
        new_face_mesh_lines = o3d.geometry.LineSet.create_from_triangle_mesh(face_mesh)
        face_mesh_lines.points = new_face_mesh_lines.points
        visual.update_geometry(face_mesh)
        visual.update_geometry(face_mesh_lines)

        # Draw the canonical face axes by using the final_transform
        canonical_face_axes = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]) * 5
        camera_pts_3d = transform_3d_to_3d(canonical_face_axes, result.metric_transform)
        canonical_face_axes_2d = transform_3d_to_2d(camera_pts_3d, K)
        cv2.line(draw_frame, tuple(canonical_face_axes_2d[0]), tuple(canonical_face_axes_2d[1]), (0, 0, 255), 2)
        cv2.line(draw_frame, tuple(canonical_face_axes_2d[0]), tuple(canonical_face_axes_2d[2]), (0, 255, 0), 2)
        cv2.line(draw_frame, tuple(canonical_face_axes_2d[0]), tuple(canonical_face_axes_2d[3]), (255, 0, 0), 2)

        # Update the 3d axes in the visualizer as well
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
            eye_R = np.dot(eye_R, OPEN3D_RT_SCREEN[:3, :3])

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

        # Draw the PoG in a rectangle black image
        screen_img = np.zeros((SCREEN_HEIGHT_PX, SCREEN_WIDTH_PX, 3), dtype=np.uint8)
        cv2.circle(screen_img, tuple(result.pog.pog_px.astype(np.int32)), 5, (0, 0, 255), -1)
        
        cv2.imshow("Face Mesh", draw_frame)
        cv2.imshow("Screen", screen_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()
    visual.destroy_window()
