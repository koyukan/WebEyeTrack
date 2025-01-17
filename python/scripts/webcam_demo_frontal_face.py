import cv2
import numpy as np
import open3d as o3d

from webeyetrack import WebEyeTrack
from webeyetrack.constants import *
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
    load_3d_axis,
    load_canonical_mesh,
    load_eyeball_model
)

def main():

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

    # Change the z far to 1000
    vis = visual.get_view_control()
    vis.set_constant_z_far(1000)
    params = o3d.camera.PinholeCameraParameters()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(intrinsic_matrix=K, width=width, height=height)
    params.extrinsic = np.eye(4)
    params.intrinsic = intrinsic
    vis.convert_from_pinhole_camera_parameters(parameter=params)

    face_mesh, face_mesh_lines = load_canonical_mesh(visual)
    face_coordinate_axes = load_3d_axis(visual)
    eyeball_meshes, _, eyeball_R = load_eyeball_model(visual)

    # Pipeline
    pipeline = WebEyeTrack(
        model_asset_path=str(GIT_ROOT / 'python'/ 'weights' / 'face_landmarker_v2_with_blendshapes.task'), 
        frame_height=height,
        frame_width=width,
        intrinsics=K,
        screen_R=np.deg2rad(np.array([0, 0, 0]).astype(np.float32)),
        screen_t=np.array([0, 0, 0]).astype(np.float32),
        # screen_width_mm=SCREEN_WIDTH_MM,
        # screen_height_mm=SCREEN_HEIGHT_MM,
        # screen_width_px=SCREEN_WIDTH_PX,
        # screen_height_px=SCREEN_HEIGHT_PX
    )
 
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        result, detection_results = pipeline.process_frame(frame)
        if not result:
            cv2.imshow("Face Mesh", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        draw_frame = frame.copy()

        # Draw the landmarks
        draw_frame = draw_landmarks_on_image(draw_frame, detection_results)

        # Draw the depth as text on the top-left corner
        cv2.putText(draw_frame, f"Depth: {result.metric_transform[2,3]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Compute the face mesh
        face_mesh.vertices = o3d.utility.Vector3dVector(transform_for_3d_scene(result.metric_face))
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
        face_coordinate_axes.points = o3d.utility.Vector3dVector(transform_for_3d_scene(camera_pts_3d))
        visual.update_geometry(face_coordinate_axes)

        # Compute the 3D eye origin
        for i in ['left', 'right']:
            eye_result = result.left if i == 'left' else result.right
            origin = eye_result.origin
            direction = eye_result.direction

            # final_position = transform_for_3d_scene(eye_g_o[k].reshape((-1,3))).flatten()
            final_position = transform_for_3d_scene(origin.reshape((-1,3))).flatten()
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

        # Update visualizer
        visual.poll_events()
        visual.update_renderer()

        cv2.imshow("Face Mesh", draw_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    visual.destroy_window()

if __name__ == "__main__":
    main()
