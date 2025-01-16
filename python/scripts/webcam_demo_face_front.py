from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import open3d as o3d
import pathlib
import trimesh

from webeyetrack import WebEyeTrack
from webeyetrack.constants import *
from webeyetrack.datasets.utils import draw_landmarks_on_image
from webeyetrack.utilities import (
    estimate_camera_intrinsics, 
    transform_for_3d_scene,
    transform_3d_to_3d,
    transform_3d_to_2d
)

import numpy as np

from create_canonical_face import convert_uv_to_xyz, create_perspective_matrix

MAX_STEP_CM = 5
SCALE = 2e-3

def load_canonical_mesh(visual):
    # 1) Load canonical mesh
    mesh_path = str(GIT_ROOT / 'python' / 'assets' / 'face_model_with_iris.obj')
    canonical_mesh = trimesh.load(mesh_path, force='mesh')
    face_width_cm = 14
    canonical_norm_pts_3d = np.asarray(canonical_mesh.vertices, dtype=np.float32) * face_width_cm * np.array([-1, 1, -1])
    face_mesh = o3d.geometry.TriangleMesh()
    face_mesh.vertices = o3d.utility.Vector3dVector(np.array(canonical_mesh.vertices))
    face_mesh.triangles = o3d.utility.Vector3iVector(canonical_mesh.faces)
    face_mesh_lines = o3d.geometry.LineSet.create_from_triangle_mesh(face_mesh)

    # Generate colors for the face mesh, with all being default color except for the iris
    colors = np.array([[3/256, 161/256, 252/256] for _ in range(len(canonical_mesh.vertices))])
    # for i in IRIS_LANDMARKS:
    #     colors[i] = [1, 0, 0]
    # face_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    visual.add_geometry(face_mesh)
    visual.add_geometry(face_mesh_lines)

    return face_mesh, face_mesh_lines

def load_3d_axis(visual):
    
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
    visual.add_geometry(line_set)

    return line_set

def load_eyeball_model(visual):
    
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

    visual.add_geometry(eyeball_meshes['left'])
    visual.add_geometry(eyeball_meshes['right'])
    visual.add_geometry(iris_3d_pt['left'])
    visual.add_geometry(iris_3d_pt['right'])

    return eyeball_meshes, iris_3d_pt, eyeball_R

def main():
    CWD = pathlib.Path(__file__).parent
    PYTHON_DIR = CWD.parent

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

    face_mesh, face_mesh_lines = load_canonical_mesh(visual)
    line_set = load_3d_axis(visual)
    eyeball_meshes, iris_3d_pt, eyeball_R = load_eyeball_model(visual)

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
        line_set.points = o3d.utility.Vector3dVector(camera_pts_3d)
        visual.update_geometry(line_set)

        # Compute the 3D eye origin
        # for i in ['left', 'right']:

        #     # final_position = transform_for_3d_scene(eye_g_o[k].reshape((-1,3))).flatten()
        #     final_position = transform_for_3d_scene(camera_eyeball_center[i].reshape((-1,3))).flatten()
        #     eyeball_meshes[i].translate(final_position, relative=False)

        #     # Rotation
        #     current_eye_R = eyeball_R[i]
        #     eye_R = get_rotation_matrix_from_vector(gaze_vectors[i])
        #     pitch, yaw, roll = rotation_matrix_to_euler_angles(eye_R)
        #     pitch, yaw, roll = yaw, -pitch, roll # Flip the pitch and yaw
        #     eye_R = euler_angles_to_rotation_matrix(pitch, yaw, 0)

        #     # Apply the scene transformation to the new eye rotation
        #     eye_R = np.dot(eye_R, RT[:3, :3])

        #     # Compute the rotation matrix to rotate the current to the target
        #     new_eye_R = np.dot(eye_R, current_eye_R.T)
        #     eyeball_R[i] = eye_R
        #     eyeball_meshes[i].rotate(new_eye_R)

        #     # Debug, print out the mean eye gaze origin of the eyeball mesh
        #     vertices = np.array(eyeball_meshes[i].vertices)
        #     centroid = vertices.mean(axis=0)
        #     # print(f"Eye gaze origin ({k}): {centroid}")

        # Update the geometry
        visual.update_geometry(face_mesh)
        visual.update_geometry(face_mesh_lines)
        # for i in ['left', 'right']:
        #     visual.update_geometry(eyeball_meshes[i])
            # visual.update_geometry(iris_3d_pt[i])

        # Update visualizer
        visual.poll_events()
        visual.update_renderer()

        cv2.imshow("Face Mesh", draw_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
