import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import open3d as o3d
import pathlib
import trimesh

from webeyetrack.constants import GIT_ROOT
from webeyetrack.datasets.utils import draw_landmarks_on_image

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
            total_distance += v_norm
        else:
            # The line is pointing away from the center
            # Draw the line in green
            color = (0, 255, 0)
            total_distance -= v_norm

        # cv2.line(draw_frame, tuple(p1.astype(np.int32)), tuple(p2.astype(np.int32)), color, 2)

    distance_per_point = total_distance / len(final_projected_pts)
    print(f"Distance per point: {distance_per_point}")

    # Use the total distance to update the depth
    new_z = old_z + 1e-1 * distance_per_point

    return new_z, draw_frame

def partial_procrustes_translation_2d(canonical_2d, detected_2d):
    c_center = canonical_2d.mean(axis=0)
    d_center = detected_2d.mean(axis=0)
    return d_center - c_center

def image_shift_to_3d(shift_2d, depth_z, fx=1000.0, fy=1000.0):
    dx_3d = shift_2d[0] * (depth_z / fx)
    dy_3d = shift_2d[1] * (depth_z / fy)
    return np.array([dx_3d, dy_3d, 0.0], dtype=np.float32)

def load_canonical_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.triangles))
    return mesh

def transform_canonical_mesh(canonical_points, rt_matrix, width, height):
    # same as you had before, with perspective divide...
    fx = fy = 1000.0
    intrinsics = np.array([
        [fx,     0, width / 2],
        [ 0,    fy, height / 2],
        [ 0,     0,         1]
    ])

    num_points = len(canonical_points)
    canonical_points_h = np.hstack([
        canonical_points, 
        np.ones((num_points, 1), dtype=np.float32)
    ])
    transformed_points_h = (rt_matrix @ canonical_points_h.T).T
    transformed_xyz = transformed_points_h[:, :3]
    camera_space = (intrinsics @ transformed_xyz.T).T

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
    # mesh_path = str(GIT_ROOT / 'python' / 'assets' / 'canonical_face_model.obj')
    mesh_path = str(GIT_ROOT / 'python' / 'assets' / 'myface_face_mesh.obj')
    # canonical_mesh = load_canonical_mesh(mesh_path)
    canonical_mesh = trimesh.load(mesh_path, force='mesh')
    face_width_cm = 14.0
    canonical_pts_3d = np.asarray(canonical_mesh.vertices, dtype=np.float32) * face_width_cm * np.array([-1, 1, -1])

    # Mirror the X-axis
    # canonical_pts_3d[:, 0] *= -1.0

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

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.astype(np.uint8))
        detection_results = face_landmarker.detect(mp_image)
        if not detection_results.face_landmarks:
            cv2.imshow("Face Mesh", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Draw the landmarks
        frame = draw_landmarks_on_image(frame, detection_results)

        # 3) Extract the face transformation matrix from MediaPipe
        face_rt = detection_results.facial_transformation_matrixes[0]  # shape (4,4)
        face_r = face_rt[:3, :3].copy()
        # Scale is embedded in face_r's columns
        scales = np.linalg.norm(face_r, axis=0)
        face_s = scales.mean()  # average scale
        face_r /= face_s

        # Optionally flip the x-axis
        face_r[0, :] *= -1.0

        # ---------------------------------------------------------------
        # (A) Build an initial 4x4 transform that has R, s, and some guess at Z
        #     For example, -30 in front of the camera
        # ---------------------------------------------------------------
        guess_z = -30.0
        init_transform = np.eye(4, dtype=np.float32)
        init_transform[:3, :3] = face_s * face_r
        init_transform[:3, 3]  = np.array([0, 0, guess_z], dtype=np.float32)

        # ---------------------------------------------------------------
        # (B) Project canonical mesh using this initial transform
        #     We'll get a set of 2D points in pixel space
        # ---------------------------------------------------------------
        canonical_proj_2d = transform_canonical_mesh(
            canonical_pts_3d, init_transform, width, height
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
        shift_3d = image_shift_to_3d(shift_2d, depth_z=guess_z, fx=1000.0, fy=1000.0)
        final_transform = init_transform.copy()
        final_transform[:3, 3] += shift_3d

        new_zs = [guess_z]
        for i in range(10):
            # Now do the final projection
            final_projected_pts = transform_canonical_mesh(
                canonical_pts_3d, final_transform, width, height
            )
            
            new_z, draw_frame = refine_depth_by_radial_magnitude(
                frame, final_projected_pts, detected_2d, old_z=final_transform[2, 3], alpha=0.5
            )

            # Compute the difference of the Z
            new_zs.append(new_z)
            diff_z = new_z - final_transform[2, 3]
            if np.abs(diff_z) < 1:
                break

            final_transform[2, 3] = new_z  # update the Z

        print(f"Zs: {new_zs}")

        # 7) Project again with updated Z
        final_projected_pts = transform_canonical_mesh(
            canonical_pts_3d, final_transform, width, height
        ).astype(np.int32)

        # Draw the transformed face mesh
        for triangle in canonical_mesh.faces:
            p1 = final_projected_pts[triangle[0]]
            p2 = final_projected_pts[triangle[1]]
            p3 = final_projected_pts[triangle[2]]
            cv2.line(draw_frame, p1, p2, (0, 255, 0), 1)
            cv2.line(draw_frame, p2, p3, (0, 255, 0), 1)
            cv2.line(draw_frame, p3, p1, (0, 255, 0), 1)

        # Draw the transformed face vertices only
        # for pt in final_projected_pts:
        #     cv2.circle(draw_frame, tuple(pt), 1, (0, 255, 0), -1)

        cv2.imshow("Face Mesh", draw_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
