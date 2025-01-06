import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import open3d as o3d
import pathlib

from webeyetrack.constants import GIT_ROOT
from webeyetrack.datasets.utils import draw_landmarks_on_image

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

    # 1) Load canonical mesh
    mesh_path = str(GIT_ROOT / 'python' / 'assets' / 'canonical_face_model.obj')
    canonical_mesh = load_canonical_mesh(mesh_path)
    canonical_pts_3d = np.asarray(canonical_mesh.vertices, dtype=np.float32)

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

        # Now do the final projection
        final_projected_pts = transform_canonical_mesh(
            canonical_pts_3d, final_transform, width, height
        )

        # Draw the transformed face mesh
        for triangle in canonical_mesh.triangles:
            p1 = final_projected_pts[triangle[0]]
            p2 = final_projected_pts[triangle[1]]
            p3 = final_projected_pts[triangle[2]]
            cv2.line(frame, p1, p2, (0, 255, 0), 1)
            cv2.line(frame, p2, p3, (0, 255, 0), 1)
            cv2.line(frame, p3, p1, (0, 255, 0), 1)

        cv2.imshow("Face Mesh", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
