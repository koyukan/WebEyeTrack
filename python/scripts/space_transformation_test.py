import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import open3d as o3d
import pathlib

from webeyetrack.constants import GIT_ROOT
from webeyetrack.datasets.utils import draw_landmarks_on_image

CWD = pathlib.Path(__file__).parent
PYTHON_DIR = CWD.parent

# Load the canonical face mesh
def load_canonical_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.triangles))
    return mesh

# Transform the canonical face mesh to screen space
# def transform_canonical_mesh(canonical_points, rt_matrix, width, height):
#     transformed_points = []
#     for point in canonical_points:
#         # Convert to homogeneous coordinates
#         canonical_point_h = np.append(point, 1)
#         # Apply rotation-translation matrix
#         transformed_point = np.dot(rt_matrix, canonical_point_h)
#         # Perspective divide and convert to screen space
#         screen_x = (transformed_point[0] / transformed_point[2]) * width / 2 + width / 2
#         screen_y = (-transformed_point[1] / transformed_point[2]) * height / 2 + height / 2
#         transformed_points.append((int(screen_x), int(screen_y)))
#     return transformed_points

def transform_canonical_mesh(canonical_points, rt_matrix, width, height):
    """
    Projects canonical 3D points into 2D screen coordinates using a
    virtual perspective camera.

    Args:
        canonical_points: (N, 3) numpy array of 3D points in canonical space
        rt_matrix: (4, 4) rigid (or similarity) transformation matrix
        width, height: Image size for the principal point in intrinsics

    Returns:
        projected_points: (N, 2) int32 numpy array of pixel coordinates
    """
    # Intrinsic camera matrix (example)
    # fx = fy = 1000, cx = width/2, cy = height/2
    intrinsics = np.array([
        [1000,    0, width / 2],
        [   0, 1000, height / 2],
        [   0,    0,          1]
    ])

    # 1) Convert canonical 3D points to homogeneous coordinates: (x, y, z, 1)
    num_points = len(canonical_points)
    canonical_points_h = np.hstack([
        canonical_points, 
        np.ones((num_points, 1), dtype=np.float32)
    ])  # shape: (N, 4)

    # 2) Apply the 4x4 rigid (or similarity) transformation
    #    transformed_points_h will be shape (N, 4).
    transformed_points_h = (rt_matrix @ canonical_points_h.T).T

    # 3) Multiply by the 3x4 (intrinsics) if needed, or 3x3 if you treat the top-left 3x3
    #    But usually we do 3x4 by ignoring the last column of intrinsics. 
    #    Here we treat intrinsics as 3x3, so we only multiply the (x, y, z) part:
    #    Note: we only need the first 3 components (x,y,z) from the transformed points for intrinsics multiplication.
    transformed_xyz = transformed_points_h[:, :3]  # shape (N, 3)
    # shape for multiplication: (3x3) @ (3xN) => (3xN)
    camera_space = (intrinsics @ transformed_xyz.T).T  # shape (N, 3)

    # 4) Perspective divide: X'/Z', Y'/Z'
    #    Make sure we don't divide by zero (if Z' is zero or very close to zero).
    eps = 1e-6
    zs = np.where(np.abs(camera_space[:, 2]) < eps, eps, camera_space[:, 2])
    camera_space[:, 0] /= zs
    camera_space[:, 1] /= zs

    # 5) The resulting (x, y) are the pixel coordinates in the image plane
    projected_points = camera_space[:, :2]

    # 6) Convert to int (pixel) coordinates
    projected_points = projected_points.astype(np.int32)

    return projected_points

# Compute rigid transform between canonical and detected landmarks
def estimate_rigid_transform(canonical_points, detected_points):
    src = np.array(canonical_points, dtype=np.float32)
    dst = np.array(detected_points, dtype=np.float32)
    matrix, _ = cv2.estimateAffine3D(src, dst)
    if matrix is None:
        raise ValueError("Could not compute a valid rigid transform")
    return np.vstack([matrix, [0, 0, 0, 1]])

# Main function
def main():
    # Load the canonical face mesh
    mesh_path = str(GIT_ROOT / 'python' / 'assets' / 'canonical_face_model.obj')
    canonical_mesh = load_canonical_mesh(mesh_path)

    # Setup MediaPipe Face Landmark model
    base_options = python.BaseOptions(model_asset_path=str(PYTHON_DIR / 'weights' / 'face_landmarker_v2_with_blendshapes.task'))
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    # Open webcam
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip and convert frame to RGB
        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape

        # Process the frame with MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.astype(np.uint8))
        detection_results = face_landmarker.detect(mp_image)
        if len(detection_results.face_landmarks) == 0:
            cv2.imshow("Face Mesh", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Draw the landmarks
        frame = draw_landmarks_on_image(frame, detection_results)

        # Given the transformation matrix, transform the canonical face mesh to screen space
        face_rt = detection_results.facial_transformation_matrixes[0]

        # Decouple the rotation, translation and scale of the face_rt
        face_r = face_rt[:3, :3]
        face_t = face_rt[:3, 3]
        face_s = np.linalg.norm(face_r, axis=0)
        face_r /= face_s

        # Only use the rotation of the transformation matrix
        new_face_rt = np.ones((4, 4))
        new_face_rt[:3, :3] = np.eye(3)
        new_face_rt[:3, 3] = np.array([0, 0, -30])
        transformed_points = transform_canonical_mesh(canonical_mesh.vertices, new_face_rt, width, height)

        # Draw the transformed face mesh
        for triangle in canonical_mesh.triangles:
            p1 = transformed_points[triangle[0]]
            p2 = transformed_points[triangle[1]]
            p3 = transformed_points[triangle[2]]
            cv2.line(frame, p1, p2, (0, 255, 0), 1)
            cv2.line(frame, p2, p3, (0, 255, 0), 1)
            cv2.line(frame, p3, p1, (0, 255, 0), 1)

        # Show the webcam frame
        cv2.imshow("Face Mesh", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
