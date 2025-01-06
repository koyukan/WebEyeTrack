import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import open3d as o3d
import pathlib

from webeyetrack.constants import GIT_ROOT
from webeyetrack.datasets.utils import draw_landmarks_on_image

# According to https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/graphs/face_effect/face_effect_gpu.pbtxt#L61-L65
# vertical_fov_degrees = 50
vertical_fov_degrees = 60
# vertical_fov_degrees = 63.0 
# vertical_fov_degrees = 90.0
near = 1.0 # 1cm
far = 10000 # 100m 

origin_point_location = 'BOTTOM_LEFT_CORNER'

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

def create_perspective_matrix(aspect_ratio):
    k_degrees_to_radians = np.pi / 180.0

    # Initialize a 4x4 matrix filled with zeros
    perspective_matrix = np.zeros((4, 4), dtype=np.float32)

    # Standard perspective projection matrix calculations
    f = 1.0 / np.tan(k_degrees_to_radians * vertical_fov_degrees / 2.0)
    denom = 1.0 / (near - far)

    # Populate the matrix values
    perspective_matrix[0, 0] = f / aspect_ratio
    perspective_matrix[1, 1] = f
    perspective_matrix[2, 2] = (near + far) * denom
    perspective_matrix[2, 3] = -1.0
    perspective_matrix[3, 2] = 2.0 * far * near * denom

    # Flip Y-axis if origin point location is top-left corner
    if origin_point_location == 'TOP_LEFT_CORNER':
        perspective_matrix[1, 1] *= -1.0

    return perspective_matrix

if __name__ == "__main__":
    CWD = pathlib.Path(__file__).parent
    PYTHON_DIR = CWD.parent

    # Load the facemesh triangles
    facemesh_triangles = np.load(GIT_ROOT / 'python' / 'assets' / 'facemesh_triangles.npy')
    canonical_mesh = o3d.io.read_triangle_mesh(str(GIT_ROOT / 'python' / 'assets' / 'canonical_face_model.obj'))
    canonical_mesh.vertices = o3d.utility.Vector3dVector(np.array(canonical_mesh.vertices))
    canonical_mesh.triangles = o3d.utility.Vector3iVector(facemesh_triangles)
    
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    perspective_matrix = create_perspective_matrix(aspect_ratio=width / height)
    inv_perspective_matrix = np.linalg.inv(perspective_matrix)

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

        # Compute the face bounding box based on the MediaPipe landmarks
        try:
            face_landmarks_proto = detection_results.face_landmarks[0]
        except:
            continue

        # Extract information fro the results
        face_landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility, lm.presence] for lm in face_landmarks_proto])

        # Convert uvz to xyz
        relative_face_mesh = np.array([convert_uv_to_xyz(perspective_matrix, x[0], x[1], x[2]) for x in face_landmarks[:, :3]])

        # Draw the landmarks
        frame = draw_landmarks_on_image(frame, detection_results)

        cv2.imshow("Face Mesh", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # If character is 's', save the mesh
        elif cv2.waitKey(1) & 0xFF == ord("s"):
            # Center to the nose 
            nose = relative_face_mesh[4]
            relative_face_mesh = relative_face_mesh - nose
            
            # make the width of the face length=1
            leftmost = np.min(relative_face_mesh[:, 0])
            rightmost = np.max(relative_face_mesh[:, 0])
            relative_face_mesh[:, :] /= rightmost - leftmost

            # Save
            canonical_mesh.vertices = o3d.utility.Vector3dVector(relative_face_mesh)
            o3d.io.write_triangle_mesh('face_mesh.obj', canonical_mesh)
            break