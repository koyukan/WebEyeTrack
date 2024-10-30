import pathlib
import time
from screeninfo import get_monitors

import open3d as o3d
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import imutils
import math

from webeyetrack.constants import GIT_ROOT
from webeyetrack.datasets.utils import draw_landmarks_on_image
from webeyetrack import vis
from webeyetrack.pipelines.flge import FLGE

EYE_TRACKING_APPROACH = "model-based"
# EYE_TRACKING_APPROACH = "landmark2d"
# EYE_TRACKING_APPROACH = "blendshape"

# Screen dimensions
m = get_monitors()[0]
SCREEN_HEIGHT_MM = m.height_mm
SCREEN_WIDTH_MM = m.width_mm
SCALE = 2e-3

if __name__ == '__main__':
    
    # Load the webcam 
    cap = cv2.VideoCapture(0)

    # Determine the size of the webcam frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_size = max(width, height)
    w_ratio, h_ratio = width/max_size, height/max_size

    # Initialize Open3D Visualizer
    visual = o3d.visualization.Visualizer()
    visual.create_window()

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
    frustum.points = o3d.utility.Vector3dVector(points)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector(colors)

    # Add the frustum to the visualizer
    visual.add_geometry(frustum)

    # Screen Display
    # Define rectangle corner points
    rw, rh = SCREEN_WIDTH_MM, SCREEN_HEIGHT_MM
    rectangle_points = np.array([
        [-rw/2,-rh,0],
        [rw/2,-rh,0],
        [rw/2,0,0],
        [-rw/2,0,0]
    ]).astype(np.float32)

    # Define triangles using indices to the points (two triangles to form a rectangle)
    triangles = np.array([
        [0, 1, 2],  # Triangle 1
        [0, 2, 3]   # Triangle 2
    ])

    # Create the TriangleMesh object
    rectangle_mesh = o3d.geometry.TriangleMesh()
    rectangle_mesh.vertices = o3d.utility.Vector3dVector(rectangle_points * SCALE)
    rectangle_mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Set the color for each vertex
    rectangle_mesh.paint_uniform_color([0, 0, 0])  # Red color

    # Add the rectangle mesh to the visualizer
    visual.add_geometry(rectangle_mesh)

    # Face Mesh
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array([[0,0,0], [1, 1, 1]]).reshape(-1, 3))
    point_cloud.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0], [0, 1, 0]]).reshape(-1, 3))
    visual.add_geometry(point_cloud)
    
    # Pipeline
    pipeline = FLGE(str(GIT_ROOT / 'python'/ 'weights' / 'face_landmarker_v2_with_blendshapes.task'), EYE_TRACKING_APPROACH)

    # Update the visualizer to patch the camera position
    parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_DEFAULT.json")
    K = np.eye(4) 
    K[:3, :3] = np.array([[0,1,0],
                         [1,0,0],
                         [0,0,-1]])
    K[:3, -1] = np.array([10,10,10])
    parameters.extrinsic = K
    ctr = visual.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(parameters)

    # Load the frames and draw the landmarks
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Define intrinsics based on the frame
        width, height = frame.shape[:2]
        intrinsics = np.array([[width, 0, width // 2], [0, height, height // 2], [0, 0, 1]])

        result = pipeline.process_frame(
            frame, 
            intrinsics, 
            smooth=True,
            screen_R=np.deg2rad(np.array([0, -180, 0]).astype(np.float32)),
            screen_t=np.array([0.5*m.width_mm, 0, 0]).astype(np.float32),
            screen_width_mm=m.width_mm,
            screen_height_mm=m.height_mm,
            screen_width_px=m.width,
            screen_height_px=m.height
        )

        if result:

            # Get 3D landmark positions
            # landmarks = results.multi_face_landmarks[0]
            # points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            points = result.tf_facial_landmarks[:, :3]
            # Center points at 0,0,0
            # points -= np.mean(points, axis=0)
            colors = np.array([[1, 0, 0] for _ in range(points.shape[0])])
            # points = np.array([[0,0,0], [1, 1, 1]])
            # colors = np.array([[1, 0, 0], [0, 1, 0]])

            # Update Point Cloud in Open3D
            point_cloud.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
            point_cloud.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))
            visual.update_geometry(point_cloud)
            ctr.convert_from_pinhole_camera_parameters(parameters)
            visual.poll_events()
            visual.update_renderer()
            
            # Render the PoG
            # screen = np.zeros((m.height, m.width, 3), dtype=np.uint8)
            # result.pog_px[1] = m.height/2
            # screen = vis.draw_pog(screen, result.pog_px, size=100)
            # cv2.imshow('screen', screen)

            if EYE_TRACKING_APPROACH == "model-based":
                img = vis.model_based_gaze_render(frame, result)
                if type(img) == np.ndarray:
                    cv2.imshow('frame', img) 
            elif EYE_TRACKING_APPROACH == "landmark2d":
                img = vis.landmark_gaze_render(frame, result)
                if type(img) == np.ndarray:
                    cv2.imshow('frame', img)
            elif EYE_TRACKING_APPROACH == 'blendshape':
                img = vis.blendshape_gaze_render(frame, result)
                if type(img) == np.ndarray:
                    cv2.imshow('frame', img)

        # cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
    visual.destroy_window()
