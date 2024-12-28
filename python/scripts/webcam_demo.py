import pathlib
import time
import platform

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

# Based on platform, use different approaches for determining size
# For Windows and Linux, use the screeninfo library
# For MacOS, use the Quartz library
if platform.system() == 'Windows' or platform.system() == 'Linux':
    from screeninfo import get_monitors
    m = get_monitors()[0]
    SCREEN_HEIGHT_MM = m.height_mm
    SCREEN_WIDTH_MM = m.width_mm
    SCREEN_HEIGHT_PX = m.height
    SCREEN_WIDTH_PX = m.width
elif platform.system() == 'Darwin':
    import Quartz
    main_display_id = Quartz.CGMainDisplayID()
    width_mm, height_mm = Quartz.CGDisplayScreenSize(main_display_id)
    width_px, height_px = Quartz.CGDisplayPixelsWide(main_display_id), Quartz.CGDisplayPixelsHigh(main_display_id)
    SCREEN_HEIGHT_MM = height_mm
    SCREEN_WIDTH_MM = width_mm
    SCREEN_HEIGHT_PX = height_px
    SCREEN_WIDTH_PX = width_px

SCALE = 2e-3
print(f"Screen Height: {SCREEN_HEIGHT_MM} mm, Screen Width: {SCREEN_WIDTH_MM} mm")

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
    visual.create_window(width=1920, height=1080)
    visual.get_render_option().background_color = [0.1, 0.1, 0.1]

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
    print(rectangle_points)

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

    # Eyeballs
    left_eyeball = o3d.geometry.TriangleMesh.create_sphere(radius=12 * SCALE)
    right_eyeball = o3d.geometry.TriangleMesh.create_sphere(radius=12 * SCALE)
    left_eyeball.paint_uniform_color([1, 1, 1])
    right_eyeball.paint_uniform_color([1, 1, 1])
    visual.add_geometry(left_eyeball)
    visual.add_geometry(right_eyeball)

    # PoG
    left_pog = o3d.geometry.TriangleMesh.create_sphere(radius=12 * SCALE)
    right_pog = o3d.geometry.TriangleMesh.create_sphere(radius=12 * SCALE)
    left_pog.paint_uniform_color([0, 1, 0])
    right_pog.paint_uniform_color([0, 0, 1])
    visual.add_geometry(left_pog)
    visual.add_geometry(right_pog)

    # Initial Setup for Gaze Vectors
    left_gaze_vector = o3d.geometry.LineSet()
    left_gaze_vector.paint_uniform_color([0, 1, 0])  # Green color for left
    right_gaze_vector = o3d.geometry.LineSet()
    right_gaze_vector.paint_uniform_color([0, 0, 1])  # Blue color for right

    # Add the gaze vectors to the visualizer
    visual.add_geometry(left_gaze_vector)
    visual.add_geometry(right_gaze_vector)

    # Draw an xyz coordinate axis with lines of length 10mm
    axis_length = rw #mm
    axis_x= o3d.geometry.LineSet()
    axis_x.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
    points = np.array([[0,0,0], [axis_length,0,0]]) * SCALE
    lines = np.array([[0, 1]])
    axis_x.points = o3d.utility.Vector3dVector(points)
    axis_x.lines = o3d.utility.Vector2iVector(lines)
    visual.add_geometry(axis_x)

    axis_y= o3d.geometry.LineSet()
    axis_y.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
    points = np.array([[0,0,0], [0,axis_length,0]]) * SCALE
    lines = np.array([[0, 1]])
    axis_y.points = o3d.utility.Vector3dVector(points)
    axis_y.lines = o3d.utility.Vector2iVector(lines)
    visual.add_geometry(axis_y)

    axis_z= o3d.geometry.LineSet()
    axis_z.colors = o3d.utility.Vector3dVector([[0, 0, 1]])
    points = np.array([[0,0,0], [0,0,axis_length]]) * SCALE
    lines = np.array([[0, 1]])
    axis_z.points = o3d.utility.Vector3dVector(points)
    axis_z.lines = o3d.utility.Vector2iVector(lines)
    visual.add_geometry(axis_z)
    
    # Pipeline
    pipeline = FLGE(str(GIT_ROOT / 'python'/ 'weights' / 'face_landmarker_v2_with_blendshapes.task'), EYE_TRACKING_APPROACH)

    # Update the visualizer to patch the camera position
    # parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_DEFAULT.json")
    # K = np.eye(4) 
    # K[:3, :3] = np.array([[0,1,0],
    #                      [1,0,0],
    #                      [0,0,-1]])
    # K[:3, -1] = np.array([10,10,10])
    # parameters.extrinsic = K
    # ctr = visual.get_view_control()
    # ctr.convert_from_pinhole_camera_parameters(parameters)

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
            # screen_R=np.deg2rad(np.array([0, -180, 0]).astype(np.float32)),
            # screen_t=np.array([0.5*SCREEN_WIDTH_MM, 0, 0]).astype(np.float32),
            screen_R=np.deg2rad(np.array([0, 0, 0]).astype(np.float32)),
            # screen_t=np.array([SCREEN_WIDTH_MM/2, 0, 0]).astype(np.float32),
            screen_t=np.array([0, 0, 0]).astype(np.float32),
            screen_width_mm=SCREEN_WIDTH_MM,
            screen_height_mm=SCREEN_HEIGHT_MM,
            screen_width_px=SCREEN_WIDTH_PX,
            screen_height_px=SCREEN_HEIGHT_PX
        )

        if result:

            # Get 3D landmark positions for the Face Mesh
            points = result.tf_facial_landmarks[:, :3] * SCALE
            colors = np.array([[1, 0, 0] for _ in range(points.shape[0])])
            point_cloud.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
            point_cloud.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))
            visual.update_geometry(point_cloud)

            # Compare left and right PoG
            # print(f"Left {result.left.pog_mm}, Right {result.right.pog_mm}")

            # Draw the 3D eyeball and gaze vector
            for side in ['left', 'right']:
                e = result.left if side == 'left' else result.right
                ball = left_eyeball if side == 'left' else right_eyeball
                gaze_vector = left_gaze_vector if side == 'left' else right_gaze_vector
                pog = left_pog if side == 'left' else right_pog

                # Eyeball
                ball.translate(e.origin * SCALE, relative=False)
                visual.update_geometry(ball)

                # Gaze vector
                # direction = e.direction # unit xyz vector
                points = np.array([e.origin, e.origin + e.direction * np.array([-1, -1, 1]) * 1e3]) * SCALE
                lines = np.array([[0, 1]])

                # Update geometry in the visualizer
                gaze_vector.points = o3d.utility.Vector3dVector(points)
                gaze_vector.lines = o3d.utility.Vector2iVector(lines)
                if side == 'left':
                    gaze_vector.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
                else:
                    gaze_vector.colors = o3d.utility.Vector3dVector([[0, 0, 1]])
                visual.update_geometry(gaze_vector)

                print(f"Origin: {e.origin}, Direction: {e.direction}, PoG: {e.pog_mm}") 
                # Transform the PoG to match the 3D coordinate space
                # e.pog_mm[0] = -e.pog_mm[0] + SCREEN_WIDTH_MM/2 # x-axis
                # e.pog_mm[1] = -e.pog_mm[1]

                pog.translate(np.array([e.pog_mm[0], e.pog_mm[1], 0]) * SCALE, relative=False)
                visual.update_geometry(pog)

            # Update visualizer
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
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # 27 is the ESC key
            break

    cv2.destroyAllWindows()
    cap.release()
    visual.destroy_window()
