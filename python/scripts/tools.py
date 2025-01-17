import open3d as o3d
import trimesh
import numpy as np

from webeyetrack.constants import *
from webeyetrack.utilities import transform_for_3d_scene

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

def load_camera_frustrum(w_ratio, h_ratio, visual):
    
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
    transformed_pts = transform_for_3d_scene(points)

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
    visual.add_geometry(frustum)

def load_screen_rect(visual, screen_width_mm, screen_height_mm):
    
    # Screen Display
    rw, rh = screen_width_mm / 10, screen_height_mm / 10
    rectangle_points = np.array([
        [-rw/2, 0, 0],
        [rw/2, 0, 0],
        [rw/2, rh, 0],
        [-rw/2, rh, 0]
    ]).astype(np.float32)

    # Define triangles using indices to the points (two triangles to form a rectangle)
    triangles = np.array([
        [0, 1, 2],  # Triangle 1
        [0, 2, 3]   # Triangle 2
    ])

    # Apply the Open3D transformation
    transformed_pts = transform_for_3d_scene(rectangle_points)

    # Create the TriangleMesh object
    rectangle_mesh = o3d.geometry.TriangleMesh()
    rectangle_mesh.vertices = o3d.utility.Vector3dVector(transformed_pts)
    rectangle_mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Set the color for each vertex
    rectangle_mesh.paint_uniform_color([0, 0, 0])  # Red color

    visual.add_geometry(rectangle_mesh)

def load_pog_balls(visual, scale):
    
    # PoG
    left_pog = o3d.geometry.TriangleMesh.create_sphere(radius=12 * scale)
    right_pog = o3d.geometry.TriangleMesh.create_sphere(radius=12 * scale)
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