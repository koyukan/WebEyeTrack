import numpy as np
import trimesh

def compute_point_of_gaze(gaze_origin, gaze_direction, screen_rotation, screen_translation, screen_normal, screen_point):
    # Convert inputs to numpy arrays
    o = np.array(gaze_origin, dtype=np.float32)
    r = np.array(gaze_direction, dtype=np.float32)
    R = np.array(screen_rotation, dtype=np.float32)
    t = np.array(screen_translation, dtype=np.float32)
    n_s = np.array(screen_normal, dtype=np.float32)
    a_s = np.array(screen_point, dtype=np.float32)

    # Transform gaze origin and direction to screen coordinates
    o_s = np.dot(R, o) + t
    r_s = np.dot(R, r)

    # Calculate the distance to the screen plane
    lambda_ = np.dot((a_s - o_s), n_s) / np.dot(r_s, n_s)
    # lambda_ = np.dot(r_s, n_s) / np.dot((a_s - o_s), n_s)
    print(f"Distance to screen plane: {lambda_}")

    # Find the point of gaze
    print(f"o_s: {o_s}")
    print(f"r_s: {r_s}")
    p = o_s + lambda_ * r_s

    return p

# def flip_y_axis(p):
#     p[1] = p[1] * -1

# Example usage
gaze_origin = [0, 0, 0]  # Example values
# gaze_direction = [0, 0, 1]
gaze_direction = np.array([0, 0.2, 0.8])
gaze_direction = gaze_direction / np.linalg.norm(gaze_direction)
screen_rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
screen_translation = [0, 0, 1]
screen_normal = [0, 0, 1]
screen_point = [0, 0, 0]

p = compute_point_of_gaze(gaze_origin, gaze_direction, screen_rotation, screen_translation, screen_normal, screen_point)
print("Point of Gaze:", p)

# Convert the point of gaze to the camera coordinate system to visualize it
R = np.linalg.inv(screen_rotation)
t = -np.dot(R, screen_translation)
p_c = np.dot(R, p - t)
# p_c[1] = p_c[1]*-1

# Visualization with trimesh
scene = trimesh.Scene()

# Add camera (gaze origin)
camera = trimesh.creation.axis(origin_size=0.01, origin_color=[1, 0, 0])
scene.add_geometry(camera)

# Add screen coordinate system
screen_cs = trimesh.creation.axis(origin_size=0.01, origin_color=[0, 1, 0])
transform = np.eye(4)
transform[:3, :3] = screen_rotation
transform[:3, 3] = screen_translation
screen_cs.apply_transform(transform)
scene.add_geometry(screen_cs)

# Add screen plane
screen = trimesh.creation.box(extents=[1, 1, 0.001])  # Create a thin box to represent the screen
screen.apply_transform(transform)
scene.add_geometry(screen)

# Add gaze direction
gaze_vector = np.array(gaze_direction) * 0.5
gaze_line = trimesh.load_path([gaze_origin, gaze_origin + gaze_vector])
scene.add_geometry(gaze_line)

# Add point of gaze
point_gaze_sphere = trimesh.creation.icosphere(radius=0.02, color=(255,0,0))
point_gaze_sphere.apply_translation(p_c)
scene.add_geometry(point_gaze_sphere)

# Show the scene
scene.show()
