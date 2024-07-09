import torch

def compute_point_of_gaze(gaze_origin, gaze_direction, screen_rotation, screen_translation, screen_normal, screen_point):
    # Convert inputs to tensors
    o = torch.tensor(gaze_origin, dtype=torch.float32)
    r = torch.tensor(gaze_direction, dtype=torch.float32)
    R = torch.tensor(screen_rotation, dtype=torch.float32)
    t = torch.tensor(screen_translation, dtype=torch.float32)
    n_s = torch.tensor(screen_normal, dtype=torch.float32)
    a_s = torch.tensor(screen_point, dtype=torch.float32)

    # Transform gaze origin and direction to screen coordinates
    o_s = torch.matmul(R, o) + t
    r_s = torch.matmul(R, r)

    # Calculate the distance to the screen plane
    lambda_ = torch.dot((a_s - o_s), n_s) / torch.dot(r_s, n_s)

    # Find the point of gaze
    p = o_s + lambda_ * r_s

    return p

# Example usage
gaze_origin = [0, 0, 0]  # Example values
gaze_direction = [0, 0, -1]
screen_rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
screen_translation = [0, 0, 1]
screen_normal = [0, 0, 1]
screen_point = [0, 0, 0]

p = compute_point_of_gaze(gaze_origin, gaze_direction, screen_rotation, screen_translation, screen_normal, screen_point)
print("Point of Gaze:", p)
