import torch

def angular_error(gaze, label):
    """
    Computes the angular error between the gaze vectors and the label vectors.
    
    Parameters:
    - gaze: Tensor of shape (batch_size, 3) containing the gaze vectors.
    - label: Tensor of shape (batch_size, 3) containing the label vectors.
    
    Returns:
    - angular_error: Tensor of shape (batch_size) containing the angular errors in degrees.
    """
    # Compute the dot product between the gaze and label vectors
    total = torch.sum(gaze * label, dim=1)
    
    # Compute the norms of the gaze and label vectors
    gaze_norm = torch.norm(gaze, dim=1)
    label_norm = torch.norm(label, dim=1)
    
    # Compute the cosine of the angle
    cos_theta = total / (gaze_norm * label_norm)
    
    # Ensure the value is within the valid range for arccos
    cos_theta = torch.clamp(cos_theta, max=0.9999999)
    
    # Compute the angular error in radians and then convert to degrees
    angular_error_rad = torch.acos(cos_theta)
    angular_error_deg = torch.rad2deg(angular_error_rad)
    
    return angular_error_deg


def generate_2d_gaussian_heatmap_torch(gaze_origins, img_size, sigma=1):
    """
    Generate a batch of 2D Gaussian heatmaps for the gaze origins.
    
    Parameters:
    - gaze_origins: Tensor of shape (batch_size, 2) containing the xy coordinates of the gaze origins.
    - img_size: Tuple containing the size of the output heatmap (height, width).
    - sigma: Standard deviation of the Gaussian.
    
    Returns:
    - heatmaps: Tensor of shape (batch_size, height, width) containing the Gaussian heatmaps.
    """
    batch_size = gaze_origins.shape[0]
    
    # Create a meshgrid
    x = torch.arange(0, img_size[0], 1, dtype=torch.float32, device=gaze_origins.device)
    y = torch.arange(0, img_size[1], 1, dtype=torch.float32, device=gaze_origins.device)
    xx, yy = torch.meshgrid(x, y)
    xx = xx.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (batch_size, height, width)
    yy = yy.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (batch_size, height, width)
    
    # Reshape gaze_origins to (batch_size, 2, 1, 1) for broadcasting
    gaze_origins = gaze_origins.unsqueeze(-1).unsqueeze(-1)
    
    # Compute the Gaussian
    heatmaps = torch.exp(-((xx - gaze_origins[:, 1]) ** 2 + (yy - gaze_origins[:, 0]) ** 2) / (2 * sigma ** 2))
    
    return heatmaps

def rodrigues_rotation_matrix_batch(rotation_vectors):
    """
    Convert a batch of rotation vectors to rotation matrices using the Rodrigues' rotation formula.

    Args:
        rotation_vectors (torch.Tensor): A tensor of shape (N, 3) representing the batch of rotation vectors (axis-angle).

    Returns:
        torch.Tensor: A tensor of shape (N, 3, 3) representing the batch of rotation matrices.
    """
    batch_size = rotation_vectors.shape[0]
    theta = torch.norm(rotation_vectors, dim=1, keepdim=True)
    r = rotation_vectors / (theta + 1e-8)  # Prevent division by zero
    r = r.unsqueeze(2)  # Shape (N, 3, 1)
    
    zero = torch.zeros(batch_size, 1, 1, dtype=torch.float32, device=rotation_vectors.device)
    K = torch.cat([
        zero, -r[:, 2:3], r[:, 1:2],
        r[:, 2:3], zero, -r[:, 0:1],
        -r[:, 1:2], r[:, 0:1], zero
    ], dim=1).view(batch_size, 3, 3)  # Shape (N, 3, 3)
    
    I = torch.eye(3, device=rotation_vectors.device).unsqueeze(0).repeat(batch_size, 1, 1)  # Shape (N, 3, 3)
    K2 = torch.bmm(K, K)
    
    sin_theta = torch.sin(theta).view(batch_size, 1, 1)
    cos_theta = torch.cos(theta).view(batch_size, 1, 1)
    
    R = I + sin_theta * K + (1 - cos_theta) * K2
    
    return R


def inverse_rotation_matrix_batch(rotation_matrices):
    """
    Compute the inverse of a batch of rotation matrices.

    Args:
        rotation_matrices (torch.Tensor): A tensor of shape (N, 3, 3) representing the batch of rotation matrices.

    Returns:
        torch.Tensor: A tensor of shape (N, 3, 3) representing the batch of inverse rotation matrices.
    """
    # Compute the transpose of each rotation matrix
    inverse_matrices = rotation_matrices.transpose(1, 2)
    
    return inverse_matrices


def reprojection_3d(xy_points, depth, intrinsic_matrices):
    """
    Reproject 2D points with depth information to 3D points using the camera intrinsics.
    
    Parameters:
    - xy_points: Tensor of shape (batch_size, 2) containing the xy coordinates.
    - depth: Tensor of shape (batch_size) containing the corresponding depth values.
    - intrinsic_matrices: Tensor of shape (batch_size, 3, 3) containing the camera intrinsic matrices for each batch.
    
    Returns:
    - xyz_points: Tensor of shape (batch_size, 3) containing the reprojected 3D points.
    """
    
    # Extract intrinsic parameters for each batch
    fx = intrinsic_matrices[:, 0, 0]  # Shape: (batch_size)
    fy = intrinsic_matrices[:, 1, 1]  # Shape: (batch_size)
    cx = intrinsic_matrices[:, 0, 2]  # Shape: (batch_size)
    cy = intrinsic_matrices[:, 1, 2]  # Shape: (batch_size)
    
    # Get the xy coordinates
    x = xy_points[:, 0]
    y = xy_points[:, 1]
    
    # Reproject to 3D
    X = (x - cx) * depth[:, 0] / fx
    Y = (y - cy) * depth[:, 0] / fy
    Z = depth[:, 0]
    
    # Stack to get the final 3D points
    xyz_points = torch.stack((X, Y, Z), dim=1)
    
    return xyz_points

def get_intersect_with_zero(o, g):
    """Intersects a given gaze ray (origin o and direction g) with z = 0."""
    device = o.device

    nn_plane_normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(1, 3, 1)
    nn_plane_other = torch.tensor([0, 0, 0], dtype=torch.float32, device=device).view(1, 3, 1)

    # Define plane to intersect with
    n = nn_plane_normal
    a = nn_plane_other
    g = g.view(-1, 3, 1)
    o = o.view(-1, 3, 1)
    numer = torch.sum(torch.mul(a - o, n), dim=1)

    # Intersect with plane using provided 3D origin
    denom = torch.sum(torch.mul(g, n), dim=1) + 1e-7
    t = torch.div(numer, denom).view(-1, 1, 1)
    return (o + torch.mul(t, g))

def screen_plane_intersection(o, d, screen_R, screen_t):

    # Obtain rotation and inverse matrix matrix 
    R_matrix = rodrigues_rotation_matrix_batch(screen_R[:, : ,0])
    inv_R_matrix = inverse_rotation_matrix_batch(R_matrix)

    # Transform gaze origin and direction to screen coordinates
    o_s = torch.bmm(inv_R_matrix, (o - screen_t[:,:,0]).unsqueeze(-1)).squeeze(-1)
    d_s = torch.bmm(inv_R_matrix, d.unsqueeze(-1)).squeeze(-1)
    
    # Screen plane
    a_s = torch.tensor([0, 0, 0], dtype=torch.float32, device=o.device).view(1, 3) # point
    n_s = torch.tensor([0, 0, 1], dtype=torch.float32, device=o.device).view(1, 3) # normal

    # Calculate the distance to the screen plane
    lambda_ = torch.sum((a_s - o_s) * n_s, dim=1) / torch.sum(d_s * n_s, dim=1)

    # Calculate the intersection point
    p = o_s + lambda_.unsqueeze(-1) * d_s

    # Keep only the x and y coordinates
    pog_mm = p[:, :2]

    return pog_mm