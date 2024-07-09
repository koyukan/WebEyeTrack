import torch

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
    nn_plane_other = torch.tensor([1, 0, 0], dtype=torch.float32, device=device).view(1, 3, 1)

    # Define plane to intersect with
    n = nn_plane_normal
    a = nn_plane_other
    g = g.view(-1, 3, 1)
    o = o.view(-1, 3, 1)
    numer = torch.sum(torch.mul(a - o, n), dim=1)

    # Intersect with plane using provided 3D origin
    denom = torch.sum(torch.mul(g, n), dim=1) + 1e-7
    t = torch.div(numer, denom).view(-1, 1, 1)
    return (o + torch.mul(t, g))[:, :2, 0]

def screen_plane_intersection(o, d, ppm_w, ppm_h, screen_size):
    
    # Determine the intersection with the z=0 plane
    pog_mm = get_intersect_with_zero(o, d)

    # Convert to pixels
    pog_px = torch.stack([
        torch.clamp(pog_mm[:, 0] * ppm_w,
                    0.0, float(screen_size[0])),
        torch.clamp(pog_mm[:, 1] * ppm_h,
                    0.0, float(screen_size[1]))
    ], axis=-1)

    return pog_mm, pog_px