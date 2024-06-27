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
    heatmaps = torch.exp(-((xx - gaze_origins[:, 0]) ** 2 + (yy - gaze_origins[:, 1]) ** 2) / (2 * sigma ** 2))
    
    return heatmaps