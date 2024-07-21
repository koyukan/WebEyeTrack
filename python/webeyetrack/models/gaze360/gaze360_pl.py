import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2

from .loss import PinBallLoss
from .gaze360 import Gaze360
from ... import vis
from ..funcs import angular_error

def spherical_to_3d_unit(spherical):
    """
    Converts gaze from spherical coordinates (phi, theta) to a 3D unit vector.
    
    Parameters:
    - gaze: Tensor of shape (batch_size, 2) containing (phi, theta) values.
    
    Returns:
    - gaze_3d: Tensor of shape (batch_size, 3) containing the 3D unit vectors.
    """
    phi = spherical[:, 0]
    theta = spherical[:, 1]
    
    gaze_3d = torch.zeros((spherical.shape[0], 3), device=spherical.device, dtype=spherical.dtype)
    gaze_3d[:, 0] = -torch.cos(theta) * torch.sin(phi)
    gaze_3d[:, 1] = -torch.sin(theta)
    gaze_3d[:, 2] = -torch.cos(theta) * torch.cos(phi)
    
    return gaze_3d

def spherical_from_3d_unit(gaze):
    """
    Converts gaze from a 3D unit vector to spherical coordinates (phi, theta).
    
    Parameters:
    - gaze_3d: Tensor of shape (batch_size, 3) containing the 3D unit vectors.
    
    Returns:
    - spherical: Tensor of shape (batch_size, 2) containing (phi, theta) values.
    """
    x = gaze[:, 0]
    y = gaze[:, 1]
    z = gaze[:, 2]
    
    theta = torch.asin(-y)
    phi = torch.atan2(-x, -z)
    
    return torch.stack((phi, theta), dim=1)

class Gaze360_PL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        # Save parameters
        self.config = config

        # Loss operation
        self.loss_op = PinBallLoss().to(self.device)

        # Create the base model
        self.base_model = Gaze360()
    
    def compute_loss(self, output, batch):

        # Decouple the output
        gaze, gaze_bias = output

        # Compute the loss
        gt_spherical = spherical_from_3d_unit(batch['gaze_direction_3d'])
        loss = self.loss_op(gaze, gt_spherical, gaze_bias)

        # Measure the angular error
        pred_gaze_3d = spherical_to_3d_unit(gaze)
        angular_errors = angular_error(pred_gaze_3d, batch['gaze_direction_3d'])

        new_output = {
            'losses': {
                'complete_loss': loss
            },
            'artifacts': {
                'pred_gaze_3d': pred_gaze_3d
            },
            'metrics': {
                'angular_error': angular_errors.mean().item()
            }
        }

        return new_output

    def training_step(self, batch, batch_idx):
        output = self.base_model.forward({'face': batch['face_image']})
        losses_output = self.compute_loss(output, batch)

        # Logging the losses
        for k, v in losses_output['losses'].items():
            self.log(f'train_{k}', v, batch_size=batch['image'].shape[0])
        for k, v in losses_output['metrics'].items():
            self.log(f'train_{k}', v, batch_size=batch['image'].shape[0])

        if batch_idx % 100 == 0:
            self.log_tb_images('train', batch, output, losses_output)
        
        return {'loss': losses_output['losses']['complete_loss'], 'log': losses_output['losses']}

    def validation_step(self, batch, batch_idx):
        output = self.base_model.forward({'face': batch['face_image']})
        losses_output = self.compute_loss(output, batch)

        self.log('val_loss', losses_output['losses']['complete_loss'])
        for k, v in losses_output['losses'].items():
            if k != 'complete_loss':
                self.log(f'val_{k}', v, batch_size=batch['image'].shape[0])
        for k, v in losses_output['metrics'].items():
            self.log(f'val_{k}', v, batch_size=batch['image'].shape[0])
        
        if batch_idx % 100 == 0:
            self.log_tb_images('val', batch, output, losses_output)

    def log_tb_images(self, prefix, batch, output, losses_output):

        tb_logger = self.logger.experiment
        
        # Log the images (Give them different names)
        np_cpu_images = batch['face_image'].cpu().numpy()

        gaze_direction_imgs = []
        for i in range(np_cpu_images.shape[0]):
            img = np.moveaxis(np_cpu_images[i], 0, -1)

            # Compute center of the image
            center = np.array([img.shape[1] / 2, img.shape[0] / 2], dtype=np.float32)
            
            # Create intrinsics based on face image
            intrinsics = np.array([
                [img.shape[1], 0, img.shape[1] / 2],
                [0, img.shape[0], img.shape[0] / 2],
                [0, 0, 1]
            ])
            
            # Create gaze_target_2d via the direction and a fixed distance
            gaze_target_3d_semi = batch['face_origin_3d'][i] + batch['gaze_direction_3d'][i] * 100
            gaze_target_3d_semi = gaze_target_3d_semi.detach().cpu().numpy()
            gaze_target_2d, _ = cv2.projectPoints(
                gaze_target_3d_semi, 
                np.array([0, 0, 0], dtype=np.float32),
                np.array([0, 0, 0], dtype=np.float32),
                intrinsics,
                batch['dist_coeffs'][i].cpu().numpy(),
            )

            gt_gaze = vis.draw_gaze_direction(
                img,
                center,
                gaze_target_2d.flatten(),
                color=(0, 0, 255)
            )

            # Create gaze_target_2d via the direction and a fixed distance
            gaze_target_3d_semi = batch['face_origin_3d'][i] + losses_output['artifacts']['pred_gaze_3d'][i] * 100
            gaze_target_3d_semi = gaze_target_3d_semi.detach().cpu().numpy()
            gaze_target_2d, _ = cv2.projectPoints(
                gaze_target_3d_semi, 
                np.array([0, 0, 0], dtype=np.float32),
                np.array([0, 0, 0], dtype=np.float32),
                intrinsics,
                batch['dist_coeffs'][i].cpu().numpy(),
            )

            gt_pred_gaze = vis.draw_gaze_direction(
                gt_gaze,
                center,
                gaze_target_2d.flatten(),
                color=(255, 0, 0)
            )
            gaze_direction_imgs.append(gt_pred_gaze)

        tb_logger.add_images(f"{prefix}_gaze_direction", np.moveaxis(np.stack(gaze_direction_imgs), -1, 1), self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)