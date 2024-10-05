from collections import defaultdict

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .model import EyeNet
from ..funcs import angular_error
from ... import vis

class EyeNet_PL(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = EyeNet()

    def compute_loss(self, output, batch):
        
        # Compute the angular loss for the gaze direction
        # https://github.com/swook/EVE/blob/master/src/losses/angular.py#L29
        gt_label = batch['gaze_direction_3d']
        # gt_label = batch['relative_gaze_vector']
        sim = F.cosine_similarity(output['gaze_direction'], gt_label, dim=1, eps=1e-8)
        # sim = F.cosine_similarity(output['gaze_direction'], batch['relative_gaze_vector'], dim=1, eps=1e-8)
        sim = F.hardtanh(sim, min_val=-1.0+1e-8, max_val=1.0-1e-8)
        angular_loss = torch.acos(sim)
        angular_loss = torch.mean(angular_loss)

        angular_degree_error = angular_error(output['gaze_direction'], gt_label)

        losses = [
            angular_loss,
        ]

        complete_loss = torch.sum(torch.stack(losses))

        return {
            'angular_loss': angular_loss,
            'complete_loss': complete_loss,
            'angular_error': angular_degree_error.mean().item(),
        }

    def training_step(self, batch, batch_idx):
        output = self.model(batch['face_image'])
        output_metrics = self.compute_loss(output, batch)

        # Logging the losses
        for k, v in output_metrics.items():
            self.log(f'train_{k}', v, batch_size=batch['image'].shape[0], on_step=True, on_epoch=True)

        return {'loss': output_metrics['complete_loss'], 'log': output_metrics}

    def validation_step(self, batch, batch_idx):
        output = self.model(batch['face_image'])
        output_metrics = self.compute_loss(output, batch)

        self.log('val_loss', output_metrics['complete_loss'])
        for k, v in output_metrics.items():
            if k != 'complete_loss':
                self.log(f'val_{k}', v, batch_size=batch['image'].shape[0], on_step=True, on_epoch=True)

        if batch_idx % 100 == 0:
            self.log_tb_images('val', batch, output, output_metrics)

        return {'loss': output_metrics['complete_loss'], 'log': output_metrics}
    
    def log_tb_images(self, prefix, batch, output, losses_output):

        tb_logger = self.logger.experiment
        
        # Log the images (Give them different names)
        np_cpu_images = batch['image'].cpu().numpy()

        gaze_direction_imgs = []
        for i in range(np_cpu_images.shape[0]):
            img = np.moveaxis(np_cpu_images[i], 0, -1)

            # Visualizing the gaze direction
            gt_gaze = vis.draw_gaze_direction(img, batch['face_origin_2d'][i], batch['gaze_target_2d'][i], color=(0, 0, 255))

            # The model now predicts gaze relative to the head pose, compute the true gaze
            # true_gaze_direction = output['gaze_direction'][i] + batch['mediapipe_head_vector'][i]
            # true_gaze_direction = true_gaze_direction / torch.norm(true_gaze_direction)

            # Create gaze_target_2d via the direction and a fixed distance
            gaze_target_3d_semi = batch['face_origin_3d'][i] + output['gaze_direction'][i] * 100
            # gaze_target_3d_semi = batch['face_origin_3d'][i] + true_gaze_direction * 100
            gaze_target_3d_semi = gaze_target_3d_semi.detach().cpu().numpy()
            gaze_target_2d, _ = cv2.projectPoints(
                gaze_target_3d_semi, 
                np.array([0, 0, 0], dtype=np.float32),
                np.array([0, 0, 0], dtype=np.float32),
                batch['intrinsics'][i].cpu().numpy(), 
                batch['dist_coeffs'][i].cpu().numpy(),
            )

            gt_pred_gaze = vis.draw_gaze_direction(
                gt_gaze,
                batch['face_origin_2d'][i],
                gaze_target_2d.flatten(),
                color=(255, 0, 0)
            )
            gaze_direction_imgs.append(gt_pred_gaze)

        tb_logger.add_images(f"{prefix}_gaze_direction", np.moveaxis(np.stack(gaze_direction_imgs), -1, 1), self.current_epoch)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

