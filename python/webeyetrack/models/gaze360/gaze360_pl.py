import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2

from .gaze360 import Gaze360
from ... import vis

class Gaze360_PL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        # Save parameters
        self.config = config

        # Create the base model
        self.base_model = Gaze360()
    
    def compute_loss(self, output, batch):

        # Combine all losess together
        # losses = [
        # ] 
        # complete_loss = torch.sum(torch.stack(losses))

        new_output = {
            'losses': {
                # 'complete_loss': complete_loss
            },
            'artifacts': {
            },
            'metrics': {
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
        
        # return {'loss': losses_output['losses']['complete_loss'], 'log': losses_output['losses']}

    def validation_step(self, batch, batch_idx):
        output = self.base_model.forward({'face': batch['face_image']})
        losses_output = self.compute_loss(output, batch)

        # self.log('val_loss', losses_output['losses']['complete_loss'])
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

            # Visualizing the gaze direction
            # gt_gaze = vis.draw_gaze_direction(img, batch['face_origin_2d'][i], batch['gaze_target_2d'][i], color=(0, 0, 255))

            # Compute center of the image
            center = np.array([img.shape[1] / 2, img.shape[0] / 2], dtype=np.float32)
            
            # Create gaze_target_2d via the direction and a fixed distance
            gaze_target_3d_semi = batch['face_origin_3d'][i] + batch['gaze_direction_3d'][i] * 100
            gaze_target_3d_semi = gaze_target_3d_semi.detach().cpu().numpy()
            gaze_target_2d, _ = cv2.projectPoints(
                gaze_target_3d_semi, 
                np.array([0, 0, 0], dtype=np.float32),
                np.array([0, 0, 0], dtype=np.float32),
                batch['intrinsics'][i].cpu().numpy(), 
                batch['dist_coeffs'][i].cpu().numpy(),
            )

            gt_gaze = vis.draw_gaze_direction(
                img,
                center,
                gaze_target_2d.flatten(),
                color=(255, 0, 0)
            )

            # Create gaze_target_2d via the direction and a fixed distance
            # gaze_target_3d_semi = batch['face_origin_3d'][i] + output['gaze_direction'][i] * 100
            # gaze_target_3d_semi = gaze_target_3d_semi.detach().cpu().numpy()
            # gaze_target_2d, _ = cv2.projectPoints(
            #     gaze_target_3d_semi, 
            #     np.array([0, 0, 0], dtype=np.float32),
            #     np.array([0, 0, 0], dtype=np.float32),
            #     batch['intrinsics'][i].cpu().numpy(), 
            #     batch['dist_coeffs'][i].cpu().numpy(),
            # )

            # gt_pred_gaze = vis.draw_gaze_direction(
            #     gt_gaze,
            #     batch['face_origin_2d'][i],
            #     gaze_target_2d.flatten(),
            #     color=(255, 0, 0)
            # )
            gaze_direction_imgs.append(gt_gaze)

        tb_logger.add_images(f"{prefix}_gaze_direction", np.moveaxis(np.stack(gaze_direction_imgs), -1, 1), self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)