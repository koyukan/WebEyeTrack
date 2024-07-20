import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

from .efemodel import EFEModel
from ..funcs import generate_2d_gaussian_heatmap_torch, reprojection_3d, screen_plane_intersection
from .unet_efficientnet_v2 import UNetEfficientNetV2Small
from .components import ResNetBlock
from ... import vis

# References:
# https://github.com/swook/EVE/blob/master/src/models/common.py#L129

class EFEModel_PL(pl.LightningModule):
    def __init__(self, config, img_size=(480, 640)):
        super().__init__()

        # Save parameters
        self.config = config
        self.img_size = img_size

        # Create the base model
        self.base_model = EFEModel(config, img_size)
    
    def compute_loss(self, output, batch):
 
        # Compute the gaze origin loss (Heatmap MSE Loss)
        # gt_heatmap = 100.0*generate_2d_gaussian_heatmap_torch(batch['face_origin_2d'], self.img_size, sigma=12.0)
        gt_heatmap = generate_2d_gaussian_heatmap_torch(batch['face_origin_2d'], self.img_size, sigma=12.0)
        gaze_origin_loss = F.mse_loss(output['gaze_origin'][:, 0], gt_heatmap)
        gaze_origin_xy_loss = F.mse_loss(output['gaze_origin_xy'], batch['face_origin_2d'])

        # Before computing the z-origin loss, we need to standardize the input and the label, based on the mean and std
        gt_gaze_origin_z = batch['face_origin_3d'][:, 2, None]
        standard_gt_gaze_origin_z = (gt_gaze_origin_z - batch['gaze_origin_depth_mean']) / batch['gaze_origin_depth_std']
        standard_pred_gaze_origin_z = (output['gaze_depth'] - batch['gaze_origin_depth_mean']) / batch['gaze_origin_depth_std']
        gaze_origin_z_loss = F.l1_loss(standard_pred_gaze_origin_z, standard_gt_gaze_origin_z)

        # Compute the gaze z-origin location (L1 Loss)
        # gaze_origin_z_loss = F.l1_loss(torch.log(output['gaze_origin_z'] + 1e6), torch.log(batch['face_origin_3d'][:, 2, None] + 1e6))
        # gaze_origin_z_loss = F.l1_loss(output['gaze_origin_z'], batch['face_origin_3d'][:, 2, None])
        # gaze_origin_z_distance = F.l1_loss(output['gaze_origin_z'], batch['face_origin_3d'][:, 2, None])

        # Compute the angular loss for the gaze direction
        # https://github.com/swook/EVE/blob/master/src/losses/angular.py#L29
        sim = F.cosine_similarity(output['gaze_direction'], batch['gaze_direction_3d'], dim=1, eps=1e-8)
        sim = F.hardtanh(sim, min_val=-1.0+1e-8, max_val=1.0-1e-8)
        angular_loss = torch.acos(sim)
        angular_loss = torch.mean(angular_loss)

        # Compute the PoG MSE Loss
        # This requires multiple steps: 3D reprojection and screen plane intersection
        gaze_origin_3d = reprojection_3d(output['gaze_origin_xy'], output['gaze_origin_z'], batch['intrinsics'])
        pog_mm = screen_plane_intersection(
            gaze_origin_3d.float(),
            output['gaze_direction'].float(),
            batch['screen_R'],
            batch['screen_t']
        )
        pog_norm = torch.stack([pog_mm[:, 0] / batch['screen_width_mm'][:, 0], pog_mm[:, 1] / batch['screen_height_mm'][:, 0]], dim=1)[:,:,0]
        pog_px = torch.stack([pog_norm[:, 0] * batch['screen_width_px'][:, 0], pog_norm[:, 1] * batch['screen_height_px'][:, 0]], dim=1)[:,:,0]

        # Before computing PoG loss, we need to standardize the input and the label, based on the mean and std
        standard_gt_pog_px = (batch['pog_px'] - batch['pog_px_mean']) / batch['pog_px_std']
        standard_pred_pog_px = (pog_px - batch['pog_px_mean']) / batch['pog_px_std']
        pog_loss = F.mse_loss(standard_pred_pog_px, standard_gt_pog_px)

        # Combine all losess together
        losses = [
            gaze_origin_loss * self.config['hparams']['xy_heatmap_loss'], 
            gaze_origin_xy_loss * self.config['hparams']['xy_loss'], 
            gaze_origin_z_loss * self.config['hparams']['z_loss'], 
            angular_loss * self.config['hparams']['angular_loss'], 
            # pog_loss * self.config['hparams']['pog_loss']
        ] 
        complete_loss = torch.sum(torch.stack(losses))

        new_output = {
            'losses': {
                'gaze_origin_heatmap_loss': gaze_origin_loss,
                'gaze_origin_xy_loss': gaze_origin_xy_loss,
                'gaze_origin_z_loss': gaze_origin_z_loss,
                'gaze_angular_loss': angular_loss,
                'pog_loss': pog_loss,
                'complete_loss': complete_loss
            },
            'artifacts': {
                'gaze_origin_heatmap': gt_heatmap,
                'pog_px': pog_px,
            },
            'metrics': {
                # 'gaze_origin_z_distance': gaze_origin_z_distance,
            }
        }

        return new_output

    def training_step(self, batch, batch_idx):
        output = self.base_model.forward(batch['image'])
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
        output = self.base_model.forward(batch['image'])
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
        np_cpu_images = batch['image'].cpu().numpy()

        gaze_origin = []
        gaze_origin_heatmaps = []
        # gaze_origin_heatmaps_gt = []
        gaze_direction_imgs = []
        gaze_depth_imgs = []
        gaze_pog_imgs = []
        for i in range(np_cpu_images.shape[0]):
            img = np.moveaxis(np_cpu_images[i], 0, -1)

            # Visualize the gaze origin xy
            gaze_origin_xy = batch['face_origin_2d'][i].cpu().numpy()
            vis_gt_gaze_origin = vis.draw_gaze_origin(img, gaze_origin_xy, color=(0, 0, 255))
            vis_gaze_origin = vis.draw_gaze_origin(vis_gt_gaze_origin, output['gaze_origin_xy'][i].detach().cpu().numpy(), color=(255, 0, 0))
            gaze_origin.append(vis_gaze_origin)

            # Visualizing the gaze origin heatmaps
            gt_gaze_origin_heatmap = losses_output['artifacts']['gaze_origin_heatmap'][i].detach().cpu().numpy()
            vis_gt_gaze_origin_heatmap = vis.draw_gaze_origin_heatmap(img, gt_gaze_origin_heatmap)
            gaze_origin_heatmap = output['gaze_origin'][i].detach().cpu().numpy()
            vis_gaze_origin_heatmap = vis.draw_gaze_origin_heatmap(img, gaze_origin_heatmap[0])

            # Concatenate the heatmaps together
            vis_gaze_origin_heatmaps = np.concatenate([vis_gt_gaze_origin_heatmap, vis_gaze_origin_heatmap], axis=1)
            gaze_origin_heatmaps.append(vis_gaze_origin_heatmaps)

            # Visualize the sparse depth map
            gaze_depth = output['gaze_depth'][i].detach().cpu().numpy()
            vis_gaze_depth = vis.draw_gaze_depth_map(img, gaze_depth[0])
            gaze_depth_imgs.append(vis_gaze_depth)

            # Visualizing the gaze direction
            gt_gaze = vis.draw_gaze_direction(img, batch['face_origin_2d'][i], batch['gaze_target_2d'][i], color=(0, 0, 255))

            # Create gaze_target_2d via the direction and a fixed distance
            gaze_target_3d_semi = batch['face_origin_3d'][i] + output['gaze_direction'][i] * 100
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

            # Draw the PoG Image
            pog_px = batch['pog_px'][i].detach().cpu().numpy()
            screen_height_px = batch['screen_height_px'][i].detach().cpu().numpy().squeeze()
            screen_width_px = batch['screen_width_px'][i].detach().cpu().numpy().squeeze()
            screen_height_mm = batch['screen_height_mm'][i].detach().cpu().numpy().squeeze()
            screen_width_mm = batch['screen_width_mm'][i].detach().cpu().numpy().squeeze()
            pog_norm = np.array([pog_px[0] / screen_width_px, pog_px[1] / screen_height_px])

            vis_screen_height = screen_height_px // 4
            vis_screen_width = screen_width_px // 4
            pog_point = (int(vis_screen_width * pog_norm[0]), int(vis_screen_height * pog_norm[1]))
            pog_img = np.ones((int(screen_height_px), int(screen_width_px), 3), dtype=np.uint8) * 255
            
            pog_px_pred = losses_output['artifacts']['pog_px'][i].detach().cpu().numpy().squeeze()
            pog_norm_pred = np.array([pog_px_pred[0] / screen_width_px, pog_px_pred[1] / screen_height_px])
            pog_point_pred = (int(vis_screen_width * pog_norm_pred[0]), int(vis_screen_height * pog_norm_pred[1]))
            # pog_mm_norm = np.array([pog_mm[0] / screen_width_mm, pog_mm[1] / screen_height_mm])
            # pog_point_pred = (int(vis_screen_width * pog_mm_norm[0]), int(vis_screen_height * pog_mm_norm[1]))

            # Make the center of the image black (with a white border equal to 10)
            pog_img[10:-10, 10:-10] = 0
            vis_gt_pog = vis.draw_pog(pog_img, pog_point, color=(0, 0, 255))
            vis_pred_pog = vis.draw_pog(vis_gt_pog, pog_point_pred, color=(255, 0, 0))
            gaze_pog_imgs.append(vis_pred_pog)

        tb_logger.add_images(f"{prefix}_gaze_origin", np.moveaxis(np.stack(gaze_origin), -1, 1), self.current_epoch)
        tb_logger.add_images(f"{prefix}_pred_gaze_origin_heatmaps", np.moveaxis(np.stack(gaze_origin_heatmaps), -1, 1), self.current_epoch)
        # tb_logger.add_images(f"{prefix}_gt_gaze_origin_heatmaps", np.moveaxis(np.stack(gaze_origin_heatmaps_gt), -1, 1), self.current_epoch)
        tb_logger.add_images(f"{prefix}_gaze_depth", np.moveaxis(np.stack(gaze_depth_imgs), -1, 1), self.current_epoch)
        tb_logger.add_images(f"{prefix}_gaze_direction", np.moveaxis(np.stack(gaze_direction_imgs), -1, 1), self.current_epoch)
        # tb_logger.add_images(f"{prefix}_pog", np.moveaxis(np.stack(gaze_pog_imgs), -1, 1), self.current_epoch)

        # Pog Images can be different sizes (since the target screen dimensions can be different)
        for i, img in enumerate(gaze_pog_imgs):
            tb_logger.add_image(f"{prefix}_pog_{i}", img, self.current_epoch, dataformats='HWC')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)