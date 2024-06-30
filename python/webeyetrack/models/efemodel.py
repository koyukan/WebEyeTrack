import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

from .funcs import generate_2d_gaussian_heatmap_torch, reprojection_3d, screen_plane_intersection
from .unet_efficientnet_v2 import UNetEfficientNetV2Small
from .components import ResNetBlock
from .. import vis

# References:
# https://github.com/swook/EVE/blob/master/src/models/common.py#L129

class EFEModel(pl.LightningModule):
    def __init__(self, img_size=(480, 640)):
        super().__init__()

        # Save the image size
        self.img_size = img_size

        # Create components
        self.unet = UNetEfficientNetV2Small(num_classes=1, pretrained=True)
        self.resnet_gaze_origin = ResNetBlock(in_channels=1, out_channels=1)
        self.resnet_gaze_depth = ResNetBlock(in_channels=1, out_channels=1)

        # Computing the size of the bottleneck
        bottleneck_size = 160 * (img_size[0] // 16) * (img_size[1] // 16)

        self.gaze_direction_fc = nn.Sequential(
            nn.Linear(bottleneck_size, 512),
            nn.SELU(inplace=True),
            nn.Linear(512, 256),
            nn.SELU(inplace=True),
            nn.Linear(256, 3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, intrinsics):
        batch_size = x.shape[0]

        # Predict
        logits, bottleneck = self.unet(x)
        gaze_origin = self.resnet_gaze_origin(logits)
        gaze_depth = self.resnet_gaze_depth(logits)
        gaze_direction = self.gaze_direction_fc(bottleneck.view(batch_size, -1))

        # Compute the gaze origin location
        softmax_gaze_origin = F.softmax(gaze_origin.view(batch_size, -1), dim=1)
        gaze_origin_xy_idx = torch.argmax(softmax_gaze_origin, dim=1)
        gaze_origin_xy = torch.stack([gaze_origin_xy_idx % self.img_size[1], gaze_origin_xy_idx // self.img_size[1]], dim=1)
        
        # First, we apply the dot product between the softmax_gaze_origin and the depth map to get the z-origin
        gaze_depth = gaze_depth.view(batch_size, -1)
        gaze_origin_z = torch.sum(gaze_depth * softmax_gaze_origin, dim=1)

        # Normalize the gaze direction
        gaze_direction = F.normalize(gaze_direction, p=2, dim=1)

        # This requires multiple steps: 3D reprojection and screen plane intersection
        gaze_origin_3d = reprojection_3d(gaze_origin_xy, gaze_origin_z, intrinsics)
        # pog_mm, pog_px = screen_plane_intersection(
        #     gaze_origin_3d, 
        #     gaze_direction_unit_vector, 
        # )

        return {
            'gaze_origin': gaze_origin,
            'gaze_depth': gaze_depth,
            'gaze_direction': gaze_direction,
            "gaze_origin_xy": gaze_origin_xy,
            "gaze_origin_z": gaze_origin_z,
            "gaze_origin_3d": gaze_origin_3d,
        }
    
    def compute_loss(self, output, batch):
        
        # Compute the gaze origin loss (Heatmap MSE Loss)
        gt_heatmap = 100*generate_2d_gaussian_heatmap_torch(batch['face_origin_2d'], self.img_size, sigma=12.0)
        gaze_origin_loss = F.mse_loss(output['gaze_origin'][:, 0], gt_heatmap)
        # gaze_origin_xy_loss = F.mse_loss(output['gaze_origin_xy'], batch['face_origin_2d'])

        # Compute the gaze z-origin location (L1 Loss)
        # gaze_origin_z_loss = F.l1_loss(output['gaze_origin_z'], batch['face_origin_3d'][:, 2])

        # Compute the angular loss for the gaze direction
        # https://github.com/swook/EVE/blob/master/src/losses/angular.py#L29
        # gaze_direction_unit_vector_gt = F.normalize(batch['gaze_direction_3d'], p=2, dim=1)
        # angular_loss = torch.acos(torch.clamp(torch.sum(output['gaze_direction'] * gaze_direction_unit_vector_gt, dim=1), -1.0, 1.0))
        # angular_loss = torch.mean(angular_loss)

        # Compute the PoG MSE Loss

        # losses = [gaze_origin_loss, gaze_origin_xy_loss, gaze_origin_z_loss, angular_loss] 
        losses = [gaze_origin_loss]
        complete_loss = torch.sum(torch.stack(losses))

        output = {
            'losses': {
                'gaze_origin_heatmap_loss': gaze_origin_loss,
                # 'gaze_origin_xy_loss': gaze_origin_xy_loss,
                'complete_loss': complete_loss
            },
            'artifacts': {
                'gaze_origin_heatmap': gt_heatmap,
            }
        }

        return output

    def training_step(self, batch, batch_idx):
        output = self.forward(batch['image'], batch['intrinsics'])
        losses_output = self.compute_loss(output, batch)

        # Logging the losses
        for k, v in losses_output['losses'].items():
            self.log(f'train_{k}', v, batch_size=batch['image'].shape[0])

        # if self.current_epoch == 2: import pdb; pdb.set_trace()
        self.log_tb_images('train', batch, output, losses_output)
        
        return {'loss': losses_output['losses']['complete_loss'], 'log': losses_output['losses']}

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch['image'], batch['intrinsics'])
        losses_output = self.compute_loss(output, batch)

        self.log('val_loss', losses_output['losses']['complete_loss'])
        for k, v in losses_output['losses'].items():
            if k != 'complete_loss':
                self.log(f'val_{k}', v, batch_size=batch['image'].shape[0])
        
        self.log_tb_images('val', batch, output, losses_output)

    def log_tb_images(self, prefix, batch, output, losses_output):

        tb_logger = self.logger.experiment
        
        # Log the images (Give them different names)
        np_cpu_images = batch['image'].cpu().numpy()

        gaze_origin_gt = []
        gaze_origin_heatmaps = []
        gaze_origin_heatmaps_gt = []
        gaze_direction_imgs = []
        for i in range(np_cpu_images.shape[0]):
            img = np.moveaxis(np_cpu_images[i], 0, -1)

            # Visualize the gaze origin xy
            gaze_origin_xy = batch['face_origin_2d'][i].cpu().numpy()
            vis_gt_gaze_origin = vis.draw_gaze_origin(img, gaze_origin_xy)
            gaze_origin_gt.append(vis_gt_gaze_origin)

            # Visualizing the gaze origin heatmaps
            gt_gaze_origin_heatmap = losses_output['artifacts']['gaze_origin_heatmap'][i].detach().cpu().numpy()
            vis_gt_gaze_origin_heatmap = vis.draw_gaze_origin_heatmap(img, gt_gaze_origin_heatmap)
            gaze_origin_heatmaps_gt.append(vis_gt_gaze_origin_heatmap)

            gaze_origin_heatmap = output['gaze_origin'][i].detach().cpu().numpy()
            vis_gaze_origin_heatmap = vis.draw_gaze_origin_heatmap(img, gaze_origin_heatmap[0])
            gaze_origin_heatmaps.append(vis_gaze_origin_heatmap)

            # Visualizing the gaze direction
            gt_gaze = vis.draw_gaze_direction(img, batch['face_origin_2d'][i], batch['gaze_target_2d'][i])

            # Create gaze_target_2d via the direction and a fixed distance
            # gaze_target_3d_semi = batch['face_origin_3d'][i] + output['gaze_direction'][i] / 5
            # gaze_target_3d_semi = gaze_target_3d_semi.cpu().numpy()
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
            #     color=(0, 0, 255)
            # )
            gaze_direction_imgs.append(gt_gaze)

        tb_logger.add_images(f"{prefix}_gt_gaze_origin", np.moveaxis(np.stack(gaze_origin_gt), -1, 1), self.current_epoch)
        tb_logger.add_images(f"{prefix}_pred_gaze_origin_heatmaps", np.moveaxis(np.stack(gaze_origin_heatmaps), -1, 1), self.current_epoch)
        tb_logger.add_images(f"{prefix}_gt_gaze_origin_heatmaps", np.moveaxis(np.stack(gaze_origin_heatmaps_gt), -1, 1), self.current_epoch)
        tb_logger.add_images(f"{prefix}_gaze_direction", np.moveaxis(np.stack(gaze_direction_imgs), -1, 1), self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
if __name__ == '__main__':
    model = EFEModel()
    input_tensor = torch.randn(1, 3, 480, 640)
    # print(timeit.timeit(lambda: model(input_tensor), number=10))
    output = model(input_tensor)
    print([o.shape for o in output.values()])