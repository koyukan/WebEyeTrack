import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from .funcs import generate_2d_gaussian_heatmap_torch
from .unet_efficientnet_v2 import UNetEfficientNetV2Small
from .components import ResNetBlock
from ..vis import draw_gaze_origin, draw_gaze_direction

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

    def forward(self, x):
        batch_size = x.shape[0]
        logits, bottleneck = self.unet(x)
        gaze_origin = self.resnet_gaze_origin(logits)
        gaze_depth = self.resnet_gaze_depth(logits)
        gaze_direction = self.gaze_direction_fc(bottleneck.view(batch_size, -1))

        return {
            'gaze_origin': gaze_origin,
            'gaze_depth': gaze_depth,
            'gaze_direction': gaze_direction
        }
    
    def compute_loss(self, output, batch):
        
        # Compute the gaze origin loss (Heatmap MSE Loss)
        gt_heatmap = generate_2d_gaussian_heatmap_torch(batch['face_origin_2d'], self.img_size)
        gaze_origin_loss = F.mse_loss(output['gaze_origin'], gt_heatmap)

        # Compute the gaze origin location (XY MSE Loss)
        softmax_gaze_origin = F.softmax(output['gaze_origin'].view(batch['face_origin_2d'].shape[0], -1), dim=1)
        gaze_origin_xy_idx = torch.argmax(softmax_gaze_origin, dim=1)
        gaze_origin_xy = torch.stack([gaze_origin_xy_idx % self.img_size[1], gaze_origin_xy_idx // self.img_size[1]], dim=1)
        gaze_origin_xy_loss = F.mse_loss(gaze_origin_xy, batch['face_origin_2d'])

        # Compute the gaze z-origin location (L1 Loss)
        # First, we apply the dot product between the softmax_gaze_origin and the depth map to get the z-origin
        gaze_depth = output['gaze_depth'].view(batch['face_origin_2d'].shape[0], -1)
        gaze_z_origin = torch.sum(gaze_depth * softmax_gaze_origin, dim=1)
        gaze_z_origin_loss = F.l1_loss(gaze_z_origin, batch['face_origin_3d'][:, 2])

        # Compute the angular loss for the gaze direction
        gaze_direction_unit_vector = output['gaze_direction'] / (torch.norm(output['gaze_direction'], dim=1, keepdim=True) + 1e-6)
        angular_loss = torch.acos(torch.sum(gaze_direction_unit_vector * batch['gaze_direction_3d'], dim=1))
        angular_loss = torch.mean(angular_loss)

        return {
            'gaze_origin_heatmap_loss': gaze_origin_loss, 
            'gaze_origin_xy_loss': gaze_origin_xy_loss,
            'gaze_z_origin_loss': gaze_z_origin_loss,
            'angular_loss': angular_loss
        }

    def training_step(self, batch, batch_idx):
        output = self.forward(batch['image'])
        losses = self.compute_loss(output, batch)
        complete_loss = torch.sum(torch.stack(list(losses.values())))
        return {'loss': complete_loss, 'log': losses}

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch['image'])
        losses = self.compute_loss(output, batch)

        self.log('val_loss', losses['gaze_origin_heatmap_loss'])
        for k, v in losses.items():
            self.log(f'val_{k}', v, batch_size=batch['image'].shape[0])
        
        if batch_idx % 10:
            self.log_tb_images(batch)

    def log_tb_images(self, batch):

        tb_logger = self.logger.experiment
        
        # Log the images (Give them different names)
        np_cpu_images = batch['image'].cpu().numpy()

        gaze_origin_imgs = []
        gaze_direction_imgs = []
        for i in range(np_cpu_images.shape[0]):
            img = np.moveaxis(np_cpu_images[i], 0, -1)
            gaze_origin_imgs.append(draw_gaze_origin(img, batch['face_origin_2d'][i]))
            gaze_direction_imgs.append(draw_gaze_direction(img, batch['face_origin_2d'][i], batch['gaze_target_2d'][i]))

        tb_logger.add_images(f"gaze_origin", np.moveaxis(np.stack(gaze_origin_imgs), -1, 1), self.current_epoch)
        tb_logger.add_images(f"gaze_direction", np.moveaxis(np.stack(gaze_direction_imgs), -1, 1), self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
if __name__ == '__main__':
    model = EFEModel()
    input_tensor = torch.randn(1, 3, 480, 640)
    output = model(input_tensor)
    print([o.shape for o in output.values()])