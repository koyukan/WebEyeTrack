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
        
        # Compute the gaze origin loss
        gt_heatmap = generate_2d_gaussian_heatmap_torch(batch['face_origin_2d'], self.img_size)
        gaze_origin_loss = F.mse_loss(output['gaze_origin'], gt_heatmap)

        return {'gaze_origin_loss': gaze_origin_loss}

    def training_step(self, batch, batch_idx):
        output = self.forward(batch['image'])
        losses = self.compute_loss(output, batch)
        return {'loss': losses['gaze_origin_loss'], 'log': losses}

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch['image'])
        losses = self.compute_loss(output, batch)
        self.log('gaze_origin_loss', losses['gaze_origin_loss'])
        # self.log('val_loss', 0.0)
        
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