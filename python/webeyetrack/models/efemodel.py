import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import torch.nn as nn
import numpy as np

# from .unet import UNet
from .unet_efficientnet_v2 import UNetEfficientNetV2Small
from .utils import ResNetBlock
from ..vis import draw_gaze_origin

class EFEModel(pl.LightningModule):
    def __init__(self, img_size=(640, 480)):
        super().__init__()

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
        logits, bottleneck = self.unet(x)
        gaze_origin = self.resnet_gaze_origin(logits)
        gaze_depth = self.resnet_gaze_depth(logits)
        gaze_direction = self.gaze_direction_fc(bottleneck.view(-1))

        return {
            'gaze_origin': gaze_origin,
            'gaze_depth': gaze_depth,
            'gaze_direction': gaze_direction
        }

    def training_step(self, batch, batch_idx):
        ...
        # output = self.forward(batch['image'])
        # import pdb; pdb.set_trace()

    def validation_step(self, batch, batch_idx):

        # Add fake val_loss
        self.log('val_loss', 0.0)
        
        if batch_idx % 10:
            self.log_tb_images(batch)

    def log_tb_images(self, batch):

        tb_logger = self.logger.experiment
        
        # Log the images (Give them different names)
        # import pdb; pdb.set_trace()
        np_cpu_images = batch['image'].cpu().numpy()
        viz_images = []
        for i in range(np_cpu_images.shape[0]):
            viz_images.append(draw_gaze_origin(np.moveaxis(np_cpu_images[i], 0, -1), batch['face_origin_2d'][i]))
        tb_logger.add_images(f"gaze_origin", np.moveaxis(np.stack(viz_images), -1, 1), self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
if __name__ == '__main__':
    model = EFEModel()
    input_tensor = torch.randn(1, 3, 640, 480)
    output = model(input_tensor)
    print([o.shape for o in output.values()])