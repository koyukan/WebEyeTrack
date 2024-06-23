import pytorch_lightning as pl
import torch
import torch.nn as nn

from .unet import UNet
from .utils import ResNetBlock

class EFEModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Create components
        self.unet = UNet(n_channels=3, n_classes=1)
        self.resnet_gaze_origin = ResNetBlock(in_channels=3, out_channels=3)
        self.resnet_gaze_depth = ResNetBlock(in_channels=3, out_channels=3)

        self.gaze_direction_fc = nn.Sequential(
            nn.Linear(3, 3),
            nn.SELU(inplace=True),
            nn.Linear(3, 3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        ...

    def training_step(self, batch, batch_idx):
        import pdb; pdb.set_trace()
        ...

    def validation_step(self, batch, batch_idx):
        ...

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)