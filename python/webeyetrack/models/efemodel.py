import pytorch_lightning as pl
import torch
import torch.nn as nn

# from .unet import UNet
from .unet_efficientnet_v2 import UNetEfficientNetV2Small
from .utils import ResNetBlock

class EFEModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Create components
        self.unet = UNetEfficientNetV2Small(num_classes=1, pretrained=True)
        self.resnet_gaze_origin = ResNetBlock(in_channels=3, out_channels=3)
        self.resnet_gaze_depth = ResNetBlock(in_channels=3, out_channels=3)

        self.gaze_direction_fc = nn.Sequential(
            nn.Linear(3, 3),
            nn.SELU(inplace=True),
            nn.Linear(3, 3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        logits, bottleneck = self.unet(x)
        import pdb; pdb.set_trace()
        # gaze_origin = self.resnet_gaze_origin(logits)
        # gaze_depth = self.resnet_gaze_depth(logits)
        # gaze_direction = self.gaze_direction_fc(bottleneck)

        return {
            # 'gaze_origin': gaze_origin,
            # 'gaze_depth': gaze_depth,
            # 'gaze_direction': gaze_direction
        }

    def training_step(self, batch, batch_idx):
        output = self.forward(batch['image'])

    def validation_step(self, batch, batch_idx):
        ...

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
if __name__ == '__main__':
    model = EFEModel()
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)
    print(output.shape)