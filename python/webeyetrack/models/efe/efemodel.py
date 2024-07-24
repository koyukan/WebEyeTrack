import torch
import torch.nn as nn
from torch.nn import functional as F

from .unet_efficientnet_v2 import UNetEfficientNetV2Small
from .components import ResNetBlock

# References:
# https://github.com/swook/EVE/blob/master/src/models/common.py#L129

class EFEModel(nn.Module):
    def __init__(self, config, img_size=(480, 640)):
        super().__init__()

        # Save parameters
        self.config = config
        self.img_size = img_size

        # Create components
        self.unet = UNetEfficientNetV2Small(num_classes=1, pretrained=True)
        # self.resnet_gaze_origin = ResNetBlock(in_channels=1, out_channels=1)
        # self.resnet_gaze_depth = ResNetBlock(in_channels=1, out_channels=1)

        # Computing the size of the bottleneck
        # bottleneck_size = 160 * (img_size[0] // 16) * (img_size[1] // 16)
        bottleneck_size = 160 * 16 * 16

        self.gaze_direction_fc = nn.Sequential(
            nn.Linear(bottleneck_size, 512),
            nn.SELU(inplace=True),
            nn.Linear(512, 256),
            nn.SELU(inplace=True),
            nn.Linear(256, 3),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Predict
        logits, bottleneck = self.unet(x)
        # gaze_origin = self.resnet_gaze_origin(logits)
        # gaze_depth = self.resnet_gaze_depth(logits)
        gaze_direction = self.gaze_direction_fc(bottleneck.view(batch_size, -1))

        # Compute the gaze origin location
        # softmax_gaze_origin = F.softmax(self.config['hparams']['heatmap_coef']*gaze_origin.view(batch_size, -1), dim=1)
        # gaze_origin_xy_idx = torch.argmax(softmax_gaze_origin, dim=1)
        # gaze_origin_xy = torch.stack([gaze_origin_xy_idx % self.img_size[1], gaze_origin_xy_idx // self.img_size[1]], dim=1)
        
        # First, we apply the dot product between the softmax_gaze_origin and the depth map to get the z-origin
        # Then, we use the gaze_origin_xy_idx to get the z-origin
        # softmax_gaze_origin_formatted = softmax_gaze_origin.view(batch_size, 1, gaze_depth.shape[2], gaze_depth.shape[3])
        # gaze_origin_z_map = torch.sum(gaze_depth * softmax_gaze_origin_formatted, dim=1)
        # gaze_origin_z = torch.gather(gaze_origin_z_map.view(batch_size, -1), 1, gaze_origin_xy_idx.unsqueeze(1))
        # gaze_origin_z = torch.sum(gaze_depth * softmax_gaze_origin_formatted, dim=(2, 3))

        # Normalize the gaze direction
        gaze_direction = F.normalize(gaze_direction, p=2, dim=1)

        return {
            # 'gaze_origin': gaze_origin,
            # 'gaze_depth': gaze_depth,
            'gaze_direction': gaze_direction,
            # "gaze_origin_xy": gaze_origin_xy,
            # "gaze_origin_z": gaze_origin_z,
        }
    
if __name__ == '__main__':
    model = EFEModel()
    input_tensor = torch.randn(1, 3, 480, 640)
    # print(timeit.timeit(lambda: model(input_tensor), number=10))
    output = model(input_tensor)
    print([o.shape for o in output.values()])