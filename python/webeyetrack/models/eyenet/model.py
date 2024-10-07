import torch
import torch.nn as nn
import torch.nn.functional as F

from ..efe.unet_efficientnet_v2 import UNetEfficientNetV2Small

from math import pi

import torch


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

class EyeNet(nn.Module):
    def __init__(self, config = None, face_size=(244, 244)):
        super().__init__()

        self.config = config

        self.fourier = GaussianFourierFeatureTransform(3, mapping_size=256, scale=10)
        # self.unet = UNetEfficientNetV2Small(num_classes=1, pretrained=True)
        self.encoder = nn.Sequential(
            # Conv Block 1: Reduce from (3, 480, 64) to (64, 240, 32)
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            # Conv Block 2: Reduce from (64, 240, 32) to (128, 120, 16)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            # Conv Block 3: Reduce from (128, 120, 16) to (256, 60, 8)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            # Conv Block 4: Reduce from (256, 60, 8) to (512, 30, 4)
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            # Conv Block 5: Final reduction, e.g., to (16, 15, 2)
            nn.Conv2d(512, 16, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid(),
        )


        # Computing the size of the bottleneck
        self.face_size = face_size
        # self.bottleneck_size = 160 * (face_size[0] // 16) * (face_size[1] // 16)
        # self.bottleneck_size = 160 * 16 * 16
        self.bottleneck_size = 16*8*8
        # import pdb; pdb.set_trace()

        self.gaze_direction_fc = nn.Sequential(
            nn.Linear(self.bottleneck_size, 512),
            nn.SELU(inplace=True),
            nn.Linear(512, 256),
            nn.SELU(inplace=True),
            nn.Linear(256, 3),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Perform gaussian fourier feature transform
        x = self.fourier(x)

        # Predict
        # logits, bottleneck = self.unet(x)
        x = self.encoder(x)
        gaze_direction = self.gaze_direction_fc(x.reshape(batch_size, -1))

        # Normalize the gaze direction
        gaze_direction = F.normalize(gaze_direction, p=2, dim=1)

        return {
            'gaze_direction': gaze_direction,
        }

if __name__ == '__main__':
    model = EyeNet(face_size=(244, 244))
    input_tensor = torch.randn(1, 3, 244, 244)
    output = model(input_tensor)
    print(output['gaze_direction'].shape)