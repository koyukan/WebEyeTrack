import torch
import torch.nn as nn
import torch.nn.functional as F

from ..efe.unet_efficientnet_v2 import UNetEfficientNetV2Small

class EyeNet(nn.Module):
    def __init__(self, config = None, face_size=(244, 244)):
        super().__init__()

        self.config = config

        self.unet = UNetEfficientNetV2Small(num_classes=1, pretrained=True)

        # Computing the size of the bottleneck
        self.face_size = face_size
        # self.bottleneck_size = 160 * (face_size[0] // 16) * (face_size[1] // 16)
        self.bottleneck_size = 160 * 16 * 16
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

        # Predict
        logits, bottleneck = self.unet(x)
        gaze_direction = self.gaze_direction_fc(bottleneck.view(batch_size, -1))

        # Normalize the gaze direction
        gaze_direction = F.normalize(gaze_direction, p=2, dim=1)

        return {
            'gaze_direction': gaze_direction,
        }

if __name__ == '__main__':
    model = EyeTrackModel(img_size=(480, 640))
    input_tensor = torch.randn(1, 3, 480, 640)
    output = model(input_tensor)
    print(output['gaze_direction'].shape)