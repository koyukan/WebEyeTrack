import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import math
import time

from .resnet import resnet18

# Credit to
# https://github.com/yihuacheng/Gaze360

class Gaze360(nn.Module):
    def __init__(self):
        super(Gaze360, self).__init__()
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame
        self.base_model = resnet18(pretrained=True)

        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)

        # The linear layer that maps the LSTM with the 3 outputs
        self.last_layer = nn.Linear(self.img_feature_dim, 3)

    def forward(self, x_in):

        base_out = self.base_model(x_in["face"])
        base_out = torch.flatten(base_out, start_dim=1)
        output = self.last_layer(base_out)

        angular_output = output[:,:2]
        angular_output[:,0:1] = math.pi*nn.Tanh()(angular_output[:,0:1])
        angular_output[:,1:2] = (math.pi/2)*nn.Tanh()(angular_output[:,1:2])

        var = math.pi*nn.Sigmoid()(output[:,2:3])
        var = var.view(-1,1).expand(var.size(0), 2)

        return angular_output,var

if __name__ == "__main__":
    model = Gaze360()

    # Test the model
    x = torch.randn(1, 3, 224, 224)
    y = model({'face': x})
    print(y, y[0].shape, y[1].shape)
