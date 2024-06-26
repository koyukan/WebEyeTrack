import torch
import torch.nn as nn
import torchvision.models as models

class UNetEfficientNetV2Small(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(UNetEfficientNetV2Small, self).__init__()
        
        # Load the EfficientNetV2 small model from torchvision
        self.encoder = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
        
        # Extract layers
        self.encoder_stem = nn.Sequential(
            self.encoder.features[0]
        )
        
        self.encoder_blocks = nn.ModuleList([
            self.encoder.features[1],  # stage 1
            self.encoder.features[2],  # stage 2
            self.encoder.features[3],  # stage 3
            self.encoder.features[4],  # stage 4
            self.encoder.features[5],  # stage 5
            self.encoder.features[6],  # stage 6
        ])
        
        # Decoder layers
        self.decoder1 = self.conv_block(256 + 128, 128)
        self.decoder2 = self.conv_block(128 + 64, 64)
        self.decoder3 = self.conv_block(64 + 48, 48)
        self.decoder4 = self.conv_block(48 + 24, 24)
        
        # Final conv layer
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)
        
        # Up-sample layers
        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(48, 48, kernel_size=2, stride=2)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder_stem(x)  # (batch_size, 24, H/2, W/2)
        
        enc2 = self.encoder_blocks[0](enc1)  # (batch_size, 48, H/4, W/4)
        enc3 = self.encoder_blocks[1](enc2)  # (batch_size, 64, H/8, W/8)
        enc4 = self.encoder_blocks[2](enc3)  # (batch_size, 128, H/16, W/16)
        enc5 = self.encoder_blocks[3](enc4)  # (batch_size, 256, H/32, W/32)
        enc6 = self.encoder_blocks[4](enc5)  # (batch_size, 256, H/32, W/32)
        bottleneck = self.encoder_blocks[5](enc6)  # (batch_size, 256, H/32, W/32)
        
        # Decoder with skip connections
        dec1 = self.up1(bottleneck)
        dec1 = torch.cat((dec1, enc5), dim=1)
        dec1 = self.decoder1(dec1)
        
        dec2 = self.up2(dec1)
        dec2 = torch.cat((dec2, enc4), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec3 = self.up3(dec2)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec4 = self.up4(dec3)
        dec4 = torch.cat((dec4, enc2), dim=1)
        dec4 = self.decoder4(dec4)
        
        return self.final_conv(dec4)

# Example usage
model = UNetEfficientNetV2Small(num_classes=1, pretrained=True)
input_tensor = torch.randn(1, 3, 256, 256)
output = model(input_tensor)
print(output.shape)