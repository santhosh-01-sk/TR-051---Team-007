import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SiameseUNetDamageAssessor(nn.Module):
    """
    Siamese U-Net architecture for Disaster Damage Assessment.
    Takes Pre and Post disaster images, fuses them with a disaster type embedding,
    and jointly predicts building footprint (segmentation) and damage classification maps.
    """
    def __init__(self, in_channels=3, num_classes=4, num_disaster_types=10):
        super().__init__()
        
        # 1. Disaster Type Embedding
        self.tag_embedding = nn.Embedding(num_disaster_types, 64)
        
        # 2. Shared Encoder (Siamese branch weights are shared)
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # 3. Bottleneck and Fusion Layer
        # Concatenate pre-features(512), post-features(512), and expanded tag embedding (64)
        self.bottleneck = DoubleConv(512 * 2 + 64, 1024)
        
        # 4. Shared Decoder (Upsampling and concatenating skip connections from BOTH pre and post images)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512 + 512 * 2, 512)  # upsampled + pre_skip + post_skip
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256 + 256 * 2, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128 + 128 * 2, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64 + 64 * 2, 64)
        
        # 5. Output Heads
        # Segmentation Head outputs logits for binary building presence (1 channel)
        self.segmentation_head = nn.Conv2d(64, 1, kernel_size=1)
        
        # Damage Head outputs logits for 4 classes (No Damage, Minor, Major, Destroyed)
        self.damage_head = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def encode(self, x):
        """Passes input through the shared encoder"""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.pool4(e4)
        return e1, e2, e3, e4, b
        
    def forward(self, pre_img, post_img, disaster_tag):
        # Base Siamese Encodings
        pre1, pre2, pre3, pre4, pre_b = self.encode(pre_img)
        post1, post2, post3, post4, post_b = self.encode(post_img)
        
        # Process Disater Tag Embedding
        b_size, _, h, w = pre_b.size()
        tag_emb = self.tag_embedding(disaster_tag)  # Shape: [batch_size, 64]
        tag_emb_expanded = tag_emb.view(b_size, 64, 1, 1).expand(b_size, 64, h, w)
        
        # Bottleneck Fusion
        fused_bottleneck = torch.cat([pre_b, post_b, tag_emb_expanded], dim=1)
        b_out = self.bottleneck(fused_bottleneck)
        
        # Decoder path with dense skip connections
        u4 = self.up4(b_out)
        d4 = self.dec4(torch.cat([u4, pre4, post4], dim=1))
        
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, pre3, post3], dim=1))
        
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, pre2, post2], dim=1))
        
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, pre1, post1], dim=1))
        
        # Output Maps (Raw Logits)
        seg_logits = self.segmentation_head(d1)
        damage_logits = self.damage_head(d1)
        
        return seg_logits, damage_logits
