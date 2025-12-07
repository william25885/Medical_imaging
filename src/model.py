import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class DoubleConv_simpleUNet(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_simpleUNet(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv_simpleUNet(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """
    Simple-CNN UNet for medical image segmentation
    
    Architecture:
    - Encoder: 4 downsampling blocks (64 -> 128 -> 256 -> 512 -> 1024)
    - Decoder: 4 upsampling blocks with skip connections
    - Output: 1 channel with sigmoid activation for binary segmentation
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv_simpleUNet(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return torch.sigmoid(logits)


class SimCLREncoder(nn.Module):
    def __init__(
        self,
        img_size=(448, 576),
        in_channels=1,
        vit_embed_dim=768,
        proj_dim=128,
        resnet_pretrained=True,
    ):
        super().__init__()

        self.img_size = img_size
        self.embed_dim = vit_embed_dim

        self.encoder = ResNetEncoder(in_channels=in_channels, pretrained=resnet_pretrained)

        feat_h = img_size[0] // 16
        feat_w = img_size[1] // 16

        self.patch_proj = nn.Linear(1024, vit_embed_dim)

        self.projector = nn.Sequential(
            nn.Linear(vit_embed_dim, vit_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(vit_embed_dim, proj_dim),
        )

    def forward(self, x):
        x0, x1, x2, x3 = self.encoder(x)
        tokens = x3.flatten(2).transpose(1, 2)
        h = self.patch_proj(tokens).mean(dim=1)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return z



class ProjectionHead(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, output_dim=128):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        z = self.projection(x)
        z = F.normalize(z, dim=1)
        return z


class SimCLRModel(nn.Module):
    def __init__(
        self,
        img_size=(448, 576),
        in_channels=1,
        vit_embed_dim=768,
        resnet_pretrained=True,
        projection_dim=128,
    ):
        super().__init__()
        
        self.encoder = SimCLREncoder(
            img_size=img_size,
            in_channels=in_channels,
            vit_embed_dim=vit_embed_dim,
            proj_dim=projection_dim,
            resnet_pretrained=resnet_pretrained
        )
    
    def forward(self, x, return_projection=True):
        return self.encoder(x)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        """
        in_ch:  上一層 feature 的 channel 數
        skip_ch: 對應 skip feature 的 channel 數
        out_ch: 輸出 channel
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=1, pretrained=False):
        super().__init__()
        base = resnet50(weights=None)  # 不使用 ImageNet 預訓練權重

        if in_channels != 3:
            old_conv = base.conv1
            self.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            # 使用隨機初始化
            nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        else:
            self.conv1 = base.conv1

        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)

        x = self.maxpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        return x0, x1, x2, x3


class ViTEncoder(nn.Module):
    """
    簡化的 ViT Encoder：
      - 不使用 CLS token
      - sequence = flatten(H,W) 的 patch tokens
    """
    def __init__(self, num_patches, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        x: (B, N, C)  N = num_patches
        """
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.encoder(x)   # (B, N, C)
        return x


class TransUNet(nn.Module):
    """
    TransUNet: ResNet50 encoder + ViT bottleneck + U-Net decoder

    - Encoder: ResNet50 到 layer3, 提供 multi-scale CNN skip
    - ViT encoder: 在 1/16 resolution 上做 global self-attention
    - Decoder: U-Net 風格，上採樣 + concat CNN skip
    """

    def __init__(
        self,
        img_size=(448, 576),
        in_channels=1,
        num_classes=1,
        vit_embed_dim=768,
        vit_depth=12,
        vit_num_heads=12,
        vit_mlp_ratio=4.0,
        vit_dropout=0.1,
        resnet_pretrained=False,
    ):
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.encoder = ResNetEncoder(in_channels=in_channels, pretrained=False)

        feat_h = img_size[0] // 16
        feat_w = img_size[1] // 16
        self.feat_size = (feat_h, feat_w)
        num_patches = feat_h * feat_w

        self.patch_proj = nn.Linear(1024, vit_embed_dim)

        self.vit_encoder = ViTEncoder(
            num_patches=num_patches,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
            mlp_ratio=vit_mlp_ratio,
            dropout=vit_dropout,
        )

        self.vit_conv = nn.Conv2d(vit_embed_dim, 512, kernel_size=3, padding=1)

        self.up1 = UpBlock(in_ch=512, skip_ch=512, out_ch=256)
        self.up2 = UpBlock(in_ch=256, skip_ch=256, out_ch=128)
        self.up3 = UpBlock(in_ch=128, skip_ch=64,  out_ch=64)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        """
        x: (B, C, H, W)  H, W 必須能被 16 整除
        return: logits (B, num_classes, H, W)
        """
        B, C, H, W = x.shape
        assert H % 16 == 0 and W % 16 == 0, f"Input size {H,W} 必須可以被 16 整除，才對得上 ResNet layer3 的 1/16 特徵。"

        x0, x1, x2, x3 = self.encoder(x)
        B, C3, Hf, Wf = x3.shape

        tokens = x3.flatten(2).transpose(1, 2)
        tokens = self.patch_proj(tokens)

        vit_out = self.vit_encoder(tokens)

        vit_out = vit_out.transpose(1, 2).reshape(B, -1, Hf, Wf)
        x_vit = self.vit_conv(vit_out)

        x = self.up1(x_vit, x2)
        x = self.up2(x, x1)
        x = self.up3(x, x0)

        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        logits = self.out_conv(x)

        return logits
