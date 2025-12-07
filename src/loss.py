import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """Binary Cross-Entropy Loss"""
    
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCELoss()
    
    def forward(self, pred, target):
        return self.bce(pred, target)


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        # Return Dice loss (1 - Dice coefficient)
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice Loss
    
    注意：此 loss 函數接受 logits（未經過 sigmoid），內部會自動應用 sigmoid
    """
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()  # 使用 BCEWithLogitsLoss，接受 logits
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(self, pred, target):
        # pred 是 logits，需要先經過 sigmoid 用於 Dice loss
        pred_sigmoid = torch.sigmoid(pred)
        
        # BCE loss 使用 logits（BCEWithLogitsLoss 內部會處理）
        bce_loss = self.bce(pred, target)
        
        # Dice loss 使用 sigmoid 後的概率值
        dice_loss = self.dice(pred_sigmoid, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class NTXentLoss(nn.Module):
    """
    Official NT-Xent loss used in SimCLR.
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1, z2: (N, dim) - 已經在 ProjectionHead 中進行了 L2 normalization
        """
        N = z1.shape[0]
        device = z1.device

        # 注意：z1 和 z2 已經在 ProjectionHead 中進行了 L2 normalization
        # 因此這裡不需要再次 normalize

        # Concatenate: (2N, dim)
        z = torch.cat([z1, z2], dim=0)

        # Similarity: (2N, 2N)
        sim = torch.matmul(z, z.T) / self.temperature

        # Remove diagonal (self similarity)
        mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, -1e4)

        # Positive pair indices:
        # For i in [0, N-1], positive is i+N
        # For i in [N, 2N-1], positive is i-N
        pos_index = torch.cat([
            torch.arange(N, 2 * N, device=device),
            torch.arange(N, device=device)
        ])

        # Cross entropy requires labels
        labels = pos_index

        # Compute loss
        loss = F.cross_entropy(sim, labels)

        return loss