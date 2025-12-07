import torch


class DiceScore:
    def __init__(self, smooth: float = 1e-5, threshold: float = 0.5):
        self.smooth = smooth
        self.threshold = threshold

    def __call__(self, pred: torch.Tensor, label: torch.Tensor, mean: bool = True) -> torch.Tensor:
        # Ensure pred and label have the same shape
        if pred.dim() == 4:
            pred = pred.squeeze(1)  # (N, 1, H, W) -> (N, H, W)
        if label.dim() == 4:
            label = label.squeeze(1)  # (N, 1, H, W) -> (N, H, W)
        
        # Apply sigmoid to logits (pred 可能是 logits，需要轉換為概率)
        # 如果 pred 已經是概率值（範圍在 [0, 1]），sigmoid 不會有太大影響
        pred_prob = torch.sigmoid(pred)
        
        # Binarize predictions
        pred_binary = (pred_prob > self.threshold).float()
        label_binary = label.float()
        
        # Flatten tensors: (N, H, W) -> (N, H*W)
        pred_flat = pred_binary.view(pred_binary.size(0), -1)
        label_flat = label_binary.view(label_binary.size(0), -1)
        
        # Calculate intersection and union for each sample in batch
        intersection = (pred_flat * label_flat).sum(dim=1)  # (N,)
        pred_sum = pred_flat.sum(dim=1)  # (N,)
        label_sum = label_flat.sum(dim=1)  # (N,)
        union = pred_sum + label_sum  # (N,)
        
        # Calculate Dice coefficient for each sample
        # Handle case where both pred and label are empty (union == 0)
        dice = torch.where(
            union == 0,
            torch.ones_like(intersection),  # If both empty, Dice = 1
            (2. * intersection + self.smooth) / (union + self.smooth)  # Normal case
        )
        
        return dice.mean() if mean else dice
