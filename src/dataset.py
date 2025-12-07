import glob
import os
import random
from typing import Optional, Union, List, Dict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import (
    to_tensor, resize, hflip, vflip, rotate,
    adjust_brightness, adjust_contrast, affine
)
from torchvision.transforms import RandomResizedCrop
import torch.nn.functional as F
from torchvision.transforms import functional as TF



class MedicalImageDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        has_label: bool = True,
        transform: Optional[callable] = None
    ):
        self.data_dir = data_dir
        self.data_list = glob.glob(f"{data_dir}/image/*.tif")
        self.has_label = has_label
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        if self.has_label:
            data =  {
                "filename": os.path.basename(self.data_list[idx]),
                "image": self.data_list[idx],
                "label": self.data_list[idx].replace("image", "label"),
            }
        else:
            data = {
                "filename": os.path.basename(self.data_list[idx]),
                "image": self.data_list[idx],
            }

        if self.transform:
            return self.transform(data)

        return data


class BaseTransform:
    def __init__(self, keys: List[str], *args, **kwargs):
        self.keys = keys
        self._parse_var(*args, **kwargs)

    def _parse_var(self, *args, **kwargs):
        pass

    def _process(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        raise NotImplementedError

    def __call__(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        for key in self.keys:
            if key in data:
                data[key] = self._process(data[key], *args, **kwargs)
            else:
                raise KeyError(f"{key} is not a key in data.")
        return data


class LoadImaged(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(LoadImaged, self).__init__(keys, **kwargs)

    def _process(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        return Image.open(data)


class ToTensord(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(ToTensord, self).__init__(keys, **kwargs)

    def _process(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        tensor = to_tensor(data)
        
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor


class ResizeImaged(BaseTransform):
    def __init__(self, keys, size, save_original_size=False, **kwargs):
        super(ResizeImaged, self).__init__(keys, **kwargs)
        self.size = size
        self.save_original_size = save_original_size  # 是否保存原始尺寸

    def __call__(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        for key in self.keys:
            if key in data:
                # 保存原始尺寸（在 resize 之前，如果需要）
                if self.save_original_size and key == "image":
                    if isinstance(data[key], torch.Tensor):
                        # Tensor: (C, H, W)
                        original_size = (data[key].shape[-2], data[key].shape[-1])
                    else:
                        # PIL Image
                        original_size = (data[key].height, data[key].width)
                    data["original_size"] = original_size
                
                # Use nearest-neighbor for labels to preserve binary values
                # Use bilinear for images
                if key == "label":
                    data[key] = resize(data[key], self.size, interpolation=0)  # 0 = nearest-neighbor
                else:
                    data[key] = resize(data[key], self.size, interpolation=2)  # 2 = bilinear
            else:
                raise KeyError(f"{key} is not a key in data.")
        return data


class RandomHorizontalFlipd(BaseTransform):
    """隨機水平翻轉（圖像和標籤同步）"""
    def __init__(self, keys, prob=0.5, **kwargs):
        super(RandomHorizontalFlipd, self).__init__(keys, **kwargs)
        self.prob = prob
        self.should_flip = None

    def __call__(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        # 決定是否翻轉（所有 keys 使用相同的決定）
        self.should_flip = random.random() < self.prob
        return super().__call__(data, *args, **kwargs)

    def _process(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.should_flip:
            return hflip(data)
        return data


class RandomRotationd(BaseTransform):
    """隨機旋轉（圖像和標籤同步）"""
    def __init__(self, keys, degrees=15, prob=1.0, **kwargs):
        super(RandomRotationd, self).__init__(keys, **kwargs)
        self.degrees = degrees
        self.prob = prob
        self.angle = None
        self.should_rotate = None

    def __call__(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        # 決定是否旋轉
        self.should_rotate = random.random() < self.prob
        # 決定旋轉角度（所有 keys 使用相同的角度）
        self.angle = random.uniform(-self.degrees, self.degrees) if self.should_rotate else 0.0
        for key in self.keys:
            if key in data:
                # Use nearest-neighbor for labels to preserve binary values
                # Use bilinear for images
                if key == "label":
                    data[key] = rotate(data[key], self.angle, interpolation=0)  # 0 = nearest-neighbor
                else:
                    data[key] = rotate(data[key], self.angle, interpolation=2)  # 2 = bilinear
            else:
                raise KeyError(f"{key} is not a key in data.")
        return data


class RandomNoiseInjectiond(BaseTransform):
    """隨機噪聲注入（僅對圖像）"""
    def __init__(self, keys, std=0.01, prob=0.5, **kwargs):
        super(RandomNoiseInjectiond, self).__init__(keys, **kwargs)
        self.std = std
        self.prob = prob

    def _process(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if random.random() < self.prob:
            noise = torch.randn_like(data) * self.std
            data = data + noise
            data = torch.clamp(data, 0, 1)  # 確保值在 [0, 1] 範圍內
        return data


class RandomVerticalFlipd(BaseTransform):
    """隨機垂直翻轉（圖像和標籤同步）"""
    def __init__(self, keys, prob=0.5, **kwargs):
        super(RandomVerticalFlipd, self).__init__(keys, **kwargs)
        self.prob = prob
        self.should_flip = None

    def __call__(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        self.should_flip = random.random() < self.prob
        return super().__call__(data, *args, **kwargs)

    def _process(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.should_flip:
            return vflip(data)
        return data


class RandomBrightnessd(BaseTransform):
    """隨機亮度調整（僅對圖像）"""
    def __init__(self, keys, brightness_factor_range=(0.8, 1.2), **kwargs):
        super(RandomBrightnessd, self).__init__(keys, **kwargs)
        self.brightness_factor_range = brightness_factor_range
        self.factor = None

    def __call__(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        self.factor = random.uniform(self.brightness_factor_range[0], self.brightness_factor_range[1])
        return super().__call__(data, *args, **kwargs)

    def _process(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # adjust_brightness 需要 PIL Image 或 tensor，這裡使用 tensor
        # brightness_factor: 1.0 表示不變，<1.0 變暗，>1.0 變亮
        data = adjust_brightness(data, self.factor)
        data = torch.clamp(data, 0, 1)
        return data


class RandomContrastd(BaseTransform):
    """隨機對比度調整（僅對圖像）"""
    def __init__(self, keys, contrast_factor_range=(0.8, 1.2), **kwargs):
        super(RandomContrastd, self).__init__(keys, **kwargs)
        self.contrast_factor_range = contrast_factor_range
        self.factor = None

    def __call__(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        self.factor = random.uniform(self.contrast_factor_range[0], self.contrast_factor_range[1])
        return super().__call__(data, *args, **kwargs)

    def _process(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # adjust_contrast: 1.0 表示不變，<1.0 降低對比度，>1.0 提高對比度
        data = adjust_contrast(data, self.factor)
        data = torch.clamp(data, 0, 1)
        return data


class RandomGammad(BaseTransform):
    """隨機伽馬校正（僅對圖像）"""
    def __init__(self, keys, gamma_range=(0.8, 1.2), **kwargs):
        super(RandomGammad, self).__init__(keys, **kwargs)
        self.gamma_range = gamma_range
        self.gamma = None

    def __call__(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        self.gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        return super().__call__(data, *args, **kwargs)

    def _process(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 伽馬校正：gamma < 1 變亮，gamma > 1 變暗
        data = torch.pow(data, self.gamma)
        data = torch.clamp(data, 0, 1)
        return data


class RandomAffined(BaseTransform):
    """隨機仿射變換（縮放和平移，圖像和標籤同步）"""
    def __init__(self, keys, scale_range=(0.9, 1.1), translate_range=(0.05, 0.05), **kwargs):
        super(RandomAffined, self).__init__(keys, **kwargs)
        self.scale_range = scale_range
        self.translate_range = translate_range
        self.angle = None
        self.translate = None
        self.scale = None
        self.shear = None

    def __call__(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        # 隨機生成變換參數（所有 keys 使用相同的參數）
        self.angle = 0.0  # 不使用旋轉（已有 RandomRotationd）
        self.scale = random.uniform(self.scale_range[0], self.scale_range[1])
        self.translate = (
            random.uniform(-self.translate_range[0], self.translate_range[0]),
            random.uniform(-self.translate_range[1], self.translate_range[1])
        )
        self.shear = (0.0, 0.0)  # 不使用剪切
        return super().__call__(data, *args, **kwargs)

    def _process(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 獲取圖像尺寸
        _, h, w = data.shape
        
        # 計算仿射矩陣
        # translate 需要轉換為像素值
        translate_px = (self.translate[0] * w, self.translate[1] * h)
        
        # 使用 affine 變換
        # interpolation: 0=nearest for labels, 2=bilinear for images
        interpolation = 0 if "label" in str(self.keys) else 2
        data = affine(data, angle=self.angle, translate=translate_px, scale=self.scale, shear=self.shear, interpolation=interpolation)
        return data


class RandomCropd(BaseTransform):
    """隨機裁剪（圖像和標籤同步）"""
    def __init__(self, keys, crop_size, **kwargs):
        super(RandomCropd, self).__init__(keys, **kwargs)
        self.crop_size = crop_size if isinstance(crop_size, (list, tuple)) else (crop_size, crop_size)
        self.top = None
        self.left = None

    def __call__(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        # 獲取第一個 key 的尺寸來決定裁剪位置（所有 keys 使用相同位置）
        first_key = self.keys[0]
        if first_key in data:
            _, h, w = data[first_key].shape
            crop_h, crop_w = self.crop_size
            
            # 確保裁剪尺寸不超過圖像尺寸
            crop_h = min(crop_h, h)
            crop_w = min(crop_w, w)
            
            # 隨機選擇裁剪位置
            self.top = random.randint(0, max(0, h - crop_h))
            self.left = random.randint(0, max(0, w - crop_w))
        return super().__call__(data, *args, **kwargs)

    def _process(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, h, w = data.shape
        crop_h, crop_w = self.crop_size
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)
        
        # 裁剪
        data = data[:, self.top:self.top + crop_h, self.left:self.left + crop_w]
        return data


class ElasticDeformationd(BaseTransform):
    """彈性變形（圖像和標籤同步）"""
    def __init__(self, keys, alpha=100, sigma=10, prob=0.5, **kwargs):
        super(ElasticDeformationd, self).__init__(keys, **kwargs)
        self.alpha = alpha  # 變形強度（像素單位）
        self.sigma = sigma  # 高斯核標準差（控制變形平滑度）
        self.prob = prob  # 應用變形的概率
        self.displacement = None
        self.apply_deformation = False

    def __call__(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        # 決定是否應用變形
        self.apply_deformation = random.random() < self.prob
        
        if self.apply_deformation:
            # 獲取第一個 key 的尺寸來生成變形場（所有 keys 使用相同的變形場）
            first_key = self.keys[0]
            if first_key in data:
                _, h, w = data[first_key].shape
                
                # 生成隨機位移場（標準正態分佈）
                dx = torch.randn(1, h, w)
                dy = torch.randn(1, h, w)
                
                # 使用高斯濾波平滑位移場
                kernel_size = int(6 * self.sigma) + 1
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel_size = min(kernel_size, min(h, w))  # 確保不超過圖像尺寸
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                # 使用平均池化近似高斯平滑
                padding = kernel_size // 2
                dx = F.avg_pool2d(dx.unsqueeze(0), kernel_size=kernel_size, stride=1, padding=padding).squeeze(0)
                dy = F.avg_pool2d(dy.unsqueeze(0), kernel_size=kernel_size, stride=1, padding=padding).squeeze(0)
                
                # 縮放位移場到 alpha 強度
                dx = dx * self.alpha
                dy = dy * self.alpha
                
                self.displacement = (dx, dy)
        return super().__call__(data, *args, **kwargs)

    def _process(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if not self.apply_deformation or self.displacement is None:
            return data
        
        dx, dy = self.displacement
        _, h, w = data.shape
        
        # 創建網格座標
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=data.device),
            torch.arange(w, dtype=torch.float32, device=data.device),
            indexing='ij'
        )
        
        # 應用位移
        x_coords = x_coords + dx.squeeze(0)
        y_coords = y_coords + dy.squeeze(0)
        
        # 歸一化到 [-1, 1] 範圍（grid_sample 的要求）
        x_coords = 2.0 * x_coords / (w - 1) - 1.0
        y_coords = 2.0 * y_coords / (h - 1) - 1.0
        
        # 創建採樣網格 (N, H, W, 2)
        grid = torch.stack([x_coords, y_coords], dim=-1).unsqueeze(0)
        
        # 使用 grid_sample 進行變形
        mode = 'nearest' if "label" in str(self.keys) else 'bilinear'
        padding_mode = 'border'
        
        data = data.unsqueeze(0)  # (1, C, H, W)
        data = F.grid_sample(data, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
        data = data.squeeze(0)  # (C, H, W)
        
        return data


class NormalizeLabeld(BaseTransform):
    """確保標籤歸一化到 [0, 1] 範圍"""
    def __init__(self, keys, **kwargs):
        super(NormalizeLabeld, self).__init__(keys, **kwargs)

    def _process(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 如果最大值 > 1，則歸一化
        if data.max() > 1.0:
            data = data / 255.0
        # 確保值在 [0, 1] 範圍內
        data = torch.clamp(data, 0.0, 1.0)
        return data


class SplitDataset(MedicalImageDataset):
    def __init__(self, data_list, has_label, transform):
        self.data_list = data_list
        self.has_label = has_label
        self.transform = transform


# ============================================================================
# SimCLR Data Augmentation Components
# ============================================================================
class RandomResizedCropd(BaseTransform):

    def __init__(self, keys, size, scale=(0.08, 1.0), ratio=(0.75, 1.33), **kwargs):
        super().__init__(keys, **kwargs)
        # 確保 size 是 tuple 格式 (H, W)
        self.size = size if isinstance(size, (tuple, list)) else (size, size)
        self.scale = scale
        self.ratio = ratio

    def __call__(self, data):
        img = data[self.keys[0]]

        # 取得 crop 參數（由 torchvision 產生）
        i, j, h, w = RandomResizedCrop.get_params(
            img if isinstance(img, Image.Image) else TF.to_pil_image(img),
            scale=self.scale,
            ratio=self.ratio
        )

        for key in self.keys:
            arr = data[key]

            if isinstance(arr, torch.Tensor):
                # tensor crop
                cropped = arr[:, i:i+h, j:j+w]

                # label 使用 nearest，image 使用 bilinear
                mode = "nearest" if key == "label" else "bilinear"

                cropped = F.interpolate(
                    cropped.unsqueeze(0),
                    size=self.size,
                    mode=mode,
                    align_corners=False if mode == "bilinear" else None
                ).squeeze(0)
            else:
                # PIL 版
                interpolation = Image.NEAREST if key == "label" else Image.BILINEAR
                cropped = TF.resized_crop(arr, i, j, h, w, self.size, interpolation)

            data[key] = cropped

        return data

    
class GaussianBlurd(BaseTransform):
    """
    使用 torchvision 官方 GaussianBlur
    """
    def __init__(self, keys, sigma=(0.1, 2.0), prob=0.5, kernel_size=23, **kwargs):
        super().__init__(keys, **kwargs)
        self.sigma = sigma
        self.prob = prob
        self.kernel_size = kernel_size

    def _process(self, data, **kwargs):
        if random.random() > self.prob:
            return data

        # double-check 是 tensor
        if not isinstance(data, torch.Tensor):
            data = TF.to_tensor(data)

        sigma = random.uniform(*self.sigma)

        # kernel size 必須是奇數
        return TF.gaussian_blur(data, kernel_size=self.kernel_size, sigma=sigma)


class SimCLRTransform:
    """
    Strong augmentation for SimCLR (image only)
    """
    def __init__(self, size=(448,576), gaussian_blur_prob=0.5, gaussian_noise_std=0.01):
        self.crop = RandomResizedCropd(
            keys=["image"],
            size=size,
            scale=(0.08, 1.0),
            ratio=(0.75, 1.33)
        )

        self.transforms = [
            RandomHorizontalFlipd(keys=["image"], prob=0.5),
            RandomVerticalFlipd(keys=["image"], prob=0.5),
            RandomRotationd(keys=["image"], degrees=15),
            RandomAffined(
                keys=["image"],
                scale_range=(0.95, 1.05),
                translate_range=(0.02, 0.02),
            ),
            GaussianBlurd(keys=["image"], prob=gaussian_blur_prob),
            RandomNoiseInjectiond(keys=["image"], std=gaussian_noise_std),
        ]

        self.to_tensor = ToTensord(keys=["image"])

    def __call__(self, image):
        data = {"image": image}
        data = self.to_tensor(data)
        data = self.crop(data)
        for t in self.transforms:
            data = t(data)
        return data["image"]


class SegmentationTransform:
    """
    Gentle augmentations for downstream segmentation.
    Always applied to both image + label.
    """
    def __init__(self, size=(448,576)):
        self.transforms = [
            ToTensord(keys=["image", "label"]),
            RandomResizedCropd(
                keys=["image","label"], 
                size=size,
                scale=(1.0,1.0),     # keep scale stable
                ratio=(1.0,1.0)      # keep aspect ratio
            ),
            RandomHorizontalFlipd(keys=["image","label"], prob=0.5),
            RandomRotationd(keys=["image","label"], degrees=15),
            RandomAffined(
                keys=["image","label"],
                scale_range=(0.95,1.05),
                translate_range=(0.02,0.02)
            )
        ]

    def __call__(self, sample):
        data = {"image": sample["image"], "label": sample["label"]}
        for t in self.transforms:
            data = t(data)
        return data


class SimCLRDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = SimCLRTransform() if transform is None else transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        image = sample["image"] if isinstance(sample, dict) else sample
        v1 = self.transform(image)
        v2 = self.transform(image)
        return {"view1": v1, "view2": v2, "index": idx}
