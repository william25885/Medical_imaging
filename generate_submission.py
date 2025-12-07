from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision.transforms import Compose
from tqdm import tqdm

from src.dataset import MedicalImageDataset, LoadImaged, ToTensord, ResizeImaged
from src.model import UNet, TransUNet
from src.utils import dict_to_device, set_random_seed


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--public_dir', type=str, required=True)
    parser.add_argument('--private_dir', type=str, required=True)
    return parser.parse_args()


def rle(image: np.ndarray) -> str:
    indices = np.where(image.flatten(order='F'))[0] + 1
    if len(indices) == 0:
        return "-1"

    prev, code = -1, []
    for ind in indices:
        if ind > prev + 1:
            code.append(ind)
            code.append(0)
        code[-1] += 1
        prev = ind

    return ' '.join(map(str, code))


def infer(dataset, model, device):
    pred_dict = {}
    model.eval()
    with torch.inference_mode():
        for data in tqdm(dataset, desc="Inference"):
            original_size = data.get("original_size", None)
            if original_size is not None:
                if isinstance(original_size, torch.Tensor):
                    original_size = tuple(original_size.cpu().numpy().tolist())
                elif isinstance(original_size, (list, tuple)):
                    original_size = tuple(original_size)
            
            data = dict_to_device(data, device)
            
            images = data["image"]
            if images.dim() == 3:
                images = images.unsqueeze(0)

            outputs = model(images)
            
            if isinstance(model, TransUNet):
                prob = torch.sigmoid(outputs)
            else:
                prob = outputs
            
            if original_size is not None:
                prob = F.interpolate(prob, size=original_size, mode="bilinear", align_corners=False)

            pred_binary = (prob > 0.45).float()
            pred_np = pred_binary.squeeze(0).squeeze(0).cpu().numpy()
            pred_np = (pred_np * 255).astype(np.uint8)
            code = rle(pred_np)

            pred_dict[data["filename"]] = code
    return pred_dict



if __name__ == "__main__":
    set_random_seed()
    args = parse_arguments()
    config = OmegaConf.load(Path("configs", f"{args.config}.yaml"))
    transform = Compose(
        [
            LoadImaged(keys=["image"]),
            ToTensord(keys=["image"]),
            ResizeImaged(keys=["image"], save_original_size=True, **config.transform.resized_imaged),
        ]
    )
    public_dataset = MedicalImageDataset(data_dir=args.public_dir, has_label=False, transform=transform)
    private_dataset = MedicalImageDataset(data_dir=args.private_dir, has_label=False, transform=transform)

    if hasattr(config, 'model') and config.model.get('type') == 'TransUNet':
        model = TransUNet(
            img_size=tuple(config.model.img_size),
            in_channels=config.model.in_channels,
            num_classes=config.model.get('num_classes', 1),
            vit_embed_dim=config.model.get('vit_embed_dim', 768),
            vit_depth=config.model.get('vit_depth', 16),
            vit_num_heads=config.model.get('vit_num_heads', 12),
            vit_mlp_ratio=config.model.get('vit_mlp_ratio', 4),
            vit_dropout=config.model.get('vit_dropout', 0.1),
            resnet_pretrained=False
        )
    else:
        in_channels = config.model.in_channels if hasattr(config, 'model') and hasattr(config.model, 'in_channels') else 1
        out_channels = config.model.out_channels if hasattr(config, 'model') and hasattr(config.model, 'out_channels') else 1
        model = UNet(in_channels=in_channels, out_channels=out_channels)
    
    model.load_state_dict(torch.load(config.checkpoint.path, map_location='cpu'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    public_pred = infer(public_dataset, model, device)
    private_pred = infer(private_dataset, model, device)
    all_pred = public_pred | private_pred
    df = pd.DataFrame.from_dict(all_pred, orient="index", columns=["label"])
    df.index.name = "row ID"
    df.reset_index().to_csv("submission.csv", index=False)
