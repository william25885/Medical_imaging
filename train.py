from argparse import ArgumentParser, Namespace
from pathlib import Path
import random
import math

import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from src.dataset import (
    MedicalImageDataset, 
    SplitDataset, 
    LoadImaged, 
    ToTensord, 
    ResizeImaged,
    RandomHorizontalFlipd,
    RandomNoiseInjectiond,
    RandomBrightnessd,
    RandomContrastd,
    NormalizeLabeld,
)
from src.metric import DiceScore
from src.model import UNet, TransUNet
from src.loss import BCEDiceLoss
from src.trainer import Trainer
from src.utils import set_random_seed

def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='cnn_unet')
    return parser.parse_args()


class WarmupReduceLROnPlateau:
    """
    - 前 warmup_epochs：線性 warmup（lr 從 warmup_start_lr → base_lr）
    - 後續：ReduceLROnPlateau（看 validation 指標自動降 lr）
    """
    def __init__(self, optimizer, warmup_epochs=3, 
                 warmup_start_lr=1e-6,
                 mode='max', factor=0.5, patience=5):
        
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_start_lr
        
        self.current_epoch = 0
        self.reduce_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience
        )
    
    def step(self, metric=None):
        if self.current_epoch < self.warmup_epochs:
            warmup_ratio = (self.current_epoch + 1) / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                target_lr = self.base_lrs[i]
                param_group['lr'] = self.warmup_start_lr + (target_lr - self.warmup_start_lr) * warmup_ratio
            
            self.current_epoch += 1
            return
        
        if metric is not None:
            self.reduce_lr_scheduler.step(metric)
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]



if __name__ == "__main__":
    set_random_seed()
    args = parse_arguments()
    config = OmegaConf.load(Path("configs", f"{args.config}.yaml"))
    
    full_dataset = MedicalImageDataset(
        data_dir=config.dataset.data_dir,
        has_label=config.dataset.has_label
    )
    dataset_size = len(full_dataset)
    train_size = int(config.dataset.train_split * dataset_size)
    valid_size = dataset_size - train_size
    
    random.seed(config.dataset.random_seed)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]
    
    train_data_list = [full_dataset.data_list[i] for i in train_indices]
    valid_data_list = [full_dataset.data_list[i] for i in valid_indices]
    
    train_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
        ResizeImaged(keys=["image", "label"], **config.transform.resized_imaged),
        RandomHorizontalFlipd(keys=["image", "label"], prob=0.05),
        RandomNoiseInjectiond(keys=["image"], std=0.01, prob=0.2),
        RandomBrightnessd(keys=["image"], brightness_factor_range=(0.85, 1.15)),
        RandomContrastd(keys=["image"], contrast_factor_range=(0.85, 1.15)),
        NormalizeLabeld(keys=["label"]),  
    ])

    valid_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ToTensord(keys=["image", "label"]),
            ResizeImaged(keys=["image", "label"], **config.transform.resized_imaged),
            NormalizeLabeld(keys=["label"]),  
        ]
    )
    
    train_dataset = SplitDataset(train_data_list, config.dataset.has_label, train_transform)
    valid_dataset = SplitDataset(valid_data_list, config.dataset.has_label, valid_transform)
    
    train_loader = DataLoader(train_dataset, **config.train_loader)
    valid_loader = DataLoader(valid_dataset, **config.valid_loader)
    
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
        
        if config.model.get('pretrained_encoder_path'):
            pretrained_path = config.model.pretrained_encoder_path
            print(f"Loading pretrained SimCLR encoder from: {pretrained_path}")
            
            try:
                pretrained_state = torch.load(pretrained_path, map_location='cpu')
                
                encoder_state = {}
                matched_layers = []
                skipped_layers = []
                
                for key, value in pretrained_state.items():
                    if key.startswith('encoder.'):
                        resnet_key = key.replace('encoder.', '', 1)
                        
                        if resnet_key in model.encoder.state_dict():
                            if model.encoder.state_dict()[resnet_key].shape == value.shape:
                                encoder_state[resnet_key] = value
                                matched_layers.append(f'encoder.{resnet_key}')
                            else:
                                skipped_layers.append(f"{key} (shape mismatch: {value.shape} vs {model.encoder.state_dict()[resnet_key].shape})")
                        else:
                            skipped_layers.append(f"{key} (not in model.encoder)")
                    elif key.startswith('patch_proj.'):
                        skipped_layers.append(f"{key} (skipped: patch_proj dimension mismatch)")
                    else:
                        skipped_layers.append(f"{key} (unknown layer)")
                
                if encoder_state:
                    model.encoder.load_state_dict(encoder_state, strict=False)
                    print(f"Successfully loaded {len(matched_layers)} ResNet encoder layers from pretrained encoder")
                else:
                    print("  Warning: No matching encoder layers found")
                
                if skipped_layers:
                    print(f"  Skipped {len(skipped_layers)} layers:")
                    # 只顯示前幾個跳過的層，避免輸出過多
                    for layer in skipped_layers[:10]:
                        print(f"    - {layer}")
                    if len(skipped_layers) > 10:
                        print(f"    ... and {len(skipped_layers) - 10} more")
                        
            except Exception as e:
                print(f" Warning: Failed to load pretrained encoder: {e}")
                import traceback
                traceback.print_exc()
                print("  Continuing with random initialization...")
    else:
        in_channels = config.model.in_channels if hasattr(config, 'model') and hasattr(config.model, 'in_channels') else 1
        out_channels = config.model.out_channels if hasattr(config, 'model') and hasattr(config.model, 'out_channels') else 1
        model = UNet(in_channels=in_channels, out_channels=out_channels)

    loss_func = BCEDiceLoss()

    if config.optimizer.type == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.get("weight_decay", 0)
        )
    elif config.optimizer.type == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.get("weight_decay", 0)
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr)

    if config.lr_scheduler.type == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.lr_scheduler.T_max
        )
    elif config.lr_scheduler.type == "WarmupReduceLROnPlateau":
        lr_scheduler = WarmupReduceLROnPlateau(
            optimizer,
            warmup_epochs=config.lr_scheduler.get("warmup_epochs", 5),
            warmup_start_lr=config.lr_scheduler.get("warmup_start_lr", 1e-6),
            mode=config.lr_scheduler.get("mode", "max"),
            factor=config.lr_scheduler.get("factor", 0.5),
            patience=config.lr_scheduler.get("patience", 5)
        )
    elif config.lr_scheduler.type == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.trainer.n_epoch
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="ntucsie_ml_hw2",)
    wandb.watch(model, log="all")

    save_path = config.checkpoint.path
    if hasattr(config, 'model'):
        model_type = config.model.get('type')
        if model_type == 'TransUNet':
            has_pretrained = config.model.get('pretrained_encoder_path') is not None
            if has_pretrained:
                save_path = "simclr_transunet_best.pth"
            else:
                save_path = "transunet_best.pth"
        elif model_type == 'UNet':
            if save_path == "best_model.pth":
                save_path = "cnn_unet_best.pth"

    trainer = Trainer(
        model=model.to(device),
        device=device,
        criterion=loss_func,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        eval_func=DiceScore(),
        train_loader=train_loader,
        valid_loader=valid_loader,
        save_path=save_path,
        logger=wandb,
        early_stopping_patience=config.trainer.get("early_stopping_patience", None),
    )
    trainer.fit(n_epoch=config.trainer.n_epoch)
