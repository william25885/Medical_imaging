from argparse import ArgumentParser, Namespace
from pathlib import Path
import random
import math

import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.dataset import (
    MedicalImageDataset,
    SimCLRDataset,
    LoadImaged,
)
from torchvision.transforms import Compose
from src.model import SimCLRModel
from src.loss import NTXentLoss
from src.trainer import SimCLRTrainer
from src.utils import set_random_seed


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='simclr')
    return parser.parse_args()

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1
        if epoch < self.warmup_epochs:
            return [
                base_lr * epoch / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        cos_epoch = epoch - self.warmup_epochs
        cos_total = self.max_epochs - self.warmup_epochs
        return [
            base_lr * 0.5 * (1 + math.cos(math.pi * cos_epoch / cos_total))
            for base_lr in self.base_lrs
        ]


if __name__ == "__main__":
    set_random_seed()
    args = parse_arguments()
    config = OmegaConf.load(Path("configs", f"{args.config}.yaml"))
    
    base_transform = Compose([
        LoadImaged(keys=["image"]),
    ])
    
    base_dataset = MedicalImageDataset(
        data_dir=config.dataset.data_dir,
        has_label=False,
        transform=base_transform
    )
    
    simclr_dataset = SimCLRDataset(base_dataset)
    
    train_loader = DataLoader(
        simclr_dataset,
        batch_size=config.train_loader.batch_size,
        num_workers=config.train_loader.num_workers,
        shuffle=config.train_loader.shuffle,
        pin_memory=True
    )
    
    model = SimCLRModel(
        img_size=tuple(config.model.img_size),
        in_channels=config.model.in_channels,
        vit_embed_dim=config.model.vit_embed_dim,
        resnet_pretrained=config.model.get('resnet_pretrained', True),
        projection_dim=config.model.projection_dim,
    )
    
    criterion = NTXentLoss(temperature=config.trainer.temperature)
    
    if config.optimizer.type == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay
        )
    elif config.optimizer.type == "Adam":
        optimizer = torch.optim.Adam(
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
    else:
        lr_scheduler = WarmupCosineLR(
            optimizer,
            warmup_epochs=config.lr_scheduler.warmup_epochs,
            max_epochs=config.trainer.n_epoch
        )

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wandb.init(project="ntucsie_ml_hw2", name="simclr_pretrain")
    wandb.watch(model, log="all")
    
    checkpoint_path = config.checkpoint.path
    checkpoint_best = checkpoint_path.replace(".pth", "_best.pth")
    checkpoint_last = checkpoint_path.replace(".pth", "_last.pth")
    
    trainer = SimCLRTrainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        save_path_best=checkpoint_best,
        save_path_last=checkpoint_last,
        logger=wandb,
    )
    
    trainer.fit(n_epoch=config.trainer.n_epoch)

