import torch
from torch import nn
from tqdm import tqdm

from src.utils import dict_to_device


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        eval_func: callable,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        save_path: str = "best_model.pth",
        logger = None,
        early_stopping_patience: int = None,
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_func = eval_func
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.save_path = save_path
        self.logger = logger
        self.early_stopping_patience = early_stopping_patience
        
        self.use_amp = device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_data in tqdm(self.train_loader, desc="Training"):
            batch_data = dict_to_device(batch_data, self.device)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    images = batch_data["image"]
                    labels = batch_data["label"].float()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                images = batch_data["image"]
                labels = batch_data["label"].float()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if self.logger:
                self.logger.log({"train/loss": loss.item()})
                if hasattr(self.lr_scheduler, 'get_last_lr'):
                    self.logger.log({"train/lr": self.lr_scheduler.get_last_lr()[0]})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        if self.logger:
            self.logger.log({"train/avg_loss": avg_loss})

    def valid_one_epoch(self):
        scores = []
        self.model.eval()
        with torch.inference_mode():
            for batch_data in tqdm(self.valid_loader, desc="Validation"):
                batch_data = dict_to_device(batch_data, self.device)

                images = batch_data["image"]
                labels = batch_data["label"].float()
                outputs = self.model(images)

                dice_score = self.eval_func(outputs, labels, mean=False)
                scores.extend(dice_score.cpu().numpy().tolist())

        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"Validation Dice Score: {avg_score:.4f}")
        if self.logger:
            self.logger.log({"valid/dice_score": avg_score})
        return avg_score

    def fit(self, n_epoch):
        best_score = 0
        epochs_without_improvement = 0
        self.model.to(self.device)
        
        for epoch in range(n_epoch):
            print(f"\nEpoch {epoch + 1}/{n_epoch}")
            self.train_one_epoch()
            valid_score = self.valid_one_epoch()

            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(valid_score)
            elif hasattr(self.lr_scheduler, 'step') and hasattr(self.lr_scheduler, 'warmup_epochs'):
                self.lr_scheduler.step(valid_score)
            else:
                self.lr_scheduler.step()

            if valid_score > best_score:
                best_score = valid_score
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"Successfully saved best model with score: {best_score:.4f}")
            else:
                epochs_without_improvement += 1
                print(f"No improvement. Best score: {best_score:.4f}, Epochs without improvement: {epochs_without_improvement}")
                
                if self.early_stopping_patience is not None and epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered! No improvement for {self.early_stopping_patience} epochs.")
                    print(f"Best validation Dice Score: {best_score:.4f}")
                    break
        
        print(f"\nTraining completed. Best validation Dice Score: {best_score:.4f}")


class SimCLRTrainer:
    """
    Trainer for SimCLR self-supervised pre-training (AMP)
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_loader: torch.utils.data.DataLoader,
        save_path_best: str = "simclr_encoder_best.pth",
        save_path_last: str = "simclr_encoder_last.pth",
        logger = None,
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.save_path_best = save_path_best
        self.save_path_last = save_path_last
        self.logger = logger

        self.scaler = torch.amp.GradScaler('cuda')

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        loop = tqdm(self.train_loader, desc=f"Training Epoch {epoch}")

        for batch_data in loop:
            view1 = batch_data["view1"].to(self.device)
            view2 = batch_data["view2"].to(self.device)

            self.optimizer.zero_grad()

            if self.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    z1 = self.model(view1, return_projection=True)
                    z2 = self.model(view2, return_projection=True)
                    loss = self.criterion(z1, z2)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                z1 = self.model(view1, return_projection=True)
                z2 = self.model(view2, return_projection=True)
                loss = self.criterion(z1, z2)
                
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            loop.set_postfix({"loss": loss.item()})

            if self.logger:
                self.logger.log({"train/loss": loss.item()})
                if hasattr(self.lr_scheduler, 'get_last_lr'):
                    self.logger.log({"train/lr": self.lr_scheduler.get_last_lr()[0]})
                else:
                    self.logger.log({"train/lr": self.optimizer.param_groups[0]["lr"]})

        avg_loss = total_loss / num_batches

        if self.logger:
            lr = (
                self.lr_scheduler.get_last_lr()[0]
                if hasattr(self.lr_scheduler, "get_last_lr")
                else self.optimizer.param_groups[0]["lr"]
            )
            self.logger.log({
                "train/avg_loss": avg_loss,
                "train/lr": lr
            })

        return avg_loss

    def save_encoder(self, path):
        encoder_state = {}
        for k, v in self.model.state_dict().items():
            if k.startswith("encoder."):
                new_key = k.replace("encoder.", "", 1)
                if not new_key.startswith("projector."):
                    encoder_state[new_key] = v
        torch.save(encoder_state, path)

    def fit(self, n_epoch):
        best_loss = float('inf')
        self.model.to(self.device)

        for epoch in range(1, n_epoch + 1):
            avg_loss = self.train_one_epoch(epoch)

            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(avg_loss)
            else:
                self.lr_scheduler.step()

            print(f"Epoch {epoch}/{n_epoch} | Avg NT-Xent Loss = {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_encoder(self.save_path_best)
                print(f"✓ Saved BEST encoder (loss={best_loss:.4f})")

        self.save_encoder(self.save_path_last)
        print(f"\nTraining finished. Saved LAST encoder → {self.save_path_last}")
        print(f"Best loss: {best_loss:.4f}")
