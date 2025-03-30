from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from collections import OrderedDict
import os

from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class TrainerConfig:
    max_epochs: int = None
    batch_size: int = None
    data_loader_workers: int = None
    grad_norm_clip: float = None
    snapshot_path: Optional[str] = None
    save_every: int = None
    use_amp: bool = False



class Trainer:

    def __init__(self, trainer_config: TrainerConfig, model, optimizer, train_dataset, test_dataset=None):
        self.config = trainer_config
        self.local_rank = int(os.environ.get('LOCAL_RANK'))
        self.global_rank = int(os.environ.get('RANK'))

        self.train_dataset = train_dataset
        self.train_loader = self._prepare_dataloader(self.train_dataset)
        self.test_loader = self._prepare_dataloader(test_dataset) if test_dataset else None

        self.epochs_run = 0
        self.model = model.to(self.local_rank)
        self.optimizer = optimizer
        self.save_every = self.config.save_every

        if self.config.snapshot_path is None:
            self.config.snapshot_path = "snapshot.pth"
        
        self._load_snapshot()

        self.model = DDP(self.model, device_ids=[self.local_rank])
        
    
    
    def _prepare_dataloader(self, dataset):
        return DataLoader(
            dataset, 
            batch_size=self.config.batch_size,
            num_workers=self.config.data_loader_workers,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(dataset),
        )


    def _load_snapshot(self):
        if os.path.exists(self.config.snapshot_path):
            snapshot = torch.load(self.config.snapshot_path)
            self.model.load_state_dict(snapshot.model_state)
            self.optimizer.load_state_dict(snapshot.optimizer_state)
            self.epochs_run = snapshot.finished_epoch
            print(f"Resuming training from epoch {self.epochs_run}")
            

    def _run_batch(self, source, targets, train: bool = True) -> float:
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(self.config.use_amp)):
            _, loss = self.model(source, targets)
        
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
            self.optimizer.step()
        
        return loss.item()
    
    def _run_epoch(self, epoch: int, data_loader: DataLoader, train: bool = True):
        data_loader.sampler.set_epoch(epoch)
        for iter, (source, targets) in enumerate(data_loader):
            step_type = "Train" if train else "Eval"
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            batch_loss = self._run_batch(source, targets, train)
            if iter % 100 == 0:
                print(f"[GPU{self.global_rank}] Epoch {epoch} | Iter {iter} | {step_type} Loss: {batch_loss:.5f}")

    def _save_snapshot(self, epoch):
        model = self.model 
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = {
            "model_state": raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "finished_epoch": epoch,
        }
        torch.save(snapshot, self.config.snapshot_path)
        print(f"Snapshot saved to at epoch {epoch} at {self.config.snapshot_path}")
    
    def train(self):
        for epoch in range(self.epochs_run, self.config.max_epochs):
            epoch = epoch + 1
            self._run_epoch(epoch, self.train_loader, train=True)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            if self.test_loader is not None:
                self._run_epoch(epoch, self.test_loader, train=False)
