"""
Author: YAG
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import PolynomialLR
import numpy as np
from tqdm import tqdm
import logging
import json
from segformer_model import create_segformer_b5, SegFormer
from data_loader import create_dataloaders
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix

class BinarySegmentationMetrics:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_tn = 0
        self.total_fn = 0
        self.total_pixels = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred_binary = (pred > self.threshold).float()
        
        pred_flat = pred_binary.cpu().numpy().flatten()
        target_flat = target.cpu().numpy().flatten()
        
        tp = ((pred_flat == 1) & (target_flat == 1)).sum()
        fp = ((pred_flat == 1) & (target_flat == 0)).sum()
        tn = ((pred_flat == 0) & (target_flat == 0)).sum()
        fn = ((pred_flat == 0) & (target_flat == 1)).sum()
        
        self.total_tp += tp
        self.total_fp += fp
        self.total_tn += tn
        self.total_fn += fn
        self.total_pixels += len(pred_flat)

class SegFormerTrainer:
    def __init__(self, 
                 model: SegFormer,
                 train_loader,
                 val_loader,
                 device: torch.device,
                 config: Dict):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self._setup_optimizer()
        self._setup_scheduler()
        
        self.criterion = nn.BCELoss()
        
        self._setup_logging()
        
        self.train_metrics = BinarySegmentationMetrics()
        self.val_metrics = BinarySegmentationMetrics()
        
        self.current_iter = 0
        self.best_miou = 0.0
        self.training_history = {
            'train_loss': [], 
            'val_loss': [], 
            'val_miou': [],
            'val_mdice': [],
            'val_macc': []
        }
    
    def _setup_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],  # 4e-5
            weight_decay=self.config['weight_decay'],  # 0.01
            betas=(0.9, 0.999)  # Paper specification
        )
    
    def _setup_scheduler(self):
        self.scheduler = PolynomialLR(
            self.optimizer, 
            total_iters=self.config['max_iters'],
            power=1.0
        )
    
    def _setup_logging(self):
        log_dir = Path(self.config['work_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self) -> float:
        self.model.train()
        self.train_metrics.reset()
        epoch_loss = 0.0
        num_batches = 0
        pbar = tqdm(self.train_loader, desc=f"Training (Iter {self.current_iter})")
        
        for batch in pbar:
            if self.current_iter >= self.config['max_iters']:
                break
                
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)  # Shape: (B, H, W), values in [0, 1]
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.train_metrics.update(outputs, masks)
            epoch_loss += loss.item()
            num_batches += 1
            self.current_iter += 1
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'lr': f"{current_lr:.2e}",
                'iter': f"{self.current_iter}/{self.config['max_iters']}"
            })
            
            if self.current_iter % self.config['val_interval'] == 0:
                val_results = self.validate()
                self._save_checkpoint(val_results['mIoU'])
                self.model.train()  # Resume training mode
            
            if self.current_iter >= self.config['max_iters']:
                break
        
        return epoch_loss / max(num_batches, 1)
    
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        self.val_metrics.reset()
        
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                self.val_metrics.update(outputs, masks)
                val_loss += loss.item()
                num_batches += 1
        
        metrics = self.val_metrics.compute()
        metrics['val_loss'] = val_loss / max(num_batches, 1)
        
        self.logger.info(
            f"Iter {self.current_iter}: "
            f"Val Loss: {metrics['val_loss']:.4f}, "
            f"mIoU: {metrics['mIoU']:.4f}, "
            f"mAcc: {metrics['mAcc']:.4f}, "
            f"mDice: {metrics['mDice']:.4f}, "
        )
        
        self.training_history['val_loss'].append(metrics['val_loss'])
        self.training_history['val_miou'].append(metrics['mIoU'])
        self.training_history['val_mdice'].append(metrics['mDice'])
        self.training_history['val_macc'].append(metrics['mAcc'])
        
        return metrics
    
    def _save_checkpoint(self, current_miou: float):
        checkpoint_dir = Path(self.config['work_dir']) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest_path = checkpoint_dir / 'latest.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'current_iter': self.current_iter,
            'best_miou': self.best_miou,
            'config': self.config,
            'training_history': self.training_history
        }, latest_path)
        
        if current_miou > self.best_miou:
            self.best_miou = current_miou
            best_path = checkpoint_dir / f'best_mIoU_iter_{self.current_iter}.pth'
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'current_iter': self.current_iter,
                'best_miou': self.best_miou,
                'config': self.config
            }, best_path)
            self.logger.info(f"New best mIoU: {self.best_miou:.4f} at iteration {self.current_iter}")
    
    def train(self):
        self.logger.info("Starting training with paper configuration...")
        self.logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        start_time = datetime.now()
        
        while self.current_iter < self.config['max_iters']:
            train_loss = self.train_epoch()
           
            train_metrics = self.train_metrics.compute()
            self.training_history['train_loss'].append(train_loss)
            
            self.logger.info(
                f"Epoch completed at iter {self.current_iter}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Train mIoU: {train_metrics['mIoU']:.4f}"
            )
            
            if self.current_iter >= self.config['max_iters']:
                break
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        self.logger.info(f"Training completed! Total time: {training_time}")
        self.logger.info(f"Best mIoU achieved: {self.best_miou:.4f}")
        
        final_metrics = self.validate()
        self.logger.info(f"Final validation metrics: {json.dumps(final_metrics, indent=2)}")
        
        self._save_training_curves()
        return final_metrics
    

def main():
    config = {
        'data_root': 'data',
        'img_size': 1000,
        'batch_size': 4,
        'num_workers': 4,
        'max_iters': 60000,  
        'val_interval': 500,
        'lr': 4e-5, 
        'weight_decay': 0.01, 
        'work_dir': ,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {gpu_memory:.1f} GB")
        if gpu_memory < 16:
            print("Warning: GPU memory < 16GB. Batch size 4 with 1000x1000 images may cause OOM.")
            print("Consider reducing batch size or using gradient accumulation.")
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=config['data_root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        
        print(f"Dataset sizes - Train: {len(train_loader.dataset)}, "
              f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
        
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return
    
    model = create_segformer_b5(img_size=config['img_size'])
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    try:
        trainer = SegFormerTrainer(model, train_loader, val_loader, device, config)
        final_metrics = trainer.train()
        
        print("\nTraining completed successfully!")
        print(f"Final metrics: mIoU={final_metrics['mIoU']:.4f}, "
              f"mDice={final_metrics['mDice']:.4f}, mAcc={final_metrics['mAcc']:.4f}")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"GPU out of memory error: {e}")
            print("Suggestions:")
            print("1. Reduce batch_size to 2 or 1")
            print("2. Use gradient accumulation to maintain effective batch size")
            print("3. Use a GPU with more memory")
        else:
            print(f"Runtime error: {e}")
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
