"""
Author: YAG
"""
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict
import glob
from pathlib import Path
import random

class SegmentationDataset(Dataset):
    def __init__(self, 
                 data_root: str,
                 img_dir: str, 
                 mask_dir: str,
                 img_size: int = 1000,
                 is_training: bool = False):

        self.data_root = Path(data_root)
        self.img_dir = self.data_root / img_dir
        self.mask_dir = self.data_root / mask_dir
        self.img_size = img_size
        self.is_training = is_training
        
        self.img_files = self._get_image_files()
        self.mask_files = self._get_corresponding_masks()
        
        assert len(self.img_files) == len(self.mask_files), \
            f"Number of images ({len(self.img_files)}) != number of masks ({len(self.mask_files)})"
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        print(f"Loaded {len(self.img_files)} samples from {self.img_dir}")
    
    def _get_image_files(self):

        extensions = ['*.png']
        img_files = []
        for ext in extensions:
            img_files.extend(glob.glob(str(self.img_dir / ext)))
        return sorted(img_files)
    
    def _get_corresponding_masks(self):

        mask_files = []
        for img_file in self.img_files:
            img_name = Path(img_file).name
            mask_file = self.mask_dir / img_name
            if mask_file.exists():
                mask_files.append(str(mask_file))
            else:
                raise FileNotFoundError(f"Mask file not found: {mask_file}")
        return mask_files
    
    def _random_resize(self, image: np.ndarray, mask: np.ndarray):
        if not self.is_training:
            return image, mask
            
        scale_factor = random.uniform(0.5, 2.0)
        
        # Calculate new size
        h, w = image.shape[:2]
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        return image, mask
    
    def _random_crop(self, image: np.ndarray, mask: np.ndarray):
        h, w = image.shape[:2]
        target_h, target_w = self.img_size, self.img_size
        
        if h <= target_h and w <= target_w:

            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, 
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, 
                                    cv2.BORDER_CONSTANT, value=0)
            h, w = image.shape[:2]
        
        if self.is_training and (h > target_h or w > target_w):
            start_h = random.randint(0, max(0, h - target_h))
            start_w = random.randint(0, max(0, w - target_w))
        else:
            start_h = (h - target_h) // 2
            start_w = (w - target_w) // 2
        
        end_h = start_h + target_h
        end_w = start_w + target_w
        
        image = image[start_h:end_h, start_w:end_w]
        mask = mask[start_h:end_h, start_w:end_w]
        
        return image, mask
    
    def _photometric_distortion(self, image: np.ndarray):
        if not self.is_training:
            return image
        
        image = image.astype(np.float32)
        
        if random.random() < 0.5:
            delta = random.uniform(-32, 32)
            image += delta
        
        if random.random() < 0.5:
            alpha = random.uniform(0.5, 1.5)
            image *= alpha
        
        if random.random() < 0.5:
            image_hsv = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            
            if random.random() < 0.5:
                sat_factor = random.uniform(0.5, 1.5)
                image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1] * sat_factor, 0, 255)
            
            if random.random() < 0.5:
                hue_delta = random.uniform(-18, 18)
                image_hsv[:, :, 0] = (image_hsv[:, :, 0] + hue_delta) % 180
            
            image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
        
        image = np.clip(image, 0, 255)
        
        return image.astype(np.uint8)
    
    def _apply_augmentation(self, image: np.ndarray, mask: np.ndarray):

        if not self.is_training:
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            return image, mask
        
        image, mask = self._random_resize(image, mask)
        image, mask = self._random_crop(image, mask)
        
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        image = self._photometric_distortion(image)
        
        return image, mask
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255.0
        
        image = (image - self.mean) / self.std
        
        return image
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]
        
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Cannot load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot load mask: {mask_path}")
        
        image, mask = self._apply_augmentation(image, mask)
        
        mask = (mask > 127).astype(np.float32)
        
        image = self._normalize_image(image)
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        mask = torch.from_numpy(mask)  # Keep as float for BCE loss
        
        return {
            'image': image,
            'mask': mask,
            'img_path': img_path,
            'mask_path': mask_path
        }


if __name__ == "__main__":
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=data_root
        )
        
        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Validation dataset size: {len(val_loader.dataset)}")
        print(f"Test dataset size: {len(test_loader.dataset)}")
        
        for batch in train_loader:
            print(f"Image batch shape: {batch['image'].shape}")
            print(f"Mask batch shape: {batch['mask'].shape}")
            print(f"Image dtype: {batch['image'].dtype}")
            print(f"Mask dtype: {batch['mask'].dtype}")
            print(f"Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
            print(f"Mask range: [{batch['mask'].min():.3f}, {batch['mask'].max():.3f}]")
            break
            
    except Exception as e:
        print(f"Error: {e}")
