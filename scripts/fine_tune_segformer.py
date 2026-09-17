#!/usr/bin/env python3
"""
Fine-tune SegFormer-B5 untuk data Indonesia.

Pipeline:
1. Label manual ~100-200 foto menggunakan tool segmentasi (CVAT/LabelMe)
2. Train dengan PyTorch Lightning / HuggingFace Trainers
3. Evaluasi pada held-out test set

Research Reference:
- "Domain Adaptation for Urban Scene Understanding" (CVPR 2024)
- Learning rate: 1e-5 (low to preserve pre-trained knowledge)
- Epochs: 10-20
- Batch size: 8-16 (depends on GPU memory)
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Import SegFormer
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchvision import transforms

class UrbanSceneDataset(Dataset):
    """Custom dataset untuk fine-tuning."""
    
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        
        # Get all image files
        self.images = list(self.image_dir.glob("*.jpg")) + \
                      list(self.image_dir.glob("*.png"))
        
        print(f"Found {len(self.images)} images in {image_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name.replace('.jpg', '.png').replace('.png', '.png')
        
        # Load image and mask
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']
        
        return image, mask


def prepare_data(args):
    """Prepare training and validation data."""
    
    # Split data: 80% train, 20% val
    from sklearn.model_selection import train_test_split
    
    image_dirs = [d for d in args.data_dir.iterdir() if d.is_dir()]
    
    train_images, val_images = train_test_split(
        image_dirs, 
        test_size=args.val_ratio, 
        random_state=42
    )
    
    print(f"\n✓ Training samples: {len(train_images)}")
    print(f"✓ Validation samples: {len(val_images)}")
    
    return train_images, val_images


def fine_tune_model(args):
    """Fine-tune SegFormer-B5."""
    
    print("\n" + "="*70)
    print("FINE-TUNING SEGFOMER-B5 FOR INDONESIA URBAN SCENES")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")
    
    # Load pre-trained model
    print("Loading pre-trained SegFormer-B5...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        ignore_mismatched_sizes=True
    ).to(device)
    
    # Freeze early layers (optional - preserves pre-trained features)
    if args.freeze_layers > 0:
        print(f"Freezing first {args.freeze_layers} layers...")
        for name, param in model.named_parameters():
            if 'layer.{0,' + str(args.freeze_layers) + '}' in name or \
               'patch_embed' in name:
                param.requires_grad = False
    
    # Create dataloaders
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = UrbanSceneDataset(args.train_dir, args.mask_dir, transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    
    # Training loop
    print("\nStarting training...")
    print("-" * 70)
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            images = images.to(device)
            masks = masks.long().to(device)
            
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            total += masks.numel()
            correct += (preds == masks).sum().item()
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        print(f"\nEpoch {epoch+1}/{args.num_epochs}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = args.output_dir / f"segformer_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': accuracy,
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path}")
    
    # Final save
    final_checkpoint = args.output_dir / "segformer_finetuned_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_checkpoint)
    print(f"\n✓ Final model saved: {final_checkpoint}")
    
    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SegFormer-B5 for Indonesian urban scenes")
    parser.add_argument("--train-dir", type=str, required=True,
                       help="Directory containing training images")
    parser.add_argument("--mask-dir", type=str, required=True,
                       help="Directory containing corresponding segmentation masks")
    parser.add_argument("--output-dir", type=str, default="models/finetuned",
                       help="Output directory for trained models")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--num-epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                       help="Learning rate (default: 1e-5 for fine-tuning)")
    parser.add_argument("--freeze-layers", type=int, default=0,
                       help="Number of layers to freeze (0 = none)")
    parser.add_argument("--save-interval", type=int, default=1,
                       help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    fine_tune_model(args)


if __name__ == "__main__":
    main()
