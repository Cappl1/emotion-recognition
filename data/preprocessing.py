import torch
from torch.utils.data import random_split, DataLoader

def create_dataloaders(dataset, batch_size=32, train_split=0.8, val_split=0.2, seed=42):
    """
    Create training and validation dataloaders from a dataset.
    
    Args:
        dataset: The full dataset
        batch_size: Batch size for dataloaders
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Calculate lengths
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_dataloader, val_dataloader