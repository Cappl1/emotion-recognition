from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import torch
from torch.utils.data import Subset, DataLoader
from dataclasses import dataclass
import json
import logging
from datetime import datetime
from data.dataset import PreprocessedGazePupilDataset
from collections import Counter

@dataclass
class CVSplitInfo:
    """Information about a single cross-validation split."""
    fold_number: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_valence_dist: Dict[int, int]
    train_arousal_dist: Dict[int, int]
    val_valence_dist: Dict[int, int]
    val_arousal_dist: Dict[int, int]

class CrossValidator:
    def __init__(
        self, 
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        results_dir: Optional[str] = None
    ):
        """
        Initialize cross-validation handler.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
            results_dir: Directory to save CV results
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.results_dir = Path(results_dir) if results_dir else Path("cv_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.splits = None
        self.splits: List[CVSplitInfo] = []
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger("CrossValidator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            fh = logging.FileHandler(self.results_dir / "cv.log")
            fh.setLevel(logging.INFO)
            
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            logger.addHandler(fh)
            logger.addHandler(ch)
        
        return logger

    def _get_label_distribution(
        self, 
        dataset: torch.utils.data.Dataset, 
        indices: np.ndarray
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Get distribution of valence and arousal labels in a subset of data.
        
        Args:
            dataset: Full dataset
            indices: Indices to consider
            
        Returns:
            Tuple of (valence distribution, arousal distribution)
        """
        valence_dist = {0: 0, 1: 0, 2: 0}
        arousal_dist = {0: 0, 1: 0, 2: 0}
        
        for idx in indices:
            _, valence, arousal = dataset[idx]
            valence_dist[valence.item()] += 1
            arousal_dist[arousal.item()] += 1
            
        return valence_dist, arousal_dist

    def analyze_class_distribution(self, dataset):
        """Analyze class distribution in the dataset."""
        valence_labels = []
        arousal_labels = []
        
        for _, valence, arousal in dataset:
            valence_labels.append(valence.item())
            arousal_labels.append(arousal.item())
            
        print("\nDataset Class Distribution:")
        print("==========================")
        print("\nValence Distribution:")
        v_dist = Counter(valence_labels)
        for label, count in sorted(v_dist.items()):
            percentage = (count / len(valence_labels)) * 100
            print(f"Class {label}: {count} samples ({percentage:.2f}%)")
            
        print("\nArousal Distribution:")
        a_dist = Counter(arousal_labels)
        for label, count in sorted(a_dist.items()):
            percentage = (count / len(arousal_labels)) * 100
            print(f"Class {label}: {count} samples ({percentage:.2f}%)")
            
        return v_dist, a_dist

    def create_splits(self, dataset):
        """Create stratified splits considering both labels."""
        # First analyze distribution
        v_dist, a_dist = self.analyze_class_distribution(dataset)
        
        # Create combined labels for stratification
        combined_labels = []
        for _, valence, arousal in dataset:
            # Create a combined label that preserves both distributions
            combined_label = valence.item() * 3 + arousal.item()  # Assuming 3 classes each
            combined_labels.append(combined_label)
        
        combined_labels = np.array(combined_labels)
        
        # Create stratified splits
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        
        self.splits = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), combined_labels)):
            # Analyze fold distribution
            train_v_dist = Counter([dataset[i][1].item() for i in train_idx])
            train_a_dist = Counter([dataset[i][2].item() for i in val_idx])
            val_v_dist = Counter([dataset[i][1].item() for i in val_idx])
            val_a_dist = Counter([dataset[i][2].item() for i in val_idx])
            """
            print(f"\nFold {fold} Distribution:")
            print("=====================")
            print("\nTraining Set:")
            print("Valence:", {k: f"{v} ({v/len(train_idx)*100:.2f}%)" for k, v in train_v_dist.items()})
            print("Arousal:", {k: f"{v} ({v/len(train_idx)*100:.2f}%)" for k, v in train_a_dist.items()})
            print("\nValidation Set:")
            print("Valence:", {k: f"{v} ({v/len(val_idx)*100:.2f}%)" for k, v in val_v_dist.items()})
            print("Arousal:", {k: f"{v} ({v/len(val_idx)*100:.2f}%)" for k, v in val_a_dist.items()})
            """
            self.splits.append({
                'train_indices': train_idx,
                'val_indices': val_idx,
                'train_stats': {
                    'valence': train_v_dist,
                    'arousal': train_a_dist
                },
                'val_stats': {
                    'valence': val_v_dist,
                    'arousal': val_a_dist
                }
            })
        
        # Check for problematic splits
        min_samples_per_class = 3  # Set minimum acceptable number of samples
        for fold, split in enumerate(self.splits):
            for label_type in ['valence', 'arousal']:
                for class_idx in range(3):  # Assuming 3 classes
                    train_count = split['train_stats'][label_type].get(class_idx, 0)
                    val_count = split['val_stats'][label_type].get(class_idx, 0)
                    
                    if train_count < min_samples_per_class or val_count < min_samples_per_class:
                        print(f"\nWARNING: Fold {fold} has low representation for {label_type} class {class_idx}")
                        print(f"Training samples: {train_count}")
                        print(f"Validation samples: {val_count}")
        
        return self.splits

    def _save_splits(self):
        """Save split information to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.results_dir / f"splits_{timestamp}.json"
        
        splits_data = []
        for split in self.splits:
            split_dict = {
                'fold_number': split.fold_number,
                'train_indices': split.train_indices.tolist(),
                'val_indices': split.val_indices.tolist(),
                'train_valence_dist': split.train_valence_dist,
                'train_arousal_dist': split.train_arousal_dist,
                'val_valence_dist': split.val_valence_dist,
                'val_arousal_dist': split.val_arousal_dist
            }
            splits_data.append(split_dict)
        
        with open(save_path, 'w') as f:
            json.dump({
                'n_splits': self.n_splits,
                'shuffle': self.shuffle,
                'random_state': self.random_state,
                'splits': splits_data
            }, f, indent=2)
        
        self.logger.info(f"Saved split information to {save_path}")

    def get_fold_dataloaders(
        self,
        dataset: torch.utils.data.Dataset,
        fold: int,
        batch_size: int,
        num_workers: int = 2
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Get train and validation dataloaders for a specific fold.
        
        Args:
            dataset: The full dataset
            fold: Fold number
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
            
        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        if not self.splits:
            raise ValueError("No splits found. Run create_splits first.")
        
        if fold >= len(self.splits):
            raise ValueError(f"Fold {fold} not found. Only {len(self.splits)} folds available.")
        
        split_info = self.splits[fold]
        
        train_dataset = Subset(dataset, self.splits[fold]["train_indices"])
        val_dataset = Subset(dataset, self.splits[fold]["val_indices"])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader

# Example usage
if __name__ == "__main__":
    # Assuming we have our dataset
    dataset = PreprocessedGazePupilDataset("preprocessed_data")
    
    # Initialize cross-validator
    cv = CrossValidator(
        n_splits=5,
        results_dir="cv_results"
    )
    
    # Create splits
    splits = cv.create_splits(dataset)
    
    # Get dataloaders for first fold
    train_loader, val_loader = cv.get_fold_dataloaders(
        dataset,
        fold=0,
        batch_size=32
    )