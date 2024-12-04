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
    labeling_mode: str = 'dual'
    
    # Dual-head label distributions
    train_valence_dist: Optional[Dict[int, int]] = None
    train_arousal_dist: Optional[Dict[int, int]] = None
    val_valence_dist: Optional[Dict[int, int]] = None
    val_arousal_dist: Optional[Dict[int, int]] = None
    
    # State label distributions
    train_state_dist: Optional[Dict[int, int]] = None
    val_state_dist: Optional[Dict[int, int]] = None

class CrossValidator:
    def __init__(
        self, 
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        results_dir: Optional[str] = None,
        labeling_mode: str = 'dual'
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.results_dir = Path(results_dir) if results_dir else Path("cv_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.splits = None
        self.labeling_mode = labeling_mode
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
        """Analyze class distribution based on labeling mode."""
        if self.labeling_mode == 'dual':
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
        else:
            state_labels = []
            state_names = ['HANV', 'MAMV', 'LAPV', 'LANV']
            
            for _, state in dataset:
                state_labels.append(state.item())
            
            print("\nDataset State Distribution:")
            print("==========================")
            s_dist = Counter(state_labels)
            for label, count in sorted(s_dist.items()):
                percentage = (count / len(state_labels)) * 100
                print(f"{state_names[label]}: {count} samples ({percentage:.2f}%)")
            
            return s_dist

    def create_splits(self, dataset):
        """Create stratified splits considering labeling mode."""
        # First analyze distribution
        if self.labeling_mode == 'dual':
            v_dist, a_dist = self.analyze_class_distribution(dataset)
            
            # Create combined labels for stratification
            combined_labels = []
            for _, valence, arousal in dataset:
                # Create a combined label that preserves both distributions
                combined_label = valence.item() * 3 + arousal.item()
                combined_labels.append(combined_label)
        else:
            s_dist = self.analyze_class_distribution(dataset)
            
            # Get state labels directly
            combined_labels = []
            for _, state in dataset:
                combined_labels.append(state.item())
        
        combined_labels = np.array(combined_labels)
        
        # Create stratified splits
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        
        self.splits = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), combined_labels)):
            if self.labeling_mode == 'dual':
                # Analyze fold distribution for dual-head
                train_v_dist = Counter([dataset[i][1].item() for i in train_idx])
                train_a_dist = Counter([dataset[i][2].item() for i in train_idx])
                val_v_dist = Counter([dataset[i][1].item() for i in val_idx])
                val_a_dist = Counter([dataset[i][2].item() for i in val_idx])
                
                split_info = {
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
                }
            else:
                # Analyze fold distribution for state classification
                train_s_dist = Counter([dataset[i][1].item() for i in train_idx])
                val_s_dist = Counter([dataset[i][1].item() for i in val_idx])
                
                split_info = {
                    'train_indices': train_idx,
                    'val_indices': val_idx,
                    'train_stats': {
                        'state': train_s_dist
                    },
                    'val_stats': {
                        'state': val_s_dist
                    }
                }
            
            self.splits.append(split_info)
            
            # Check for problematic splits
            self._check_split_distribution(split_info, fold)
        
        return self.splits
    
    def _check_split_distribution(self, split_info: dict, fold: int):
        """Check for problematic class distributions in splits."""
        min_samples_per_class = 3
        
        if self.labeling_mode == 'dual':
            for label_type in ['valence', 'arousal']:
                for class_idx in range(3):
                    train_count = split_info['train_stats'][label_type].get(class_idx, 0)
                    val_count = split_info['val_stats'][label_type].get(class_idx, 0)
                    
                    if train_count < min_samples_per_class or val_count < min_samples_per_class:
                        self.logger.warning(
                            f"Fold {fold} has low representation for {label_type} "
                            f"class {class_idx} (train: {train_count}, val: {val_count})"
                        )
        else:
            state_names = ['HANV', 'MAMV', 'LAPV', 'LANV']
            for class_idx in range(4):
                train_count = split_info['train_stats']['state'].get(class_idx, 0)
                val_count = split_info['val_stats']['state'].get(class_idx, 0)
                
                if train_count < min_samples_per_class or val_count < min_samples_per_class:
                    self.logger.warning(
                        f"Fold {fold} has low representation for state "
                        f"{state_names[class_idx]} (train: {train_count}, val: {val_count})"
                    )
                        
    def _save_splits(self):
        """Save split information to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.results_dir / f"splits_{timestamp}.json"
        
        splits_data = []
        for split in self.splits:
            split_dict = {
                'train_indices': split['train_indices'].tolist(),
                'val_indices': split['val_indices'].tolist(),
                'labeling_mode': self.labeling_mode,
            }
            
            # Add mode-specific distribution information
            if self.labeling_mode == 'dual':
                split_dict.update({
                    'train_stats': {
                        'valence': dict(split['train_stats']['valence']),
                        'arousal': dict(split['train_stats']['arousal'])
                    },
                    'val_stats': {
                        'valence': dict(split['val_stats']['valence']),
                        'arousal': dict(split['val_stats']['arousal'])
                    }
                })
            else:
                state_names = ['HANV', 'MAMV', 'LAPV', 'LANV']
                split_dict.update({
                    'train_stats': {
                        'state': {state_names[k]: v for k, v in split['train_stats']['state'].items()}
                    },
                    'val_stats': {
                        'state': {state_names[k]: v for k, v in split['val_stats']['state'].items()}
                    }
                })
            
            splits_data.append(split_dict)
        
        with open(save_path, 'w') as f:
            json.dump({
                'n_splits': self.n_splits,
                'shuffle': self.shuffle,
                'random_state': self.random_state,
                'labeling_mode': self.labeling_mode,
                'splits': splits_data
            }, f, indent=2)
        
        self.logger.info(f"Saved split information to {save_path}")
        return save_path

    def get_fold_dataloaders(
        self,
        dataset: torch.utils.data.Dataset,
        fold: int,
        batch_size: int,
        num_workers: int = 2
    ) -> Tuple[DataLoader, DataLoader]:
        """Get train and validation dataloaders for a specific fold."""
        if not self.splits:
            raise ValueError("No splits found. Run create_splits first.")
        
        if fold >= len(self.splits):
            raise ValueError(f"Fold {fold} not found. Only {len(self.splits)} folds available.")
        
        split_info = self.splits[fold]
        
        train_dataset = Subset(dataset, split_info['train_indices'])
        val_dataset = Subset(dataset, split_info['val_indices'])
        
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
        
        # Log distribution information
        self._log_fold_distribution(split_info, fold)
        
        return train_loader, val_loader

    def _log_fold_distribution(self, split_info: dict, fold: int):
        """Log class distribution information for the fold."""
        if self.labeling_mode == 'dual':
            self.logger.info(f"\nFold {fold} Distribution:")
            self.logger.info("Training Set:")
            self.logger.info(f"Valence: {dict(split_info['train_stats']['valence'])}")
            self.logger.info(f"Arousal: {dict(split_info['train_stats']['arousal'])}")
            self.logger.info("Validation Set:")
            self.logger.info(f"Valence: {dict(split_info['val_stats']['valence'])}")
            self.logger.info(f"Arousal: {dict(split_info['val_stats']['arousal'])}")
        else:
            state_names = ['HANV', 'MAMV', 'LAPV', 'LANV']
            self.logger.info(f"\nFold {fold} State Distribution:")
            self.logger.info("Training Set:")
            for class_idx, count in split_info['train_stats']['state'].items():
                self.logger.info(f"{state_names[class_idx]}: {count}")
            self.logger.info("Validation Set:")
            for class_idx, count in split_info['val_stats']['state'].items():
                self.logger.info(f"{state_names[class_idx]}: {count}")
                
    def load_splits(self, splits_file: str):
        """Load previously saved splits."""
        with open(splits_file, 'r') as f:
            data = json.load(f)
        
        # Verify compatibility
        if data['labeling_mode'] != self.labeling_mode:
            raise ValueError(
                f"Incompatible labeling mode in saved splits "
                f"(saved: {data['labeling_mode']}, current: {self.labeling_mode})"
            )
        
        # Convert lists back to numpy arrays
        self.splits = []
        for split in data['splits']:
            split['train_indices'] = np.array(split['train_indices'])
            split['val_indices'] = np.array(split['val_indices'])
            self.splits.append(split)
        
        self.n_splits = data['n_splits']
        self.shuffle = data['shuffle']
        self.random_state = data['random_state']
        
        self.logger.info(f"Loaded {len(self.splits)} splits from {splits_file}")
        return self.splits
    
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