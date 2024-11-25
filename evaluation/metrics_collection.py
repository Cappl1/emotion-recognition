from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass, field
import torch
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from collections import defaultdict

@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    valence_train_acc: float
    valence_val_acc: float
    arousal_train_acc: float
    arousal_val_acc: float
    valence_val_confusion: np.ndarray
    arousal_val_confusion: np.ndarray
    learning_rate: float

@dataclass
class FoldMetrics:
    """Collection of metrics for a single fold."""
    fold: int
    epochs: List[EpochMetrics] = field(default_factory=list)
    best_valence_acc: float = 0.0
    best_arousal_acc: float = 0.0
    best_epoch: int = 0
    train_time: float = 0.0
    
    # Per-class metrics
    valence_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    arousal_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

class MetricsCollector:
    def __init__(
        self,
        save_dir: str,
        experiment_name: Optional[str] = None,
        num_classes: int = 3
    ):
        """
        Initialize metrics collector.
        
        Args:
            save_dir: Directory to save metrics
            experiment_name: Name of the experiment
            num_classes: Number of classes for the task
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.save_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_classes = num_classes
        self.folds = []
        self.current_fold = None
        
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger(f"MetricsCollector_{self.experiment_name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            fh = logging.FileHandler(self.experiment_dir / "metrics.log")
            fh.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            
            logger.addHandler(fh)
        
        return logger

    def start_fold(self, fold: int):
        """Start collecting metrics for a new fold."""
        self.current_fold = FoldMetrics(fold=fold)
        self.folds.append(self.current_fold)
        self.logger.info(f"Started collecting metrics for fold {fold}")

    def update_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        valence_train_acc: float,
        valence_val_acc: float,
        arousal_train_acc: float,
        arousal_val_acc: float,
        valence_val_preds: np.ndarray,  # Changed from torch.Tensor to np.ndarray
        valence_val_true: np.ndarray,   # Changed from torch.Tensor to np.ndarray
        arousal_val_preds: np.ndarray,  # Changed from torch.Tensor to np.ndarray
        arousal_val_true: np.ndarray,   # Changed from torch.Tensor to np.ndarray
        learning_rate: float
    ):
        """Update metrics for current epoch."""
        if self.current_fold is None:
            raise ValueError("No active fold. Call start_fold first.")
        
        # Print incoming metrics for debugging
        """
        print(f"\nUpdating metrics for fold {self.current_fold.fold}, epoch {epoch}")
        print(f"Incoming metrics:")
        print(f"Valence val acc: {valence_val_acc:.4f}")
        print(f"Arousal val acc: {arousal_val_acc:.4f}")
        """
        # Calculate confusion matrices
        try:
            valence_conf = confusion_matrix(
                valence_val_true,
                valence_val_preds,
                labels=range(self.num_classes)
            )
            arousal_conf = confusion_matrix(
                arousal_val_true,
                arousal_val_preds,
                labels=range(self.num_classes)
            )
        except Exception as e:
            print(f"Error calculating confusion matrices: {e}")
            print(f"Valence shapes - Pred: {valence_val_preds.shape}, True: {valence_val_true.shape}")
            print(f"Arousal shapes - Pred: {arousal_val_preds.shape}, True: {arousal_val_true.shape}")
            # Create empty confusion matrices if calculation fails
            valence_conf = np.zeros((self.num_classes, self.num_classes))
            arousal_conf = np.zeros((self.num_classes, self.num_classes))
        
        # Create epoch metrics
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            valence_train_acc=float(valence_train_acc),
            valence_val_acc=float(valence_val_acc),
            arousal_train_acc=float(arousal_train_acc),
            arousal_val_acc=float(arousal_val_acc),
            valence_val_confusion=valence_conf,
            arousal_val_confusion=arousal_conf,
            learning_rate=float(learning_rate)
        )
        
        # Store epoch metrics
        self.current_fold.epochs.append(epoch_metrics)
        
        # Update best metrics with proper type conversion
        current_valence_acc = float(valence_val_acc)
        current_arousal_acc = float(arousal_val_acc)
        
        # Update best metrics if current is better
        if current_valence_acc > self.current_fold.best_valence_acc:
            self.current_fold.best_valence_acc = current_valence_acc
            self.current_fold.best_epoch = epoch
            print(f"New best valence accuracy: {current_valence_acc:.4f}")
        
        if current_arousal_acc > self.current_fold.best_arousal_acc:
            self.current_fold.best_arousal_acc = current_arousal_acc
            print(f"New best arousal accuracy: {current_arousal_acc:.4f}")
        
        # Print current state
        print(f"Current state:")
        print(f"Best valence accuracy: {self.current_fold.best_valence_acc:.4f}")
        print(f"Best arousal accuracy: {self.current_fold.best_arousal_acc:.4f}")
        print(f"Best epoch: {self.current_fold.best_epoch}")
        
        # Print confusion matrices for debugging
        print("\nValence Confusion Matrix:")
        print(valence_conf)
        print("\nArousal Confusion Matrix:")
        print(arousal_conf)
        
        # Log metrics
        self.logger.info(
            f"Fold {self.current_fold.fold}, Epoch {epoch}: "
            f"Val Loss: {val_loss:.4f}, "
            f"Valence Acc: {current_valence_acc:.4f} (Best: {self.current_fold.best_valence_acc:.4f}), "
            f"Arousal Acc: {current_arousal_acc:.4f} (Best: {self.current_fold.best_arousal_acc:.4f})"
        )

    def end_fold(
        self,
        valence_val_preds: torch.Tensor,
        valence_val_true: torch.Tensor,
        arousal_val_preds: torch.Tensor,
        arousal_val_true: torch.Tensor,
        train_time: float
    ):
        """
        Finish collecting metrics for current fold.
        Calculate final per-class metrics.
        """
        if self.current_fold is None:
            raise ValueError("No active fold.")
            
        # Calculate per-class metrics with zero_division parameter
        self.current_fold.valence_class_metrics = classification_report(
            valence_val_true,
            valence_val_preds,
            output_dict=True,
            zero_division=0  # Set to 0 when there are no predictions
        )
        
        self.current_fold.arousal_class_metrics = classification_report(
            arousal_val_true,
            arousal_val_preds,
            output_dict=True,
            zero_division=0  # Set to 0 when there are no predictions
        )
        
        self.current_fold.train_time = train_time
        
        # Calculate additional metrics with zero handling
        self._calculate_additional_metrics(
            valence_val_preds,
            valence_val_true,
            arousal_val_preds,
            arousal_val_true
        )
        
        # Save fold metrics
        self._save_fold_metrics(self.current_fold)
        self.current_fold = None

    def _calculate_additional_metrics(
        self, 
        valence_preds: np.ndarray,
        valence_true: np.ndarray,
        arousal_preds: np.ndarray,
        arousal_true: np.ndarray
    ):
        """Calculate additional metrics with proper zero handling."""
        if self.current_fold is None:
            return

        # For each class
        for task, (preds, true) in [
            ('valence', (valence_preds, valence_true)),
            ('arousal', (arousal_preds, arousal_true))
        ]:
            class_metrics = {}
            for class_idx in range(self.num_classes):
                # Convert to binary classification problem for each class
                binary_true = (true == class_idx)
                binary_pred = (preds == class_idx)
                
                if np.sum(binary_true) == 0:
                    # No samples of this class in ground truth
                    metrics = {
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1-score': 0.0,
                        'support': 0
                    }
                else:
                    # Calculate metrics with zero_division parameter
                    precision = precision_score(
                        binary_true, 
                        binary_pred, 
                        zero_division=0
                    )
                    recall = recall_score(
                        binary_true, 
                        binary_pred, 
                        zero_division=0
                    )
                    f1 = f1_score(
                        binary_true, 
                        binary_pred, 
                        zero_division=0
                    )
                    support = np.sum(binary_true)
                    
                    metrics = {
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1-score': float(f1),
                        'support': int(support)
                    }
                
                class_metrics[str(class_idx)] = metrics
            
            # Add macro and weighted averages
            for average in ['macro', 'weighted']:
                precision = precision_score(
                    true, 
                    preds, 
                    average=average, 
                    zero_division=0
                )
                recall = recall_score(
                    true, 
                    preds, 
                    average=average, 
                    zero_division=0
                )
                f1 = f1_score(
                    true, 
                    preds, 
                    average=average, 
                    zero_division=0
                )
                
                class_metrics[f'{average} avg'] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1-score': float(f1),
                    'support': int(np.sum(true != -1))
                }
            
            # Store in fold metrics
            setattr(self.current_fold, f'{task}_class_metrics', class_metrics)

    def _save_fold_metrics(self, fold_metrics: FoldMetrics):
        """Save metrics for a single fold."""
        fold_dir = self.experiment_dir / f"fold_{fold_metrics.fold}"
        print(f"saved at {fold_dir}")
        fold_dir.mkdir(exist_ok=True)
        
        # Save metrics as JSON with explicit float conversion
        metrics_dict = {
            'fold': fold_metrics.fold,
            'best_valence_acc': float(fold_metrics.best_valence_acc),
            'best_arousal_acc': float(fold_metrics.best_arousal_acc),
            'best_epoch': fold_metrics.best_epoch,
            'train_time': float(fold_metrics.train_time),
            'valence_class_metrics': fold_metrics.valence_class_metrics,
            'arousal_class_metrics': fold_metrics.arousal_class_metrics,
            'epochs': [
                {
                    'epoch': e.epoch,
                    'train_loss': float(e.train_loss),
                    'val_loss': float(e.val_loss),
                    'valence_train_acc': float(e.valence_train_acc),
                    'valence_val_acc': float(e.valence_val_acc),
                    'arousal_train_acc': float(e.arousal_train_acc),
                    'arousal_val_acc': float(e.arousal_val_acc),
                    'learning_rate': float(e.learning_rate),
                    'valence_val_confusion': e.valence_val_confusion.tolist(),
                    'arousal_val_confusion': e.arousal_val_confusion.tolist()
                }
                for e in fold_metrics.epochs
            ]
        }
        
        # Save individual fold metrics
        with open(fold_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        # Save final summary after each fold
        self._save_final_summary()
        
        # Plot and save confusion matrices
        self._plot_confusion_matrices(fold_metrics, fold_dir)
        
        # Plot and save learning curves
        self._plot_learning_curves(fold_metrics, fold_dir)

    def _save_final_summary(self):
        """Save final summary of all folds."""
        summary = {
            'raw_data': {
                'folds': [
                    {
                        'fold': fold.fold,
                        'best_valence_acc': float(fold.best_valence_acc),
                        'best_arousal_acc': float(fold.best_arousal_acc),
                        'best_epoch': fold.best_epoch,
                        'train_time': float(fold.train_time),
                        'n_epochs': len(fold.epochs)
                    }
                    for fold in self.folds
                ],
            },
            'summary': {
                'valence_accuracy': {
                    'values': [float(fold.best_valence_acc) for fold in self.folds],
                    'mean': float(np.mean([fold.best_valence_acc for fold in self.folds])),
                    'std': float(np.std([fold.best_valence_acc for fold in self.folds]))
                },
                'arousal_accuracy': {
                    'values': [float(fold.best_arousal_acc) for fold in self.folds],
                    'mean': float(np.mean([fold.best_arousal_acc for fold in self.folds])),
                    'std': float(np.std([fold.best_arousal_acc for fold in self.folds]))
                }
            }
        }
        
        with open(self.experiment_dir / 'final_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    def _plot_confusion_matrices(self, fold_metrics: FoldMetrics, save_dir: Path):
        """Plot and save confusion matrices for the fold."""
        # Get final epoch confusion matrices
        final_epoch = fold_metrics.epochs[-1]
        
        # Valence confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            final_epoch.valence_val_confusion,
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        plt.title(f'Valence Confusion Matrix - Fold {fold_metrics.fold}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(save_dir / 'valence_confusion.png')
        plt.close()
        
        # Arousal confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            final_epoch.arousal_val_confusion,
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        plt.title(f'Arousal Confusion Matrix - Fold {fold_metrics.fold}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(save_dir / 'arousal_confusion.png')
        plt.close()

    def _plot_learning_curves(self, fold_metrics: FoldMetrics, save_dir: Path):
        """Plot and save learning curves for the fold."""
        epochs = range(len(fold_metrics.epochs))
        metrics = fold_metrics.epochs
        
        # Loss curves
        plt.figure(figsize=(10, 6))
        plt.plot([m.train_loss for m in metrics], label='Train Loss')
        plt.plot([m.val_loss for m in metrics], label='Validation Loss')
        plt.title(f'Loss Curves - Fold {fold_metrics.fold}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(save_dir / 'loss_curves.png')
        plt.close()
        
        # Accuracy curves
        plt.figure(figsize=(10, 6))
        plt.plot([m.valence_train_acc for m in metrics], label='Valence Train')
        plt.plot([m.valence_val_acc for m in metrics], label='Valence Val')
        plt.plot([m.arousal_train_acc for m in metrics], label='Arousal Train')
        plt.plot([m.arousal_val_acc for m in metrics], label='Arousal Val')
        plt.title(f'Accuracy Curves - Fold {fold_metrics.fold}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(save_dir / 'accuracy_curves.png')
        plt.close()

    def summarize_results(self) -> Dict:
        """Generate summary statistics across all folds."""
        
        
        # Collect accuracies directly from folds
        valence_accs = []
        arousal_accs = []
        
        for fold in self.folds:
            """
            print(f"\nFold {fold.fold}:")
            print(f"Best valence accuracy: {fold.best_valence_acc}")
            print(f"Best arousal accuracy: {fold.best_arousal_acc}")
            print(f"Number of epochs: {len(fold.epochs)}")
            """
            # Only add if we have valid accuracies (> 0)
            if fold.best_valence_acc > 0:
                valence_accs.append(fold.best_valence_acc)
            if fold.best_arousal_acc > 0:
                arousal_accs.append(fold.best_arousal_acc)
        
        # Print what we collected
        """
        print("\nCOLLECTED ACCURACIES:")
        print(f"Valence accuracies: {valence_accs}")
        print(f"Arousal accuracies: {arousal_accs}")
        """
        # Calculate means and stds directly
        valence_mean = np.mean(valence_accs) if valence_accs else 0.0
        valence_std = np.std(valence_accs) if len(valence_accs) > 1 else 0.0
        arousal_mean = np.mean(arousal_accs) if arousal_accs else 0.0
        arousal_std = np.std(arousal_accs) if len(arousal_accs) > 1 else 0.0
        
        # Print calculations
        """
        print("\nCALCULATIONS:")
        print(f"Valence mean: {valence_mean:.4f}")
        print(f"Valence std: {valence_std:.4f}")
        print(f"Arousal mean: {arousal_mean:.4f}")
        print(f"Arousal std: {arousal_std:.4f}")
        """
        summary = {
            'valence_accuracy': {
                'mean': float(valence_mean),
                'std': float(valence_std),
                'per_fold': valence_accs
            },
            'arousal_accuracy': {
                'mean': float(arousal_mean),
                'std': float(arousal_std),
                'per_fold': arousal_accs
            }
        }
        
        # Save everything
        with open(self.experiment_dir / 'final_summary.json', 'w') as f:
            json.dump({
                'summary': summary,
                'raw_data': {
                    'folds': [{
                        'fold': fold.fold,
                        'best_valence': fold.best_valence_acc,
                        'best_arousal': fold.best_arousal_acc,
                        'n_epochs': len(fold.epochs)
                    } for fold in self.folds],
                    'collected_accuracies': {
                        'valence': valence_accs,
                        'arousal': arousal_accs
                    }
                }
            }, f, indent=2)
        
        return summary

    
    def print_final_metrics(self):
        """Print final metrics before summary."""
        print("\nFINAL METRICS CHECK:")
        print("=" * 50)
        for fold in self.folds:
            print(f"\nFold {fold.fold}:")
            print(f"Best valence: {fold.best_valence_acc:.4f}")
            print(f"Best arousal: {fold.best_arousal_acc:.4f}")
            if fold.epochs:
                print(f"Last epoch metrics:")
                last_epoch = fold.epochs[-1]
                print(f"  Valence acc: {last_epoch.valence_val_acc:.4f}")
                print(f"  Arousal acc: {last_epoch.arousal_val_acc:.4f}")

    