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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px 

@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    labeling_mode: str = 'dual'
    
    # Dual-head metrics
    valence_train_acc: float = 0.0
    valence_val_acc: float = 0.0
    arousal_train_acc: float = 0.0
    arousal_val_acc: float = 0.0
    valence_val_confusion: Optional[np.ndarray] = None
    arousal_val_confusion: Optional[np.ndarray] = None
    
    # Single-head state metrics
    state_train_acc: float = 0.0
    state_val_acc: float = 0.0
    state_val_confusion: Optional[np.ndarray] = None
    
    learning_rate: float = 0.0

@dataclass
class FoldMetrics:
    """Collection of metrics for a single fold."""
    fold: int
    labeling_mode: str = 'dual'
    epochs: List[EpochMetrics] = field(default_factory=list)
    
    # Dual-head metrics
    best_valence_acc: float = 0.0
    best_arousal_acc: float = 0.0
    
    # Single-head state metrics
    best_state_acc: float = 0.0
    
    best_epoch: int = 0
    train_time: float = 0.0
    
    # Per-class metrics
    valence_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    arousal_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    state_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

class MetricsCollector:
    def __init__(
        self,
        save_dir: str,
        experiment_name: Optional[str] = None,
        labeling_mode: str = 'dual'
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.labeling_mode = labeling_mode
        # Set num_classes based on labeling mode
        self.num_classes = 4 if labeling_mode == 'state' else 3
        
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.save_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
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
        learning_rate: float,
        **kwargs
    ):
        """Update metrics for current epoch."""
        if self.current_fold is None:
            raise ValueError("No active fold. Call start_fold first.")
        
        if self.labeling_mode == 'dual':
            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                valence_train_acc=kwargs.get('valence_train_acc', 0.0),
                valence_val_acc=kwargs.get('valence_val_acc', 0.0),
                arousal_train_acc=kwargs.get('arousal_train_acc', 0.0),
                arousal_val_acc=kwargs.get('arousal_val_acc', 0.0),
                valence_val_confusion=kwargs.get('valence_val_confusion'),  # No default
                arousal_val_confusion=kwargs.get('arousal_val_confusion'),  # No default
                learning_rate=learning_rate,
                labeling_mode='dual'
            )
            
            # Update best metrics
            if epoch_metrics.valence_val_acc > self.current_fold.best_valence_acc:
                self.current_fold.best_valence_acc = epoch_metrics.valence_val_acc
                self.current_fold.best_epoch = epoch
            
            if epoch_metrics.arousal_val_acc > self.current_fold.best_arousal_acc:
                self.current_fold.best_arousal_acc = epoch_metrics.arousal_val_acc
        
        else:  # state classification
            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                state_train_acc=kwargs.get('state_train_acc', 0.0),
                state_val_acc=kwargs.get('state_val_acc', 0.0),
                state_val_confusion=kwargs.get('state_val_confusion'),  # No default
                learning_rate=learning_rate,
                labeling_mode='state'
            )
            
            # Update best metrics
            if epoch_metrics.state_val_acc > self.current_fold.best_state_acc:
                self.current_fold.best_state_acc = epoch_metrics.state_val_acc
                self.current_fold.best_epoch = epoch
        
        self.current_fold.epochs.append(epoch_metrics)

    def end_fold(
        self,
        preds: Optional[np.ndarray] = None,
        true: Optional[np.ndarray] = None,
        valence_preds: Optional[np.ndarray] = None,
        valence_true: Optional[np.ndarray] = None,
        arousal_preds: Optional[np.ndarray] = None,
        arousal_true: Optional[np.ndarray] = None,
        train_time: float = 0.0
    ):
        """Finish collecting metrics for current fold."""
        if self.current_fold is None:
            raise ValueError("No active fold.")
        
        # Ensure train_time is set
        self.current_fold.train_time = float(train_time)
        
        if self.labeling_mode == 'dual':
            self._calculate_additional_metrics(
                None, None, train_time,
                valence_preds, valence_true,
                arousal_preds, arousal_true
            )
        else:
            self._calculate_additional_metrics(
                preds, true, train_time
            )
        
        # Save the metrics
        self._save_fold_metrics(self.current_fold)
        self.current_fold = None


    def _calculate_additional_metrics(
        self,
        preds: Optional[np.ndarray] = None,
        true: Optional[np.ndarray] = None,
        train_time: float = 0.0,
        valence_preds: Optional[np.ndarray] = None,
        valence_true: Optional[np.ndarray] = None,
        arousal_preds: Optional[np.ndarray] = None,
        arousal_true: Optional[np.ndarray] = None
    ):
        """Calculate additional metrics with proper zero handling."""
        if self.current_fold is None:
            return

        if self.labeling_mode == 'dual':
            # ... dual mode code remains the same ...
            pass
        else:
            # State classification metrics calculation
            if preds is not None and true is not None:
                class_metrics = {
                    'precision': [],
                    'recall': [],
                    'f1-score': [],
                    'support': []
                }
                
                # Per-class metrics
                for class_idx in range(self.num_classes):
                    binary_true = (true == class_idx)
                    binary_pred = (preds == class_idx)
                    
                    if np.sum(binary_true) == 0:
                        class_metrics['precision'].append(0.0)
                        class_metrics['recall'].append(0.0)
                        class_metrics['f1-score'].append(0.0)
                        class_metrics['support'].append(0)
                    else:
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
                        
                        class_metrics['precision'].append(float(precision))
                        class_metrics['recall'].append(float(recall))
                        class_metrics['f1-score'].append(float(f1))
                        class_metrics['support'].append(int(support))
                
                # Calculate macro averages
                class_metrics['macro_avg'] = {
                    'precision': float(np.mean([p for p in class_metrics['precision'] if p > 0])),
                    'recall': float(np.mean([r for r in class_metrics['recall'] if r > 0])),
                    'f1-score': float(np.mean([f for f in class_metrics['f1-score'] if f > 0])),
                    'support': int(np.sum(class_metrics['support']))
                }
                
                # Store metrics
                self.current_fold.state_class_metrics = class_metrics
            
        # Store training time
        self.current_fold.train_time = train_time

    def _save_fold_metrics(self, fold_metrics: FoldMetrics):
        """Save metrics for a single fold."""
        fold_dir = self.experiment_dir / f"fold_{fold_metrics.fold}"
        print(f"saved at {fold_dir}")
        fold_dir.mkdir(exist_ok=True)
        
        # Prepare metrics dictionary with safe type conversion
        metrics_dict = {
            'fold': fold_metrics.fold,
            'train_time': float(fold_metrics.train_time) if fold_metrics.train_time is not None else 0.0,
        }
        
        if self.labeling_mode == 'dual':
            metrics_dict.update({
                'best_valence_acc': float(fold_metrics.best_valence_acc),
                'best_arousal_acc': float(fold_metrics.best_arousal_acc),
                'best_epoch': fold_metrics.best_epoch,
                'valence_class_metrics': fold_metrics.valence_class_metrics,
                'arousal_class_metrics': fold_metrics.arousal_class_metrics,
            })
        else:
            metrics_dict.update({
                'best_state_acc': float(fold_metrics.best_state_acc) if hasattr(fold_metrics, 'best_state_acc') else 0.0,
                'best_epoch': fold_metrics.best_epoch,
                'state_class_metrics': fold_metrics.state_class_metrics if hasattr(fold_metrics, 'state_class_metrics') else {},
            })
        
        # Add epochs data with safe type conversion
        metrics_dict['epochs'] = []
        for e in fold_metrics.epochs:
            epoch_dict = {
                'epoch': e.epoch,
                'train_loss': float(e.train_loss),
                'val_loss': float(e.val_loss),
                'learning_rate': float(e.learning_rate)
            }
            
            if self.labeling_mode == 'dual':
                epoch_dict.update({
                    'valence_train_acc': float(e.valence_train_acc),
                    'valence_val_acc': float(e.valence_val_acc),
                    'arousal_train_acc': float(e.arousal_train_acc),
                    'arousal_val_acc': float(e.arousal_val_acc),
                    'valence_val_confusion': e.valence_val_confusion.tolist() if e.valence_val_confusion is not None else None,
                    'arousal_val_confusion': e.arousal_val_confusion.tolist() if e.arousal_val_confusion is not None else None
                })
            else:
                epoch_dict.update({
                    'state_train_acc': float(e.state_train_acc) if hasattr(e, 'state_train_acc') else 0.0,
                    'state_val_acc': float(e.state_val_acc) if hasattr(e, 'state_val_acc') else 0.0,
                    'state_val_confusion': e.state_val_confusion.tolist() if hasattr(e, 'state_val_confusion') and e.state_val_confusion is not None else None
                })
            
            metrics_dict['epochs'].append(epoch_dict)
        
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
                        'train_time': float(fold.train_time) if fold.train_time is not None else 0.0,
                    }
                    for fold in self.folds
                ],
            },
            'summary': {}
        }

        # Add mode-specific metrics
        if self.labeling_mode == 'dual':
            # Update each fold's data with dual-head metrics
            for idx, fold_data in enumerate(summary['raw_data']['folds']):
                fold_data.update({
                    'best_valence': float(self.folds[idx].best_valence_acc),
                    'best_arousal': float(self.folds[idx].best_arousal_acc),
                    'n_epochs': len(self.folds[idx].epochs)
                })
            
            # Calculate summary statistics for dual-head mode
            summary['summary'].update({
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
            })
        else:
            # Update each fold's data with state metrics
            for idx, fold_data in enumerate(summary['raw_data']['folds']):
                fold_data.update({
                    'best_state': float(self.folds[idx].best_state_acc),
                    'n_epochs': len(self.folds[idx].epochs)
                })
            
            # Calculate summary statistics for state mode
            summary['summary'].update({
                'state_accuracy': {
                    'values': [float(fold.best_state_acc) for fold in self.folds],
                    'mean': float(np.mean([fold.best_state_acc for fold in self.folds])),
                    'std': float(np.std([fold.best_state_acc for fold in self.folds]))
                }
            })

        # Add training time summary
        train_times = [float(fold.train_time) if fold.train_time is not None else 0.0 
                    for fold in self.folds]
        summary['summary']['training_time'] = {
            'total': float(np.sum(train_times)),
            'mean_per_fold': float(np.mean(train_times)),
            'std': float(np.std(train_times))
        }

        # Save the summary
        with open(self.experiment_dir / 'final_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        return summary


    def _plot_confusion_matrices(self, fold_metrics: FoldMetrics, save_dir: Path):
        """Plot and save confusion matrices for the fold."""
        try:
            if self.labeling_mode == 'dual':
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Check if we have valid confusion matrices
                if hasattr(fold_metrics.epochs[-1], 'valence_val_confusion') and \
                fold_metrics.epochs[-1].valence_val_confusion is not None and \
                fold_metrics.epochs[-1].valence_val_confusion.size > 0:
                    
                    sns.heatmap(
                        fold_metrics.epochs[-1].valence_val_confusion,
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        ax=ax1,
                        xticklabels=['NV', 'MV', 'PV'],
                        yticklabels=['NV', 'MV', 'PV']
                    )
                    ax1.set_title('Valence Confusion Matrix')
                    ax1.set_xlabel('Predicted')
                    ax1.set_ylabel('True')
                else:
                    self.logger.warning("No valid valence confusion matrix available")
                    ax1.text(0.5, 0.5, 'No Data Available', 
                            horizontalalignment='center',
                            verticalalignment='center')
                
                if hasattr(fold_metrics.epochs[-1], 'arousal_val_confusion') and \
                fold_metrics.epochs[-1].arousal_val_confusion is not None and \
                fold_metrics.epochs[-1].arousal_val_confusion.size > 0:
                    
                    sns.heatmap(
                        fold_metrics.epochs[-1].arousal_val_confusion,
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        ax=ax2,
                        xticklabels=['LA', 'MA', 'HA'],
                        yticklabels=['LA', 'MA', 'HA']
                    )
                    ax2.set_title('Arousal Confusion Matrix')
                    ax2.set_xlabel('Predicted')
                    ax2.set_ylabel('True')
                else:
                    self.logger.warning("No valid arousal confusion matrix available")
                    ax2.text(0.5, 0.5, 'No Data Available', 
                            horizontalalignment='center',
                            verticalalignment='center')
                
                plt.tight_layout()
                plt.savefig(save_dir / 'confusion_matrices.png')
                plt.close()
                
            else:
                # State classification confusion matrix
                plt.figure(figsize=(10, 8))
                
                if hasattr(fold_metrics.epochs[-1], 'state_val_confusion') and \
                fold_metrics.epochs[-1].state_val_confusion is not None and \
                fold_metrics.epochs[-1].state_val_confusion.size > 0:
                    
                    sns.heatmap(
                        fold_metrics.epochs[-1].state_val_confusion,
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        xticklabels=['HANV', 'MAMV', 'LAPV', 'LANV'],
                        yticklabels=['HANV', 'MAMV', 'LAPV', 'LANV']
                    )
                    plt.title('State Classification Confusion Matrix')
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                else:
                    self.logger.warning("No valid state confusion matrix available")
                    plt.text(0.5, 0.5, 'No Data Available', 
                            horizontalalignment='center',
                            verticalalignment='center')
                
                plt.tight_layout()
                plt.savefig(save_dir / 'confusion_matrix.png')
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrices: {str(e)}")
            self.logger.error("Confusion matrix data:")
            if self.labeling_mode == 'dual':
                if hasattr(fold_metrics.epochs[-1], 'valence_val_confusion'):
                    self.logger.error(f"Valence confusion matrix shape: {fold_metrics.epochs[-1].valence_val_confusion.shape if fold_metrics.epochs[-1].valence_val_confusion is not None else 'None'}")
                if hasattr(fold_metrics.epochs[-1], 'arousal_val_confusion'):
                    self.logger.error(f"Arousal confusion matrix shape: {fold_metrics.epochs[-1].arousal_val_confusion.shape if fold_metrics.epochs[-1].arousal_val_confusion is not None else 'None'}")
            else:
                if hasattr(fold_metrics.epochs[-1], 'state_val_confusion'):
                    self.logger.error(f"State confusion matrix shape: {fold_metrics.epochs[-1].state_val_confusion.shape if fold_metrics.epochs[-1].state_val_confusion is not None else 'None'}")

    def _plot_confusion_matrices_matplotlib(self):
        """Fallback matplotlib version for confusion matrix plotting."""
        if self.labeling_mode == 'dual':
            tasks = ['valence', 'arousal']
            plt.figure(figsize=(12, 5))
            
            for task_idx, task in enumerate(tasks, 1):
                cms = []
                for fold in self.folds:
                    if fold.epochs:
                        cms.append(fold.epochs[-1][f'{task}_val_confusion'])
                
                if cms:
                    avg_cm = np.mean(cms, axis=0)
                    avg_cm = np.nan_to_num(avg_cm, 0)
                    
                    plt.subplot(1, 2, task_idx)
                    sns.heatmap(
                        avg_cm,
                        annot=True,
                        fmt='.2f',
                        cmap='Blues',
                        xticklabels=['Class 0', 'Class 1', 'Class 2'],
                        yticklabels=['Class 0', 'Class 1', 'Class 2']
                    )
                    plt.title(f'{task.capitalize()} Confusion Matrix')
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
        else:
            plt.figure(figsize=(10, 8))
            cms = []
            for fold in self.folds:
                if fold.epochs and hasattr(fold.epochs[-1], 'state_val_confusion'):
                    conf_matrix = fold.epochs[-1].state_val_confusion
                    if conf_matrix is not None and conf_matrix.size > 0:
                        cms.append(conf_matrix)
            
            if cms:
                avg_cm = np.mean(cms, axis=0)
                avg_cm = np.nan_to_num(avg_cm, 0)
                
                sns.heatmap(
                    avg_cm,
                    annot=True,
                    fmt='.2f',
                    cmap='Blues',
                    xticklabels=['HANV', 'MAMV', 'LAPV', 'LANV'],
                    yticklabels=['HANV', 'MAMV', 'LAPV', 'LANV']
                )
                plt.title('State Classification Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('True')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "confusion_matrices.png")
        plt.close()

    def _plot_learning_curves(self, fold_metrics: FoldMetrics, save_dir: Path):
        """Plot and save learning curves for the fold."""
        epochs = range(len(fold_metrics.epochs))
        metrics = fold_metrics.epochs
        
        if self.labeling_mode == 'dual':
            # Create subplot with 2 rows
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Loss Curves',
                    'Accuracy Curves',
                    'Valence/Arousal Training Progress',
                    'Learning Rate'
                )
            )
            
            # Loss curves
            fig.add_trace(
                go.Scatter(name='Train Loss', 
                          y=[m.train_loss for m in metrics], 
                          mode='lines'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(name='Val Loss', 
                          y=[m.val_loss for m in metrics], 
                          mode='lines'),
                row=1, col=1
            )
            
            # Accuracy curves
            fig.add_trace(
                go.Scatter(name='Valence Train', 
                          y=[m.valence_train_acc for m in metrics],
                          mode='lines'),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(name='Valence Val', 
                          y=[m.valence_val_acc for m in metrics],
                          mode='lines'),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(name='Arousal Train', 
                          y=[m.arousal_train_acc for m in metrics],
                          mode='lines'),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(name='Arousal Val', 
                          y=[m.arousal_val_acc for m in metrics],
                          mode='lines'),
                row=1, col=2
            )
            
            # Combined progress plot
            fig.add_trace(
                go.Scatter(name='Valence Progress', 
                          y=[m.valence_val_acc for m in metrics],
                          mode='lines'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(name='Arousal Progress', 
                          y=[m.arousal_val_acc for m in metrics],
                          mode='lines'),
                row=2, col=1
            )
            
        else:
            # Create subplot with 2 rows for state classification
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Loss Curves',
                    'State Accuracy',
                    'State Progress',
                    'Learning Rate'
                )
            )
            
            # Loss curves
            fig.add_trace(
                go.Scatter(name='Train Loss', 
                          y=[m.train_loss for m in metrics],
                          mode='lines'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(name='Val Loss', 
                          y=[m.val_loss for m in metrics],
                          mode='lines'),
                row=1, col=1
            )
            
            # State accuracy curves
            fig.add_trace(
                go.Scatter(name='State Train Acc', 
                          y=[m.state_train_acc for m in metrics],
                          mode='lines'),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(name='State Val Acc', 
                          y=[m.state_val_acc for m in metrics],
                          mode='lines'),
                row=1, col=2
            )
            
            # State progress
            fig.add_trace(
                go.Scatter(name='State Progress', 
                          y=[m.state_val_acc for m in metrics],
                          mode='lines'),
                row=2, col=1
            )
        
        # Learning rate (common for both modes)
        fig.add_trace(
            go.Scatter(name='Learning Rate', 
                      y=[m.learning_rate for m in metrics],
                      mode='lines'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Learning Curves - Fold {fold_metrics.fold}",
            showlegend=True
        )
        
        # Save plot
        fig.write_html(save_dir / 'learning_curves.html')

    def summarize_results(self) -> Dict:
        """Generate summary statistics across all folds."""
        if self.labeling_mode == 'dual':
            valence_accs = []
            arousal_accs = []
            
            for fold in self.folds:
                if fold.best_valence_acc > 0:
                    valence_accs.append(fold.best_valence_acc)
                if fold.best_arousal_acc > 0:
                    arousal_accs.append(fold.best_arousal_acc)
            
            summary = {
                'valence_accuracy': {
                    'mean': float(np.mean(valence_accs)) if valence_accs else 0.0,
                    'std': float(np.std(valence_accs)) if len(valence_accs) > 1 else 0.0,
                    'per_fold': valence_accs
                },
                'arousal_accuracy': {
                    'mean': float(np.mean(arousal_accs)) if arousal_accs else 0.0,
                    'std': float(np.std(arousal_accs)) if len(arousal_accs) > 1 else 0.0,
                    'per_fold': arousal_accs
                }
            }
        else:
            state_accs = []
            
            for fold in self.folds:
                if fold.best_state_acc > 0:
                    state_accs.append(fold.best_state_acc)
            
            summary = {
                'state_accuracy': {
                    'mean': float(np.mean(state_accs)) if state_accs else 0.0,
                    'std': float(np.std(state_accs)) if len(state_accs) > 1 else 0.0,
                    'per_fold': state_accs
                }
            }
        
        return summary

    
    def print_final_metrics(self):
        """Print final metrics before summary."""
        print("\nFINAL METRICS CHECK:")
        print("=" * 50)
        print(f"Labeling Mode: {self.labeling_mode}")
        
        for fold in self.folds:
            print(f"\nFold {fold.fold}:")
            if self.labeling_mode == 'dual':
                print(f"Best valence: {fold.best_valence_acc:.4f}")
                print(f"Best arousal: {fold.best_arousal_acc:.4f}")
                if fold.epochs:
                    last_epoch = fold.epochs[-1]
                    print(f"Last epoch metrics:")
                    print(f"  Valence acc: {last_epoch.valence_val_acc:.4f}")
                    print(f"  Arousal acc: {last_epoch.arousal_val_acc:.4f}")
            else:
                print(f"Best state accuracy: {fold.best_state_acc:.4f}")
                if fold.epochs:
                    last_epoch = fold.epochs[-1]
                    print(f"Last epoch metrics:")
                    print(f"  State acc: {last_epoch.state_val_acc:.4f}")
            print(f"Training time: {fold.train_time:.2f}s")
    