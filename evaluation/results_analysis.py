import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
from datetime import datetime
import yaml
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback  

@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    confidence_level: float = 0.95
    min_samples_per_class: int = 10
    plot_style: str = "plotly"  # or "matplotlib"
    save_format: str = "html"   # or "png"

class ResultsAnalyzer:
    def __init__(
        self,
        results_dir: str,
        config: Optional[AnalysisConfig] = None,
        experiment_name: Optional[str] = None,
        labeling_mode: str = 'dual'  # Add labeling_mode parameter
    ):
        self.results_dir = Path(results_dir)
        self.config = config or AnalysisConfig()
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.labeling_mode = labeling_mode
        
        # Create analysis directory
        self.analysis_dir = self.results_dir / "analysis" 
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logger()
        self.results = self._load_results()
        
        # State classification labels
        self.state_labels = ['HANV', 'MAMV', 'LAPV', 'LANV']
    
    def _load_results(self) -> Dict:
        """Load results from all folds."""
        print("\nLoading results...")
        results = {'folds': [], 'config': None}
        
        # First try to load from fold directories
        fold_dirs = sorted(self.results_dir.glob("fold_*"))
        if not fold_dirs:
            # If no fold directories found, try looking in the metrics_collector output
            metrics_file = self.results_dir / "final_summary.json"
            #print(f"metrics file{metrics_file}")
            if metrics_file.exists():
                print(f"Loading from summary file: {metrics_file}")
                with open(metrics_file, 'r') as f:
                    summary_data = json.load(f)
                    if 'raw_data' in summary_data:
                        for fold_data in summary_data['raw_data']['folds']:
                            results['folds'].append({
                                'fold': fold_data['fold'],
                                'best_valence_acc': fold_data['best_valence'],
                                'best_arousal_acc': fold_data['best_arousal'],
                                'epochs': [{'epoch': 0}]  # Minimal epoch data
                            })
                        print(f"Loaded {len(results['folds'])} folds from summary")
                        return results
        
        # Load from individual fold directories if they exist
        for fold_dir in fold_dirs:
            metrics_file = fold_dir / "metrics.json"
            if metrics_file.exists():
                print(f"Loading metrics from: {metrics_file}")
                with open(metrics_file, 'r') as f:
                    fold_data = json.load(f)
                    results['folds'].append(fold_data)
        
        print(f"Loaded {len(results['folds'])} folds")
        return results
    
    def calculate_confidence_intervals(self) -> Dict:
        """Calculate confidence intervals for key metrics."""
        if self.labeling_mode == 'dual':
            valence_accs = []
            arousal_accs = []
            
            for fold in self.results['folds']:
                valence_acc = fold.get('best_valence_acc', fold.get('best_valence', None))
                arousal_acc = fold.get('best_arousal_acc', fold.get('best_arousal', None))
                
                if valence_acc is not None:
                    valence_accs.append(float(valence_acc))
                if arousal_acc is not None:
                    arousal_accs.append(float(arousal_acc))
            
            if not valence_accs or not arousal_accs:
                print("WARNING: No accuracy data found in folds")
                return {}
                
            return {
                'valence_accuracy': {
                    'mean': float(np.mean(valence_accs)),
                    'std': float(np.std(valence_accs)),
                    'values': valence_accs
                },
                'arousal_accuracy': {
                    'mean': float(np.mean(arousal_accs)),
                    'std': float(np.std(arousal_accs)),
                    'values': arousal_accs
                }
            }
        else:
            state_accs = []
            for fold in self.results['folds']:
                state_acc = fold.get('best_state_acc', None)
                if state_acc is not None:
                    state_accs.append(float(state_acc))
            
            if not state_accs:
                print("WARNING: No state accuracy data found in folds")
                return {}
            
            return {
                'state_accuracy': {
                    'mean': float(np.mean(state_accs)),
                    'std': float(np.std(state_accs)),
                    'values': state_accs
                }
            }
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger(f"ResultsAnalyzer_{self.experiment_name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            fh = logging.FileHandler(self.analysis_dir / "analysis.log")
            fh.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            
            logger.addHandler(fh)
        
        return logger
    
    
    def analyze_class_performance(self) -> Dict:
        """Analyze per-class performance across folds."""
        if self.labeling_mode == 'dual':
            class_metrics = {
                'valence': {'precision': [], 'recall': [], 'f1-score': []},
                'arousal': {'precision': [], 'recall': [], 'f1-score': []}
            }
            
            for fold in self.results['folds']:
                for task in ['valence', 'arousal']:
                    metrics = fold[f'{task}_class_metrics']
                    for class_idx in ['0', '1', '2']:
                        if class_idx in metrics:
                            for metric in ['precision', 'recall', 'f1-score']:
                                class_metrics[task][metric].append(
                                    metrics[class_idx][metric]
                                )
        else:
            class_metrics = {
                'state': {'precision': [], 'recall': [], 'f1-score': []}
            }
            
            for fold in self.results['folds']:
                metrics = fold['state_class_metrics']
                for class_idx in ['0', '1', '2', '3']:  # Four states
                    if class_idx in metrics:
                        for metric in ['precision', 'recall', 'f1-score']:
                            class_metrics['state'][metric].append(
                                metrics[class_idx][metric]
                            )
        
        return class_metrics

    

    

    def _plot_confusion_matrices_matplotlib(self):
        """Fallback matplotlib version for confusion matrix plotting."""
        tasks = ['valence', 'arousal']
        
        plt.figure(figsize=(12, 5))
        for task_idx, task in enumerate(tasks, 1):
            cms = []
            for fold in self.results['folds']:
                if fold['epochs']:
                    cms.append(fold['epochs'][-1][f'{task}_val_confusion'])
            
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
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "confusion_matrices.png")
        plt.close()
    
    def generate_summary_report(self):
        """Generate comprehensive analysis report with debug prints."""
        print("\nDEBUG: Generating Summary Report")
        print("=" * 50)
        
        try:
            report = {
                'performance_summary': {},
                'class_performance': {},
                'training_time': {}
            }

            # Get the training times from the loaded results
            train_times = [fold.get('train_time', 0.0) for fold in self.results['folds']]
            report['training_time'] = {
                'total': float(sum(train_times)),
                'mean_per_fold': float(np.mean(train_times)) if train_times else 0.0
            }

            # Generate performance metrics
            if self.labeling_mode == 'dual':
                # ... dual mode code ...
                pass
            else:
                # State mode metrics
                state_accs = []
                for fold in self.results['folds']:
                    acc = fold.get('best_state_acc', fold.get('best_state', 0.0))
                    if acc > 0:  # Only include valid accuracies
                        state_accs.append(float(acc))
                
                if state_accs:
                    report['performance_summary']['state_accuracy'] = {
                        'mean': float(np.mean(state_accs)),
                        'std': float(np.std(state_accs)),
                        'values': state_accs
                    }
                
                # Collect class performance metrics
                #report['class_performance'] = self._collect_class_performance()

            # Generate plots
            self.plot_learning_curves()
            self.plot_confusion_matrices()
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error in generate_summary_report: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def plot_learning_curves(self):
        """Plot learning curves."""
        try:
            if self.labeling_mode == 'dual':
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        'Loss Curves', 'Valence Accuracy',
                        'Arousal Accuracy', 'Learning Rate'
                    )
                )
            else:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        'Loss Curves', 'State Accuracy',
                        'Class-wise Performance', 'Learning Rate'
                    )
                )
            
            colors = px.colors.qualitative.Set1
            
            # Plot for each fold
            for fold_idx, fold in enumerate(self.results['folds']):
                if 'epochs' not in fold or not fold['epochs']:
                    continue
                    
                epochs = list(range(len(fold['epochs'])))
                
                # Loss curves
                train_losses = [e.get('train_loss', 0) for e in fold['epochs']]
                val_losses = [e.get('val_loss', 0) for e in fold['epochs']]
                
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=train_losses,
                        name=f'Train Loss Fold {fold_idx}',
                        line=dict(color=colors[fold_idx], dash='solid')
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=val_losses,
                        name=f'Val Loss Fold {fold_idx}',
                        line=dict(color=colors[fold_idx], dash='dash')
                    ),
                    row=1, col=1
                )
                
                # Accuracy curves
                if self.labeling_mode == 'dual':
                    # Plot valence accuracy
                    fig.add_trace(
                        go.Scatter(
                            x=epochs,
                            y=[e.get('valence_val_acc', 0) for e in fold['epochs']],
                            name=f'Valence Fold {fold_idx}',
                            line=dict(color=colors[fold_idx])
                        ),
                        row=1, col=2
                    )
                    
                    # Plot arousal accuracy
                    fig.add_trace(
                        go.Scatter(
                            x=epochs,
                            y=[e.get('arousal_val_acc', 0) for e in fold['epochs']],
                            name=f'Arousal Fold {fold_idx}',
                            line=dict(color=colors[fold_idx])
                        ),
                        row=2, col=1
                    )
                else:
                    # Plot state accuracy
                    fig.add_trace(
                        go.Scatter(
                            x=epochs,
                            y=[e.get('state_val_acc', 0) for e in fold['epochs']],
                            name=f'State Acc Fold {fold_idx}',
                            line=dict(color=colors[fold_idx])
                        ),
                        row=1, col=2
                    )
                
                # Learning rate
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=[e.get('learning_rate', 0) for e in fold['epochs']],
                        name=f'LR Fold {fold_idx}',
                        line=dict(color=colors[fold_idx])
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=800,
                title_text=f"Training Progress Across Folds ({self.labeling_mode.capitalize()} Mode)",
                showlegend=True
            )
            
            # Ensure the analysis directory exists
            if not self.analysis_dir.exists():
                self.analysis_dir.mkdir(parents=True)
            
            # Save the plot
            plot_path = self.analysis_dir / "learning_curves.html"
            fig.write_html(str(plot_path))
            self.logger.info(f"Saved learning curves plot to {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Error plotting learning curves: {e}")
            self.logger.error(traceback.format_exc())

    def plot_confusion_matrices(self):
        """Plot confusion matrices."""
        try:
            if self.labeling_mode == 'dual':
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=['Valence Confusion Matrix', 'Arousal Confusion Matrix']
                )
            else:
                fig = make_subplots(
                    rows=1, cols=1,
                    subplot_titles=['State Confusion Matrix']
                )
                
                # Get confusion matrix for each fold's last epoch
                matrices = []
                for fold in self.results['folds']:
                    if 'epochs' in fold and fold['epochs']:
                        last_epoch = fold['epochs'][-1]
                        if 'state_val_confusion' in last_epoch:
                            matrix = last_epoch['state_val_confusion']
                            # Validate matrix before adding
                            if matrix is not None and isinstance(matrix, (list, np.ndarray)):
                                matrix = np.array(matrix)
                                if matrix.size > 0:  # Check if matrix is not empty
                                    matrices.append(matrix)
                                    self.logger.debug(f"Added valid confusion matrix with shape {matrix.shape}")
                                else:
                                    self.logger.warning(f"Empty confusion matrix found in fold")
                            else:
                                self.logger.warning(f"Invalid confusion matrix type: {type(matrix)}")
                
                if matrices:
                    self.logger.info(f"Found {len(matrices)} valid confusion matrices")
                    # Convert matrices to numpy arrays and ensure they have the same shape
                    matrices = [m for m in matrices if m.shape == matrices[0].shape]
                    
                    if matrices:  # Check again after filtering
                        # Average the matrices
                        avg_matrix = np.mean(matrices, axis=0)
                        
                        # Create heatmap
                        heatmap = go.Heatmap(
                            z=avg_matrix,
                            x=['HANV', 'MAMV', 'LAPV', 'LANV'],
                            y=['HANV', 'MAMV', 'LAPV', 'LANV'],
                            text=[[f"{val:.2f}" for val in row] for row in avg_matrix],
                            texttemplate="%{text}",
                            colorscale="Blues"
                        )
                        
                        fig.add_trace(heatmap, row=1, col=1)
                        
                        fig.update_layout(
                            height=600,
                            title_text="Confusion Matrices",
                            showlegend=False
                        )
                        
                        # Save the plot
                        plot_path = self.analysis_dir / "confusion_matrices.html"
                        fig.write_html(str(plot_path))
                        self.logger.info(f"Saved confusion matrices plot to {plot_path}")
                    else:
                        self.logger.warning("No valid confusion matrices found after shape validation")
                else:
                    self.logger.warning("No valid confusion matrices found in results")
                    
                # Add some debugging information
                self.logger.debug("Confusion matrix data:")
                for fold_idx, fold in enumerate(self.results['folds']):
                    if 'epochs' in fold and fold['epochs']:
                        last_epoch = fold['epochs'][-1]
                        if 'state_val_confusion' in last_epoch:
                            self.logger.debug(f"Fold {fold_idx} confusion matrix: {last_epoch['state_val_confusion']}")
                        else:
                            self.logger.debug(f"Fold {fold_idx} has no confusion matrix")
                
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrices: {e}")
            self.logger.error(traceback.format_exc())
            
            # Add detailed error information
            self.logger.error("Results structure:")
            if hasattr(self, 'results'):
                for fold_idx, fold in enumerate(self.results.get('folds', [])):
                    self.logger.error(f"\nFold {fold_idx}:")
                    if 'epochs' in fold and fold['epochs']:
                        last_epoch = fold['epochs'][-1]
                        self.logger.error(f"Last epoch keys: {list(last_epoch.keys())}")
                        if 'state_val_confusion' in last_epoch:
                            self.logger.error(f"Confusion matrix type: {type(last_epoch['state_val_confusion'])}")
                            self.logger.error(f"Confusion matrix content: {last_epoch['state_val_confusion']}")
                    else:
                        self.logger.error("No epochs data")
        
    def compare_experiments(self, other_results_dirs: List[str]):
        """Compare results across different experiments."""
        comparisons = []
        
        for other_dir in other_results_dirs:
            other_analyzer = ResultsAnalyzer(
                other_dir, 
                labeling_mode=self.labeling_mode
            )
            other_intervals = other_analyzer.calculate_confidence_intervals()
            
            comparison = {
                'experiment': Path(other_dir).name,
                'metrics': {}
            }
            
            if self.labeling_mode == 'dual':
                metrics = ['valence_accuracy', 'arousal_accuracy']
            else:
                metrics = ['state_accuracy']
            
            for metric in metrics:
                current_values = [
                    fold[f'best_{metric.split("_")[0]}'] 
                    for fold in self.results['folds']
                ]
                other_values = [
                    fold[f'best_{metric.split("_")[0]}'] 
                    for fold in other_analyzer.results['folds']
                ]
                
                t_stat, p_value = stats.ttest_ind(current_values, other_values)
                
                comparison['metrics'][metric] = {
                    'difference': np.mean(current_values) - np.mean(other_values),
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            
            comparisons.append(comparison)
        
        with open(self.analysis_dir / "experiment_comparisons.json", 'w') as f:
            json.dump(comparisons, f, indent=2)
        
        return comparisons
# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ResultsAnalyzer(
        results_dir="experiment_results/test_run",
        experiment_name="analysis_1"
    )
    
    # Generate summary report
    report = analyzer.generate_summary_report()
    
    # Compare with other experiments
    comparisons = analyzer.compare_experiments([
        "experiment_results/baseline",
        "experiment_results/improved_model"
    ])