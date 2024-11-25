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
        experiment_name: Optional[str] = None
    ):
        """
        Initialize results analyzer.
        
        Args:
            results_dir: Directory containing experiment results
            config: Analysis configuration
            experiment_name: Optional name for the analysis
        """
        self.results_dir = Path(results_dir)
        self.config = config or AnalysisConfig()
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create analysis directory
        self.analysis_dir = self.results_dir / "analysis" 
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logger()
        
        # Load results
        self.results = self._load_results()
    
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
        valence_accs = []
        arousal_accs = []
        
        for fold in self.results['folds']:
            # Try different possible key names
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
    
    
        """Calculate confidence intervals for key metrics."""
        print("\nCalculating intervals from folds:")
        
        valence_accs = []
        arousal_accs = []
        
        for fold in self.results['folds']:
            print(f"\nProcessing fold data:", json.dumps(fold, indent=2))
            
            if 'epochs' in fold:
                best_valence = max(e['valence_val_acc'] for e in fold['epochs'])
                best_arousal = max(e['arousal_val_acc'] for e in fold['epochs'])
                valence_accs.append(best_valence)
                arousal_accs.append(best_arousal)
        
        if not valence_accs or not arousal_accs:
            print("No accuracy data found in folds")
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
        
    def analyze_class_performance(self) -> Dict:
        """Analyze per-class performance across folds."""
        class_metrics = {
            'valence': {'precision': [], 'recall': [], 'f1-score': []},
            'arousal': {'precision': [], 'recall': [], 'f1-score': []}
        }
        
        for fold in self.results['folds']:
            for task in ['valence', 'arousal']:
                metrics = fold[f'{task}_class_metrics']
                for class_idx in ['0', '1', '2']:  # Assuming 3 classes
                    if class_idx in metrics:
                        for metric in ['precision', 'recall', 'f1-score']:
                            class_metrics[task][metric].append(
                                metrics[class_idx][metric]
                            )
        
        return class_metrics
    
    def plot_learning_curves(self):
        """Plot learning curves across all folds."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Loss Curves', 'Valence Accuracy',
                'Arousal Accuracy', 'Learning Rate'
            )
        )
        
        colors = px.colors.qualitative.Set1
        
        for fold_idx, fold in enumerate(self.results['folds']):
            epochs = list(range(len(fold['epochs'])))
            
            # Loss curves
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=[e['train_loss'] for e in fold['epochs']],
                    name=f'Train Loss Fold {fold_idx}',
                    line=dict(color=colors[fold_idx], dash='solid'),
                    showlegend=True
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=[e['val_loss'] for e in fold['epochs']],
                    name=f'Val Loss Fold {fold_idx}',
                    line=dict(color=colors[fold_idx], dash='dash'),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Valence accuracy
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=[e['valence_val_acc'] for e in fold['epochs']],
                    name=f'Valence Fold {fold_idx}',
                    line=dict(color=colors[fold_idx]),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Arousal accuracy
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=[e['arousal_val_acc'] for e in fold['epochs']],
                    name=f'Arousal Fold {fold_idx}',
                    line=dict(color=colors[fold_idx]),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Learning rate
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=[e['learning_rate'] for e in fold['epochs']],
                    name=f'LR Fold {fold_idx}',
                    line=dict(color=colors[fold_idx]),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Training Progress Across Folds",
            showlegend=True
        )
        
        # Save plot
        fig.write_html(self.analysis_dir / "learning_curves.html")
    
    def plot_confusion_matrices(self):
        """Plot average confusion matrices across folds."""
        tasks = ['valence', 'arousal']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Valence Confusion Matrix', 'Arousal Confusion Matrix')
        )
        
        for task_idx, task in enumerate(tasks):
            # Average confusion matrices across folds
            cms = []
            for fold in self.results['folds']:
                if fold['epochs']:  # Check if there are any epochs
                    cms.append(fold['epochs'][-1][f'{task}_val_confusion'])
            
            if cms:  # Only proceed if we have confusion matrices
                avg_cm = np.mean(cms, axis=0)
                
                # Replace NaN values with 0
                avg_cm = np.nan_to_num(avg_cm, 0)
                
                # Create text matrix for annotations
                text_matrix = np.zeros_like(avg_cm, dtype=str)
                for i in range(avg_cm.shape[0]):
                    for j in range(avg_cm.shape[1]):
                        if avg_cm[i, j] > 0:
                            text_matrix[i, j] = f"{avg_cm[i, j]:.2f}"
                        else:
                            text_matrix[i, j] = "0"
                
                # Plot heatmap
                heatmap = go.Heatmap(
                    z=avg_cm,
                    x=['Class 0', 'Class 1', 'Class 2'],
                    y=['Class 0', 'Class 1', 'Class 2'],
                    text=text_matrix,
                    texttemplate="%{text}",
                    textfont={"size": 12},
                    colorscale="Blues",
                    showscale=False
                )
                
                fig.add_trace(heatmap, row=1, col=task_idx+1)
        
        fig.update_layout(
            height=400,
            title_text="Average Confusion Matrices"
        )
        
        # Save plot
        try:
            fig.write_html(self.analysis_dir / "confusion_matrices.html")
        except Exception as e:
            self.logger.warning(f"Failed to save confusion matrices: {e}")
            # Fallback to matplotlib if plotly fails
            self._plot_confusion_matrices_matplotlib()

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
            print("\nCalculating confidence intervals...")
            intervals = self.calculate_confidence_intervals()
            print(f"Intervals calculated: {intervals}")
            
            print("\nAnalyzing class performance...")
            class_metrics = self.analyze_class_performance()
            print(f"Class metrics calculated: {class_metrics}")
            
            print("\nCreating report...")
            report = {
                'performance_summary': intervals,
                'class_performance': class_metrics,
                'training_time': {
                    'total': sum(fold.get('train_time', 0) for fold in self.results['folds']),
                    'mean_per_fold': np.mean([fold.get('train_time', 0) for fold in self.results['folds']])
                }
            }
            
            print("\nSaving report...")
            report_path = self.analysis_dir / "analysis_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {report_path}")
            
            print("\nGenerating plots...")
            try:
                self.plot_learning_curves()
                print("Learning curves plotted")
            except Exception as e:
                print(f"Error plotting learning curves: {e}")
            
            try:
                self.plot_confusion_matrices()
                print("Confusion matrices plotted")
            except Exception as e:
                print(f"Error plotting confusion matrices: {e}")
            
            print("\nLogging summary...")
            if intervals:
                for metric, stats in intervals.items():
                    print(f"\n{metric}:")
                    print(f"Mean: {stats['mean']:.4f}")
                    print(f"Std: {stats['std']:.4f}")
            
            return report
        
        except Exception as e:
            print(f"\nERROR in generate_summary_report: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def compare_experiments(self, other_results_dirs: List[str]):
        """Compare results across different experiments."""
        comparisons = []
        
        for other_dir in other_results_dirs:
            other_analyzer = ResultsAnalyzer(other_dir)
            other_intervals = other_analyzer.calculate_confidence_intervals()
            
            comparison = {
                'experiment': Path(other_dir).name,
                'metrics': {}
            }
            
            for metric in ['valence_accuracy', 'arousal_accuracy']:
                # Perform statistical test
                current_values = [fold['best_' + metric] for fold in self.results['folds']]
                other_values = [fold['best_' + metric] for fold in other_analyzer.results['folds']]
                
                t_stat, p_value = stats.ttest_ind(current_values, other_values)
                
                comparison['metrics'][metric] = {
                    'difference': np.mean(current_values) - np.mean(other_values),
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            
            comparisons.append(comparison)
        
        # Save comparisons
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