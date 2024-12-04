import torch
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import argparse
import json
from tqdm import tqdm
import time
from typing import Dict, List, Optional, Tuple
import copy

from config.config_management import ConfigManager, ExperimentConfig
from data.dataset import PreprocessedGazePupilDataset, DatasetPreprocessor
from evaluation.cross_validation import CrossValidator
from evaluation.metrics_collection import MetricsCollector
from evaluation.results_analysis import ResultsAnalyzer
from models.model import HierarchicalEmotionModel
from models.trainer import HierarchicalTrainer


class ExperimentRunner:
    def __init__(
        self,
        config_path: str = None,
        base_path: str = None,
        experiment_name: str = None
    ):
        """
        Initialize experiment runner.
        
        Args:
            config_path: Path to existing config file, or None to create new
            base_path: Path to raw data directory
            experiment_name: Name for the experiment
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"experiment_{self.timestamp}"
        
        # Setup directories
        self.setup_directories()
        
        # Initialize logger
        self.logger = self.setup_logger()
        
        # Load or create configuration
        self.config_manager = ConfigManager(self.config_dir)
        self.config = self.load_or_create_config(config_path, base_path)
        
        # Set random seeds
        self.set_random_seeds()
        
        self.logger.info(f"Initialized experiment: {self.experiment_name}")
        
    def setup_directories(self):
        """Create necessary directories."""
        self.base_dir = Path(f"experiments/{self.experiment_name}")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_dir = self.base_dir / "configs"
        self.preprocessed_dir = self.base_dir / "preprocessed_data"
        self.results_dir = self.base_dir / "results"
        self.model_dir = self.base_dir / "models"
        
        for directory in [self.config_dir, self.preprocessed_dir, 
                         self.results_dir, self.model_dir]:
            directory.mkdir(exist_ok=True)
    
    def setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.base_dir / "experiment.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def load_or_create_config(
        self,
        config_path: str = None,
        base_path: str = None
    ) -> ExperimentConfig:
        """Load existing config or create new one."""
        if config_path:
            self.logger.info(f"Loading configuration from {config_path}")
            config = self.config_manager.load_config(config_path)
        else:
            self.logger.info("Creating new configuration")
            if not base_path:
                raise ValueError("base_path must be provided when creating new config")
            
            config = self.config_manager.create_config(
                name=self.experiment_name,
                base_path=base_path
            )
        
        # Save config
        self.config_manager.save_config(config)
        return config
    
    def set_random_seeds(self):
        """Set random seeds for reproducibility."""
        seed = self.config.training.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def prepare_data(self) -> PreprocessedGazePupilDataset:
        """Prepare and preprocess dataset."""
        self.logger.info("Preparing dataset...")
        
        preprocessor = DatasetPreprocessor(
            base_path=self.config.data.base_path,
            confidence_threshold=self.config.data.confidence_threshold,
            max_seq_length=self.config.data.max_seq_length,
            labeling_mode=self.config.data.labeling_mode  # Add labeling mode
        )
        
        # Preprocess dataset
        preprocessor.preprocess_dataset(cache_dir=str(self.preprocessed_dir))
        
        # Create dataset
        dataset = PreprocessedGazePupilDataset(
            str(self.preprocessed_dir),
            labeling_mode=self.config.data.labeling_mode  # Add labeling mode
        )
        
        return dataset
    
    def train_fold(
        self,
        fold: int,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        metrics_collector: MetricsCollector
    ) -> Tuple[HierarchicalEmotionModel, dict]:
        """Train model for one fold."""
        self.logger.info(f"Training fold {fold}")
        
        # Initialize model
        model = HierarchicalEmotionModel(
            input_dim=self.config.model.input_dim,
            lstm_hidden_dim=self.config.model.lstm_hidden_dim,
            lstm_layers=self.config.model.lstm_layers,
            lstm_dropout=self.config.model.lstm_dropout,
            window_size=self.config.data.window_size,
            d_model=self.config.model.d_model,
            nhead=self.config.model.nhead,
            num_transformer_layers=self.config.model.num_transformer_layers,
            dim_feedforward=self.config.model.dim_feedforward,
            transformer_dropout=self.config.model.transformer_dropout,
            num_classes=self.config.model.num_classes,
            labeling_mode=self.config.data.labeling_mode
        )
        
        trainer = HierarchicalTrainer(
            model,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Start collecting metrics for this fold
        metrics_collector.start_fold(fold)
        
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(self.config.training.num_epochs):
            # Training
            train_metrics = trainer.train_epoch(train_loader)
            
            # Validation
            val_metrics = trainer.evaluate(val_loader)
            
            # Update metrics based on labeling mode
            if self.config.data.labeling_mode == 'dual':
                metrics_collector.update_epoch_metrics(
                    epoch=epoch,
                    train_loss=train_metrics['loss'],
                    val_loss=val_metrics['loss'],
                    valence_train_acc=train_metrics['valence_acc'],
                    valence_val_acc=val_metrics['valence_acc'],
                    arousal_train_acc=train_metrics['arousal_acc'],
                    arousal_val_acc=val_metrics['arousal_acc'],
                    valence_val_confusion=val_metrics['valence_val_confusion'],
                    arousal_val_confusion=val_metrics['arousal_val_confusion'],
                    learning_rate=trainer.get_lr()
                )
            else:
                metrics_collector.update_epoch_metrics(
                    epoch=epoch,
                    train_loss=train_metrics['loss'],
                    val_loss=val_metrics['loss'],
                    state_train_acc=train_metrics['state_acc'],
                    state_val_acc=val_metrics['state_acc'],
                    state_val_confusion=val_metrics['state_val_confusion'],
                    learning_rate=trainer.get_lr()
                )
            
            # Early stopping check
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # Save best model
                torch.save(
                    model.state_dict(),
                    self.model_dir / f"model_fold_{fold}_best.pt"
                )
            else:
                patience_counter += 1
                if patience_counter >= self.config.training.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        train_time = time.time() - start_time
        
        # End fold metrics collection
        if self.config.data.labeling_mode == 'dual':
            metrics_collector.end_fold(
                val_metrics['valence_preds'],
                val_metrics['valence_true'],
                val_metrics['arousal_preds'],
                val_metrics['arousal_true'],
                train_time=train_time
            )
        else:
            metrics_collector.end_fold(
                preds=val_metrics['state_preds'],
                true=val_metrics['state_true'],
                train_time=train_time
            )
        
        return model, val_metrics
    
    def run_experiment(self):
        """Run complete experiment."""
        self.logger.info("Starting experiment")
        
        # Prepare dataset
        dataset = self.prepare_data()
        
        # Initialize cross-validation
        cv = CrossValidator(
            n_splits=self.config.cross_validation.n_splits,
            shuffle=self.config.cross_validation.shuffle,
            random_state=self.config.training.seed,
            results_dir=str(self.results_dir),
            labeling_mode=self.config.data.labeling_mode
        )
        
        # Create splits
        splits = cv.create_splits(dataset)
        
        # Initialize metrics collector
        metrics_collector = MetricsCollector(
            save_dir=str(self.results_dir),
            experiment_name=self.experiment_name,
            labeling_mode=self.config.data.labeling_mode  # Replace num_classes with labeling_mode
        )
        
        # Train and evaluate for each fold
        for fold, split_info in enumerate(splits):
            self.logger.info(f"\nProcessing fold {fold}")
            
            # Get dataloaders for this fold
            train_loader, val_loader = cv.get_fold_dataloaders(
                dataset,
                fold,
                batch_size=self.config.data.batch_size,
                num_workers=self.config.data.num_workers
            )
            
            # Train model
            model, val_metrics = self.train_fold(
                fold,
                train_loader,
                val_loader,
                metrics_collector
            )
        
        # Generate final analysis
        analyzer = ResultsAnalyzer(
            results_dir=str(self.results_dir),
            experiment_name=self.experiment_name,
            labeling_mode=self.config.data.labeling_mode  # Add labeling mode
        )
        # In your main script, right before getting the final summary:
        metrics_collector.print_final_metrics()
        final_summary = metrics_collector.summarize_results()
        print("\nFINAL STATE BEFORE SUMMARY:")
        print("=" * 50)
        for fold in metrics_collector.folds:
            print(f"\nFold {fold.fold}")
            print(f"Best valence acc: {fold.best_valence_acc}")
            print(f"Best arousal acc: {fold.best_arousal_acc}")
    
            report = analyzer.generate_summary_report()
    
            self.logger.info("\nExperiment completed!")
            self.logger.info("\nFinal Results:")
            self.logger.info("=" * 50)
            
            # Print the raw report for debugging
            print("\nDEBUG - Raw Report Contents:")
            print(json.dumps(report, indent=2))
            
            try:
                if report and 'performance_summary' in report:
                    perf_summary = report['performance_summary']
                    
                    if self.config.data.labeling_mode == 'state':
                        if perf_summary.get('state_accuracy'):
                            self.logger.info(
                                f"State Classification Accuracy: {perf_summary['state_accuracy']['mean']:.4f} "
                                f"± {perf_summary['state_accuracy']['std']:.4f}"
                            )
                            
                            # Print per-class metrics if available
                            if 'class_performance' in report and 'state' in report['class_performance']:
                                state_metrics = report['class_performance']['state']
                                state_names = ['HANV', 'MAMV', 'LAPV', 'LANV']
                                
                                # Calculate metrics for each fold
                                for i, state in enumerate(state_names):
                                    self.logger.info(f"\n{state}:")
                                    # Calculate average precision across folds
                                    precisions = [state_metrics['precision'][j] for j in range(0, len(state_metrics['precision']), 4) if state_metrics['precision'][j+i] > 0]
                                    if precisions:
                                        self.logger.info(f"Precision: {np.mean(precisions):.4f}")
                                    else:
                                        self.logger.info("Precision: N/A")
                                        
                                    # Calculate average recall across folds
                                    recalls = [state_metrics['recall'][j] for j in range(0, len(state_metrics['recall']), 4) if state_metrics['recall'][j+i] > 0]
                                    if recalls:
                                        self.logger.info(f"Recall: {np.mean(recalls):.4f}")
                                    else:
                                        self.logger.info("Recall: N/A")
                                        
                                    # Calculate average F1-score across folds
                                    f1_scores = [state_metrics['f1-score'][j] for j in range(0, len(state_metrics['f1-score']), 4) if state_metrics['f1-score'][j+i] > 0]
                                    if f1_scores:
                                        self.logger.info(f"F1-score: {np.mean(f1_scores):.4f}")
                                    else:
                                        self.logger.info("F1-score: N/A")

                                # Print macro averages
                                self.logger.info("\nMacro Averages:")
                                for metric in ['precision', 'recall', 'f1-score']:
                                    values = [x for x in state_metrics[metric] if x > 0]
                                    if values:
                                        self.logger.info(f"Average {metric}: {np.mean(values):.4f}")
                    else:
                        # Original dual-head metrics reporting
                        if perf_summary.get('valence_accuracy'):
                            self.logger.info(
                                f"Valence Accuracy: {perf_summary['valence_accuracy']['mean']:.4f} "
                                f"± {perf_summary['valence_accuracy']['std']:.4f}"
                            )
                        if perf_summary.get('arousal_accuracy'):
                            self.logger.info(
                                f"Arousal Accuracy: {perf_summary['arousal_accuracy']['mean']:.4f} "
                                f"± {perf_summary['arousal_accuracy']['std']:.4f}"
                            )

                return report
                
            except Exception as e:
                self.logger.error(f"Error processing report: {e}")
                self.logger.error("Report structure:")
                self.logger.error(str(report))
                
                # Print detailed structure for debugging
                self.logger.error("\nDetailed metrics structure:")
                if 'class_performance' in report and 'state' in report['class_performance']:
                    state_metrics = report['class_performance']['state']
                    self.logger.error(f"Precision: {state_metrics['precision']}")
                    self.logger.error(f"Recall: {state_metrics['recall']}")
                    self.logger.error(f"F1-score: {state_metrics['f1-score']}")
                
                return None
    def run_validation_experiment(self):
        """Run validation experiment for different model configurations."""
        self.logger.info("Starting validation experiment")
        
        # Configurations to test feature processing dimensions
        configs_to_test = [
            # Baseline for comparison
            {
                'window_size': 100,
                'lstm_layers': 2,
                'lstm_hidden_dim': 128,
                'd_model': 128,
                'nhead': 8,
                'dim_feedforward': 512,
                'num_transformer_layers': 2,
                'lstm_dropout': 0.1,
                'transformer_dropout': 0.1
            },
            
            # "Deep Transformer" - More layers with gradient checkpointing
            {
                'window_size': 100,
                'lstm_layers': 2,
                'lstm_hidden_dim': 256,
                'd_model': 256,
                'nhead': 8,
                'dim_feedforward': 1024,
                'num_transformer_layers': 6,
                'lstm_dropout': 0.2,
                'transformer_dropout': 0.2
            },
            
            # "Wide LSTM" - Focusing on temporal feature extraction
            {
                'window_size': 150,  # Larger window for more temporal context
                'lstm_layers': 3,
                'lstm_hidden_dim': 512,  # Much wider LSTM
                'd_model': 256,
                'nhead': 8,
                'dim_feedforward': 1024,
                'num_transformer_layers': 2,
                'lstm_dropout': 0.3,  # Higher dropout for wider model
                'transformer_dropout': 0.1
            },
            
            # "Attention Monster" - Heavy focus on multi-head attention
            {
                'window_size': 100,
                'lstm_layers': 2,
                'lstm_hidden_dim': 256,
                'd_model': 512,  # Larger embedding dimension
                'nhead': 16,    # More attention heads
                'dim_feedforward': 2048,
                'num_transformer_layers': 4,
                'lstm_dropout': 0.2,
                'transformer_dropout': 0.2
            },
            
            # "Hybrid Giant" - Large in all dimensions
            {
                'window_size': 200,  # Much larger temporal window
                'lstm_layers': 4,
                'lstm_hidden_dim': 512,
                'd_model': 512,
                'nhead': 16,
                'dim_feedforward': 2048,
                'num_transformer_layers': 4,
                'lstm_dropout': 0.3,
                'transformer_dropout': 0.3
            },
            
            # "Balanced Powerhouse" - Carefully balanced architecture
            {
                'window_size': 150,
                'lstm_layers': 3,
                'lstm_hidden_dim': 384,
                'd_model': 384,
                'nhead': 12,
                'dim_feedforward': 1536,
                'num_transformer_layers': 3,
                'lstm_dropout': 0.25,
                'transformer_dropout': 0.25
            }
        ]
        
        validation_results = []
        
        # Prepare dataset once
        dataset = self.prepare_data()
        
        for config_variant in configs_to_test:
            variant_name = f"w{config_variant['window_size']}_l{config_variant['lstm_layers']}_t{config_variant['num_transformer_layers']}"
            self.logger.info(f"\nTesting configuration: {variant_name}")
            
            # Create modified config for this variant
            variant_config = copy.deepcopy(self.config)
            
            # Update all configuration parameters
            for key, value in config_variant.items():
                if key == 'window_size':
                    variant_config.data.window_size = value
                else:
                    setattr(variant_config.model, key, value)
            
            # Initialize cross-validation
            cv = CrossValidator(
                n_splits=variant_config.cross_validation.n_splits,
                shuffle=variant_config.cross_validation.shuffle,
                random_state=variant_config.training.seed,
                results_dir=str(self.results_dir / variant_name),
                labeling_mode=variant_config.data.labeling_mode  # Add labeling mode
            )
            
            # Create splits
            splits = cv.create_splits(dataset)
            
            # Initialize metrics collector for this variant
            metrics_collector = MetricsCollector(
                save_dir=str(self.results_dir / variant_name),
                experiment_name=f"{self.experiment_name}_{variant_name}",
                labeling_mode=variant_config.data.labeling_mode  # Add labeling mode
            )
            
            # Train and evaluate for each fold
            for fold, split_info in enumerate(splits):
                self.logger.info(f"\nProcessing fold {fold} for {variant_name}")
                
                # Get dataloaders for this fold
                train_loader, val_loader = cv.get_fold_dataloaders(
                    dataset,
                    fold,
                    batch_size=variant_config.data.batch_size,
                    num_workers=variant_config.data.num_workers
                )
                
                # Train model
                model, val_metrics = self.train_fold(
                    fold,
                    train_loader,
                    val_loader,
                    metrics_collector
                )
            
            # Get results for this variant
            variant_results = metrics_collector.summarize_results()
            variant_results['config'] = config_variant
            validation_results.append(variant_results)
            
            # Log results based on labeling mode
            self.logger.info(f"\nResults for {variant_name}:")
            if variant_config.data.labeling_mode == 'dual':
                self.logger.info(f"Valence Accuracy: {variant_results['valence_accuracy']['mean']:.4f} ± {variant_results['valence_accuracy']['std']:.4f}")
                self.logger.info(f"Arousal Accuracy: {variant_results['arousal_accuracy']['mean']:.4f} ± {variant_results['arousal_accuracy']['std']:.4f}")
            else:
                self.logger.info(f"State Accuracy: {variant_results['state_accuracy']['mean']:.4f} ± {variant_results['state_accuracy']['std']:.4f}")
        
        # Save overall validation results
        with open(self.results_dir / 'validation_results.json', 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        # Print final comparison
        self.logger.info("\nValidation Results Summary:")
        self.logger.info("=" * 50)
        for result in validation_results:
            config = result['config']
            name = f"w{config['window_size']}_l{config['lstm_layers']}_t{config['num_transformer_layers']}"
            self.logger.info(f"\nConfiguration: {name}")
            if self.config.data.labeling_mode == 'dual':
                self.logger.info(f"Valence Accuracy: {result['valence_accuracy']['mean']:.4f} ± {result['valence_accuracy']['std']:.4f}")
                self.logger.info(f"Arousal Accuracy: {result['arousal_accuracy']['mean']:.4f} ± {result['arousal_accuracy']['std']:.4f}")
            else:
                self.logger.info(f"State Accuracy: {result['state_accuracy']['mean']:.4f} ± {result['state_accuracy']['std']:.4f}")
        
        return validation_results
"""
def main():
    parser = argparse.ArgumentParser(description='Run emotion recognition experiment')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_path', type=str, help='Path to raw data')
    parser.add_argument('--name', type=str, help='Experiment name')
    
    args = parser.parse_args()
    print(torch.cuda.is_available())
    runner = ExperimentRunner(
        config_path=args.config,
        base_path=args.data_path,
        experiment_name=args.name
    )
    
    results = runner.run_experiment()
"""
def main():
    parser = argparse.ArgumentParser(description='Run emotion recognition experiment')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_path', type=str, help='Path to raw data')
    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--validation', action='store_true', help='Run validation experiment')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        config_path=args.config,
        base_path=args.data_path,
        experiment_name=args.name
    )
    
    if args.validation:
        results = runner.run_validation_experiment()
    else:
        results = runner.run_experiment()
if __name__ == "__main__":
    main()