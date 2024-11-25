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
            max_seq_length=self.config.data.max_seq_length
        )
        
        # Preprocess dataset
        preprocessor.preprocess_dataset(cache_dir=str(self.preprocessed_dir))
        
        # Create dataset
        dataset = PreprocessedGazePupilDataset(str(self.preprocessed_dir))
        
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
            num_classes=self.config.model.num_classes
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

           
            
            # Update metrics
            metrics_collector.update_epoch_metrics(
                epoch=epoch,
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                valence_train_acc=train_metrics['valence_acc'],
                valence_val_acc=val_metrics['valence_acc'],
                arousal_train_acc=train_metrics['arousal_acc'],
                arousal_val_acc=val_metrics['arousal_acc'],
                valence_val_preds=val_metrics['valence_preds'],
                valence_val_true=val_metrics['valence_true'],
                arousal_val_preds=val_metrics['arousal_preds'],
                arousal_val_true=val_metrics['arousal_true'],
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
        metrics_collector.end_fold(
            val_metrics['valence_preds'],
            val_metrics['valence_true'],
            val_metrics['arousal_preds'],
            val_metrics['arousal_true'],
            train_time
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
            results_dir=str(self.results_dir)
        )
        
        # Create splits
        splits = cv.create_splits(dataset)
        
        # Initialize metrics collector
        metrics_collector = MetricsCollector(
        save_dir=str(self.results_dir),
        experiment_name=self.experiment_name,
        num_classes=self.config.model.num_classes  
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
            experiment_name=self.experiment_name
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
        
        # Safely access report values
        try:
            if report and 'performance_summary' in report:
                perf_summary = report['performance_summary']
                
                # Print available keys
                print("\nAvailable keys in performance_summary:")
                print(list(perf_summary.keys()))
                
                # Valence accuracy
                if perf_summary.get('valence_accuracy'):
                    self.logger.info(
                        f"Valence Accuracy: {perf_summary['valence_accuracy']['mean']:.4f} "
                        f"± {perf_summary['valence_accuracy']['std']:.4f}"
                    )
                else:
                    self.logger.info("Valence accuracy not available in report")
                
                # Arousal accuracy
                if perf_summary.get('arousal_accuracy'):
                    self.logger.info(
                        f"Arousal Accuracy: {perf_summary['arousal_accuracy']['mean']:.4f} "
                        f"± {perf_summary['arousal_accuracy']['std']:.4f}"
                    )
                else:
                    self.logger.info("Arousal accuracy not available in report")
            else:
                self.logger.warning("Report is empty or missing performance summary")
                
                # Use metrics collector's summary as fallback
                collector_summary = metrics_collector.summarize_results()
                self.logger.info("\nUsing metrics collector summary:")
                self.logger.info(f"Valence Accuracy: {collector_summary['valence_accuracy']['mean']:.4f} "
                               f"± {collector_summary['valence_accuracy']['std']:.4f}")
                self.logger.info(f"Arousal Accuracy: {collector_summary['arousal_accuracy']['mean']:.4f} "
                               f"± {collector_summary['arousal_accuracy']['std']:.4f}")
                
                return collector_summary
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error processing report: {e}")
            self.logger.error("Report structure:")
            self.logger.error(str(report))
            return None
            
        except Exception as e:
            self.logger.error(f"Error in experiment: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

def main():
    parser = argparse.ArgumentParser(description='Run emotion recognition experiment')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_path', type=str, help='Path to raw data')
    parser.add_argument('--name', type=str, help='Experiment name')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        config_path=args.config,
        base_path=args.data_path,
        experiment_name=args.name
    )
    
    results = runner.run_experiment()

if __name__ == "__main__":
    main()