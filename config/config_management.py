from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import yaml
import json
from pathlib import Path
import torch
from datetime import datetime
import logging
import copy
import hashlib

@dataclass
class DataConfig:
    """Configuration for dataset and data processing."""
    base_path: str
    window_size: int = 100
    max_seq_length: int = 512
    confidence_threshold: float = 0.8
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    input_dim: int = 6
    lstm_hidden_dim: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    d_model: int = 128
    nhead: int = 8
    num_transformer_layers: int = 2
    dim_feedforward: int = 512
    transformer_dropout: float = 0.1
    num_classes: int = 3

@dataclass
class TrainingConfig:
    """Configuration for training process."""
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 10
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    warmup_epochs: int = 5
    seed: int = 42

@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation."""
    n_splits: int = 5
    shuffle: bool = True
    stratify: bool = True

@dataclass
class ExperimentConfig:
    """Master configuration for entire experiment."""
    name: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    cross_validation: CrossValidationConfig
    notes: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    def __post_init__(self):
        """Generate unique identifier for config."""
        # Create dictionary of all settings except timestamp
        config_dict = asdict(self)
        config_dict.pop('timestamp')
        
        # Generate hash for reproducibility
        config_str = json.dumps(config_dict, sort_keys=True)
        self.hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]

class ConfigManager:
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory to store configurations
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger("ConfigManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            fh = logging.FileHandler(self.config_dir / "config_manager.log")
            fh.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            
            logger.addHandler(fh)
        
        return logger
    
    def create_config(
        self,
        name: str,
        base_path: str,
        **kwargs
    ) -> ExperimentConfig:
        """
        Create a new experiment configuration.
        
        Args:
            name: Name of the experiment
            base_path: Path to dataset
            **kwargs: Override default configuration values
        """
        # Create default configs
        data_config = DataConfig(base_path=base_path)
        model_config = ModelConfig()
        training_config = TrainingConfig()
        cv_config = CrossValidationConfig()
        
        # Update with provided kwargs
        for key, value in kwargs.items():
            config_name, param = key.split('.')
            if config_name == 'data':
                setattr(data_config, param, value)
            elif config_name == 'model':
                setattr(model_config, param, value)
            elif config_name == 'training':
                setattr(training_config, param, value)
            elif config_name == 'cv':
                setattr(cv_config, param, value)
        
        # Create experiment config
        config = ExperimentConfig(
            name=name,
            data=data_config,
            model=model_config,
            training=training_config,
            cross_validation=cv_config
        )
        
        return config
    
    def save_config(self, config: ExperimentConfig) -> Path:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
        
        Returns:
            Path to saved configuration
        """
        # Create experiment directory
        exp_dir = self.config_dir / f"{config.timestamp}_{config.name}_{config.hash}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as YAML
        config_path = exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False)
        
        self.logger.info(f"Saved configuration to {config_path}")
        return config_path
    
    def load_config(self, config_path: str) -> ExperimentConfig:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Reconstruct nested configs
        data_config = DataConfig(**config_dict['data'])
        model_config = ModelConfig(**config_dict['model'])
        training_config = TrainingConfig(**config_dict['training'])
        cv_config = CrossValidationConfig(**config_dict['cross_validation'])
        
        config = ExperimentConfig(
            name=config_dict['name'],
            data=data_config,
            model=model_config,
            training=training_config,
            cross_validation=cv_config,
            notes=config_dict.get('notes')
        )
        
        return config
    
    def generate_grid_configs(
        self,
        base_config: ExperimentConfig,
        param_grid: Dict[str, List[Any]]
    ) -> List[ExperimentConfig]:
        """
        Generate configurations for grid search.
        
        Args:
            base_config: Base configuration to modify
            param_grid: Dictionary of parameters to grid search over
                Format: {'data.window_size': [100, 200], 'model.lstm_hidden_dim': [64, 128]}
        
        Returns:
            List of configurations
        """
        configs = []
        
        def recursive_grid_search(param_names, current_values, base_config):
            if not param_names:
                # Create new config with current values
                new_config = copy.deepcopy(base_config)
                for param, value in current_values.items():
                    config_name, param_name = param.split('.')
                    if config_name == 'data':
                        setattr(new_config.data, param_name, value)
                    elif config_name == 'model':
                        setattr(new_config.model, param_name, value)
                    elif config_name == 'training':
                        setattr(new_config.training, param_name, value)
                    elif config_name == 'cv':
                        setattr(new_config.cross_validation, param_name, value)
                
                configs.append(new_config)
                return
            
            current_param = param_names[0]
            for value in param_grid[current_param]:
                current_values[current_param] = value
                recursive_grid_search(param_names[1:], current_values, base_config)
                current_values.pop(current_param)
        
        recursive_grid_search(list(param_grid.keys()), {}, base_config)
        return configs

# Example usage
if __name__ == "__main__":
    # Initialize config manager
    config_manager = ConfigManager()
    
    # Create basic configuration
    config = config_manager.create_config(
        name="test_experiment",
        base_path="/path/to/data",
        **{
            'data.window_size': 200,
            'model.lstm_hidden_dim': 256
        }
    )
    
    # Save configuration
    config_path = config_manager.save_config(config)
    
    # Load configuration
    loaded_config = config_manager.load_config(config_path)
    
    # Generate grid search configurations
    param_grid = {
        'data.window_size': [100, 200],
        'model.lstm_hidden_dim': [128, 256],
        'training.learning_rate': [1e-3, 1e-4]
    }
    
    grid_configs = config_manager.generate_grid_configs(config, param_grid)
    
    print(f"Generated {len(grid_configs)} configurations for grid search")
    
    # Example of accessing configuration parameters
    for i, cfg in enumerate(grid_configs):
        print(f"\nConfiguration {i+1}:")
        print(f"Window size: {cfg.data.window_size}")
        print(f"LSTM hidden dim: {cfg.model.lstm_hidden_dim}")
        print(f"Learning rate: {cfg.training.learning_rate}")