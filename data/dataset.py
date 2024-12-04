import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
from pathlib import Path

class DatasetPreprocessor:
    def __init__(self, base_path, confidence_threshold=0.8, max_seq_length=512, labeling_mode='dual'):
        self.base_path = base_path
        self.confidence_threshold = confidence_threshold
        self.max_seq_length = max_seq_length
        self.labeling_mode = labeling_mode
        self.stats = {
            'diameter': {
                'mean': None,
                'std': None,
                'min': None,
                'max': None
            },
            'filtering': {
                'sequences_length_before': [],
                'sequences_length_after': [],
            }
        }
        
        # Original dual-head categorization
        self.valence_groups = {
            'NV': [1, 5, 6, 7, 8, 10],
            'MV': [3, 9],
            'PV': [2, 4]
        }
        self.arousal_groups = {
            'LA': [2, 4, 6, 8],
            'MA': [3, 9],
            'HA': [1, 5, 7, 10]
        }
        
        # New single-head state categorization
        self.state_categories = {
            'HANV': [1, 5, 7, 10],
            'MAMV': [3, 9],
            'LAPV': [2, 4],
            'LANV': [6, 8]
        }
    
    def _get_labels(self, video_num):
        """Get labels based on labeling mode."""
        if self.labeling_mode == 'dual':
            valence_label = None
            arousal_label = None
            
            for label, videos in self.valence_groups.items():
                if video_num in videos:
                    valence_label = {'NV': 0, 'MV': 1, 'PV': 2}[label]
                    break
            
            for label, videos in self.arousal_groups.items():
                if video_num in videos:
                    arousal_label = {'LA': 0, 'MA': 1, 'HA': 2}[label]
                    break
            
            return valence_label, arousal_label
        else:  # single head state classification
            state_label = None
            for label, videos in self.state_categories.items():
                if video_num in videos:
                    state_label = {'HANV': 0, 'MAMV': 1, 'LAPV': 2, 'LANV': 3}[label]
                    break
            return state_label

    def calculate_diameter_stats(self):
        """Calculate normalization statistics for pupil diameter."""
        print("Calculating pupil diameter statistics...")
        all_diameters = []
        
        for participant_folder in tqdm(os.listdir(self.base_path)):
            participant_path = os.path.join(self.base_path, participant_folder)
            if not os.path.isdir(participant_path):
                continue
            
            for video_folder in os.listdir(participant_path):
                video_path = os.path.join(participant_path, video_folder)
                if not os.path.isdir(video_path):
                    continue
                
                pupil_file = os.path.join(video_path, 'pupil.csv')
                if not os.path.exists(pupil_file):
                    continue
                
                pupil_data = pd.read_csv(pupil_file)
                valid_data = pupil_data[pupil_data['confidence'] > self.confidence_threshold]
                all_diameters.extend(valid_data['diameter'].values)
        
        all_diameters = np.array(all_diameters)
        self.stats['diameter']['mean'] = float(np.mean(all_diameters))
        self.stats['diameter']['std'] = float(np.std(all_diameters))
        self.stats['diameter']['min'] = float(np.min(all_diameters))
        self.stats['diameter']['max'] = float(np.max(all_diameters))
        
    def process_sequence(self, data, eye_id):
        """Process a single eye sequence."""
        # Filter by eye and confidence
        eye_data = data[data['eye_id'] == eye_id].copy()
        
        # Store sequence lengths for analysis
        if eye_id == 0:  # Only store once per sequence
            self.stats['filtering']['sequences_length_before'].append(len(eye_data))
        
        valid_data = eye_data[eye_data['confidence'] > self.confidence_threshold]
        
        if eye_id == 0:  # Only store once per sequence
            self.stats['filtering']['sequences_length_after'].append(len(valid_data))
        
        if len(valid_data) == 0:
            return None
            
        # Extract features
        features = valid_data[[
            'pupil_timestamp',
            'norm_pos_x',
            'norm_pos_y',
            'diameter'
        ]].copy()
        
        # Normalize diameter using pre-calculated stats
        features['diameter'] = (features['diameter'] - self.stats['diameter']['mean']) / self.stats['diameter']['std']
        
        # Sort by timestamp and drop it
        features = features.sort_values('pupil_timestamp')
        return features.drop('pupil_timestamp', axis=1).values
    
    def preprocess_dataset(self, cache_dir='preprocessed_data'):
        """Preprocess entire dataset and cache results."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        
        # First calculate diameter statistics for normalization
        self.calculate_diameter_stats()
        
        processed_samples = []
        print("\nProcessing and saving sequences...")
        
        for participant_folder in tqdm(os.listdir(self.base_path)):
            participant_path = os.path.join(self.base_path, participant_folder)
            if not os.path.isdir(participant_path):
                continue
            
            for video_folder in os.listdir(participant_path):
                video_path = os.path.join(participant_path, video_folder)
                if not os.path.isdir(video_path):
                    continue
                
                video_num = int(video_folder.split('_')[-1])
                
                # Get labels based on labeling mode
                if self.labeling_mode == 'dual':
                    valence_label, arousal_label = self._get_labels(video_num)
                    labels = {
                        'valence_label': valence_label,
                        'arousal_label': arousal_label
                    }
                else:
                    state_label = self._get_labels(video_num)
                    labels = {
                        'state_label': state_label
                    }
                
                pupil_file = os.path.join(video_path, 'pupil.csv')
                if not os.path.exists(pupil_file):
                    continue
                
                # Load and process data
                pupil_data = pd.read_csv(pupil_file)
                
                # Process each eye
                eye0_features = self.process_sequence(pupil_data, 0)
                eye1_features = self.process_sequence(pupil_data, 1)
                
                # Only save if we have data for both eyes
                if eye0_features is not None and eye1_features is not None:
                    # Handle sequence length
                    if self.max_seq_length:
                        eye0_features = self._pad_or_truncate(eye0_features)
                        eye1_features = self._pad_or_truncate(eye1_features)
                    
                    # Save processed sequence
                    sample_id = f"{participant_folder}_{video_folder}"
                    cache_file = cache_dir / f"{sample_id}.pkl"
                    
                    sample_data = {
                        'eye0_features': eye0_features,
                        'eye1_features': eye1_features,
                        **labels  # Unpack the labels dict
                    }
                    
                    with open(cache_file, 'wb') as f:
                        pickle.dump(sample_data, f)
                    
                    processed_samples.append({
                        'sample_id': sample_id,
                        'cache_file': str(cache_file),
                        **labels
                    })
        
        # Save metadata
        metadata = {
            'stats': self.stats,
            'samples': processed_samples
        }
        
        with open(cache_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        self.print_analysis()
        return processed_samples, self.stats
    
    def _pad_or_truncate(self, features):
        """Pad or truncate sequence to max_seq_length."""
        if len(features) > self.max_seq_length:
            return features[:self.max_seq_length]
        elif len(features) < self.max_seq_length:
            padding = np.zeros((self.max_seq_length - len(features), features.shape[1]))
            return np.vstack([features, padding])
        return features
    
    def print_analysis(self):
        """Print analysis of the preprocessing."""
        print("\nPreprocessing Analysis Report")
        print("=" * 50)
        
        # Diameter statistics
        print("\nPupil Diameter Statistics:")
        print(f"Mean: {self.stats['diameter']['mean']:.2f}")
        print(f"Std: {self.stats['diameter']['std']:.2f}")
        print(f"Min: {self.stats['diameter']['min']:.2f}")
        print(f"Max: {self.stats['diameter']['max']:.2f}")
        
        # Sequence length analysis
        seq_before = self.stats['filtering']['sequences_length_before']
        seq_after = self.stats['filtering']['sequences_length_after']
        
        print("\nSequence Length Analysis:")
        print(f"Average length before filtering: {np.mean(seq_before):.2f}")
        print(f"Average length after filtering: {np.mean(seq_after):.2f}")
        print(f"Median length before filtering: {np.median(seq_before):.2f}")
        print(f"Median length after filtering: {np.median(seq_after):.2f}")
        print(f"Average retention rate: {np.mean(np.array(seq_after) / np.array(seq_before)):.2%}")
        print("=" * 50)


class PreprocessedGazePupilDataset(Dataset):
    def __init__(self, cache_dir, labeling_mode='dual'):
        self.cache_dir = Path(cache_dir)
        self.labeling_mode = labeling_mode
        
        # Load metadata
        with open(self.cache_dir / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        self.stats = metadata['stats']
        self.samples = metadata['samples']
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load preprocessed data
        with open(sample['cache_file'], 'rb') as f:
            data = pickle.load(f)
        
        # Combine features from both eyes
        combined_features = np.concatenate([data['eye0_features'], data['eye1_features']], axis=1)
        features = torch.tensor(combined_features, dtype=torch.float32)
        
        if self.labeling_mode == 'dual':
            valence_label = torch.tensor(data['valence_label'], dtype=torch.long)
            arousal_label = torch.tensor(data['arousal_label'], dtype=torch.long)
            return features, valence_label, arousal_label
        else:
            state_label = torch.tensor(data['state_label'], dtype=torch.long)
            return features, state_label
    


