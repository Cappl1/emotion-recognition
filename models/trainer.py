import torch
import torch.nn as nn
import math
import numpy as np


class HierarchicalTrainer:
    def __init__(self, model, learning_rate=1e-4, weight_decay=1e-5, device='mps' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def train_step(self, batch):
        self.model.train()
        data, valence_labels, arousal_labels = [x.to(self.device) for x in batch]
        
        # Forward pass
        valence_logits, arousal_logits = self.model(data)
        
        # Calculate losses
        valence_loss = self.criterion(valence_logits, valence_labels)
        arousal_loss = self.criterion(arousal_logits, arousal_labels)
        total_loss = valence_loss + arousal_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'valence_loss': valence_loss.item(),
            'arousal_loss': arousal_loss.item(),
            'valence_acc': (valence_logits.argmax(dim=1) == valence_labels).float().mean().item(),
            'arousal_acc': (arousal_logits.argmax(dim=1) == arousal_labels).float().mean().item()
        }

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'valence_acc': 0.0,
            'arousal_acc': 0.0,
            'valence_loss': 0.0,
            'arousal_loss': 0.0
        }
        
        n_batches = len(train_loader)
        
        for batch in train_loader:
            batch_metrics = self.train_step(batch)
            
            # Accumulate metrics
            epoch_metrics['loss'] += batch_metrics['total_loss']
            epoch_metrics['valence_loss'] += batch_metrics['valence_loss']
            epoch_metrics['arousal_loss'] += batch_metrics['arousal_loss']
            epoch_metrics['valence_acc'] += batch_metrics['valence_acc']
            epoch_metrics['arousal_acc'] += batch_metrics['arousal_acc']
        
        # Calculate averages
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
        
        return epoch_metrics

    def evaluate(self, val_loader):
        """Evaluate the model."""
        self.model.eval()
        total_samples = 0
        running_loss = 0.0
        all_valence_preds = []
        all_valence_true = []
        all_arousal_preds = []
        all_arousal_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                data, valence_labels, arousal_labels = [x.to(self.device) for x in batch]
                batch_size = data.size(0)
                total_samples += batch_size
                
                # Forward pass
                valence_logits, arousal_logits = self.model(data)
                
                # Calculate losses
                valence_loss = self.criterion(valence_logits, valence_labels)
                arousal_loss = self.criterion(arousal_logits, arousal_labels)
                total_loss = valence_loss + arousal_loss
                
                running_loss += total_loss.item() * batch_size
                
                # Get predictions
                valence_preds = valence_logits.argmax(dim=1)
                arousal_preds = arousal_logits.argmax(dim=1)
                
                # Store predictions and true labels
                all_valence_preds.extend(valence_preds.cpu().numpy())
                all_valence_true.extend(valence_labels.cpu().numpy())
                all_arousal_preds.extend(arousal_preds.cpu().numpy())
                all_arousal_true.extend(arousal_labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_valence_preds = np.array(all_valence_preds)
        all_valence_true = np.array(all_valence_true)
        all_arousal_preds = np.array(all_arousal_preds)
        all_arousal_true = np.array(all_arousal_true)
        
        # Calculate accuracies
        valence_acc = np.mean(all_valence_preds == all_valence_true)
        arousal_acc = np.mean(all_arousal_preds == all_arousal_true)
        avg_loss = running_loss / total_samples
        
        print(f"\nEvaluation Summary:")
        print(f"Total samples: {total_samples}")
        print(f"Valence Accuracy: {valence_acc:.4f}")
        print(f"Arousal Accuracy: {arousal_acc:.4f}")
        print(f"Average Loss: {avg_loss:.4f}")
        
        return {
            'loss': avg_loss,
            'valence_acc': float(valence_acc),  # Ensure float
            'arousal_acc': float(arousal_acc),  # Ensure float
            'valence_preds': all_valence_preds,
            'valence_true': all_valence_true,
            'arousal_preds': all_arousal_preds,
            'arousal_true': all_arousal_true
        }
    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']