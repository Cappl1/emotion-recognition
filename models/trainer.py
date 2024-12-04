import torch
import torch.nn as nn
import math
import numpy as np
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class HierarchicalTrainer:
    def __init__(self, model, learning_rate=1e-4, weight_decay=1e-5, 
                 device='cuda:2' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        # Add StepLR scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,  # Decrease LR every 10 epochs
            gamma=0.1      # Multiply LR by 0.1 at each step
        )
        self.labeling_mode = model.labeling_mode

    def train_step(self, batch):
        self.model.train()
        
        if self.labeling_mode == 'dual':
            data, valence_labels, arousal_labels = [x.to(self.device) for x in batch]
            valence_logits, arousal_logits = self.model(data)
            
            valence_loss = self.criterion(valence_logits, valence_labels)
            arousal_loss = self.criterion(arousal_logits, arousal_labels)
            total_loss =   valence_loss# + arousal_loss
            
            metrics = {
                'total_loss': total_loss.item(),
                'valence_loss': valence_loss.item(),
                'arousal_loss': arousal_loss.item(),
                'valence_acc': (valence_logits.argmax(dim=1) == valence_labels).float().mean().item(),
                'arousal_acc': (arousal_logits.argmax(dim=1) == arousal_labels).float().mean().item()
            }
        else:
            data, state_labels = [x.to(self.device) for x in batch]
            state_logits = self.model(data)
            
            loss = self.criterion(state_logits, state_labels)
            
            metrics = {
                'total_loss': loss.item(),
                'state_acc': (state_logits.argmax(dim=1) == state_labels).float().mean().item()
            }
            total_loss = loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return metrics

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        
        # Initialize metrics based on labeling mode
        if self.labeling_mode == 'dual':
            epoch_metrics = {
                'loss': 0.0,
                'valence_acc': 0.0,
                'arousal_acc': 0.0,
                'valence_loss': 0.0,
                'arousal_loss': 0.0
            }
        else:  # state classification
            epoch_metrics = {
                'loss': 0.0,
                'state_acc': 0.0
            }
        
        n_batches = len(train_loader)
        
        for batch in train_loader:
            batch_metrics = self.train_step(batch)
            
            # Accumulate metrics based on labeling mode
            if self.labeling_mode == 'dual':
                epoch_metrics['loss'] += batch_metrics['total_loss']
                epoch_metrics['valence_loss'] += batch_metrics['valence_loss']
                epoch_metrics['arousal_loss'] += batch_metrics['arousal_loss']
                epoch_metrics['valence_acc'] += batch_metrics['valence_acc']
                epoch_metrics['arousal_acc'] += batch_metrics['arousal_acc']
            else:
                epoch_metrics['loss'] += batch_metrics['total_loss']
                epoch_metrics['state_acc'] += batch_metrics['state_acc']
        
        # Calculate averages
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
        
        # Log training progress
        if self.labeling_mode == 'dual':
            print(f"\nTraining Metrics:")
            print(f"Loss: {epoch_metrics['loss']:.4f}")
            print(f"Valence Accuracy: {epoch_metrics['valence_acc']:.4f}")
            print(f"Arousal Accuracy: {epoch_metrics['arousal_acc']:.4f}")
        else:
            print(f"\nTraining Metrics:")
            print(f"Loss: {epoch_metrics['loss']:.4f}")
            print(f"State Accuracy: {epoch_metrics['state_acc']:.4f}")
        #self.scheduler.step()
        return epoch_metrics

    def evaluate(self, val_loader):
        """Evaluate the model."""
        self.model.eval()
        total_samples = 0
        running_loss = 0.0
        
        if self.labeling_mode == 'dual':
            all_valence_preds = []
            all_valence_true = []
            all_arousal_preds = []
            all_arousal_true = []
        else:
            all_state_preds = []
            all_state_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                if self.labeling_mode == 'dual':
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
                else:
                    data, state_labels = [x.to(self.device) for x in batch]
                    batch_size = data.size(0)
                    total_samples += batch_size
                    
                    # Forward pass
                    state_logits = self.model(data)
                    
                    # Calculate loss
                    total_loss = self.criterion(state_logits, state_labels)
                    running_loss += total_loss.item() * batch_size
                    
                    # Get predictions
                    state_preds = state_logits.argmax(dim=1)
                    
                    # Store predictions and true labels
                    all_state_preds.extend(state_preds.cpu().numpy())
                    all_state_true.extend(state_labels.cpu().numpy())
        
        # Convert lists to numpy arrays
        if self.labeling_mode == 'dual':
            all_valence_preds = np.array(all_valence_preds)
            all_valence_true = np.array(all_valence_true)
            all_arousal_preds = np.array(all_arousal_preds)
            all_arousal_true = np.array(all_arousal_true)
            
            # Calculate accuracies
            valence_acc = np.mean(all_valence_preds == all_valence_true)
            arousal_acc = np.mean(all_arousal_preds == all_arousal_true)
            avg_loss = running_loss / total_samples

            # Create confusion matrices with explicit labels
            valence_conf = confusion_matrix(
                all_valence_true, 
                all_valence_preds,
                labels=np.arange(3)  # Explicitly specify 3 classes
            )
            arousal_conf = confusion_matrix(
                all_arousal_true, 
                all_arousal_preds,
                labels=np.arange(3)  # Explicitly specify 3 classes
            )
            
            val_metrics = {
                'loss': avg_loss,
                'valence_acc': float(valence_acc),
                'arousal_acc': float(arousal_acc),
                'valence_preds': all_valence_preds,
                'valence_true': all_valence_true,
                'arousal_preds': all_arousal_preds,
                'arousal_true': all_arousal_true,
                'valence_val_confusion': valence_conf,
                'arousal_val_confusion': arousal_conf
            }
        else:
            all_state_preds = np.array(all_state_preds)
            all_state_true = np.array(all_state_true)
            
            # Calculate accuracy
            state_acc = np.mean(all_state_preds == all_state_true)
            avg_loss = running_loss / total_samples

            # Create confusion matrix with explicit labels for 4 states
            state_conf = confusion_matrix(
                all_state_true, 
                all_state_preds,
                labels=np.arange(4)  # Explicitly specify 4 classes
            )
            
            val_metrics = {
                'loss': avg_loss,
                'state_acc': float(state_acc),
                'state_preds': all_state_preds,
                'state_true': all_state_true,
                'state_val_confusion': state_conf
            }

        return val_metrics
    
    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']