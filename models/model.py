import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class HierarchicalEmotionModel(nn.Module):
    def __init__(
        self,
        input_dim=6,
        lstm_hidden_dim=128,
        lstm_layers=2,
        lstm_dropout=0.1,
        window_size=100,
        d_model=128,
        nhead=8,
        num_transformer_layers=2,
        dim_feedforward=512,
        transformer_dropout=0.1,
        num_classes=3,
        labeling_mode='dual'
    ):
        super().__init__()
        self.labeling_mode = labeling_mode
        
        self.window_size = window_size
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # LSTM for processing windows of sequence
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Linear layer to match transformer dimension
        self.lstm_to_transformer = nn.Linear(lstm_hidden_dim * 2, d_model)
        
        # Positional encoding for transformer
        self.pos_encoder = PositionalEncoding(d_model, dropout=transformer_dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_transformer_layers
        )
        
        # Global attention pooling
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        
        # Classification heads
        if labeling_mode == 'dual':
            # Dual-head for valence and arousal
            self.valence_head = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(transformer_dropout),
                nn.Linear(dim_feedforward, num_classes)
            )
            
            self.arousal_head = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(transformer_dropout),
                nn.Linear(dim_feedforward, num_classes)
            )
        else:
            # Single head for state classification
            self.state_head = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(transformer_dropout),
                nn.Linear(dim_feedforward, 4)  # 4 classes for states
            )
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly."""
        def _init_layer(layer):
            if isinstance(layer, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LSTM):
                for name, param in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
        
        self.apply(_init_layer)
    
    def process_sequence_windows(self, x):
        """Process sequence in windows using LSTM."""
        batch_size, seq_len, feat_dim = x.shape
        
        # Calculate number of windows
        num_windows = (seq_len + self.window_size - 1) // self.window_size
        
        # Pad sequence if needed
        if seq_len % self.window_size != 0:
            pad_len = num_windows * self.window_size - seq_len
            padding = torch.zeros(batch_size, pad_len, feat_dim, device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        # Reshape into windows
        x = x.view(batch_size * num_windows, self.window_size, feat_dim)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(x)  # [batch*windows, window_size, lstm_hidden*2]
        
        # Get last output for each window
        lstm_out = lstm_out[:, -1]  # [batch*windows, lstm_hidden*2]
        
        # Reshape back to batch dimension
        lstm_out = lstm_out.view(batch_size, num_windows, -1)  # [batch, windows, lstm_hidden*2]
        
        return lstm_out
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Input tensor [batch_size, seq_length, input_dim]
            attention_mask: Optional mask for transformer
        """
        # Process sequence windows with LSTM
        lstm_out = self.process_sequence_windows(x)  # [batch, windows, lstm_hidden*2]
        
        # Project to transformer dimension
        transformer_input = self.lstm_to_transformer(lstm_out)  # [batch, windows, d_model]
        
        # Add positional encoding
        transformer_input = self.pos_encoder(transformer_input)
        
        # Transform through encoder
        transformer_out = self.transformer_encoder(
            transformer_input,
            src_key_padding_mask=attention_mask
        )  # [batch, windows, d_model]
        
        # Apply attention pooling
        attention_weights = self.attention(transformer_out)  # [batch, windows, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended = torch.sum(attention_weights * transformer_out, dim=1)  # [batch, d_model]
        
        if self.labeling_mode == 'dual':
            valence_logits = self.valence_head(attended)
            arousal_logits = self.arousal_head(attended)
            return valence_logits, arousal_logits
        else:
            state_logits = self.state_head(attended)
            return state_logits