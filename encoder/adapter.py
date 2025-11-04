import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    """
    2-layer MLP adapter with ReLU activation and k-based downsampling.
    Downsamples by concatenating every k frames.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, ds_rate=5):
        super().__init__()
        self.k = ds_rate
        self.fc1 = nn.Linear(input_dim * self.k, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        
        remainder = seq_len % self.k
        if remainder > 0:
            pad_size = self.k - remainder
            x = F.pad(x, (0, 0, 0, pad_size), value=0.0)
            seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

