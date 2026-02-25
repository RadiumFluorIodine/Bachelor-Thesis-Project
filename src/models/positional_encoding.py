import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """
    Standard Sinusoidal Positional Encoding.
    Adds temporal position information to embeddings since attention is permutation-invariant.
    
    Formula:
        PE(pos, 2i)   = sin(pos / T^(2i/d))
        PE(pos, 2i+1) = cos(pos / T^(2i/d))
    
    Args:
        d (int): Dimension of the embedding (d_model).
        T (int): Period or maximum sequence length (default: 1000).
        repeat (int): Number of times to repeat the encoding (matches n_head).
        offset (int): Offset for the position indices.
    """
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        # Create positional encoding matrix
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True

        # Safety check
        if self.denom.device != batch_positions.device:
            self.denom = self.denom.to(batch_positions.device)
            
        # batch_positions: B x T
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table