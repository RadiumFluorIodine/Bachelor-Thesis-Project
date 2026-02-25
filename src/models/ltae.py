"""
LTAE (Lightweight Temporal Attention Encoder)
Standalone Implementation based on Garnot & Landrieu (2021)
Ref: https://github.com/VSainteuf/utae-paps

Features:
1. Multi-Head Attention (Modified for Temporal Pooling)
2. LTAE2d Wrapper (Spatial-aware processing)
"""

import copy
import numpy as np
import torch
import torch.nn as nn

try:
    from positional_encoding import PositionalEncoder
except ImportError:
    try:
        from models.positional_encoding import PositionalEncoder
    except ImportError:
        from src.models.positional_encoding import PositionalEncoder


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    Computes attention weights based on Query-Key similarity and applies them to Values.
    
    Formula:
        Attention(Q, K, V) = softmax((Q . K^T) / sqrt(d_k)) . V
    
    Args:
        temperature (float): Scaling factor (usually sqrt(d_k)).
        attn_dropout (float): Dropout probability for attention weights.

    Forward Args:
        q: (n*b, 1, d_k) - Query vectors
        k: (n*b, T, d_k) - Key vectors
        v: (n*b, T, d_v) - Value vectors
        pad_mask: (n*b, T) - Padding mask
    
    Returns:
        output: (n*b, 1, d_v) - Attention output
        attn: (n*b, 1, T) - Attention weights
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        # q: (n*b) x 1 x d_k
        # k: (n*b) x T x d_k
        # v: (n*b) x T x d_v
        
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) # (n*b) x 1 x T
        attn = attn / self.temperature
        
        if pad_mask is not None:
            # pad_mask: (n*b) x T
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, v) # (n*b) x 1 x d_v

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module for L-TAE.
    Uses a learnable 'Master Query' instead of an input-derived query to summarize temporal features.
    
    Formula:
        Head_i = Attention(Q_master, K_i, V_i)
        Output = Concat(Head_1, ..., Head_h)
        
    Where Q_master is a learned parameter initialized with normal distribution.
    
    Args:
        n_head (int): Number of attention heads (G in paper).
        d_k (int): Dimension of key/query per head.
        d_in (int): Dimension of input features.
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        # Query is a learnable parameter (Global Temporal Query)
        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, return_comp=False):
        # v: (b) x T x d_in
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        # 1. Prepare Query (Repeat for batch)
        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(
            -1, d_k
        )  # (n_head*b) x d_k

        # 2. Prepare Key
        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n_head*b) x T x d_k

        # 3. Prepare Mask
        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # (n_head*b) x T

        # 4. Prepare Value (Split channels per head)
        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        ) # (n_head*b) x T x (d_in/n_head)

        # 5. Attention
        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        
        # Reshape Attention Weights
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2) # n_head x b x T

        # Reshape Output
        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2) # n_head x b x (d_in/n_head)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn



class LTAE(nn.Module):
    """
    Lightweight Temporal Attention Encoder (L-TAE).
    Encodes a sequence of images into a single feature map using temporal self-attention.
    Also produces temporal attention masks for use in the decoder.
    
    Formula:
        1. Encoding:  E = PosEnc(Input)
        2. Attention: a_g = Attention(Q_master, K(E), V(E))
        3. Features:  Out = MLP(Concat(a_g . V(E)))
        
    Args:
        in_channels (int): Number of input channels.
        n_head (int): Number of attention heads (G).
        d_k (int): Dimension of the key and query vectors.
        mlp (List[int]): Widths of the MLP layers [d_model, ..., output_dim].
        dropout (float): Dropout rate.
        d_model (int): Input embedding dimension.
        n_temporal (int): Max sequence length for positional encoding.
    """
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=[256, 128],
        dropout=0.2,
        d_model=256,
        n_temporal=1000,
        return_att=True,
        positional_encoding=True,
    ):
        super(LTAE, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        # Input projection if needed
        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        
        # Adjust MLP input dimension to match d_model
        if self.mlp[0] != self.d_model:
             self.mlp[0] = self.d_model

        # Positional Encoding
        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=n_temporal, repeat=n_head
            )
        else:
            self.positional_encoder = None

        # Multi-Head Attention
        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )

        # Channel checker 
        if self.in_channels % n_head == 0:
            norm_groups_in = n_head
        elif self.in_channels % 2 == 0:
            norm_groups_in = 2 
        else:
            norm_groups_in = 1

        # Norm Layers
        self.in_norm = nn.GroupNorm(
            num_groups=norm_groups_in, 
            num_channels=self.in_channels,
        )
        
        
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        # Feed Forward Network (MLP)
        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        """
        Args:
            x: Input tensor (Batch*H*W, T, C) or (Batch, T, C, H, W)
            batch_positions: Temporal positions (Batch, T).
            pad_mask: Padding mask (Batch, T).
        
        Returns:
            out: Temporally collapsed features (Batch, C_out, H, W).
            attn: Attention weights (n_head, Batch, T, H, W).
        """
        # Handle input dimensions
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )
            pad_mask = (
                pad_mask.permute(0,2,3,1).contiguous().view(sz_b * h * w, seq_len)
            )

        # 1. Input Normalization
        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        # 2. Input Projection (1x1 Conv)
        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        # 3. Positional Encoding
        if self.positional_encoder is not None:
            if batch_positions is None:
                batch_positions = torch.arange(
                    1, seq_len + 1, device=out.device, dtype=torch.float32
                ).unsqueeze(0).repeat(sz_b, 1)

            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        # 4. Attention Heads
        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        # 5. Concatenate Heads & MLP
        out = out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        
        # MLP Processing
        out = self.dropout(self.mlp(out))
        
        # Output Normalization
        out = self.out_norm(out) if self.out_norm is not None else out
        
        # Reshape Output if input was 5D
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)
        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(0, 1, 4, 2, 3)

        if self.return_att:
            return out, attn
        else:
            return out


# Unit Test Code
if __name__ == "__main__":
    import sys
    import os
    from torch.utils.data import DataLoader

    print("=" * 80)
    print("LTAE Unit Test")
    print("=" * 80)

    # Setup Path
    current_file_path = os.path.abspath(__file__)
    src_models_dir = os.path.dirname(current_file_path)
    src_dir = os.path.dirname(src_models_dir)
    root_dir = os.path.dirname(src_dir)

    sys.path.append(root_dir)
    
    try:
        from src.data.dataset import BiomassDataset, collate_fn_biomass
    except ImportError as e:
        print("Error Import !")
        print(f"Detail: {e}")
        exit()

    # Config and Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Running on: {device}")

    # Path to data processed
    data_path = os.path.join(root_dir, 'data', 'processed', 'lampung', 'version_2')

    # Load Data
    print(f"\nLoading 1 Batch from: {data_path}")
    try:
        dataset = BiomassDataset(root_dir=data_path, mode='train')
        
        # Ambil 1 batch (Batch size = 2)
        loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_biomass)
        batch = next(iter(loader))
        
        images = batch['image'].to(device)          # (B, T, C, H, W)
        positions = batch['batch_positions'].to(device) # (B, T)

        if positions.dim() == 1:
            positions = positions.unsqueeze(0).repeat(images.size(0), 1)
        
        # Ambil dimensi otomatis dari data asli
        B, T, C, H, W = images.shape
        
        print(f"✅ Data Loaded!")
        print(f"   Input Shape: {images.shape}")
        print(f"   Positions  : {positions.shape}")
        print(f"   Channels   : {C} (Real Sentinel-2 Bands)")
        
    except FileNotFoundError:
        print(f"❌ Data tidak ditemukan di {data_path}")
        print("   Pastikan Anda sudah menjalankan preprocessing.")
        exit()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        exit()

    # Initialize and Run Model
    print("\nInitializing LTAE...")

    try:
        d_model = 128
        model = LTAE(
            in_channels=C,      
            d_model=d_model, 
            n_head=4, 
            n_temporal=T,       
            d_k=4
        ).to(device)
        
        print("⚡ Running Forward Pass...")
        
        # Forward Pass
        out = model(images, batch_positions=positions)
        
        # Output check
        # If return tuple (out, attn), take first element
        if isinstance(out, tuple):
            out = out[0]
            
        print("\n✅ TEST PASSED!")
        print("-" * 30)
        print(f"Input  : {images.shape} (Time dimension T={T})")
        print(f"Output : {out.shape}     (Time collapsed, Features={d_model})")
        
        # Validasi: Dimensi output harus (B, d_model, H, W)
        assert out.shape == (B, d_model, H, W), "❌ Dimensi Output salah!"
        print("   -> Dimensi output valid.")
        
    except Exception as e:
        print(f"\n❌ MODEL ERROR: {e}")
        import traceback
        traceback.print_exc()

    print("="*60)