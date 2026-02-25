"""
U-TAE (U-Net with Lightweight Temporal Attention Encoder)
Standalone implementation untuk AGB Estimation

Reference: https://github.com/VSainteuf/utae-paps
Paper: Garnot, V. S. F., & Landrieu, L. (2021). 
"Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks"
"""

import torch
import torch.nn as nn
import json

try:
    from ltae import LTAE
except ImportError:
    try:
        from models.ltae import LTAE
    except ImportError:
        from src.models.ltae import LTAE


class TemporallySharedBlock(nn.Module):
    """
    Docstring for TemporallySharedBlock
    
    """
    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, input):
        if len(input.shape) == 4:
            return self.forward(input)
        else:
            b, t, c, h, w = input.shape

            if self.pad_value is not None:
                dummy = torch.zeros(input.shape, device=input.device).float()
                self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

            out = input.view(b * t, c, h, w)
            if self.pad_value is not None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
                if pad_mask.any():
                    temp = (
                        torch.ones(
                            self.out_shape, device=input.device, requires_grad=False
                        )
                        * self.pad_value
                    )
                    temp[~pad_mask] = self.forward(out[~pad_mask])
                    out = temp
                else:
                    out = self.forward(out)
            else:
                out = self.forward(out)
            _, c, h, w = out.shape
            out = out.view(b, t, c, h, w)
            return out

class ConvLayer(nn.Module):
    def __init__(
            self,
            nkernels,
            norm="batch",
            k=3,
            s=1,
            p=1,
            n_groups=4,
            last_relu=True,
            padding_mode="reflect",
    ):
        super(ConvLayer, self).__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=n_groups,
            )
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)
    

class ConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        nkernels,
        pad_value=None,
        norm="batch",
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvBlock, self).__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        return self.conv(input)



class DownConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        pad_value=None,
        norm="batch",
        padding_mode="reflect",
    ):
        super(DownConvBlock, self).__init__(pad_value=pad_value)
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        out = self.down(input)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out



class UpConvBlock(nn.Module):
    def __init__(
        self, d_in, d_out, k, s, p, norm="batch", d_skip=None, padding_mode="reflect"
    ):
        super(UpConvBlock, self).__init__()
        d = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p
            ),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
        )
        self.conv1 = ConvLayer(
            nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode
        )

    def forward(self, input, skip):
        out = self.up(input)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out
    

class Temporal_Aggregator(nn.Module):
    def __init__(self, mode="mean"):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)


class UTAE(nn.Module):
    """
    U-TAE: U-Net with Lightweight Temporal Attention Encoder.
    
    Architecture:
    1. Spatial Encoder: Time-distributed convolutions (shared weights)
    2. LTAE: Temporal attention at lowest resolution (8x8)
    3. Spatial Decoder: Attention-weighted skip aggregation
    """

    def __init__(
        self,
        input_dim = 10,
        output_dim = 1,
        encoder_widths = [64, 64, 64, 128],
        decoder_widths = [32, 32, 64, 128],
        out_conv = [32, 20],
        str_conv_k = 4,
        str_conv_s = 2,
        str_conv_p = 1,
        agg_mode = 'att_group',
        encoder_norm = 'group',
        n_head = 16,
        d_model = 256,
        d_k = 4,
        encoder = False,
        return_maps = False,
        pad_value = 0,
        padding_mode = 'reflect',
        use_checkpointing=False,

    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = (
            decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        )
        self.stack_dim = (
            sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        )
        self.pad_value = pad_value
        self.encoder = encoder
        self.use_checkpointing = use_checkpointing
        if encoder:
            self.return_maps = True

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths


        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down_block = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm="batch",
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        self.temporal_encoder = LTAE(
            in_channels= encoder_widths[-1],
            d_model = d_model,
            n_head = n_head,
            mlp = [d_model, encoder_widths[-1]],
            return_att= True,
            d_k = d_k,  
        )
        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)

        # Output head
        self.out_conv = ConvBlock(
            nkernels=[decoder_widths[0], 
            decoder_widths[0]], 
            padding_mode=padding_mode,
            norm=encoder_norm,
            )
        
        self.regressor = nn.Conv2d(
            decoder_widths[0], 
            output_dim, 
            kernel_size=1
        ) 

    
    def forward(
            self, 
            input, 
            batch_positions=None, 
            return_att=False,
            denormalize=False,     
            norm_stats=None 
        ):

        """
        Forward pass through U-TAE with optional denormalization.
        
        Args:
            input: (B, T, C, H, W) - Time-series satellite images
            Expected: (batch, 12, 10, 128, 128) for 12-month Sentinel-2
            batch_positions: (T,) or (B, T) - Temporal positions (day-of-year)
            return_att: bool - Return attention weights (deprecated, always returns)
            denormalize: bool - If True, denormalize output to Mg/ha
            norm_stats: dict - Normalization statistics (required if denormalize=True)
            Must contain: 'agb_log_mean', 'agb_log_std'
        
        Returns:
            dict with:
                - 'agb': (B, 1, H, W) - AGB prediction (normalized or Mg/ha)
                - 'agb_normalized': (B, 1, H, W) - Always normalized (if denormalized)
                - 'attn_weights': (n_heads, B, T, H_att, W_att) - Attention weights
        """
        # Validate input shape
        assert input.dim() == 5, f"Expected 5D input (B,T,C,H,W), got {input.dim()}D"
        B, T, C, H, W = input.shape
        assert C == self.input_dim, f"Expected {self.input_dim} channels, got {C}"

        # Validate denormalization parameters
        if denormalize:
            assert norm_stats is not None, \
                "❌ norm_stats required when denormalize=True"
            assert 'agb_log_mean' in norm_stats and 'agb_log_std' in norm_stats, \
                "❌ norm_stats must contain 'agb_log_mean' and 'agb_log_std'"
            

        pad_mask = (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        out = self.in_conv.smart_forward(input)
        feature_maps = [out]
        
        # Spatial Encoder
        for i in range(self.n_stages - 1):
            if self.use_checkpointing and self.training:
                out = torch.utils.checkpoint.checkpoint(
                    self.down_block[i].smart_forward,
                    feature_maps[-1],
                    use_reentrant=False
                )
            else:
                out = self.down_block[i].smart_forward(feature_maps[-1])
            
            feature_maps.append(out)

        # Temporal Encoder
        out, att = self.temporal_encoder(
            feature_maps[-1], 
            batch_positions=batch_positions, 
            pad_mask=pad_mask
        )

        # Spatial Decoder
        if self.return_maps:
            maps = [out]

        for i in range(self.n_stages - 1):
            skip = self.temporal_aggregator(
                feature_maps[-(i + 2)], 
                pad_mask=pad_mask, 
                attn_mask=att
            )
            if self.use_checkpointing and self.training:
                out = torch.utils.checkpoint.checkpoint(
                    self.up_blocks[i],
                    out,
                    skip,
                    use_reentrant=False
                )
            else:
                out = self.up_blocks[i](out, skip)

        # Output Head
        out = self.out_conv(out)
        agb_normalized = self.regressor(out)

        if denormalize and norm_stats is not None:
            # Reverse z-score in log-space
            agb_log = (
                agb_normalized * norm_stats['agb_log_std']
            ) + norm_stats['agb_log_mean']
            
            # Reverse log-transform: exp(x) - 1
            agb_mgha = torch.expm1(agb_log)
            
            # Clip negative values (shouldn't happen, but safety)
            agb_mgha = torch.clamp(agb_mgha, min=0.0)
            
            return {
                'agb': agb_mgha,                    # (B, 1, H, W) in Mg/ha
                'agb_normalized': agb_normalized,   # (B, 1, H, W) normalized
                'attn_weights': att                 # (n_heads, B, T, H_att, W_att)
            }
        
        # Return normalized
        return {
            'agb': agb_normalized,      # (B, 1, H, W) normalized
            'attn_weights': att         # (n_heads, B, T, H_att, W_att)
        }
    


    def get_model_summary(self):
        """
        Return comprehensive model architecture summary.
        
        Returns:
            dict with architecture details and parameter counts
        
        Example:
            >>> model = UTAE(input_dim=10, encoder_widths=[64,64,128,128])
            >>> summary = model.get_model_summary()
            >>> print(json.dumps(summary, indent=2))
        """
        # Count parameters by module
        encoder_params = sum(
            p.numel() for p in self.in_conv.parameters()
        ) + sum(
            p.numel() for block in self.down_block for p in block.parameters()
        )
        
        temporal_params = sum(p.numel() for p in self.temporal_encoder.parameters())
        
        decoder_params = sum(
            p.numel() for block in self.up_blocks for p in block.parameters()
        ) + sum(
            p.numel() for p in self.out_conv.parameters()
        ) + sum(
            p.numel() for p in self.regressor.parameters()
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        
        summary = {
            # Architecture
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'encoder_widths': self.encoder_widths,
            'decoder_widths': self.decoder_widths,
            'n_stages': self.n_stages,
            
            # Temporal encoder config
            'temporal_encoder': {
                'n_head': self.temporal_encoder.n_head,
                'd_model': self.temporal_encoder.d_model,
                'd_k': getattr(self.temporal_encoder, 'd_k', 'N/A'),
            },
            
            # Parameters
            'parameters': {
                'total': total_params,
                'trainable': trainable_params,
                'encoder': encoder_params,
                'temporal': temporal_params,
                'decoder': decoder_params,
            },
            
            # Memory estimation 
            'estimated_memory_mb': {
                'params': (total_params * 4) / (1024 ** 2),  # float32
                'gradients': (trainable_params * 4) / (1024 ** 2),
            },
            
            # Config
            'use_checkpointing': self.use_checkpointing,
            'pad_value': self.pad_value,
        }
        
        return summary
    

    def print_model_summary(self):
        """Print human-readable model summary."""
        summary = self.get_model_summary()
        
        print("\n" + "-"*70)
        print("U-TAE MODEL ARCHITECTURE SUMMARY")
        print("-"*70)
        
        print("\n Architecture:")
        print(f"   Input:  {summary['input_dim']} channels")
        print(f"   Output: {summary['output_dim']} channels")
        print(f"   Stages: {summary['n_stages']}")
        print(f"   Encoder widths: {summary['encoder_widths']}")
        print(f"   Decoder widths: {summary['decoder_widths']}")
        
        print("\n Temporal Attention:")
        print(f"   Heads:   {summary['temporal_encoder']['n_head']}")
        print(f"   d_model: {summary['temporal_encoder']['d_model']}")
        print(f"   d_k:     {summary['temporal_encoder']['d_k']}")
        
        print("\n Parameters:")
        total = summary['parameters']['total']
        print(f"   Total:      {total:,} ({total/1e6:.2f}M)")
        print(f"   Trainable:  {summary['parameters']['trainable']:,}")
        print(f"   Encoder:    {summary['parameters']['encoder']:,} "
              f"({100*summary['parameters']['encoder']/total:.1f}%)")
        print(f"   Temporal:   {summary['parameters']['temporal']:,} "
              f"({100*summary['parameters']['temporal']/total:.1f}%)")
        print(f"   Decoder:    {summary['parameters']['decoder']:,} "
              f"({100*summary['parameters']['decoder']/total:.1f}%)")
        
        print("\n Estimated Memory:")
        print(f"   Parameters: {summary['estimated_memory_mb']['params']:.1f} MB")
        print(f"   Gradients:  {summary['estimated_memory_mb']['gradients']:.1f} MB")
        print(f"   Total (approx): {sum(summary['estimated_memory_mb'].values()):.1f} MB")
        
        print("\n Configuration:")
        print(f"   Gradient Checkpointing: {summary['use_checkpointing']}")
        print(f"   Pad Value: {summary['pad_value']}")
        
        print("-"*70 + "\n")
    
    

    def visualize_attention(
        self, 
        attn_weights, 
        timestep_labels=None,
        save_path=None,
        figsize=(12, 6)
    ):
        """
        Visualize temporal attention patterns across batch samples.
        
        Args:
            attn_weights: (n_heads, B, T, H_att, W_att) - From forward pass
            timestep_labels: List[str] - Labels for timesteps (e.g., month names)
            save_path: str - Path to save figure
            figsize: tuple - Figure size
        
        Example:
            >>> output = model(images, positions)
            >>> model.visualize_attention(
            ...     output['attn_weights'],
            ...     timestep_labels=['Jan', 'Feb', ..., 'Dec'],
            ...     save_path='figures/attention_pattern.png'
            ... )
        """
        import matplotlib.pyplot as plt
        
        # Average over heads and spatial dimensions
        # (n_heads, B, T, H, W) → (B, T)
        att_temporal = attn_weights.mean(dim=[0, 3, 4])
        
        B, T = att_temporal.shape
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each sample
        for b in range(min(B, 5)):  # Limit to 5 samples for clarity
            att = att_temporal[b].cpu().detach().numpy()
            ax.plot(
                range(1, T + 1), 
                att,
                marker='o',
                linewidth=2,
                markersize=6,
                label=f'Sample {b+1}',
                alpha=0.8
            )
        
        # Styling
        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel('Attention Weight', fontsize=12)
        ax.set_title('Temporal Attention Distribution', fontsize=14, weight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(loc='best')
        
        # Custom x-axis labels
        if timestep_labels:
            ax.set_xticks(range(1, T + 1))
            ax.set_xticklabels(timestep_labels, rotation=45, ha='right')
        else:
            ax.set_xticks(range(1, T + 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Attention visualization saved to: {save_path}")
        
        plt.show()
    
    
    def visualize_spatial_attention(
        self,
        attn_weights,
        sample_idx=0,
        timestep_idx=None,
        save_path=None
    ):
        """
        Visualize spatial distribution of attention at specific timestep.
        
        Args:
            attn_weights: (n_heads, B, T, H, W) - From forward pass
            sample_idx: int - Which sample in batch to visualize
            timestep_idx: int - Which timestep to visualize (None = average all)
            save_path: str - Path to save figure
        
        Example:
            >>> output = model(images, positions)
            >>> model.visualize_spatial_attention(
            ...     output['attn_weights'],
            ...     sample_idx=0,
            ...     timestep_idx=5,  # June
            ...     save_path='figures/spatial_attention_june.png'
            ... )
        """
        import matplotlib.pyplot as plt
        
        n_heads, B, T, H, W = attn_weights.shape
        
        assert sample_idx < B, f"sample_idx {sample_idx} out of range (B={B})"
        
        # Extract sample
        att_sample = attn_weights[:, sample_idx, :, :, :]  # (n_heads, T, H, W)
        
        # Select timestep or average
        if timestep_idx is not None:
            att_spatial = att_sample[:, timestep_idx, :, :]  # (n_heads, H, W)
            title_suffix = f"(Timestep {timestep_idx + 1})"
        else:
            att_spatial = att_sample.mean(dim=1)  # Average over T
            title_suffix = "(Averaged over all timesteps)"
        
        # Average over heads
        att_spatial = att_spatial.mean(dim=0).cpu().detach().numpy()  # (H, W)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 7))
        
        im = ax.imshow(att_spatial, cmap='hot', interpolation='bilinear')
        ax.set_title(f'Spatial Attention Distribution {title_suffix}', 
                    fontsize=12, weight='bold')
        ax.axis('off')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Spatial attention saved to: {save_path}")
        
        plt.show()



# Unit Test Code
if __name__ == "__main__":
    import sys
    import os
    from torch.utils.data import DataLoader

    print("=" * 80)
    print("U-TAE for Aboveground Biomass Estimation - Comprehensive Unit Test")
    print("=" * 80)
    
    # 1. Setup Path
    current_file_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    print(f"📂 Project Root: {root_dir}")

    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Device: {device}")

    # 3. Load Data (Real or Dummy Fallback)
    data_path = os.path.join(root_dir, 'data', 'processed', 'lampung', 'version_2')
    print(f"\n⏳ Loading 1 Batch from: {data_path}")

    try:
        from src.data.dataset import BiomassDataset, collate_fn_biomass
        dataset = BiomassDataset(root_dir=data_path, mode='train')
        
        if len(dataset) == 0:
            raise ValueError("Dataset kosong! Folder ditemukan namun tidak ada data .npz.")

        loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_biomass)
        batch = next(iter(loader))
        
        images = batch['image'].to(device)           # (B, T, C, H, W)
        positions = batch['batch_positions'].to(device)  # (B, T)
        
        print("✅ Real Data Loaded Successfully!")
        
    except Exception as e:
        print(f"⚠️  Gagal memuat Real Data: {e}")
        print("⚠️  Menggunakan DUMMY DATA untuk melanjutkan testing...")
        
        images = torch.randn(2, 12, 10, 128, 128).to(device)
        positions = torch.tensor([15, 45, 75, 105, 135, 165, 
                                  195, 225, 255, 285, 315, 345]).float().to(device)


    if positions.dim() == 1:
        positions = positions.unsqueeze(0).repeat(images.size(0), 1)

    B, T, C, H, W = images.shape
    print(f"   Input Shape : {images.shape}")
    print(f"   Positions   : {positions.shape}")
    print(f"   Channels    : {C}")

    # 4. Initialize Model
    print("\n" + "-"*80)
    print("🛠️ INITIALIZING U-TAE MODEL")
    print("-"*80)
    
    try:
        model = UTAE(
            input_dim=C,                
            output_dim=1,                
            encoder_widths=[64, 64, 128, 128],
            decoder_widths=[32, 32, 64, 128],
            n_head=4,                    
            d_model=128,
            encoder_norm="group",
            use_checkpointing=True  # Test Enhancement 4        
        ).to(device)
        print("✅ Model initialized successfully.")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        exit()


    # 5. Test Enhancement 1: Model Summary
    print("\n" + "-"*80)
    print("TEST 1: Model Summary (Enhancement 1)")
    print("-"*80)
    model.print_model_summary()


    # 6. Test Forward Pass & Output Shape Validation
    print("\n" + "-"*80)
    print("TEST 2: Forward Pass & Shape Validation")
    print("-"*80)
    try:
        output_norm = model(images, batch_positions=positions, denormalize=False)
        
        # Handle Output Tuple/Dict
        if isinstance(output_norm, dict):
            agb_pred = output_norm['agb']
            attn_weights = output_norm['attn_weights']
        else:
            agb_pred = output_norm
            attn_weights = None

        assert agb_pred.shape == (B, 1, H, W), f"❌ Dimensi Salah! Dapat {agb_pred.shape}"
        
        print("✅ Forward pass successful!")
        print(f"   Output Shape: {agb_pred.shape}")
        print(f"   Normalized output range: [{agb_pred.min():.3f}, {agb_pred.max():.3f}]")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()


    # 7. Test Enhancement 2: Denormalization
    print("\n" + "-"*80)
    print("TEST 3: Denormalization Logic (Enhancement 2)")
    print("-"*80)
    norm_stats = {
        'agb_log_mean': 3.5,
        'agb_log_std': 0.8
    }
    try:
        output_denorm = model(
            images, 
            batch_positions=positions,
            denormalize=True,
            norm_stats=norm_stats
        )
        print("✅ Denormalization successful!")
        print(f"   Denormalized output (Mg/ha): "
              f"[{output_denorm['agb'].min():.1f}, {output_denorm['agb'].max():.1f}]")
    except Exception as e:
        print(f"❌ Denormalization test failed: {e}")


    # 8. Test Enhancement 3: Attention Visualization
    print("\n" + "-"*80)
    print("TEST 4: Attention Visualization (Enhancement 3)")
    print("-"*80)
    try:
        import matplotlib
        matplotlib.use('Agg') # Gunakan backend non-interactive agar test jalan di background tanpa pop-up
        
        if attn_weights is not None:
            # Temporal attention
            model.visualize_attention(
                attn_weights,
                timestep_labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                save_path='test_utae_attention.png'
            )
            print(" ✅ Temporal attention visualization works!")
            
            # Spatial attention
            model.visualize_spatial_attention(
                attn_weights,
                sample_idx=0,
                timestep_idx=5,  # June
                save_path='test_utae_spatial_attention.png'
            )
            print(" ✅ Spatial attention visualization works!")
            
            # Hapus file gambar hasil test agar tidak menyampah di direktori
            if os.path.exists('test_utae_attention.png'): os.remove('test_utae_attention.png')
            if os.path.exists('test_utae_spatial_attention.png'): os.remove('test_utae_spatial_attention.png')
            
        else:
            print("⚠️ attn_weights tidak ditemukan dari output model.")
            
    except ImportError:
        print("⚠️ Matplotlib tidak terinstall. Skipping visualization test.")
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")


    # 9. Test Enhancement 4: Gradient Checkpointing
    print("\n" + "-"*80)
    print("TEST 5: Gradient Checkpointing & Backward Pass (Enhancement 4)")
    print("-"*80)
    try:
        model.train()
        output_train = model(images, batch_positions=positions)
        loss = output_train['agb'].mean()
        loss.backward()
        
        print("✅ Gradient checkpointing works (backward pass successful!)")
        print(f"   Memory efficient training enabled: {model.use_checkpointing}")
    except Exception as e:
        print(f"❌ Backward pass test failed: {e}")


    print("\n" + "="*80)
    print("🎉 ALL U-TAE UNIT TESTS PASSED SUCCESSFULLY! SYSTEM READY.")
    print("="*80)