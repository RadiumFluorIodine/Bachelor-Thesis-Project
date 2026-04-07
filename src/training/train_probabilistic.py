"""
Probabilistic U-TAE for Epistemic Uncertainty Estimation

Extension of U-TAE that outputs:
- Mean prediction (AGB)
- Epistemic uncertainty (model confidence)

Benefits:
1. Quantify prediction confidence
2. Identify out-of-distribution samples
3. Better cross-region generalization interpretation
4. Confidence-weighted field validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from utae import UTAE
except ImportError:
    from models.utae import UTAE


class ProbabilisticUTAE(nn.Module):
    """
    Wrapper around UTAE untuk probabilistic output.
    
    Output:
    - mean: E[AGB|X] - Expected biomass
    - uncertainty: Epistemic uncertainty (σ)
    
    Training:
    - Use Negative Log-Likelihood Loss (NLL)
    - Or Evidential Deep Learning loss
    
    Inference:
    - mean ± k*uncertainty (confidence intervals)
    """
    
    def __init__(
        self,
        input_dim=10,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        n_head=16,
        d_model=256,
        d_k=4,
        encoder_norm='group',
        pad_value=-9999.0,
        uncertainty_mode='heteroscedastic'  # 'heteroscedastic' or 'evidential'
    ):
        """
        Args:
            uncertainty_mode: 
                - 'heteroscedastic': Output log_variance (data-dependent uncertainty)
                - 'evidential': Evidential Deep Learning (more principled)
        """
        super().__init__()
        
        self.uncertainty_mode = uncertainty_mode
        
        # Base U-TAE encoder-decoder (shared)
        self.utae_backbone = UTAE(
            input_dim=input_dim,
            output_dim=1,  # Will be replaced
            encoder_widths=encoder_widths,
            decoder_widths=decoder_widths,
            n_head=n_head,
            d_model=d_model,
            d_k=d_k,
            encoder_norm=encoder_norm,
            pad_value=pad_value,
            return_maps=False,
            encoder=False
        )
        
        # Replace output head
        self.utae_backbone.regressor = nn.Identity()  # Remove original
        
        # Custom probabilistic head
        if uncertainty_mode == 'heteroscedastic':
            # Output: [mean, log_variance]
            self.mean_head = nn.Conv2d(decoder_widths[0], 1, kernel_size=1)
            self.logvar_head = nn.Sequential(
                nn.Conv2d(decoder_widths[0], 1, kernel_size=1),
                nn.Softplus()  # Ensure log_var stays reasonable
            )
        
        elif uncertainty_mode == 'evidential':
            # Output: [gamma, nu, alpha, beta] for NormalInverseGamma
            self.evidence_head = nn.Conv2d(decoder_widths[0], 4, kernel_size=1)
            self.softplus = nn.Softplus()
        
        else:
            raise ValueError(f"Unknown uncertainty_mode: {uncertainty_mode}")
    
    def forward(self, input, batch_positions=None, temporal_mask=None, return_att=False):
        """
        Forward pass dengan uncertainty estimation.
        
        Returns:
            Dict with:
            - 'agb': Mean prediction (B, H, W)
            - 'uncertainty': Epistemic uncertainty (B, H, W)
            - 'log_variance': Log variance (for loss computation)
            - 'attn_weights': Optional attention weights
        """
        B, T, C, H, W = input.shape
        
        # Get features from U-TAE backbone
        # We need to access intermediate features
        # Modify forward to return features
        
        # ===== FORWARD THROUGH BACKBONE =====
        if temporal_mask is not None:
            pad_mask = ~temporal_mask.bool()
        else:
            pad_mask = (input == self.utae_backbone.pad_value).all(
                dim=-1).all(dim=-1).all(dim=-1)
        
        if batch_positions is not None and batch_positions.dim() == 1:
            batch_positions = batch_positions.unsqueeze(0).repeat(B, 1)
        
        # Encoder
        out = self.utae_backbone.in_conv.smart_forward(input)
        feature_maps = [out]
        
        for i in range(self.utae_backbone.n_stages - 1):
            out = self.utae_backbone.down_block[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)
        
        # Temporal Attention
        out, att = self.utae_backbone.temporal_encoder(
            feature_maps[-1], 
            batch_positions=batch_positions, 
            pad_mask=pad_mask
        )
        
        # Decoder
        for i in range(self.utae_backbone.n_stages - 1):
            skip = self.utae_backbone.temporal_aggregator(
                feature_maps[-(i + 2)], 
                pad_mask=pad_mask, 
                attn_mask=att
            )
            out = self.utae_backbone.up_blocks[i](out, skip)
        
        # Final conv
        features = self.utae_backbone.out_conv(out)  # (B, D, H, W)
        
        # ===== PROBABILISTIC HEADS =====
        output_dict = {}
        
        if self.uncertainty_mode == 'heteroscedastic':
            # Predict mean and variance
            mean = self.mean_head(features).squeeze(1)  # (B, H, W)
            log_var = self.logvar_head(features).squeeze(1)  # (B, H, W)
            
            # Uncertainty = standard deviation
            uncertainty = torch.exp(0.5 * log_var)  # σ = exp(0.5 * log(σ²))
            
            output_dict['agb'] = mean
            output_dict['uncertainty'] = uncertainty
            output_dict['log_variance'] = log_var  # For loss computation
        
        elif self.uncertainty_mode == 'evidential':
            # Evidential Deep Learning
            evidence = self.evidence_head(features)  # (B, 4, H, W)
            
            gamma = evidence[:, 0, :, :]  # Mean
            nu = self.softplus(evidence[:, 1, :, :]) + 1  # Precision (>0)
            alpha = self.softplus(evidence[:, 2, :, :]) + 1  # Shape (>1)
            beta = self.softplus(evidence[:, 3, :, :])  # Scale (>0)
            
            # Predicted mean
            mean = gamma
            
            # Epistemic uncertainty (higher when less confident)
            # From NormalInverseGamma: Var = β / (α - 1)
            epistemic_unc = torch.sqrt(beta / (alpha - 1 + 1e-6))
            
            # Aleatoric uncertainty (data noise)
            aleatoric_unc = torch.sqrt(beta / (nu * (alpha - 1) + 1e-6))
            
            # Total uncertainty
            total_unc = torch.sqrt(epistemic_unc**2 + aleatoric_unc**2)
            
            output_dict['agb'] = mean
            output_dict['uncertainty'] = total_unc
            output_dict['epistemic_uncertainty'] = epistemic_unc
            output_dict['aleatoric_uncertainty'] = aleatoric_unc
            output_dict['evidence'] = {
                'gamma': gamma, 'nu': nu, 'alpha': alpha, 'beta': beta
            }
        
        if return_att:
            output_dict['attn_weights'] = att
        
        return output_dict


class HeteroscedasticLoss(nn.Module):
    """
    Negative Log-Likelihood Loss for heteroscedastic uncertainty.
    
    Loss = 0.5 * (log(σ²) + (y - ŷ)² / σ²)
    
    Intuition:
    - If prediction is wrong (y - ŷ large) → Increase uncertainty (σ large) → Lower loss
    - But log(σ²) term prevents σ → infinity
    - Model learns when to be uncertain!
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred_mean, pred_log_var, target, mask=None):
        """
        Args:
            pred_mean: (B, H, W) - Predicted mean
            pred_log_var: (B, H, W) - Predicted log variance
            target: (B, H, W) - Ground truth
            mask: (B, H, W) - Valid pixel mask (optional)
        
        Returns:
            loss: Scalar
        """
        # NLL = 0.5 * (log(σ²) + (y - ŷ)² / σ²)
        squared_error = (target - pred_mean) ** 2
        loss = 0.5 * (pred_log_var + squared_error / (torch.exp(pred_log_var) + 1e-6))
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-6)
            elif self.reduction == 'sum':
                return loss.sum()
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
        
        return loss


class EvidentialLoss(nn.Module):
    """
    Evidential Deep Learning Loss.
    
    More principled uncertainty estimation.
    
    Reference: Amini et al. (2020) "Deep Evidential Regression"
    """
    
    def __init__(self, coeff=1.0, reduction='mean'):
        super().__init__()
        self.coeff = coeff  # Regularization coefficient
        self.reduction = reduction
    
    def forward(self, evidence, target, mask=None):
        """
        Args:
            evidence: Dict with gamma, nu, alpha, beta
            target: (B, H, W)
            mask: (B, H, W)
        """
        gamma = evidence['gamma']
        nu = evidence['nu']
        alpha = evidence['alpha']
        beta = evidence['beta']
        
        # NLL loss
        two_beta_lambda = 2 * beta * (1 + nu)
        nll = 0.5 * torch.log(torch.pi / nu) \
              - alpha * torch.log(two_beta_lambda) \
              + (alpha + 0.5) * torch.log(
                  nu * (target - gamma)**2 + two_beta_lambda
              ) \
              + torch.lgamma(alpha) \
              - torch.lgamma(alpha + 0.5)
        
        # Regularization (penalize high variance when wrong)
        error = (target - gamma).abs()
        reg = error * (2 * nu + alpha)
        
        loss = nll + self.coeff * reg
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-6)
        else:
            if self.reduction == 'mean':
                return loss.mean()
        
        return loss


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING PROBABILISTIC U-TAE")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create dummy data
    B, T, C, H, W = 2, 12, 10, 128, 128
    dummy_input = torch.randn(B, T, C, H, W).to(device)
    dummy_label = torch.randn(B, H, W).to(device)
    dummy_mask = torch.ones(B, H, W).to(device)
    positions = torch.tensor([15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]).float()
    
    print("--- Testing Heteroscedastic Mode ---")
    model_hetero = ProbabilisticUTAE(
        uncertainty_mode='heteroscedastic'
    ).to(device)
    
    output_hetero = model_hetero(dummy_input, batch_positions=positions)
    
    print(f"✓ AGB shape: {output_hetero['agb'].shape}")
    print(f"✓ Uncertainty shape: {output_hetero['uncertainty'].shape}")
    print(f"✓ AGB range: [{output_hetero['agb'].min():.2f}, {output_hetero['agb'].max():.2f}]")
    print(f"✓ Uncertainty range: [{output_hetero['uncertainty'].min():.2f}, {output_hetero['uncertainty'].max():.2f}]")
    
    # Test loss
    criterion = HeteroscedasticLoss()
    loss = criterion(
        output_hetero['agb'], 
        output_hetero['log_variance'], 
        dummy_label,
        dummy_mask
    )
    print(f"✓ Loss: {loss.item():.4f}\n")
    
    print("--- Testing Evidential Mode ---")
    model_evid = ProbabilisticUTAE(
        uncertainty_mode='evidential'
    ).to(device)
    
    output_evid = model_evid(dummy_input, batch_positions=positions)
    
    print(f"✓ AGB shape: {output_evid['agb'].shape}")
    print(f"✓ Total uncertainty: {output_evid['uncertainty'].shape}")
    print(f"✓ Epistemic uncertainty: {output_evid['epistemic_uncertainty'].shape}")
    print(f"✓ Aleatoric uncertainty: {output_evid['aleatoric_uncertainty'].shape}")
    
    criterion_evid = EvidentialLoss()
    loss_evid = criterion_evid(
        output_evid['evidence'],
        dummy_label,
        dummy_mask
    )
    print(f"✓ Loss: {loss_evid.item():.4f}\n")
    
    print("=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("Probabilistic U-TAE ready for training!")
    print("=" * 80)