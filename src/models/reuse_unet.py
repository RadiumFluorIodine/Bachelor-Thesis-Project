"""
ReUse: REgressive Unet for Carbon Storage and Above-Ground Biomass Estimation.

Adapted from: Pascarella et al. (2023)
Original Repository: https://github.com/priamus-lab/ReUse [cite: 148]
Paper: ReUse: REgressive Unet for Carbon Storage and Above-Ground Biomass Estimation [cite: 2, 5, 6, 8]
Paper DOI: 10.3390/jimaging9030061


Adaptations:
- Input size fixed to 128x128
- Integrated into PyTorch framework (Original was Keras/TensorFlow)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

class ReUseUNet(nn.Module):
    """
    PyTorch implementation of ReUse Regressive U-Net.
    
    Architecture: 4-level U-Net encoder-decoder with skip connections.
    Task: Pixel-wise regression of Above-Ground Biomass (AGB) in Mg/ha.
    Input:  (B, C=10, H=128, W=128) - single-date Sentinel-2 median composite
    Output: (B, 1, H=128, W=128)   - AGB map
    """
    def __init__(self, input_channels=10, n_filters=16, dropout=0.1):
        super(ReUseUNet, self).__init__()
        
        # Contracting Path (Encoder)
        self.c1 = self.conv_block(input_channels, n_filters)
        self.p1 = nn.MaxPool2d(2)
        self.d1 = nn.Dropout(dropout)
        
        self.c2 = self.conv_block(n_filters, n_filters * 2)
        self.p2 = nn.MaxPool2d(2)
        self.d2 = nn.Dropout(dropout)
        
        self.c3 = self.conv_block(n_filters * 2, n_filters * 4)
        self.p3 = nn.MaxPool2d(2)
        self.d3 = nn.Dropout(dropout)
        
        self.c4 = self.conv_block(n_filters * 4, n_filters * 8)
        self.p4 = nn.MaxPool2d(2)
        self.d4 = nn.Dropout(dropout)
        
        # Bottleneck
        self.c5 = self.conv_block(n_filters * 8, n_filters * 16)
        
        # Expansive Path (Decoder)
        self.u6 = nn.ConvTranspose2d(n_filters * 16, n_filters * 8, 
                                     kernel_size=3, stride=2, 
                                     padding=1, output_padding=1)
        self.c6 = self.conv_block(n_filters * 16, n_filters * 8)
        self.d6 = nn.Dropout(dropout)
         
        self.u7 = nn.ConvTranspose2d(n_filters * 8, n_filters * 4, 
                                     kernel_size=3, stride=2, 
                                     padding=1, output_padding=1)
        self.c7 = self.conv_block(n_filters * 8, n_filters * 4)
        self.d7 = nn.Dropout(dropout)
        
        self.u8 = nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 
                                     kernel_size=3, stride=2, 
                                     padding=1, output_padding=1)
        self.c8 = self.conv_block(n_filters * 4, n_filters * 2)
        self.d8 = nn.Dropout(dropout)
        
        self.u9 = nn.ConvTranspose2d(n_filters * 2, n_filters, 
                                     kernel_size=3, stride=2, 
                                     padding=1, output_padding=1)
        self.c9 = self.conv_block(n_filters * 2, n_filters)
        self.d9 = nn.Dropout(dropout)
        
        
        # Output Layer (Regression)
        self.output = nn.Conv2d(n_filters, 1, kernel_size=1)
        
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        c1 = self.c1(x)
        c2 = self.c2(self.d1(self.p1(c1)))
        c3 = self.c3(self.d2(self.p2(c2)))
        c4 = self.c4(self.d3(self.p3(c3)))

        
        # Bottleneck
        c5 = self.c5(self.d4(self.p4(c4)))
        
        # Decoder
        u6 = self.u6(c5)
        c6 = self.d6(self.c6(torch.cat([u6, c4], dim=1)))

        u7 = self.u7(c6)
        c7 = self.d7(self.c7(torch.cat([u7, c3], dim=1)))

        u8 = self.u8(c7)
        c8 = self.d8(self.c8(torch.cat([u8, c2], dim=1)))

        u9 = self.u9(c8)
        c9 = self.d9(self.c9(torch.cat([u9, c1], dim=1)))

        # Linear output for regression
        out = self.output(c9)
        
        return out

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    print("-" * 80)
    print("🚀 REUSE U-NET INTEGRATION TEST")
    print("-" * 80)


    # Setup path
    current_file_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    print(f"📂 Project Root: {root_dir}")

    # Import dataset
    try:
        from src.data.dataset_reuse import ReUseDataset, collate_fn_reuse
        print("✅ Import ReUseDataset & collate_fn_reuse berhasil.")
    except ImportError as e:
        print(f"\n❌ IMPORT ERROR: {e}")
        print("   Pastikan 'src/data/dataset_reuse.py' sudah dibuat!")
        exit()

    
    # Load Data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_path = os.path.join(root_dir, 'data', 'processed', 'lampung', 'version_2')
    print(f"\n Loading 1 Batch from: {data_path}")

    try:
        dataset = ReUseDataset(root_dir=data_path, mode='train')
        
        if len(dataset) == 0:
            print("Dataset kosong !")
            exit()
            
        loader = DataLoader(
            dataset, batch_size=2, shuffle=True,
            collate_fn=collate_fn_reuse          
        )
        
        batch  = next(iter(loader))
        images = batch['image'].to(device)       # (B, C, H, W)
        labels = batch['label'].to(device)       # (B, 1, H, W)
        
        # Dimension Check
        if len(images.shape) == 4:
            B, C, H, W = images.shape
            print(f"✅ Data Loaded (Spatial Only): {images.shape}")
            print("   (Dimensi Waktu sudah di-collapse oleh ReUseDataset menggunakan Median)")
        else:
            print(f"❌ SALAH DIMENSI: {images.shape}. Harusnya 4D (B,C,H,W)")
            exit()
        
        # Initialize and Run 
        print("\n🛠️  Initializing ReUse Model...")
        model = ReUseUNet(
            input_channels=10, 
            n_filters=16, 
            dropout=0.1
        ).to(device)
        
        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
        print(f"   Total params    : {total_params:,}")
        print(f"   Trainable params: {trainable_params:,}")
        
        print("⚡ Running Forward Pass...")
        
    
        print("\n⚡ Running Forward Pass (eval mode)...")
        model.eval()
        with torch.no_grad():
            output = model(images)

        
        print("-" * 30)
        print(f"Output Shape: {output.shape}")
        
        # Validasi Regresi
        assert output.shape == (B, 1, H, W), \
            f"❌ Output shape salah! Dapat {output.shape}, harusnya {(B,1,H,W)}"

        print(f"✅ Output shape  : {tuple(output.shape)} — (B, 1, H, W)")
        print(f"   Output min    : {output.min().item():.4f} (Z-Score)")
        print(f"   Output max    : {output.max().item():.4f} (Z-Score)")
        print(f"   Output mean   : {output.mean().item():.4f} (Z-Score)")
        
        print(f"   Linear Output : ✅ (Negative values allowed for Z-Score)")

        print("\n🎉 REUSE INTEGRATION TEST PASSED!")
        print("   Pipeline ReUseDataset → collate_fn_reuse → ReUseUNet ✅")
    except Exception as e:
        print(f"\n❌ Runtime Error: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 80)