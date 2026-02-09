# Estimasi Aboveground Biomass (AGB) Hutan Menggunakan Sentinel-2 Multi-Temporal dengan Model U-LTAE

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Repositori ini berisi implementasi penelitian skripsi mengenai pengembangan model **U-Net dengan Lightweight Temporal Attention Encoder (U-LTAE)** untuk estimasi biomassa di atas permukaan tanah (*Aboveground Biomass*) menggunakan data citra satelit Sentinel-2 multi-temporal di wilayah Lampung.

## 📌 Gambaran Proyek

Penelitian ini membandingkan efektivitas model *Deep Learning* berbasis spasio-temporal (U-LTAE) dengan model konvensional (Baseline) dalam memetakan biomassa hutan. Dataset yang digunakan mencakup rentang waktu 12 bulan untuk menangkap fenologi vegetasi yang membantu meningkatkan akurasi estimasi dibandingkan hanya menggunakan citra *single-date*.

## 🏗️ Arsitektur Model

1.  **U-LTAE (Proposed):** Menggabungkan kemampuan spasial U-Net dengan *Lightweight Temporal Attention Encoder* untuk memproses urutan waktu citra satelit secara efisien.
2.  **ReUse U-Net (Comparison):** Implementasi *Regressive U-Net* berdasarkan referensi Pascarella et al. (2023).
3.  **Machine Learning Baselines:** Linear Regression, Random Forest, dan XGBoost sebagai pembanding performa.

## 💻 Spesifikasi Sistem

Eksperimen dilakukan menggunakan spesifikasi *high-performance computing* berikut:

* **GPU:** NVIDIA GeForce RTX 4090 (24GB VRAM)
* **CPU:** Intel Core i9 Gen 14 (24 Cores / 32 Threads)
* **OS:** Windows 11 / Linux
* **Environment:** Anaconda & PyTorch (CUDA 12.1)

## 📂 Struktur Folder

```text
Bachelor-Thesis/
├── data/
│   ├── raw/             # Data .tif asli (Sentinel-2 & AGB Label)
│   └── processed/       # Patch .npz dan normalization.json
├── src/
│   ├── data/            # Script Dataset & Dataloader
│   ├── models/          # Arsitektur U-LTAE & ReUse
│   ├── training/        # Script Training & Utilities
│   └── results/
│       ├── checkpoints/ # Hasil simpanan model (.pt, .joblib)
│       ├── logs/        # File log training (.log)
│       └── runs/        # TensorBoard event files
├── .gitignore
└── README.md
