"""
Baseline Models Training Pipeline.

Models:
1. Linear Regression
2. Random Forest
3. XGBoost

Features:
- Spatio-temporal data flattening (Temporal Mean -> Pixel-wise)
- Model training & evaluation
- Comprehensive metrics (MAE, RMSE, R², CCC, Pearson, etc.)
- Overfitting detection
- Results comparison table
"""

import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime
import sys

# Setup Path
current_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import Modules
from src.models.baselines_ml import BaselineModelWrapper
from src.data.preprocess_baseline import prepare_baseline_data
from src.training.training_utils import RegressionMetrics
from src.data.dataset import get_or_create_global_split

def evaluate_model(model, X, y, set_name='Validation'):
    """
    Comprehensive model evaluation using all metrics.
    
    Args:
        model: Trained model wrapper
        X: Features
        y: True targets
        set_name: Name for logging (e.g., 'Train', 'Validation')
    
    Returns:
        dict: All regression metrics
    """
    print(f"   Evaluating on {set_name} set...")
    y_pred = model.predict(X)
    
    # Comprehensive metrics
    metrics = RegressionMetrics.compute_all(y_pred, y)
    
    # Print summary
    print(f"   {set_name} Metrics:")
    print(f"     MAE:     {metrics['mae']:.4f} Mg/ha")
    print(f"     RMSE:    {metrics['rmse']:.4f} Mg/ha")
    print(f"     R²:      {metrics['r2']:.4f}")
    print(f"     CCC:     {metrics['ccc']:.4f}")
    print(f"     Pearson: {metrics['pearson']:.4f}")
    
    return metrics

def train_and_evaluate(model, X_train, y_train, X_val, y_val, model_name):
    """
    Train model and evaluate on both train and validation sets.
    
    Args:
        model: Model wrapper
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_name: Name for logging
    
    Returns:
        dict: Training and validation metrics
    """
    print(f"\n{'-'*70}")
    print(f"Training {model_name}")
    print('-'*70)
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    train_metrics = evaluate_model(model, X_train, y_train, 'Train')
    val_metrics = evaluate_model(model, X_val, y_val, 'Validation')
    
    # Check overfitting
    r2_gap = train_metrics['r2'] - val_metrics['r2']
    if r2_gap > 0.15:
        print(f"\n Warning: Significant overfitting detected!")
        print(f"Train R² - Val R² = {r2_gap:.3f}")
    elif r2_gap > 0.05:
        print(f"\n Minor overfitting (ΔR² = {r2_gap:.3f})")
    else:
        print(f"\n Good generalization (ΔR² = {r2_gap:.3f})")
    
    return {
        'train': train_metrics,
        'val': val_metrics,
        'overfitting_gap': float(r2_gap)
    }


def print_results_table(results):
    """Print formatted comparison table."""
    print("\n" + "-"*80)
    print("BASELINE MODELS COMPARISON (Validation Set)")
    print("-"*80)
    
    # Header
    print(f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'CCC':>10} "
          f"{'Pearson':>10}")
    print("-"*80)
    
    # Sort by R² (descending)
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]['val']['r2'],
        reverse=True
    )
    
    for model_name, metrics in sorted_models:
        val = metrics['val']
        print(f"{model_name:<20} "
              f"{val['mae']:>10.4f} "
              f"{val['rmse']:>10.4f} "
              f"{val['r2']:>10.4f} "
              f"{val['ccc']:>10.4f} "
              f"{val['pearson']:>10.4f}")
    
    print("-"*80)
    
    # Best model
    best_model, best_metrics = sorted_models[0]
    print(f"\n Best Model: {best_model}")
    print(f"Validation R² = {best_metrics['val']['r2']:.4f}")
    print(f"Validation MAE = {best_metrics['val']['mae']:.4f} Mg/ha")
    print(f"Overfitting Gap (ΔR²) = {best_metrics['overfitting_gap']:.3f}")
    print("-"*80 + "\n")


def main():
    # Configuration
    DATA_DIR = os.path.join(root_dir, 'data', 'processed', 'lampung', 'version_2')
    OUTPUT_DIR = os.path.join(root_dir, 'src', 'results', 'checkpoints', 'baselines')
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Save config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Prepare Data
    print("=== Data Preparation ===")
    SPLIT_DIR = os.path.join(root_dir, 'data', 'processed', 'lampung', 'splits')

    train_files, val_files = get_or_create_global_split(
        data_dir=DATA_DIR, 
        split_dir=SPLIT_DIR, 
        test_size=0.2, 
        random_state=42
    )


    X_train, y_train = prepare_baseline_data(
        DATA_DIR, 
        mode='train', 
        aggregation='median', 
        file_list=train_files, 
        max_patches=5000
    )

    X_val, y_val = prepare_baseline_data(
        DATA_DIR, 
        mode='val', 
        aggregation='median', 
        file_list=val_files, 
        max_patches=2000
    ) 

    print(f"\n Data loaded successfully!")
    print(f"   Training samples:   {X_train.shape[0]:,}")
    print(f"   Validation samples: {X_val.shape[0]:,}")
    print(f"   Features:           {X_train.shape[1]}")
    print(f"   Target range:       [{y_train.min():.2f}, {y_train.max():.2f}] Mg/ha")
    
    results = {}
    
    # 2. Linear Regression
    print("\n=== Training Linear Regression ===")
    lr = BaselineModelWrapper('linear_regression')
    results['Linear Regression'] = train_and_evaluate(
        lr, X_train, y_train, X_val, y_val, 'Linear Regression'
    )
    lr.save(os.path.join(OUTPUT_DIR, f"linear_regression_{timestamp}.joblib"))

    # 3. Random Forest
    print("\n=== Training Random Forest ===")
    # Using reduced parameters for speed demonstration; tune these for final results
    rf = BaselineModelWrapper(
        'random_forest',
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        verbose=0
    )
    results['Random Forest'] = train_and_evaluate(
        rf, X_train, y_train, X_val, y_val, 'Random Forest'
    )
    rf.save(os.path.join(OUTPUT_DIR, f"random_forest_{timestamp}.joblib"))

    # 4. XGBoost
    print("\n=== Training XGBoost ===")
    xgb_model = BaselineModelWrapper(
        'xgboost',
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )
    results['XGBoost'] = train_and_evaluate(
        xgb_model, X_train, y_train, X_val, y_val, 'XGBoost'
    )
    xgb_model.save(os.path.join(OUTPUT_DIR, f"xgboost_{timestamp}.joblib"))
    
    # 5. Print Summary
    print_results_table(results)
    
    # 6. Save Results
    results_file = os.path.join(OUTPUT_DIR, f"baseline_metrics_{timestamp}.json")

    # Convert numpy types to Python types for JSON serialization
    json_results = {}
    for model_name, metrics in results.items():
        json_results[model_name] = {
            'train': {k: float(v) for k, v in metrics['train'].items()},
            'val': {k: float(v) for k, v in metrics['val'].items()},
            'overfitting_gap': float(metrics['overfitting_gap'])
        }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print(f"Training complete!")
    print(f"Models saved to: {OUTPUT_DIR}")
    print(f"Results saved to: {results_file}")  

if __name__ == "__main__":
    main()