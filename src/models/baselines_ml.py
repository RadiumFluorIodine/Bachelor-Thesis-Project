"""
Machine Learning Baseline Models.

Wrappers for Scikit-Learn and XGBoost models for consistent interface.
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import xgboost as xgb
import joblib
import sys
import os

class BaselineModelWrapper:
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        self.model = self._get_model(model_type, **kwargs)
        
    def _get_model(self, model_type, **kwargs):
        if model_type == 'linear_regression':
            return LinearRegression(**kwargs)
        elif model_type == 'random_forest':
            return RandomForestRegressor(n_jobs=-1, **kwargs)
        elif model_type == 'xgboost':
            return xgb.XGBRegressor(n_jobs=-1, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def fit(self, X, y):
        """
        Train model.
        Args:
            X: Numpy array (N_samples, N_features)
            y: Numpy array (N_samples,)
        """
        print(f"   Training {self.model_type} with input shape {X.shape}...")
        self.model.fit(X, y)


    def predict(self, X):
        """
        Predict.
        Returns:
            Numpy array (N_samples,)
        """
        return self.model.predict(X)
        
    def save(self, path):
        joblib.dump(self.model, path)
        print(f"   Model saved to {path}")

        
    def load(self, path):
        self.model = joblib.load(path)
        print(f"   Model loaded from {path}")


# Unit Testing Code
if __name__ == "__main__":
    print("=" * 80)
    print("ML BASELINES INTEGRATION TEST")
    print("=" * 80)


    # Setup Path
    current_file_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    print(f"📂 Project Root: {root_dir}")

    # Import Preprocess Function
    try:
        from src.data.preprocess_baseline import prepare_baseline_data
        print("✅ Berhasil import prepare_baseline_data.")
    except ImportError as e:
        print("\n❌ IMPORT ERROR!")
        print(f"   Detail: {e}")
        print("   Pastikan file 'src/data/preprocess_baseline.py' sudah dibuat!")
        exit()

    
    # Prepare Data
    data_path = os.path.join(root_dir, 'data', 'processed', 'lampung', 'version_2')
    print(f"\n⏳ Preparing Data from: {data_path}")

    try:
        if not os.path.exists(data_path):
             print(f"❌ Folder data tidak ditemukan di: {data_path}")
             exit()

        X_train, y_train = prepare_baseline_data(
            data_dir=data_path,
            mode='train',
            aggregation='median', 
            max_patches=5,        
            pixel_subsample=0.1   
        )
        
        if len(X_train) == 0:
            print("❌ Data hasil preprocessing kosong!")
            exit()

        print(f"✅ Data Prepared Successfully!")
        print(f"   X_train shape: {X_train.shape} (N_Samples, N_Features)")
        print(f"   y_train shape: {y_train.shape} (N_Samples,)")
        
        # Feature Validation
        assert X_train.shape[1] == 10, f"❌ Jumlah fitur salah! Dapat {X_train.shape[1]}, Harusnya 10."

        # Test Model
        models_to_test = ['linear_regression', 'random_forest', 'xgboost']
            
        for model_name in models_to_test:
            print(f"\n🛠️  Testing Model: {model_name.upper()}")
            
            # Gunakan n_estimators kecil agar cepat
            kwargs = {'n_estimators': 5, 'random_state': 42} if model_name != 'linear_regression' else {}
            wrapper = BaselineModelWrapper(model_name, **kwargs)
            
            # Fit
            wrapper.fit(X_train, y_train)
            
            # Predict
            preds = wrapper.predict(X_train)
            
            print(f"   Prediction shape: {preds.shape}")
            
            # Validasi Dimensi Output
            assert preds.shape == y_train.shape, "❌ Dimensi prediksi tidak sesuai target!"
            
            # Metric RMSE
            rmse = np.sqrt(((preds - y_train) ** 2).mean())
            print(f"   Training RMSE: {rmse:.4f}")
            
            # Test Save/Load
            save_path = os.path.join(root_dir, 'results', f'test_{model_name}.joblib')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            wrapper.save(save_path)
            wrapper.load(save_path)
            
            # Cleanup file test
            if os.path.exists(save_path):
                os.remove(save_path)
                print("   Test file cleaned up.")

        print("\n🎉 ML BASELINES INTEGRATION TEST PASSED!")
        print("   Preprocessing pipeline dan Model Wrapper berfungsi dengan benar.")
        
    except Exception as e:
        print(f"\n❌ Runtime Error: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 80)