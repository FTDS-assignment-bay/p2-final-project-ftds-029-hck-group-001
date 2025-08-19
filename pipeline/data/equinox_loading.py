import pickle
from pathlib import Path
import json
import numpy as np
import pandas as pd

def load_bandit(save_dir):
    """Load bandit from pickle files and prepare for Airflow serialization"""
    save_dir = Path(save_dir)
    
    try:
        # 1. Load base parameters
        with open(save_dir / 'bandit_params.pkl', 'rb') as f:
            bandit_data = pickle.load(f)
        
        # 2. Reinitialize bandit (simplified representation)
        bandit_info = {
            'actions': bandit_data['actions'],
            'action_stats': bandit_data['action_stats'],
            '_imputer_fitted': bandit_data['_imputer_fitted'],
            'model_paths': [str(save_dir / f'model_action_{i}.pkl') for i in range(len(bandit_data['actions']))],
            'scaler_path': str(save_dir / 'scaler.pkl'),
            'imputer_path': str(save_dir / 'imputer.pkl'),
            'save_dir': str(save_dir)
        }
        
        # 3. Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return int(obj) if isinstance(obj, np.integer) else float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(x) for x in obj]
            return obj
        
        return convert_types(bandit_info)
    
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'save_dir': str(save_dir)
        }