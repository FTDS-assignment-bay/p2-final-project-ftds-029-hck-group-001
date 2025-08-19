from pathlib import Path
import pickle

def save_bandit(bandit, save_dir):
    """Save all components of the EQuinoxBandit using pickle"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save basic attributes
    bandit_data = {
        'actions': bandit.actions,
        'action_stats': bandit.action_stats,
        '_imputer_fitted': bandit._imputer_fitted
    }
    with open(save_dir / 'bandit_params.pkl', 'wb') as f:
        pickle.dump(bandit_data, f)
    
    # 2. Save each action's model
    for i in range(bandit.n_arms):
        with open(save_dir / f'model_action_{i}.pkl', 'wb') as f:
            pickle.dump(bandit.models[i], f)
    
    # 3. Save sklearn components
    with open(save_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(bandit.scaler, f)
    
    with open(save_dir / 'imputer.pkl', 'wb') as f:
        pickle.dump(bandit.imputer, f)
    
    print(f"Saved updated bandit to {save_dir}")
    return str(save_dir)