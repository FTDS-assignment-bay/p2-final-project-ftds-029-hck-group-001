# Import Libraries
import json
import pickle
from pathlib import Path
from equinox_bandit import EQuinoxBandit


# Bandit loading for inference
def load_bandit(save_dir):
    """Reconstruct the EQuinoxBandit from disk"""
    save_dir = Path(save_dir)
    
    # 1. Load base parameters
    with open(save_dir / 'bandit_params.pkl', 'rb') as f:
        bandit_data = pickle.load(f)
    
    # 2. Reinitialize bandit
    bandit = EQuinoxBandit(bandit_data['actions'])
    bandit.action_stats = bandit_data['action_stats']
    bandit._imputer_fitted = bandit_data['_imputer_fitted']
    
    # 3. Load models
    for i in range(bandit.n_arms):
        with open(save_dir / f'model_action_{i}.pkl', 'rb') as f:
            bandit.models[i] = pickle.load(f)
    
    # 4. Load sklearn components
    with open(save_dir / 'scaler.pkl', 'rb') as f:
        bandit.scaler = pickle.load(f)
    
    with open(save_dir / 'imputer.pkl', 'rb') as f:
        bandit.imputer = pickle.load(f)
    
    return bandit


# Predictor Implementation
class EQuinoxPredictor:
    def __init__(self, bandit):
        """
        Initialize predictor with a PRE-LOADED EQuinoxBandit instance
        
        Args:
            bandit: Already loaded EQuinoxBandit object
        """
        self.bandit = bandit
        self._init_requirements()

    def _init_requirements(self):
        """Define required input features"""
        self.required_user_feats = [
            'total_views', 'events_per_day', 'conversion_rate',
            'avg_price_viewed', 'unique_categories', 'user_tenure_days',
            'has_converted', 'avg_availability'
        ]
        self.required_context = [
            'hour', 'is_weekend', 'item_price', 'item_category'
        ]

    def validate_inputs(self, user_feats, session_context):
        """Validate input feature dictionaries"""
        missing_user = [f for f in self.required_user_feats if f not in user_feats]
        missing_context = [f for f in self.required_context if f not in session_context]
        
        if missing_user or missing_context:
            raise ValueError(
                f"Missing features:\n"
                f"- User: {missing_user}\n"
                f"- Session: {missing_context}"
            )

    def predict(self, user_feats, session_context):
        """
        Make prediction with conversion rates
        
        Returns:
            {
                'recommended_action': str,
                'expected_conversion_rate': float,
                'action_breakdown': dict,
                'baseline_rate': float
            }
        """
        self.validate_inputs(user_feats, session_context)
        
        context = self.bandit._prepare_context(user_feats, session_context)
        action_probs = {
            action: (
                self.bandit.models[i].predict_proba(context)[0][1] 
                if hasattr(self.bandit.models[i], 'coef_') 
                else 0.0
            )
            for i, action in enumerate(self.bandit.actions)
        }
        
        recommended = max(action_probs.items(), key=lambda x: x[1])[0]
        
        return {
            'recommended_action': recommended,
            'expected_conversion_rate': action_probs[recommended],
            'action_breakdown': action_probs,
            'baseline_rate': user_feats['conversion_rate']
        }

    def predict_formatted(self, user_feats, session_context):
        """User-friendly formatted prediction"""
        pred = self.predict(user_feats, session_context)
        return json.dumps({
            'recommendation': pred['recommended_action'],
            'predicted_conversion_lift': f"{pred['expected_conversion_rate'] - pred['baseline_rate']:.1%}",
            'action_breakdown': pred['action_breakdown']
        }, indent=2)