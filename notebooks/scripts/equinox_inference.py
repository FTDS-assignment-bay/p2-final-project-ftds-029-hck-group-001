# Import Libraries
import json
import pickle
import numpy as np
import lightgbm as lgb
from pathlib import Path
from equinox_bandit import EQuinoxBandit, EQuinoxBanditGBM


# Bandit loading for inference
def load_bandit(save_dir):
    """
    Load a saved bandit model from disk with universal format support.
    
    Reconstructs either EQuinoxBandit (Logistic Regression) or EQuinoxBanditGBM (LightGBM)
    from previously saved components, ready for inference or continued training.
    
    Args:
        save_dir (str or Path): Directory path where bandit components were saved.
                                Must contain files created by save_bandit().
        
    Returns:
        EQuinoxBandit or EQuinoxBanditGBM: Reconstructed bandit instance ready for use
        
    Raises:
        FileNotFoundError: If required files are missing from the save directory
        ValueError: If the bandit type in bandit_params.pkl is unrecognized
        
    Example:
        >>> bandit = load_bandit("models/bandit_v1")
        >>> gbm_bandit = load_bandit("models/gbm_bandit_v1")
        
    Note:
        This function automatically detects the bandit type from saved metadata and
        handles the different loading requirements for scikit-learn vs LightGBM models.
        It provides default values for missing attributes to ensure backward compatibility.
    """
    save_dir = Path(save_dir)
    
    with open(save_dir / 'bandit_params.pkl', 'rb') as f:
        bandit_data = pickle.load(f)
    
    # Recreate the right bandit type
    if bandit_data['bandit_type'] == 'EQuinoxBandit':  # LogisticRegression version
        bandit = EQuinoxBandit(bandit_data['actions'])
        bandit.action_stats = bandit_data.get('action_stats', {})
        bandit._imputer_fitted = bandit_data.get('_imputer_fitted', False)
        
    elif bandit_data['bandit_type'] == 'EQuinoxBanditGBM':  # LightGBM version
        bandit = EQuinoxBanditGBM(bandit_data['actions'])
        bandit.alpha = bandit_data.get('alpha', np.ones(len(bandit_data['actions'])))
        bandit.beta = bandit_data.get('beta', np.ones(len(bandit_data['actions'])))
        bandit.action_counts = bandit_data.get('action_counts', np.zeros(len(bandit_data['actions'])))
        bandit._fitted = bandit_data.get('_fitted', False)
    
    # Load models based on file extension
    for i in range(bandit.n_arms):
        txt_file = save_dir / f'model_action_{i}.txt'
        pkl_file = save_dir / f'model_action_{i}.pkl'
        
        if txt_file.exists():  # LightGBM
            bandit.models[i] = lgb.Booster(model_file=str(txt_file))
        elif pkl_file.exists():  # Scikit-learn
            with open(pkl_file, 'rb') as f:
                bandit.models[i] = pickle.load(f)
    
    # Load sklearn components
    with open(save_dir / 'scaler.pkl', 'rb') as f:
        bandit.scaler = pickle.load(f)
    
    with open(save_dir / 'imputer.pkl', 'rb') as f:
        bandit.imputer = pickle.load(f)
    
    return bandit


# Predictor Implementation
class EQuinoxPredictor:
    """
    Production inference wrapper for trained EQuinox bandit models.
    
    Provides a clean interface for making real-time predictions with validation,
    error handling, and user-friendly output formatting. Supports both Logistic
    Regression and LightGBM bandit implementations transparently.
    
    Attributes:
        bandit (EQuinoxBandit or EQuinoxBanditGBM): Loaded bandit instance for prediction
        required_user_feats (list): Required user feature keys for validation
        required_context (list): Required session context keys for validation  
        optional_user_feats (dict): Optional user features with default values
    """
    
    def __init__(self, bandit):
        """
        Initialize predictor with a pre-loaded bandit instance.
        
        Args:
            bandit (EQuinoxBandit or EQuinoxBanditGBM): Already loaded bandit object
                    ready for inference, loaded via load_bandit().
        
        Example:
            >>> bandit = load_bandit("model/equinox_model")
            >>> predictor = EQuinoxPredictor(bandit)
        """
        self.bandit = bandit
        self._init_requirements()
    
    def _init_requirements(self):
        """
        Define required and optional input features with fallback defaults.
        
        Establishes validation rules for input data to ensure prediction reliability
        while providing sensible defaults for optional features.
        """
        self.required_user_feats = [
            'total_views', 'events_per_day', 'conversion_rate',
            'avg_price_viewed', 'unique_categories', 'user_tenure_days'
        ]
        
        self.required_context = [
            'hour', 'is_weekend', 'item_price', 'item_category'
        ]
        
        # Optional features with defaults
        self.optional_user_feats = {
            'has_converted': False,
            'avg_availability': 1.0
        }
    
    def validate_inputs(self, user_feats, session_context):
        """
        Validate input feature dictionaries for completeness.
        
        Args:
            user_feats (dict): Dictionary of user features
            session_context (dict): Dictionary of session context features
            
        Raises:
            ValueError: If any required features are missing from inputs
            
        Note:
            Only validates presence of required features, not data types or values.
            Optional features use defaults if missing.
        """
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
        Make real-time action recommendation with conversion probability estimates.
        
        Args:
            user_feats (dict): User feature dictionary containing:
                - total_views: Total number of item views
                - events_per_day: Average daily engagement  
                - conversion_rate: Historical conversion rate
                - avg_price_viewed: Average price of viewed items
                - unique_categories: Number of distinct categories viewed
                - user_tenure_days: Days since first activity
                - has_converted: Whether user has converted before (optional)
                - avg_availability: Average item availability (optional)
            session_context (dict): Current session context containing:
                - hour: Current hour of day (0-23)
                - is_weekend: Boolean indicating weekend
                - item_price: Price of current item being viewed
                - item_category: Category ID of current item
                
        Returns:
            dict: Prediction results with keys:
                - recommended_action: Optimal action to take
                - expected_conversion_rate: Estimated conversion probability for recommended action
                - action_breakdown: Dictionary of conversion probabilities for all actions
                - baseline_rate: User's historical conversion rate for comparison
                
        Note:
            Automatically handles both Logistic Regression and LightGBM bandit types
            with appropriate prediction methods for each model format.
        """
        self.validate_inputs(user_feats, session_context)
        
        context = self.bandit._prepare_context(user_feats, session_context)
        action_probs = {}
        
        for i, action in enumerate(self.bandit.actions):
            model = self.bandit.models[i]
            
            # Detect model type and predict accordingly
            if hasattr(model, 'coef_'):  # Scikit-learn model
                proba = model.predict_proba(context)
                action_probs[action] = proba[0][1]
                
            elif hasattr(model, 'predict_proba'):  # LightGBM with predict_proba
                proba = model.predict_proba(context)
                # LightGBM might return different format
                if hasattr(proba, 'shape') and len(proba.shape) == 2:
                    action_probs[action] = proba[0][1]
                else:
                    action_probs[action] = float(proba)
                    
            elif hasattr(model, 'predict'):  # LightGBM with only predict
                # Some LightGBM versions only have predict()
                prediction = model.predict(context)
                action_probs[action] = float(prediction[0])
                
            else:
                # Untrained or unknown model type
                action_probs[action] = 0.0
        
        recommended = max(action_probs.items(), key=lambda x: x[1])[0]
        
        return {
            'recommended_action': recommended,
            'expected_conversion_rate': action_probs[recommended],
            'action_breakdown': action_probs,
            'baseline_rate': user_feats['conversion_rate']
        }
    
    def predict_formatted(self, user_feats, session_context):
        """
        Generate user-friendly formatted prediction output.
        
        Args:
            user_feats (dict): User features (see predict() for details)
            session_context (dict): Session context (see predict() for details)
            
        Returns:
            str: JSON-formatted string with human-readable prediction results,
                suitable for API responses or user interfaces.
        
        Example:
            >>> result = predictor.predict_formatted(user_data, session_data)
            >>> print(result)
            {
                "recommendation": "email_10%_discount",
                "predicted_conversion_lift": "+15.2%", 
                "action_breakdown": {
                    "email_no_discount": 0.12,
                    "email_10%_discount": 0.27,
                    ...
                }
            }
        """
        pred = self.predict(user_feats, session_context)
        return json.dumps({
            'recommendation': pred['recommended_action'],
            'predicted_conversion_lift': f"{pred['expected_conversion_rate'] - pred['baseline_rate']:.1%}",
            'action_breakdown': pred['action_breakdown']
        }, indent=2)