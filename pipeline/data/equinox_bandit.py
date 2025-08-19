# Import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import defaultdict

# Configuration
RANDOM_SEED = 69
np.random.seed(RANDOM_SEED)


# Bandit Implementation
class EQuinoxBandit:
    def __init__(self, actions):
        self.actions = actions
        self.n_arms = len(actions)
        self.models = defaultdict(
            lambda: LogisticRegression(
                warm_start=True,
                solver='saga',
                penalty='elasticnet',
                l1_ratio=0.5,
                max_iter=300,
                C=0.1,
                tol=1e-3,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                n_jobs=-1))
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self._imputer_fitted = False  # Flag to track fitting
        self.action_stats = {i: {'count': 0, 'rewards': []} for i in range(self.n_arms)}
    
    def _prepare_context(self, user_feats, session_context):
        """Create the standardized feature vector with NaN protection"""
        features = [
            # Behavioral
            user_feats['total_views'],
            user_feats['events_per_day'],
            user_feats['conversion_rate'],
            
            # Product Affinity
            user_feats.get('avg_price_viewed', 0) or 0,  # Fallback to 0 if NaN
            user_feats['unique_categories'],  # Shouldn't be NaN after fillna
            
            # Temporal
            session_context['hour'],
            session_context['is_weekend'],
            user_feats['user_tenure_days'],
            
            # Current Session
            session_context.get('item_price', 0) or 0,  # Fallback to 0
            int(session_context.get('item_category', -1))  # Fallback to -1
        ]
        return np.array(features).reshape(1, -1)
    
    def select_action(self, user_feats, session_context):
        """Thompson sampling with exploration"""
        context = self._prepare_context(user_feats, session_context)
        scaled_context = self.scaler.partial_fit(context).transform(context)
        
        # Initialize all action probabilities first
        action_probs = {}
        for action in range(self.n_arms):
            if not hasattr(self.models[action], 'coef_'):  # Untrained model
                prob = 0.5
            else:
                prob = self.models[action].predict_proba(scaled_context)[0][1]
            
            # Add exploration noise
            action_probs[action] = prob + np.random.normal(0, 0.1)
        
        # Business rule 1. Reduce discount probability for cheap items
        if 1 in action_probs and user_feats['avg_price_viewed'] < 50:  # Discount action
            action_probs[1] *= 0.3
        
        # Business rule 2. Reduce all non-essential actions during dead hours (7AM-12PM)
        if 7 <= session_context['hour'] <= 12:
            for action in [1, 2, 3]:  # All except email_no_discount (action 0)
                if action in action_probs:  # 80% reduction
                    action_probs[action] *= 0.2
        
        # Business rule 3. Weekend pacing - only show high-impact actions
        if session_context['is_weekend']:
            # Boost essential actions
            if 0 in action_probs: action_probs[0] *= 1.2  # Baseline email
            if 3 in action_probs: action_probs[3] *= 1.5  # Cart reminders
            
            # Suppress others
            if 1 in action_probs: action_probs[1] *= 0.3  # Discounts
            if 2 in action_probs: action_probs[2] *= 0.4  # Banners
        
        # Calibrate probabilities (squash extremes)
        for action in action_probs:
            action_probs[action] = 0.5 + 0.5 * (action_probs[action] - 0.5)  # Linear scaling
        
        return max(action_probs.items(), key=lambda x: x[1])[0]
    
    def update(self, user_feats, session_context, action, reward):
        """Online model update"""
        context = self._prepare_context(user_feats, session_context)
        if np.isnan(context).any(): # Validate if there are Missing Values
            print(f"Skipping update for action {action} due to NaN in context")
            return
        context = np.array(context).reshape(1, -1)  # Ensure it's in 2D
        
        # First-time fitting
        if not self._imputer_fitted:
            self.imputer.fit(context.reshape(1, -1))
            self.scaler.fit(self.imputer.transform(context.reshape(1, -1)))
            self._imputer_fitted = True
        
        # Then transform as normal
        context_imputed = self.imputer.transform(context.reshape(1, -1))
        scaled_context = self.scaler.transform(context_imputed)
        
        # Initialize model if first time
        model = self.models[action]
        
        if not hasattr(model, 'coef_'):
            # Warm start with dummy data
            dummy_x = np.array([scaled_context[0], scaled_context[0]])
            dummy_y = [0, 1]
            model.fit(dummy_x, dummy_y)
        
        # Update model
        try:
            # Try regular fit (works if we have both classes)
            model.fit(scaled_context, [reward])
        except ValueError:
            # If error occurs (single class), add a synthetic opposite sample
            synthetic_reward = 1 - reward
            x = np.array([scaled_context[0], scaled_context[0]])
            y = [reward, synthetic_reward]
            model.fit(x, y)
        
        # Update statistics
        self.action_stats[action]['count'] += 1
        self.action_stats[action]['rewards'].append(reward)
    
    def get_action_stats(self):
        """Return performance metrics per action"""
        stats = []
        for action, data in self.action_stats.items():
            if data['count'] > 0:
                stats.append({
                    'action': self.actions[action],
                    'count': data['count'],
                    'avg_reward': np.mean(data['rewards']),
                    'conversion_rate': np.sum(data['rewards']) / data['count']
                })
        return pd.DataFrame(stats)