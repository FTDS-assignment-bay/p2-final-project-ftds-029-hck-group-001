# Import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import defaultdict
import lightgbm as lgb

# Configuration
RANDOM_SEED = 69
np.random.seed(RANDOM_SEED)


# LogisticRegression Bandit Implementation
class EQuinoxBandit:
    """
    A contextual bandit implementation using Logistic Regression for e-commerce personalization.
    
    This bandit learns to choose optimal marketing actions (emails, discounts, banners, cart reminders)
    based on user context and session features, using online logistic regression with business rule constraints.
    
    Attributes:
        actions (list): List of available marketing actions
        n_arms (int): Number of available actions
        models (defaultdict): Dictionary of LogisticRegression models, one per action
        scaler (StandardScaler): Feature scaler for normalization
        imputer (SimpleImputer): Missing value imputer
        _imputer_fitted (bool): Flag indicating whether imputer has been fitted
        action_stats (dict): Tracking statistics for each action's performance
    """
    
    def __init__(self, actions):
        """
        Initialize the contextual bandit with available actions.
        
        Args:
            actions (list): List of marketing action names to choose from.
                            Example: ['email_no_discount', 'email_10%_discount', 
                                    'banner_limited_time_offer', 'popup_abandoned_cart_reminder']
        """
        self.actions = actions
        self.n_arms = len(actions)
        
        # Logistic regression model for each action
        self.models = defaultdict(
            lambda: LogisticRegression(
                warm_start=True,
                solver='saga',
                penalty='elasticnet',
                l1_ratio=0.5,
                max_iter=500,
                C=0.1,
                tol=1e-3,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                n_jobs=-1))
        
        # Feature processing
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self._imputer_fitted = False  # Flag to track fitting
        self.action_stats = {i: {'count': 0, 'rewards': []} for i in range(self.n_arms)}
    
    def _prepare_context(self, user_feats, session_context):
        """
        Create standardized feature vector from user and session context with robust NaN handling.
        
        Constructs a feature vector suitable for logistic regression prediction by combining
        user behavioral features with current session context.
        
        Args:
            user_feats (dict): User-level features including:
                - total_views: Total number of item views
                - events_per_day: Average daily engagement rate
                - conversion_rate: Historical conversion rate
                - avg_price_viewed: Average price of viewed items
                - unique_categories: Number of distinct categories viewed
                - user_tenure_days: Days since first activity
            session_context (dict): Current session features including:
                - hour: Current hour of day
                - is_weekend: Whether current day is weekend
                - item_price: Price of current item being viewed
                - item_category: Category of current item
                
        Returns:
            np.ndarray: Standardized feature vector of shape (1, n_features) ready for model prediction
        """
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
        """
        Select optimal action using epsilon-greedy exploration with business rule constraints.
        
        Combines model predictions with exploration noise and applies domain-specific
        business rules to ensure practical and compliant action selection.
        
        Args:
            user_feats (dict): User-level features (see _prepare_context for details)
            session_context (dict): Session-level features (see _prepare_context for details)
            
        Returns:
            int: Index of the selected action in self.actions list
            
        Business Rules Applied:
            1. Discount reduction for cheap items: Reduce discount probability for items under $50
            2. Dead hour reduction: Reduce non-essential actions during 7AM-12PM
            3. Weekend pacing: Boost high-impact actions, suppress low-impact actions on weekends
        """
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
        """
        Update the bandit's model with new observation using online learning.
        
        Performs incremental model update with robust handling of edge cases including:
        - Missing value detection and skipping
        - First-time feature processor fitting
        - Untrained model initialization with warm start
        - Single-class handling with synthetic samples
        
        Args:
            user_feats (dict): User-level features for context
            session_context (dict): Session-level features for context  
            action (int): Index of the action that was taken
            reward (int): Binary reward signal (1 = conversion, 0 = no conversion)
        """
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
        """
        Return performance metrics for each action.
        
        Provides insights into bandit performance including conversion rates,
        action frequencies, and reward statistics for monitoring and analysis.
        
        Returns:
            pd.DataFrame: Performance statistics with columns:
                - action: Action name
                - count: Number of times action was selected
                - avg_reward: Average reward obtained
                - conversion_rate: Conversion rate for this action
        """
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


# LightGBM Bandit Implementation
class EQuinoxBanditGBM:
    """
    Enhanced contextual bandit using LightGBM with hybrid exploration-exploitation.
    
    Combines LightGBM's powerful gradient boosting for exploitation with Thompson sampling
    for intelligent exploration, providing better performance than logistic regression
    while maintaining business rule constraints.
    
    Attributes:
        actions (list): List of available marketing actions
        n_arms (int): Number of available actions
        models (dict): Dictionary of LightGBM classifiers, one per action
        alpha (np.ndarray): Thompson sampling success counts per action
        beta (np.ndarray): Thompson sampling failure counts per action
        action_counts (np.ndarray): Count of how many times each action was selected
        min_exploration (float): Minimum exploration rate to maintain (0.2 = 20%)
        exploration_decay (float): Rate at which exploration decreases with more data
        X (dict): Storage of context features for each action's training data
        y (dict): Storage of rewards for each action's training data
        scaler (StandardScaler): Feature scaler for normalization
        imputer (SimpleImputer): Missing value imputer
        _fitted (bool): Flag indicating whether feature processors have been fitted
    """
    
    def __init__(self, actions):
        """
        Initialize the LightGBM hybrid bandit with exploration control.
        
        Args:
            actions (list): List of marketing action names to choose from.
                            Example: ['email_no_discount', 'email_10%_discount', 
                                    'banner_limited_time_offer', 'popup_abandoned_cart_reminder']
        """
        self.actions = actions
        self.n_arms = len(actions)
        
        # LightGBM model for each action
        self.models = {}
        for action in range(self.n_arms):
            self.models[action] = lgb.LGBMClassifier(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=5,
                verbosity=-1,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
        
        # Thompson sampling for exploration
        self.alpha = np.ones(self.n_arms)  # Success counts
        self.beta = np.ones(self.n_arms)   # Failure counts
        self.action_counts = np.zeros(self.n_arms)
        
        # Add exploration control
        self.min_exploration = 0.2  # Always keep at least 20% exploration
        self.exploration_decay = 0.999  # Slower decay
        
        # Context storage for training
        self.X = {action: [] for action in range(self.n_arms)}
        self.y = {action: [] for action in range(self.n_arms)}
        
        # Feature processing
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self._fitted = False  # Flag to track fitting
    
    def _prepare_context(self, user_feats, session_context):
        """
        Create standardized feature vector identical to the original bandit implementation.
        
        Maintains compatibility with existing feature engineering while ensuring
        consistent context representation across both bandit versions.
        
        Args:
            user_feats (dict): User-level features (see EQuinoxBandit._prepare_context)
            session_context (dict): Session-level features (see EQuinoxBandit._prepare_context)
            
        Returns:
            np.ndarray: Standardized feature vector of shape (1, n_features)
        """
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
        """
        Select action using hybrid LightGBM exploitation and Thompson sampling exploration.
        
        Intelligently balances between model predictions and exploration based on
        data availability and maintains business rule constraints for practical deployment.
        
        Args:
            user_feats (dict): User-level features (see _prepare_context for details)
            session_context (dict): Session-level features (see _prepare_context for details)
            
        Returns:
            int: Index of the selected action in self.actions list
            
        Strategy:
            - Uses LightGBM predictions when sufficient data is available (>10 samples)
            - Falls back to Thompson sampling for under-explored actions
            - Applies adaptive weighting: trusts LightGBM more with more data
            - Maintains minimum 20% exploration rate for continual learning
            - Applies same business rules as original implementation
        """
        context = self._prepare_context(user_feats, session_context)
        
        if not self._fitted:
            self.scaler.fit(context)
            self.imputer.fit(context)
            self._fitted = True
            
        scaled_context = self.scaler.transform(self.imputer.transform(context))
        
        # Get predictions from all models
        gbm_probs = []
        for action in range(self.n_arms):
            if len(self.y[action]) > 10:  # Enough data to predict
                try:
                    prob = self.models[action].predict_proba(scaled_context)[0][1]
                    gbm_probs.append(prob)
                except:
                    gbm_probs.append(0.5)
            else:
                gbm_probs.append(0.5)
        
        # Thompson sampling probabilities
        thompson_probs = [np.random.beta(self.alpha[action], self.beta[action]) 
                         for action in range(self.n_arms)]
        
        # Adaptive weighting: trust LightGBM more as we get more data
        combined_probs = []
        for action in range(self.n_arms):
            data_count = len(self.y[action])
            weight = 1 - (1 - self.min_exploration) * np.exp(-data_count / 2000.0)
            
            # Cap the weight to ensure exploration
            weight = min(weight, 0.8)  # Never more than 80% exploitation
            combined_prob = (weight * gbm_probs[action] + 
                           (1 - weight) * thompson_probs[action])
            combined_probs.append(combined_prob)
        
        # Apply business rules
        final_probs = np.array(combined_probs)
        
        # Business rules (unchanged)
        if user_feats.get('avg_price_viewed', 0) < 50:
            discount_idx = 1  # email_10%_discount
            final_probs[discount_idx] *= 0.3
        
        if session_context.get('is_weekend', False):
            final_probs[0] *= 1.2  # email_no_discount
            final_probs[3] *= 1.5  # cart_reminder
            final_probs[1] *= 0.3  # discount
            final_probs[2] *= 0.3  # banner
        
        chosen_action = np.argmax(final_probs)
        self.action_counts[chosen_action] += 1
        
        return chosen_action
    
    def update(self, user_feats, session_context, action, reward):
        """
        Update bandit with new observation using batch training approach.
        
        Stores context-reward pairs and periodically retrains LightGBM models
        while maintaining Thompson sampling counts for exploration.
        
        Args:
            user_feats (dict): User-level features for context
            session_context (dict): Session-level features for context  
            action (int): Index of the action that was taken
            reward (int): Binary reward signal (1 = conversion, 0 = no conversion)
            
        Note:
            LightGBM models are trained every 100 samples to balance computational
            efficiency with model freshness. Thompson counts are updated immediately.
        """
        context = self._prepare_context(user_feats, session_context)
        
        if not self._fitted:
            self.scaler.fit(context)
            self.imputer.fit(context)
            self._fitted = True
            
        scaled_context = self.scaler.transform(self.imputer.transform(context))
        
        # Update Thompson counts
        if reward == 1:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1
        
        # Store data for training
        self.X[action].append(scaled_context.flatten())
        self.y[action].append(reward)
        
        # Train LightGBM model periodically
        if len(self.y[action]) % 100 == 0 and len(self.y[action]) > 50:
            try:
                X_train = np.array(self.X[action])
                y_train = np.array(self.y[action])
                
                # Skip if only one class
                if len(np.unique(y_train)) > 1:
                    self.models[action].fit(X_train, y_train)
            except Exception as e:
                print(f"Training failed for action {action}: {e}")
    
    def get_action_stats(self):
        """
        Return comprehensive performance metrics for each action.
        
        Provides detailed insights including both bandit selection statistics
        and underlying Thompson sampling parameters for debugging and analysis.
        
        Returns:
            pd.DataFrame: Performance statistics with columns:
                - action: Action name
                - count: Number of times action was selected
                - avg_reward: Average reward obtained
                - conversion_rate: Conversion rate for this action
                - samples_collected: Number of training samples collected
                - thompson_alpha: Thompson sampling alpha parameter (successes + 1)
                - thompson_beta: Thompson sampling beta parameter (failures + 1)
        """
        stats = []
        for action in range(self.n_arms):
            if self.action_counts[action] > 0:
                rewards = np.array(self.y[action])
                stats.append({
                    'action': self.actions[action],
                    'count': self.action_counts[action],
                    'avg_reward': np.mean(rewards) if len(rewards) > 0 else 0,
                    'conversion_rate': np.sum(rewards) / len(rewards) if len(rewards) > 0 else 0,
                    'samples_collected': len(self.y[action]),
                    'thompson_alpha': self.alpha[action],
                    'thompson_beta': self.beta[action]
                })
            else:
                stats.append({
                    'action': self.actions[action],
                    'count': 0,
                    'avg_reward': 0,
                    'conversion_rate': 0,
                    'samples_collected': 0,
                    'thompson_alpha': self.alpha[action],
                    'thompson_beta': self.beta[action]
                })
        return pd.DataFrame(stats)