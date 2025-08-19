# Importing libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    def __init__(self):
        self.category_hierarchy = None
        self.scaler = StandardScaler()
        
    def build_features(self, events, items, categories):
        """Main feature engineering pipeline"""
        # Convert ISO strings back to datetime for processing
        events['timestamp'] = pd.to_datetime(events['timestamp'])
        items['timestamp'] = pd.to_datetime(items['timestamp'])

        # 1. Create category hierarchy
        self.category_hierarchy = categories.set_index('categoryid')['parentid'].to_dict()
        
        # 2. Mark conversions (transactions)
        transactions = events[events['event'] == 'transaction']
        events['converted'] = events['visitorid'].isin(transactions['visitorid']).astype(int)
        
        # 3. Add item features
        events = self._add_item_features(events, items)
        
        # 4. Create user features
        user_features = self._create_user_features(events)
        
        # 5. Robust missing values handling
        events = self._handle_missing_values(events)
        user_features = self._handle_missing_values(user_features)
        
        return events, user_features
    
    def _add_item_features(self, events, items):
        """Merge item properties into events"""
        # 1. Get categories (only reliable property)
        categories = (
            items[items['property'] == 'categoryid']
            .drop_duplicates(['itemid', 'value'])
        )
        categories['category'] = categories['value'].astype(int).map(self.category_hierarchy)
        
        # 2. Get availability (non-hashed property)
        availability = (
            items[items['property'] == 'available']
            .drop_duplicates(['itemid', 'value'])
        )
        availability['is_available'] = availability['value'].astype(int)
        
        # 3. Find numeric properties (more robust handling of hashed values)
        numeric_items = items[items['value'].str.startswith('n', na=False)].copy()
        
        # Extract first numeric value from each cell (handles cases like 'n5.000 n10.000')
        numeric_items['numeric_value'] = numeric_items['value'].str.extract(r'(n[-+]?\d+\.\d+)')[0]
        numeric_items = numeric_items.dropna(subset=['numeric_value'])
        numeric_items['price'] = numeric_items['numeric_value'].str[1:].astype(float)  # Strip 'n' prefix
        
        # If no numeric properties found, create dummy price
        if numeric_items.empty:
            prices = pd.DataFrame({'itemid': events['itemid'].unique(), 'price': 100})
        else:
            prices = numeric_items.drop_duplicates(['itemid', 'price'])
        
        # Merge all features
        events = events.merge(
            prices[['itemid', 'price']],
            on='itemid',
            how='left'
        ).merge(
            categories[['itemid', 'category']],
            on='itemid',
            how='left'
        ).merge(
            availability[['itemid', 'is_available']],
            on='itemid',
            how='left'
        )
        
        return events
    
    def _create_user_features(self, events):
        """Generate all user-level features"""
        # Temporal features
        events['hour'] = events['timestamp'].dt.hour
        events['is_weekend'] = events['timestamp'].dt.weekday >= 5
        events['month'] = events['timestamp'].dt.month
        
        # User-level aggregations
        user_features = events.groupby('visitorid').agg({
            'timestamp': ['min', 'max', 'count'],
            'event': lambda x: (x == 'view').sum(),
            'itemid': 'nunique',
            'category': 'nunique',
            'price': ['mean', 'max'],
            'is_available': 'mean',  # Average availability of viewed items
            'converted': 'max'
        })
        
        # Flatten multi-index columns
        user_features.columns = [
            'first_seen', 'last_seen', 'total_events',
            'total_views', 'unique_items', 'unique_categories',
            'avg_price_viewed', 'max_price_viewed', 'avg_availability',
            'has_converted'
        ]
        
        # Derived features
        user_features['user_tenure_days'] = (
            (user_features['last_seen'] - user_features['first_seen']).dt.days
        ).clip(1)  # Avoid division by zero
        
        user_features['events_per_day'] = (
            user_features['total_events'] / user_features['user_tenure_days']
        )
        
        user_features['conversion_rate'] = (
            user_features['has_converted'] / user_features['total_views'].clip(1)
        )
        
        # Fill missing values
        user_features['avg_price_viewed'] = user_features['avg_price_viewed'].fillna(0)
        user_features['max_price_viewed'] = user_features['max_price_viewed'].fillna(0)
        user_features['avg_availability'] = user_features['avg_availability'].fillna(0)
        
        return user_features
    
    def _handle_missing_values(self, df):
        """Fill missing values with reasonable defaults"""
        # For prices - use median of existing values
        if 'price' in df.columns:
            median_price = df['price'].median()
            df['price'] = df['price'].fillna(median_price)
        
        # For categories - create an "unknown" category
        if 'category' in df.columns:
            df['category'] = df['category'].fillna(-1).astype(int)
            
        # For availability - assume items are available if data is missing
        if 'is_available' in df.columns:
            df['is_available'] = df['is_available'].fillna(1).astype(int)
        
        return df