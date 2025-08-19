# Importing libraries
import pandas as pd
import numpy as np
import polars as pl

from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sqlalchemy import create_engine
from airflow import DAG
from airflow.decorators import task
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable

# different python
from equinox_inference import load_bandit
from equinox_training import train_bandit, save_bandit

# Configuration
RANDOM_SEED = 69
np.random.seed(RANDOM_SEED)


default_args= {
    'owner': 'Group',
    'start_date': datetime(2024, 11, 1)  # Start date diubah ke 1 November 2024
}

with DAG(
    'E-Quinox',
    description='from gdrive to postgres',
    schedule_interval='10,20,30 9 * * 6',
    default_args=default_args, 
    catchup=False) as dag:

    start = EmptyOperator(task_id='start')

    # ======================================================================================================

    @task()
    def new_data():
        def load_csv(table_name):
            return pd.read_csv(f"/opt/airflow/dags/{table_name}.csv")
        
        events =  load_csv("new_events")
        item_props =  load_csv("new_items")
        categories =  load_csv("new_categories")

        return {
            "status": "success",
            "data": {
                "events": events,
                "items": item_props,
                "categories": categories
            },
            "events_count": len(events),
            "items_count": len(item_props),
            "categories_count": len(categories)
        }

    # =======================================================================================

    @task()
    def load_files():
        database = "airflow"
        username = "airflow"
        password = "airflow"
        host = "host.docker.internal"
        port = "5434"

        postgres_url = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"

        engine = create_engine(postgres_url)
        conn = engine.connect()
        try:
        
            # Fungsi untuk load data dengan LIMIT
            def load_table(table_name, dtypes):
                return pd.read_sql(
                    f"SELECT * FROM {table_name} LIMIT 500000",  # Tetap pakai LIMIT 500k
                    conn,
                    dtype=dtypes
                )
            
            # Load semua tabel
            events = load_table("events", {
                "timestamp": "int64",
                "visitorid": "int32", 
                "event": "object",
                "itemid": "int32",
                "transactionid": "object"
            })
            
            item_props1 = load_table("item_properties_part1", {
                "timestamp": "int64",
                "itemid": "int32",
                "property": "object",
                "value": "object"
            })
            
            item_props2 = load_table("item_properties_part2", {
                "timestamp": "int64", 
                "itemid": "int32",
                "property": "object",
                "value": "object"
            })
            
            categories = load_table("category_tree", {
                "categoryid": "int32",
                "parentid": "float64"
            })
            
            # Gabungkan item properties
            items = pd.concat([item_props1, item_props2]).drop_duplicates()
            
            return {
                "status": "success",
                "data": {
                    "events": events,
                    "items": items,
                    "categories": categories
                },
                "events_count": len(events),
                "items_count": len(items),
                "categories_count": len(categories)
            }
        
        finally:
            conn.close()


    
    # ======================================================================================================

    @task
    def update_df(loaded_data, new_data):
        events1 = loaded_data['data']['events']
        items1 = loaded_data['data']['items']
        categories1 = loaded_data['data']['categories']

        events2 =  new_data['data']['events']
        items2 =  new_data['data']['items']
        categories2 =  new_data['data']['categories']

        events = pd.concat([events1, events2]).drop_duplicates()
        items = pd.concat([items1, items2]).drop_duplicates()
        categories = pd.concat([categories1, categories2]).drop_duplicates()
        
        print("Updating data is Success")
        return {
            "status": "success",
            "data": {
                "events": events.to_dict('records'),
                "items": items.to_dict('records'),
                "categories": categories.to_dict('records')
            },
            "events_count": len(events),
            "items_count": len(items),
            "categories_count": len(categories)
        }

    # ======================================================================================================

    @task()
    def preprocess_data(data):
        events = data['data']['events']
        items = data['data']['items']
        categories = data['data']['categories']
        
        events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')
        items['timestamp'] = pd.to_datetime(items['timestamp'], unit='ms')

        def clean_nans(df):
            return df.where(pd.notna(df), None).replace([np.nan], [None])
        
        events = clean_nans(events)
        items = clean_nans(items)
        categories = clean_nans(categories)

        print("Preprocessed data is Success")
        return {
            "events": events.to_dict('records'),
            "items": items.to_dict('records'),
            "categories": categories.to_dict('records')
        }

    # ======================================================================================================
    @task()
    def feature_engineering(processed_data):
        # Inisialisasi FeatureEngineer
        fe = FeatureEngineer()
        
        # Ekstrak data dari processed_data
        events = processed_data['events']
        items = processed_data['items']
        categories = processed_data['categories']
        
        # Bangun fitur
        processed_events, user_features = fe.build_features(events, items, categories)
        
        return {
            "processed_events": processed_events,
            "user_features": user_features
        }

    # Class FeatureEngineer tetap sama seperti yang Anda berikan
    class FeatureEngineer:
        def __init__(self):
            self.category_hierarchy = None
            self.scaler = StandardScaler()
            
        def build_features(self, events, items, categories):
            """Main feature engineering pipeline"""
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

    # ======================================================================================================

    @task()
    def loading(save_dir):
        equinox_bandit = load_bandit(f"/opt/airflow/dags/{save_dir}")

        return equinox_bandit

    # ======================================================================================================

    @task()
    def training(bandit, events, user_features):
        update_bandit, history_df = train_bandit(bandit, events, user_features)
   
        return {
            "update_bandit": update_bandit,
            "history_df": history_df
        }
    
    # ======================================================================================================

    @task()
    def saving(bandit, save_dir):
        return save_bandit(bandit, f"/opt/airflow/dags/{save_dir}")
    
    # ======================================================================================================

    new_data = new_data()

    loaded_data = load_files()
    updated_data = update_df(loaded_data, new_data)
    timestamp_change1 = preprocess_data(updated_data)
    features1 = feature_engineering(timestamp_change1)

    equinox_bandit = loading("equinox_model_v1.1")
    timestamp_change2 = preprocess_data(new_data)
    features2 = feature_engineering(timestamp_change2)
    update_bandit = training(equinox_bandit, features2["processed_events"], features2["user_features"])
    saved_bandit = saving(update_bandit["update_bandit"], "equinox_model_v1.1")

    # ======================================================================================================
    
    # end = EmptyOperator(task_id='end')

    start >> new_data >> loaded_data >>updated_data>>timestamp_change1>>features1 

    new_data >> equinox_bandit>> timestamp_change2>>features2>>update_bandit>>saved_bandit

    # ======================================================================================================
    