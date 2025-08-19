# Importing libraries
import pandas as pd
from pandas import isnull
import json
import numpy as np
import polars as pl
import pickle
from pathlib import Path
import traceback
from typing import Optional

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
from equinox_fe import FeatureEngineer
from equinox_loading import load_bandit
from equinox_bandit import EQuinoxBandit
from equinox_training import train_bandit
from equinox_saving import save_bandit

# Configuration
RANDOM_SEED = 69
np.random.seed(RANDOM_SEED)

# class SafeJSONEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, (np.float32, np.float64)):
#             return float(obj) if not isnull(obj) else None
#         return super().default(obj)


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
            "events": events.replace({np.nan: None}).to_dict(orient='records'),
            "items": item_props.replace({np.nan: None}).to_dict(orient='records'),
            "categories": categories.replace({np.nan: None}).to_dict(orient='records')
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
                "events": events.replace({np.nan: None}).to_dict(orient='records'),
                "items": items.replace({np.nan: None}).to_dict(orient='records'),
                "categories": categories.replace({np.nan: None}).to_dict(orient='records')
            }
        
        finally:
            conn.close()


    
    # ======================================================================================================

    @task
    def update_df(loaded_data, new_data):
        events = pd.concat([
            pd.DataFrame(loaded_data["events"]),
            pd.DataFrame(new_data["events"])
        ]).drop_duplicates()
        
        items = pd.concat([
            pd.DataFrame(loaded_data["items"]),
            pd.DataFrame(new_data["items"])
        ]).drop_duplicates()
        
        categories = pd.concat([
            pd.DataFrame(loaded_data["categories"]),
            pd.DataFrame(new_data["categories"])
        ]).drop_duplicates()
        
        print("Updating data is Success")
        return {
            "events": events.replace({np.nan: None}).to_dict(orient='records'),
            "items": items.replace({np.nan: None}).to_dict(orient='records'),
            "categories": categories.replace({np.nan: None}).to_dict(orient='records')
        }

    # ======================================================================================================

    @task()
    def preprocess_data(data):
        events = pd.DataFrame(data["events"])
        items = pd.DataFrame(data["items"])
        categories = pd.DataFrame(data["categories"])
        
        # Convert timestamps to datetime and then to ISO format strings
        events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms').dt.strftime('%Y-%m-%dT%H:%M:%S')
        items['timestamp'] = pd.to_datetime(items['timestamp'], unit='ms').dt.strftime('%Y-%m-%dT%H:%M:%S')

        print("Preprocessed data is Success")
        return {
            "events": events.replace({np.nan: None}).to_dict(orient='records'),
            "items": items.replace({np.nan: None}).to_dict(orient='records'),
            "categories": categories.replace({np.nan: None}).to_dict(orient='records')
        }

    # ======================================================================================================
    @task()
    def feature_engineering(processed_data):
        # Inisialisasi FeatureEngineer
        fe = FeatureEngineer()
        
        # Ekstrak data dari processed_data
        events = pd.DataFrame(processed_data["events"])
        items = pd.DataFrame(processed_data["items"])
        categories = pd.DataFrame(processed_data["categories"])
        
        # Bangun fitur
        processed_events, user_features = fe.build_features(events, items, categories)

        # Convert KNOWN timestamp columns - add any others your FeatureEngineer creates
        processed_events['timestamp'] = processed_events['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        user_features['first_seen'] = user_features['first_seen'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        user_features['last_seen'] = user_features['last_seen'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        print("Feature engineering data is Success")
        return {
            "processed_events": processed_events.replace({np.nan: None}).to_dict(orient='records'),
            "user_features": user_features.replace({np.nan: None}).to_dict(orient='records')
        }

    # ======================================================================================================

    @task()
    def loading(save_dir):
        """Task to load bandit metadata without unpickling models"""
        bandit_info = load_bandit(f"/opt/airflow/dags/{save_dir}")
        if isinstance(bandit_info, dict) and 'error' in bandit_info:
            raise ValueError(f"Failed to load bandit: {bandit_info['error']}")
        return bandit_info

    # ======================================================================================================

    @task()
    def training(bandit_info, processed_data):
        """Robust bandit training with full error handling"""
        try:
            # 1. Load and validate inputs
            events = pd.DataFrame(processed_data["processed_events"])
            user_features = pd.DataFrame(processed_data["user_features"])
            
            # Convert ISO timestamps
            for df in [events, user_features]:
                for col in df.columns:
                    if df[col].dtype == 'object' and df[col].str.contains('T').any():
                        df[col] = pd.to_datetime(df[col])
            
            # 2. Validate data
            missing_users = set(events['visitorid']) - set(user_features.index)
            if missing_users:
                print(f"Warning: {len(missing_users)} users missing features")
            
            # 3. Load bandit
            save_dir = Path(bandit_info['save_dir'])
            with open(save_dir / 'bandit_params.pkl', 'rb') as f:
                bandit_data = pickle.load(f)
            
            bandit = EQuinoxBandit(bandit_data['actions'])
            bandit.action_stats = bandit_data['action_stats']
            bandit._imputer_fitted = bandit_data['_imputer_fitted']
            
            # Load models
            for i in range(bandit.n_arms):
                with open(save_dir / f'model_action_{i}.pkl', 'rb') as f:
                    bandit.models[i] = pickle.load(f)
            
            # Load sklearn components
            with open(save_dir / 'scaler.pkl', 'rb') as f:
                bandit.scaler = pickle.load(f)
            with open(save_dir / 'imputer.pkl', 'rb') as f:
                bandit.imputer = pickle.load(f)
            
            # 4. Train
            updated_bandit, history_df = train_bandit(bandit, events, user_features)
            
            # 5. Save
            updated_path = save_bandit(updated_bandit, bandit_info['save_dir'])
            
            # 6. Prepare safe output (handle None case)
            safe_history = []
            if history_df is not None:
                safe_history = (
                    history_df.replace({np.nan: None})
                    .assign(**{
                        col: lambda x: x[col].dt.strftime('%Y-%m-%dT%H:%M:%S')
                        for col in history_df.columns
                        if pd.api.types.is_datetime64_any_dtype(history_df[col])
                    })
                    .to_dict(orient='records')
                )
            
            return {
                'status': 'success',
                'updated_path': updated_path,
                'history': safe_history,
                'users_processed': len(events['visitorid'].unique()),
                'missing_users': len(missing_users)
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc(),  # Now properly imported
                'failed_at': pd.Timestamp.now().isoformat()
            }
    
    # ======================================================================================================

    @task()
    def saving(training_result):
        if not training_result.get('save_success'):
            raise ValueError(f"Bandit not saved! Path: {training_result.get('updated_path')}")
        
        return {
            'final_path': training_result['updated_path'],
            'verified_at': pd.Timestamp.now().isoformat()
        }
    
    # ======================================================================================================

    new_data = new_data()

    loaded_data = load_files()
    updated_data = update_df(loaded_data, new_data)
    timestamp_change1 = preprocess_data(updated_data)
    features1 = feature_engineering(timestamp_change1)

    equinox_bandit = loading("equinox_model_v1.1")
    timestamp_change2 = preprocess_data(new_data)
    features2 = feature_engineering(timestamp_change2)
    update_bandit = training(equinox_bandit, features2)
    saved_bandit = saving(update_bandit)

    # ======================================================================================================
    
    # end = EmptyOperator(task_id='end')

    start >> new_data >> loaded_data >>updated_data>>timestamp_change1>>features1 

    new_data >> equinox_bandit>> timestamp_change2>>features2>>update_bandit>>saved_bandit

    # ======================================================================================================
    