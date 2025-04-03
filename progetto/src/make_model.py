from src import config
import sqlite3
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import sys
sys.path.append(os.path.abspath('..'))  # Adds the parent directory to sys.path

import logging
# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data_l():
    """Loads data from the SQLite database."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    query = f"SELECT latitudine, longitudine, prezzo_per_unità FROM {config.RAW_TABLE}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def train_rf_model_l():
    """Trains a Random Forest regression model and saves evaluation metrics to CSV."""
    df = load_data_l()  # Load and sample data
    df_indices = df.index  # Preserve original indices

    # Feature extraction
    X = df[['latitudine', 'longitudine']]
    y = df['prezzo_per_unità']

    # Train-test split
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df_indices, test_size=0.2, random_state=42
    )

    # Train RandomForest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Save the model
    logging.info('Saving model...')
    with open(os.path.join(config.MODELS_PATH, "random_forest_l.pickle"), "wb") as file:
        pickle.dump(rf, file)

    # Save test predictions
    test_df = df.loc[test_idx].copy()
    test_df['prediction'] = y_pred

    # Compute metrics
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'r2_score': r2_score(y_test, y_pred)
    }

    # Save predictions and metrics in the database
    conn = sqlite3.connect(config.DATABASE_PATH)
    test_df.to_sql(config.PREDICTIONS_TABLE, conn, if_exists='replace', index=False)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_sql(config.EVALUATION_TABLE, conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    logging.info("Training completed and results saved.")

def load_data_a():
    """Loads data from the SQLite database."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    query = f"SELECT età_casa, dist_MRT, n_negozi_convenienti, prezzo_per_unità FROM {config.RAW_TABLE}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def train_rf_model_a():
    """Trains a Random Forest regression model and saves evaluation metrics to CSV."""
    df = load_data_a()  # Load and sample data
    df_indices = df.index  # Preserve original indices

    # Feature extraction
    X = df[['età_casa', 'dist_MRT', 'n_negozi_convenienti']]
    y = df['prezzo_per_unità']

    # Train-test split
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df_indices, test_size=0.2, random_state=42
    )

    # Train RandomForest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Save the model
    logging.info('Saving model...')
    with open(os.path.join(config.MODELS_PATH, "random_forest_a.pickle"), "wb") as file:
        pickle.dump(rf, file)

    # Save test predictions
    test_df = df.loc[test_idx].copy()
    test_df['prediction'] = y_pred

    # Compute metrics
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'r2_score': r2_score(y_test, y_pred)
    }

    # Save predictions and metrics in the database
    conn = sqlite3.connect(config.DATABASE_PATH)
    test_df.to_sql(config.PREDICTIONS_TABLE, conn, if_exists='replace', index=False)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_sql(config.EVALUATION_TABLE, conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    logging.info("Training completed and results saved.")