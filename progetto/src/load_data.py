import sqlite3
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
 # Adds the parent directory to sys.path
from src import config

import logging
# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data():
    logging.info('Opening Excel Files...')
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', 'data', 'Real estate valuation data set.xlsx'))
    df = pd.read_excel(dataset_path, header=0, index_col=0)
    df.columns = ['data', 'età_casa', 'dist_MRT', 'n_negozi_convenienti', 'latitudine', 'longitudine', 'prezzo_per_unità']
    df['data'] = pd.to_datetime(df['data'], errors='coerce')
    

    # Create a connection to the SQLite database (or create if it doesn't exist)
    conn = sqlite3.connect(config.DATABASE_PATH)

    # Write the DataFrame to a table (replace 'my_table' with your desired table name)
    df.to_sql(config.RAW_TABLE, conn, if_exists='replace', index=False)

    # Commit and close the connection
    conn.commit()
    conn.close()

    logging.info(f"Data successfully written to {config.RAW_TABLE} table.")
