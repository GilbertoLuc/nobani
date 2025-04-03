import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
from src import config
from src.load_data import load_data
from src.make_model import train_rf_model_l, train_rf_model_a

logging.basicConfig(filename='../log/pipeline.log', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logging.info("Starting Sentiment Analysis Pipeline...")

    logging.info("Loading raw data...")
    load_data()

    logging.info("Training the model...")
    train_rf_model_l()
    train_rf_model_a()


if __name__ == "__main__": # questo funziona in modo che if sia accettato solo se stiamo facendo girare 
    main()  # main() nel codice in cui è stato creato, senza importarlo
    