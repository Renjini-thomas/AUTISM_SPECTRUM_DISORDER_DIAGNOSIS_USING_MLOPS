from src.components.data_preparation import DataPreparation
from src.utils.logger import logging


if __name__ == "__main__":

    logging.info("Stage 02 Data Preparation Started")

    preparation = DataPreparation()
    preparation.run()

    logging.info("Stage 02 Data Preparation Completed")