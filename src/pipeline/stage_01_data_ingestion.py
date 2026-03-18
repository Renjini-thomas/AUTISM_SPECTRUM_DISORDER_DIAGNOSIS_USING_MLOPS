from src.components.data_ingestion import DataIngestion
from src.utils.logger import logging


if __name__ == "__main__":

    logging.info("Stage 01 Data Ingestion Started")

    ingestion = DataIngestion()
    ingestion.ingest()

    logging.info("Stage 01 Completed")