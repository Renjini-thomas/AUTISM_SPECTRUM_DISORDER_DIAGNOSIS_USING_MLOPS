from src.components.preprocessing import Preprocessing
from src.utils.logger import logging


if __name__ == "__main__":

    try:

        logging.info("Stage 03: MRI Preprocessing Started")

        Preprocessing().run()

        logging.info("Stage 03: MRI Preprocessing Completed")

    except Exception as e:

        logging.error(f"Stage 03 Failed: {e}")
        raise