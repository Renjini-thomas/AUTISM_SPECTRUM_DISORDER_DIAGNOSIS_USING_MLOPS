import logging
import os
from datetime import datetime

def setup_logger():

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir,
        f"run_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    )

    logging.basicConfig(
        filename=log_file,
        format="[ %(asctime)s ] %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )

    return log_file