import yaml
from src.components.data_augmentation import DataAugmentation
from src.utils.logger import logging
from src.utils.exception import CustomException

def main():
    logging.info("Stage 04 Started")
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    augmenter = DataAugmentation(config)
    augmenter.augment()
    logging.info("Stage 04 Completed")


if __name__ == "__main__":
    main()