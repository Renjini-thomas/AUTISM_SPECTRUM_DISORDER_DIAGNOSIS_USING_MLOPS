import yaml
from src.utils.exception import CustomException
import sys


def read_yaml(file_path):

    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    except Exception as e:
        raise CustomException(e, sys)