# import os
# import shutil
# import sys
# from dataclasses import dataclass

# from src.utils.exception import CustomException
# from src.utils.logger import logging
# from src.utils.common import read_yaml


# CONFIG_PATH = "config/config.yaml"


# @dataclass
# class DataIngestionConfig:
#     source_dir: str
#     destination_dir: str


# class DataIngestion:

#     def __init__(self):

#         config = read_yaml(CONFIG_PATH)

#         self.config = DataIngestionConfig(
#             source_dir=config["data_ingestion"]["source_dir"],
#             destination_dir=config["data_ingestion"]["destination_dir"]
#         )

#     def ingest(self):

#         try:

#             logging.info("Starting Data Ingestion")

#             autistic_src = os.path.join(self.config.source_dir, "autistic")
#             non_autistic_src = os.path.join(self.config.source_dir, "non-autistic")

#             autistic_dst = os.path.join(self.config.destination_dir, "autistic")
#             non_autistic_dst = os.path.join(self.config.destination_dir, "non_autistic")

#             os.makedirs(autistic_dst, exist_ok=True)
#             os.makedirs(non_autistic_dst, exist_ok=True)

#             # copy autistic
#             for file in os.listdir(autistic_src):
#                 if file.endswith(".nii") or file.endswith(".nii.gz"):

#                     shutil.copy(
#                         os.path.join(autistic_src, file),
#                         os.path.join(autistic_dst, file)
#                     )

#             # copy non autistic
#             for file in os.listdir(non_autistic_src):
#                 if file.endswith(".nii") or file.endswith(".nii.gz"):

#                     shutil.copy(
#                         os.path.join(non_autistic_src, file),
#                         os.path.join(non_autistic_dst, file)
#                     )

#             logging.info("Data Ingestion Completed Successfully")

#         except Exception as e:
#             raise CustomException(e, sys)
import os
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.common import read_yaml


CONFIG_PATH = "config/config.yaml"


@dataclass
class DataIngestionConfig:
    s3_mri_path: str
    phenotypic_url: str
    raw_data_dir: str
    mri_dir: str
    phenotypic_file: str


class DataIngestion:

    def __init__(self):

        config = read_yaml(CONFIG_PATH)["data_ingestion"]

        self.config = DataIngestionConfig(
            s3_mri_path=config["s3_mri_path"],
            phenotypic_url=config["phenotypic_url"],
            raw_data_dir=config["raw_data_dir"],
            mri_dir=config["mri_dir"],
            phenotypic_file=config["phenotypic_file"]
        )

    def create_dirs(self):

        Path(self.config.raw_data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.mri_dir).mkdir(parents=True, exist_ok=True)

    def download_mri(self):

        logging.info("Downloading ABIDE FreeSurfer selective MRI files")

        command = [
            "aws",
            "s3",
            "sync",
            self.config.s3_mri_path,
            self.config.mri_dir,
            "--exclude", "*",
            "--include", "*/mri/brain.mgz",
            "--include", "*/stats/aseg.stats",
            "--include", "*/stats/lh.aparc.stats",
            "--include", "*/stats/rh.aparc.stats",
            "--no-sign-request",
        ]

        subprocess.run(command, check=True)

        logging.info("MRI selective sync completed")

    def download_phenotype(self):

        logging.info("Downloading phenotypic CSV")

        command = [
            "curl",
            "-L",
            self.config.phenotypic_url,
            "-o",
            self.config.phenotypic_file
        ]

        subprocess.run(command, check=True)

        logging.info("Phenotypic file downloaded")

    def ingest(self):

        try:

            logging.info("Data Ingestion started")

            self.create_dirs()
            self.download_mri()
            self.download_phenotype()

            logging.info("Data Ingestion completed successfully")

        except Exception as e:
            raise CustomException(e, sys)