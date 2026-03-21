# import os
# import sys
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from dataclasses import dataclass

# from src.utils.common import read_yaml
# from src.utils.logger import logging
# from src.utils.exception import CustomException


# CONFIG_PATH = "config/config.yaml"


# @dataclass
# class DataSplitConfig:
#     raw_data_dir: str
#     output_dir: str
#     test_size: float
#     random_state: int


# class DataSplit:

#     def __init__(self):

#         config = read_yaml(CONFIG_PATH)

#         self.config = DataSplitConfig(
#             raw_data_dir=config["data_split"]["raw_data_dir"],
#             output_dir=config["data_split"]["output_dir"],
#             test_size=config["data_split"]["test_size"],
#             random_state=config["data_split"]["random_state"]
#         )

#     def split(self):

#         try:

#             logging.info("Starting Subject Level Split")

#             autistic_dir = os.path.join(self.config.raw_data_dir, "autistic")
#             non_autistic_dir = os.path.join(self.config.raw_data_dir, "non_autistic")

#             autistic_files = [
#                 os.path.join(autistic_dir, f)
#                 for f in os.listdir(autistic_dir)
#                 if f.endswith(".nii") or f.endswith(".nii.gz")
#             ]

#             non_autistic_files = [
#                 os.path.join(non_autistic_dir, f)
#                 for f in os.listdir(non_autistic_dir)
#                 if f.endswith(".nii") or f.endswith(".nii.gz")
#             ]

#             df_autistic = pd.DataFrame({
#                 "path": autistic_files,
#                 "label": 1
#             })

#             df_non = pd.DataFrame({
#                 "path": non_autistic_files,
#                 "label": 0
#             })

#             df = pd.concat([df_autistic, df_non], ignore_index=True)

#             train_df, test_df = train_test_split(
#                 df,
#                 test_size=self.config.test_size,
#                 stratify=df["label"],
#                 random_state=self.config.random_state
#             )

#             os.makedirs(self.config.output_dir, exist_ok=True)

#             train_df.to_csv(
#                 os.path.join(self.config.output_dir, "train.csv"),
#                 index=False
#             )

#             test_df.to_csv(
#                 os.path.join(self.config.output_dir, "test.csv"),
#                 index=False
#             )

#             logging.info("Split Completed Successfully")

#         except Exception as e:
#             raise CustomException(e, sys)
import pandas as pd
import shutil
import sys
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.utils.common import read_yaml
from src.utils.logger import logging
from src.utils.exception import CustomException


CONFIG_PATH = "config/config.yaml"


# ⭐ SUBJECT BLACKLIST (NOISY / WRONG ORIENTATION)
UNWANTED_SUBJECTS = {

    # AUTISM
    "UCLA_51232","UCLA_51243","UCLA_51244","UCLA_51245",
    "UCLA_51246","UCLA_51247","UM_1_0050273","UM_1_0050277",
    "UM_1_0050283","UM_1_0050287","UM_1_0050309","UM_1_0050313",
    "UM_1_0050323","UCLA_51233","UCLA_51242","UM_1_0050272",
    "UM_1_0050274","UM_1_0050276","UM_1_0050288",

    # CONTROL
    "UCLA_2_0051312","UCLA_2_0051314","UCLA_51310",
    "UM_1_0050327","UM_1_0050328","UCLA_2_0051316",
    "UCLA_51270"
}


@dataclass
class DataPreparationConfig:
    raw_mri_dir: str
    phenotypic_file: str
    output_dir: str
    test_size: float
    random_state: int


class DataPreparation:

    def __init__(self):

        config = read_yaml(CONFIG_PATH)["data_preparation"]

        self.config = DataPreparationConfig(
            raw_mri_dir=config["raw_mri_dir"],
            phenotypic_file=config["phenotypic_file"],
            output_dir=config["output_dir"],
            test_size=config["test_size"],
            random_state=config["random_state"]
        )

    # --------------------------
    # BUILD MANIFEST
    # --------------------------

    def build_manifest(self):

        logging.info("Building ABIDE manifest")

        pheno = pd.read_csv(self.config.phenotypic_file)

        dataset = []

        skipped = 0

        for subject_dir in Path(self.config.raw_mri_dir).iterdir():

            if not subject_dir.is_dir():
                continue

            subject_id = subject_dir.name

            # ⭐ DISCARD BAD SUBJECTS
            if subject_id in UNWANTED_SUBJECTS:
                logging.info(f"Skipping unwanted subject → {subject_id}")
                skipped += 1
                continue

            brain_file = subject_dir / "mri" / "brain.mgz"

            if not brain_file.exists():
                continue

            try:
                sub_id = int(subject_id.split("_")[-1])
            except:
                continue

            row = pheno[pheno["SUB_ID"] == sub_id]

            if row.empty:
                continue

            dx = row.iloc[0]["DX_GROUP"]
            label = "autism" if dx == 1 else "control"

            dataset.append({
                "subject_id": subject_id,
                "label": label,
                "path": subject_dir
            })

        manifest = pd.DataFrame(dataset)

        logging.info(f"Total usable subjects: {len(manifest)}")
        logging.info(f"Total discarded subjects: {skipped}")

        return manifest

    # --------------------------
    # SPLIT + COPY
    # --------------------------

    def split_and_save(self, manifest):

        train, test = train_test_split(
            manifest,
            test_size=self.config.test_size,
            stratify=manifest["label"],
            random_state=self.config.random_state
        )

        for split_name, df in [("train", train), ("test", test)]:

            for _, row in df.iterrows():

                label = row["label"]
                src = Path(row["path"])

                dest = (
                    Path(self.config.output_dir)
                    / split_name
                    / label
                    / row["subject_id"]
                )

                if dest.exists():
                    continue

                dest.parent.mkdir(parents=True, exist_ok=True)

                shutil.copytree(src, dest)

        logging.info("Train-Test dataset creation completed")

    # --------------------------
    # RUN
    # --------------------------

    def run(self):

        try:

            logging.info("Data Preparation Started")

            manifest = self.build_manifest()

            self.split_and_save(manifest)

            logging.info("Data Preparation Finished")

        except Exception as e:
            raise CustomException(e, sys)