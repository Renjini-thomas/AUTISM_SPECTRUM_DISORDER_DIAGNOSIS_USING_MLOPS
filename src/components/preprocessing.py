import os
import sys
import numpy as np
import nibabel as nib
import cv2
from dataclasses import dataclass

from src.utils.common import read_yaml
from src.utils.logger import logging
from src.utils.exception import CustomException


CONFIG_PATH = "config/config.yaml"


@dataclass
class PreprocessingConfig:
    split_dir: str
    output_dir: str
    image_size: int


class Preprocessing:

    def __init__(self):

        config = read_yaml(CONFIG_PATH)

        self.config = PreprocessingConfig(
            split_dir=config["preprocessing"]["split_dir"],
            output_dir=config["preprocessing"]["output_dir"],
            image_size=config["preprocessing"]["image_size"]
        )

    # =========================================================
    # PROCESS SUBJECT (MULTI SLICE FOR TRAIN / SINGLE FOR TEST)
    # =========================================================
    def process_subject(self, mgz_path, multi_slice=True):

        img = nib.load(mgz_path)
        img = nib.as_closest_canonical(img)
        volume = img.get_fdata()

        mid = volume.shape[0] // 2
        # 11 slices
        offsets = list(range(-5,6)) if multi_slice else [0]

        # 9 slices
        # offsets = list(range(-4,5)) if multi_slice else [0]

        slices = []

        for off in offsets:

            idx = mid + off

            if idx < 0 or idx >= volume.shape[0]:
                continue

            slice_2d = volume[idx, :,: ]

            # orientation fix
            slice_2d = np.rot90(slice_2d)

            # z-score normalization
            slice_2d = (slice_2d - np.mean(slice_2d)) / (
                np.std(slice_2d) + 1e-8
            )

            # resize
            slice_2d = cv2.resize(
                slice_2d,
                (self.config.image_size, self.config.image_size)
            )

            slices.append(slice_2d)

        return slices

    # =========================================================
    # SCALE → PNG
    # =========================================================
    def save_png(self, arr):

        min_val = arr.min()
        max_val = arr.max()

        scaled = (arr - min_val) / (max_val - min_val + 1e-8)

        return (scaled * 255).astype("uint8")

    # =========================================================
    # MAIN RUN
    # =========================================================
    def run(self):

        try:

            logging.info("Starting MRI Preprocessing")

            for split_name in ["train", "test"]:

                split_root = os.path.join(
                    self.config.split_dir,
                    split_name
                )

                # ⭐ NOW BOTH TRAIN AND TEST → MULTI SLICE
                multi_slice_flag = True

                for class_folder in ["autism", "control"]:

                    class_path = os.path.join(
                        split_root,
                        class_folder
                    )

                    if not os.path.exists(class_path):
                        continue

                    subjects = os.listdir(class_path)

                    for subject in subjects:

                        mgz_path = os.path.join(
                            class_path,
                            subject,
                            "mri",
                            "brain.mgz"
                        )

                        if not os.path.exists(mgz_path):
                            continue

                        processed_slices = self.process_subject(
                            mgz_path,
                            multi_slice=multi_slice_flag
                        )

                        image_dir = os.path.join(
                            self.config.output_dir,
                            split_name,
                            class_folder
                        )

                        os.makedirs(image_dir, exist_ok=True)

                        for i, arr in enumerate(processed_slices):

                            png_img = self.save_png(arr)

                            # ⭐ ALWAYS SAVE WITH SLICE INDEX NOW
                            save_name = f"{subject}_slice{i}.png"

                            cv2.imwrite(
                                os.path.join(image_dir, save_name),
                                png_img
                            )

            logging.info("Preprocessing Completed Successfully")

        except Exception as e:
            raise CustomException(e, sys)