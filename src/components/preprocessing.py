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

            # # z-score normalization
            # slice_2d = (slice_2d - np.mean(slice_2d)) / (
            #     np.std(slice_2d) + 1e-8
            # )

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
# import os
# import sys
# import numpy as np
# import nibabel as nib
# import cv2
# from dataclasses import dataclass

# from src.utils.common import read_yaml
# from src.utils.logger import logging
# from src.utils.exception import CustomException


# CONFIG_PATH = "config/config.yaml"


# @dataclass
# class PreprocessingConfig:
#     split_dir: str
#     output_dir: str
#     image_size: int


# class Preprocessing:

#     def __init__(self):

#         config = read_yaml(CONFIG_PATH)

#         self.config = PreprocessingConfig(
#             split_dir=config["preprocessing"]["split_dir"],
#             output_dir=config["preprocessing"]["output_dir"],
#             image_size=config["preprocessing"]["image_size"]
#         )

#     # =========================================================
#     # PROCESS SUBJECT
#     # =========================================================
#     def process_subject(self, mgz_path, multi_slice=True):

#         img = nib.load(mgz_path)
#         img = nib.as_closest_canonical(img)
#         volume = img.get_fdata()

#         # ✅ FIX 1: Normalize entire volume ONCE (not per slice)
#         # Prevents empty edge slices from getting the same scale as full brain slices
#         volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)

#         mid = volume.shape[0] // 2  # sagittal midline (~91 in MNI 182-voxel space)

#         if multi_slice:
#             # ✅ FIX 2: Targeted sampling from 3 neurologically meaningful zones
#             # Zone 1 — Medial (corpus callosum, ACC, mPFC) — ASD-relevant
#             medial    = list(range(-5, 6))           # 11 slices around midline
#             # Zone 2 — Left lateral (STG, amygdala, insula) — ASD-relevant
#             lateral_l = list(range(-25, -10, 3))     # 5 slices
#             # Zone 3 — Right lateral (mirror of left)
#             lateral_r = list(range(11, 26, 3))       # 5 slices
#             offsets   = medial + lateral_l + lateral_r  # 21 slices total
#         else:
#             offsets = [0]

#         slices = []

#         for off in offsets:

#             idx = mid + off

#             if idx < 0 or idx >= volume.shape[0]:
#                 continue

#             slice_2d = volume[idx, :, :]

#             # Orientation fix
#             slice_2d = np.rot90(slice_2d)

#             # ✅ FIX 3: Clip outlier intensities after volume-level normalization
#             # Prevents extreme scanner noise spikes from skewing the PNG scaling
#             slice_2d = np.clip(slice_2d, -3, 3)

#             # Resize
#             slice_2d = cv2.resize(
#                 slice_2d,
#                 (self.config.image_size, self.config.image_size)
#             )

#             # ✅ Return (array, offset) tuple for metadata tracking
#             slices.append((slice_2d, off))

#         return slices

#     # =========================================================
#     # SCALE → PNG
#     # =========================================================
#     def save_png(self, arr):
#         """
#         arr: 2D numpy float array (volume-normalized, clipped)
#         Returns uint8 array scaled to [0, 255]
#         """
#         min_val = arr.min()
#         max_val = arr.max()
#         scaled  = (arr - min_val) / (max_val - min_val + 1e-8)
#         return (scaled * 255).astype("uint8")

#     # =========================================================
#     # MAIN RUN
#     # =========================================================
#     def run(self):

#         try:

#             logging.info("Starting MRI Preprocessing")

#             for split_name in ["train", "test"]:

#                 split_root = os.path.join(
#                     self.config.split_dir,
#                     split_name
#                 )

#                 multi_slice_flag = True  # both train and test use multi-slice

#                 for class_folder in ["autism", "control"]:

#                     class_path = os.path.join(
#                         split_root,
#                         class_folder
#                     )

#                     if not os.path.exists(class_path):
#                         continue

#                     subjects = os.listdir(class_path)

#                     logging.info(
#                         f"{split_name}/{class_folder} → {len(subjects)} subjects"
#                     )

#                     for subject in subjects:

#                         mgz_path = os.path.join(
#                             class_path,
#                             subject,
#                             "mri",
#                             "brain.mgz"
#                         )

#                         if not os.path.exists(mgz_path):
#                             continue

#                         processed_slices = self.process_subject(
#                             mgz_path,
#                             multi_slice=multi_slice_flag
#                         )

#                         image_dir = os.path.join(
#                             self.config.output_dir,
#                             split_name,
#                             class_folder
#                         )

#                         os.makedirs(image_dir, exist_ok=True)

#                         # ✅ FIX 4: Unpack (arr, offset) tuple correctly
#                         # Previously passed the whole tuple to save_png → AttributeError
#                         for i, (arr, offset) in enumerate(processed_slices):

#                             png_img = self.save_png(arr)

#                             # Save with slice index for subject-level aggregation
#                             save_name = f"{subject}_slice{i}.png"

#                             cv2.imwrite(
#                                 os.path.join(image_dir, save_name),
#                                 png_img
#                             )

#             logging.info("Preprocessing Completed Successfully")

#         except Exception as e:
#             raise CustomException(e, sys)