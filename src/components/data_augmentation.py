# import numpy as np
# import cv2
# import random
# from pathlib import Path
# from tqdm import tqdm


# class DataAugmentation:

#     def __init__(self, config):

#         self.input_dir = Path(config["augmentation"]["input_dir"])
#         self.output_dir = Path(config["augmentation"]["output_dir"])

#         self.rotation = config["augmentation"]["rotation"]
#         self.noise_std = config["augmentation"]["noise_std"]
#         self.elastic_alpha = config["augmentation"]["elastic_alpha"]
#         self.elastic_sigma = config["augmentation"]["elastic_sigma"]

#         self.output_dir.mkdir(parents=True, exist_ok=True)

#     # ================= ROTATION =================
#     def rotate(self, image):

#         angle = random.uniform(-self.rotation, self.rotation)

#         h, w = image.shape

#         M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)

#         return cv2.warpAffine(
#             image,
#             M,
#             (w, h),
#             borderMode=cv2.BORDER_REFLECT_101
#         )

#     # ================= ZOOM =================
#     def zoom(self, image):

#         scale = random.uniform(0.97, 1.03)

#         h, w = image.shape

#         resized = cv2.resize(image, None, fx=scale, fy=scale)

#         if scale > 1:

#             startx = resized.shape[1]//2 - w//2
#             starty = resized.shape[0]//2 - h//2

#             return resized[starty:starty+h, startx:startx+w]

#         else:

#             pad_x = (w - resized.shape[1]) // 2
#             pad_y = (h - resized.shape[0]) // 2

#             padded = np.pad(
#                 resized,
#                 ((pad_y, pad_y), (pad_x, pad_x)),
#                 mode="reflect"
#             )

#             return padded[:h, :w]

#     # ================= ELASTIC =================
#     def elastic(self, image):

#         random_state = np.random.RandomState(None)

#         shape = image.shape

#         dx = cv2.GaussianBlur(
#             (random_state.rand(*shape) * 2 - 1),
#             (0, 0),
#             self.elastic_sigma
#         ) * self.elastic_alpha

#         dy = cv2.GaussianBlur(
#             (random_state.rand(*shape) * 2 - 1),
#             (0, 0),
#             self.elastic_sigma
#         ) * self.elastic_alpha

#         x, y = np.meshgrid(
#             np.arange(shape[1]),
#             np.arange(shape[0])
#         )

#         map_x = (x + dx).astype(np.float32)
#         map_y = (y + dy).astype(np.float32)

#         return cv2.remap(
#             image,
#             map_x,
#             map_y,
#             interpolation=cv2.INTER_LINEAR,
#             borderMode=cv2.BORDER_REFLECT_101
#         )

#     # ================= NOISE =================
#     def add_noise(self, image):

#         noise = np.random.normal(
#             0,
#             self.noise_std,
#             image.shape
#         )

#         noisy = image + noise

#         return np.clip(noisy, 0, 1)

#     # ================= SAVE IMAGE =================
#     def save_png(self, image, path):

#         img = (image * 255).astype("uint8")

#         cv2.imwrite(str(path), img)

#     # ================= MAIN =================
#     def augment(self):

#         print("Starting MRI Image Augmentation...")

#         for cls in ["autistic", "non_autistic"]:

#             input_path = self.input_dir / cls / "images"
#             output_path = self.output_dir / cls / "images"

#             output_path.mkdir(parents=True, exist_ok=True)

#             files = list(input_path.glob("*.png"))

#             # ⭐ CLASS BALANCED
#             factor = 2 if cls == "autistic" else 1

#             for file in tqdm(files, desc=f"Augmenting {cls}"):

#                 img = cv2.imread(str(file), 0) / 255.0

#                 # save original
#                 self.save_png(img, output_path / file.name)

#                 for i in range(factor):

#                     aug = self.rotate(img)
#                     aug = self.zoom(aug)
#                     aug = self.elastic(aug)
#                     aug = self.add_noise(aug)

#                     new_name = file.stem + f"_aug{i}.png"

#                     self.save_png(
#                         aug,
#                         output_path / new_name
#                     )

#         print("Augmentation Completed ✅")
# # import cv2
# # import numpy as np
# # import os
# # from pathlib import Path
# # import random
# # from tqdm import tqdm


# # class DataAugmentation:

# #     def __init__(self, config):

# #         self.input_dir = Path(config["augmentation"]["input_dir"])
# #         self.output_dir = Path(config["augmentation"]["output_dir"])
# #         self.factor = config["augmentation"]["augment_factor"]
# #         self.rotation = config["augmentation"]["rotation"]
# #         self.noise_std = config["augmentation"]["noise_std"]

# #         self.output_dir.mkdir(parents=True, exist_ok=True)

# #     def rotate(self, image):
# #         angle = random.uniform(-self.rotation, self.rotation)
# #         h, w = image.shape
# #         M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
# #         return cv2.warpAffine(image, M, (w, h))

# #     def add_noise(self, image):
# #         noise = np.random.normal(0, self.noise_std, image.shape)
# #         return np.clip(image + noise, 0, 1)

# #     def zoom(self, image):
# #         scale = random.uniform(0.95, 1.05)
# #         h, w = image.shape
# #         resized = cv2.resize(image, None, fx=scale, fy=scale)

# #         if scale > 1:
# #             startx = resized.shape[1]//2 - w//2
# #             starty = resized.shape[0]//2 - h//2
# #             return resized[starty:starty+h, startx:startx+w]
# #         else:
# #             pad_x = (w - resized.shape[1])//2
# #             pad_y = (h - resized.shape[0])//2
# #             return np.pad(resized, ((pad_y,pad_y),(pad_x,pad_x)), mode='constant')

# #     def augment(self):

# #         for cls in ["autistic", "non_autistic"]:

# #             input_path = self.input_dir / cls / "images"
# #             output_path = self.output_dir / cls / "images"
# #             output_path.mkdir(parents=True, exist_ok=True)

# #             images = list(input_path.glob("*.png"))

# #             for img_path in tqdm(images):

# #                 img = cv2.imread(str(img_path), 0) / 255.0

# #                 # save original
# #                 cv2.imwrite(
# #                     str(output_path / img_path.name),
# #                     (img * 255).astype(np.uint8)
# #                 )

# #                 for i in range(self.factor):

# #                     aug = self.rotate(img)
# #                     aug = self.add_noise(aug)
# #                     aug = self.zoom(aug)

# #                     new_name = img_path.stem + f"_aug{i}.png"

# #                     cv2.imwrite(
# #                         str(output_path / new_name),
# #                         (aug * 255).astype(np.uint8)
# #                     )

# #         print("Augmentation Completed")
# # import cv2
# # import numpy as np
# # import os
# # from pathlib import Path
# # import random
# # from tqdm import tqdm
# # from scipy.ndimage import gaussian_filter


# # class DataAugmentation:
# #     """
# #     Medically valid data augmentation for structural MRI (sMRI) images.

# #     Augmentation strategies chosen to preserve:
# #     - Hemispheric asymmetry (critical for autism biomarkers)
# #     - Anatomical orientation (no flips)
# #     - Realistic intensity distributions (scanner variability simulation)
# #     - Natural anatomical shape variance (elastic deformation)

# #     Strategies applied:
# #         ✅ Small rotation          — simulates subject head positioning variance
# #         ✅ Elastic deformation     — simulates inter-subject anatomical shape variance
# #         ✅ Gamma correction        — simulates scanner intensity/contrast variability
# #         ✅ Bias field simulation   — simulates MRI RF field inhomogeneity artifact
# #         ✅ Gaussian noise          — simulates scanner thermal noise
# #         ✅ Small zoom              — simulates slight FOV differences

# #     Strategies intentionally excluded:
# #         ❌ Horizontal/vertical flip — destroys L/R hemispheric asymmetry (autism biomarker)
# #         ❌ Cutout / random erasing  — destroys real anatomical structures
# #         ❌ Color jitter             — grayscale MRI; not applicable
# #         ❌ Heavy rotation (>±15°)   — disrupts anatomical orientation priors
# #     """

# #     def __init__(self, config):

# #         self.input_dir  = Path(config["augmentation"]["input_dir"])
# #         self.output_dir = Path(config["augmentation"]["output_dir"])
# #         self.factor     = config["augmentation"]["augment_factor"]

# #         # Rotation: cap at ±10° to preserve hemispheric asymmetry
# #         raw_rotation = config["augmentation"]["rotation"]
# #         if raw_rotation > 10:
# #             print(
# #                 f"[WARNING] Rotation {raw_rotation}° exceeds safe sMRI limit. "
# #                 "Clamping to ±10° to preserve hemispheric asymmetry."
# #             )
# #         self.rotation  = min(raw_rotation, 10)
# #         self.noise_std = config["augmentation"]["noise_std"]

# #         # Elastic deformation parameters
# #         # alpha: deformation magnitude (keep low — 20–40 for sMRI)
# #         # sigma: smoothness of deformation field (higher = smoother/more anatomically plausible)
# #         self.elastic_alpha = config["augmentation"].get("elastic_alpha", 30)
# #         self.elastic_sigma = config["augmentation"].get("elastic_sigma", 5)

# #         self.output_dir.mkdir(parents=True, exist_ok=True)

# #     # ------------------------------------------------------------------
# #     # Individual augmentation methods
# #     # ------------------------------------------------------------------

# #     def rotate(self, image: np.ndarray) -> np.ndarray:
# #         """
# #         Small random rotation to simulate subject head positioning variance.
# #         Capped at ±10° to preserve hemispheric asymmetry, which is a known
# #         structural biomarker in autism research.
# #         """
# #         angle = random.uniform(-self.rotation, self.rotation)
# #         h, w  = image.shape
# #         M     = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
# #         # Use reflect border mode — avoids black borders that bias CNNs
# #         return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

# #     def elastic_deform(self, image: np.ndarray) -> np.ndarray:
# #         """
# #         Elastic deformation — the gold standard augmentation for brain MRI.
# #         Simulates natural inter-subject anatomical shape variance by applying
# #         a smooth random displacement field to the image.

# #         Parameters are intentionally conservative:
# #             alpha (magnitude): 20–40 keeps deformations sub-voxel to voxel scale
# #             sigma (smoothness): ≥4 ensures anatomically plausible smooth fields
# #         """
# #         shape = image.shape
# #         # Random displacement fields, smoothed with a Gaussian kernel
# #         dx = gaussian_filter(np.random.randn(*shape), self.elastic_sigma) * self.elastic_alpha
# #         dy = gaussian_filter(np.random.randn(*shape), self.elastic_sigma) * self.elastic_alpha

# #         # Build sampling grid
# #         x, y   = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
# #         map_x  = np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
# #         map_y  = np.clip(y + dy, 0, shape[0] - 1).astype(np.float32)

# #         return cv2.remap(image.astype(np.float32), map_x, map_y,
# #                          interpolation=cv2.INTER_LINEAR,
# #                          borderMode=cv2.BORDER_REFLECT)

# #     def gamma_correction(self, image: np.ndarray) -> np.ndarray:
# #         """
# #         Gamma correction to simulate scanner intensity and contrast variability.
# #         This is especially important for multi-site datasets where scanners
# #         (e.g., 1.5T vs 3T) produce different intensity profiles.

# #         Range 0.8–1.2 corresponds to mild contrast changes without
# #         altering structural information.
# #         """
# #         gamma = random.uniform(0.8, 1.2)
# #         # image is in [0,1]; power transform preserves range
# #         return np.power(np.clip(image, 1e-7, 1.0), gamma)

# #     def bias_field(self, image: np.ndarray) -> np.ndarray:
# #         """
# #         Smooth low-frequency intensity non-uniformity simulation.
# #         Models the B1 RF field inhomogeneity artifact present in real MRI scans,
# #         where signal intensity varies spatially across the image even in
# #         homogeneous tissue.

# #         Implemented as a smooth quadratic field — same family as the
# #         N4ITK bias field model.
# #         """
# #         h, w = image.shape
# #         x = np.linspace(-1, 1, w)
# #         y = np.linspace(-1, 1, h)
# #         X, Y = np.meshgrid(x, y)

# #         # Smooth quadratic field with small random coefficient
# #         coeff = random.uniform(-1, 1)
# #         bias  = 1.0 + 0.08 * coeff * (X**2 + Y**2)

# #         return np.clip(image * bias, 0.0, 1.0)

# #     def add_noise(self, image: np.ndarray) -> np.ndarray:
# #         """
# #         Additive Gaussian noise to simulate scanner thermal (Johnson) noise.
# #         Noise is zero-mean and normally distributed, matching the real
# #         noise model of MRI magnitude images in high-SNR regions.
# #         """
# #         noise = np.random.normal(0, self.noise_std, image.shape)
# #         return np.clip(image + noise, 0.0, 1.0)

# #     def zoom(self, image: np.ndarray) -> np.ndarray:
# #         """
# #         Small zoom (±5%) to simulate slight field-of-view or voxel-size differences
# #         across scanner protocols. Reflect padding used instead of zero-padding to
# #         avoid introducing artificial dark borders that a CNN may treat as
# #         scanner-specific artifacts.
# #         """
# #         scale   = random.uniform(0.95, 1.05)
# #         h, w    = image.shape
# #         resized = cv2.resize(image, None, fx=scale, fy=scale,
# #                              interpolation=cv2.INTER_LINEAR)

# #         if scale > 1:
# #             # Crop center
# #             startx = resized.shape[1] // 2 - w // 2
# #             starty = resized.shape[0] // 2 - h // 2
# #             return resized[starty:starty + h, startx:startx + w]
# #         else:
# #             # Reflect pad instead of zero pad — avoids artificial black borders
# #             pad_x = (w - resized.shape[1]) // 2
# #             pad_y = (h - resized.shape[0]) // 2
# #             return np.pad(resized, ((pad_y, pad_y), (pad_x, pad_x)),
# #                           mode='reflect')

# #     # ------------------------------------------------------------------
# #     # Probabilistic pipeline (each transform applied independently)
# #     # ------------------------------------------------------------------

# #     def augment_single(self, image: np.ndarray) -> np.ndarray:
# #         """
# #         Applies augmentations independently with per-transform probabilities.

# #         Each transform is stochastic — not all transforms are applied every
# #         time. This produces diverse, non-correlated augmented samples, which
# #         is critical to avoid overfitting on augmentation patterns rather than
# #         true anatomical features.

# #         Transform probabilities are tuned for sMRI:
# #             - Elastic deform: 0.7  (most important; applied most often)
# #             - Gamma:          0.6  (common across multi-site datasets)
# #             - Bias field:     0.5  (present in most real scanners)
# #             - Noise:          0.5  (always present but varies)
# #             - Rotation:       0.5  (common positioning variation)
# #             - Zoom:           0.3  (less common; smaller effect)
# #         """
# #         aug = image.copy()

# #         if random.random() < 0.7:
# #             aug = self.elastic_deform(aug)   # anatomical shape variance (most important)
# #         if random.random() < 0.6:
# #             aug = self.gamma_correction(aug) # scanner intensity variability
# #         if random.random() < 0.5:
# #             aug = self.bias_field(aug)       # RF inhomogeneity artifact
# #         if random.random() < 0.5:
# #             aug = self.add_noise(aug)        # thermal noise
# #         if random.random() < 0.5:
# #             aug = self.rotate(aug)           # head positioning variance
# #         if random.random() < 0.3:
# #             aug = self.zoom(aug)             # FOV differences

# #         return aug

# #     # ------------------------------------------------------------------
# #     # Main augmentation loop
# #     # ------------------------------------------------------------------

# #     def augment(self):
# #         """
# #         Iterates over each class, reads original images, saves them as-is,
# #         then generates `factor` augmented variants per image using the
# #         probabilistic pipeline.
# #         """
# #         for cls in ["autistic", "non_autistic"]:

# #             input_path  = self.input_dir  / cls / "images"
# #             output_path = self.output_dir / cls / "images"
# #             output_path.mkdir(parents=True, exist_ok=True)

# #             images = list(input_path.glob("*.png"))
# #             print(f"\n[{cls}] Found {len(images)} images → "
# #                   f"generating {len(images) * self.factor} augmented samples")

# #             for img_path in tqdm(images, desc=cls):

# #                 # Load as grayscale, normalise to [0, 1]
# #                 img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
# #                 if img is None:
# #                     print(f"[WARNING] Could not read {img_path}, skipping.")
# #                     continue
# #                 img = img.astype(np.float64) / 255.0

# #                 # Save original (no augmentation applied)
# #                 cv2.imwrite(
# #                     str(output_path / img_path.name),
# #                     (img * 255).astype(np.uint8)
# #                 )

# #                 # Generate augmented variants
# #                 for i in range(self.factor):
# #                     aug      = self.augment_single(img)
# #                     new_name = img_path.stem + f"_aug{i}.png"
# #                     cv2.imwrite(
# #                         str(output_path / new_name),
# #                         (aug * 255).astype(np.uint8)
# #                     )

# #         print("\n[✓] Augmentation completed successfully.")
import numpy as np
import cv2
import random
from pathlib import Path
from tqdm import tqdm


class DataAugmentation:

    def __init__(self, config):

        self.input_dir = Path(config["augmentation"]["input_dir"])
        self.output_dir = Path(config["augmentation"]["output_dir"])

        self.rotation = config["augmentation"]["rotation"]
        self.noise_std = config["augmentation"]["noise_std"]
        self.elastic_alpha = config["augmentation"]["elastic_alpha"]
        self.elastic_sigma = config["augmentation"]["elastic_sigma"]

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ================= ROTATION =================
    def rotate(self, image):

        angle = random.uniform(-self.rotation, self.rotation)

        h, w = image.shape

        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)

        return cv2.warpAffine(
            image,
            M,
            (w, h),
            borderMode=cv2.BORDER_REFLECT_101
        )

    # ================= ZOOM =================
    def zoom(self, image):

        scale = random.uniform(0.97, 1.03)

        h, w = image.shape

        resized = cv2.resize(image, None, fx=scale, fy=scale)

        if scale > 1:

            startx = resized.shape[1]//2 - w//2
            starty = resized.shape[0]//2 - h//2

            return resized[starty:starty+h, startx:startx+w]

        else:

            pad_x = (w - resized.shape[1]) // 2
            pad_y = (h - resized.shape[0]) // 2

            padded = np.pad(
                resized,
                ((pad_y, pad_y), (pad_x, pad_x)),
                mode="reflect"
            )

            return padded[:h, :w]

    # ================= ELASTIC =================
    def elastic(self, image):

        random_state = np.random.RandomState(None)

        shape = image.shape

        dx = cv2.GaussianBlur(
            (random_state.rand(*shape) * 2 - 1),
            (0, 0),
            self.elastic_sigma
        ) * self.elastic_alpha

        dy = cv2.GaussianBlur(
            (random_state.rand(*shape) * 2 - 1),
            (0, 0),
            self.elastic_sigma
        ) * self.elastic_alpha

        x, y = np.meshgrid(
            np.arange(shape[1]),
            np.arange(shape[0])
        )

        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        return cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

    # ================= NOISE =================
    def add_noise(self, image):

        noise = np.random.normal(
            0,
            self.noise_std,
            image.shape
        )

        noisy = image + noise

        return np.clip(noisy, 0, 1)

    # ================= SAVE =================
    def save_png(self, image, path):

        img = (image * 255).astype("uint8")

        cv2.imwrite(str(path), img)

    # ================= MAIN =================
    def augment(self):

        print("Starting MRI Augmentation")

        train_root = self.input_dir / "train"

        for cls in ["autism", "control"]:

            input_path = train_root / cls
            output_path = self.output_dir / "train" / cls

            output_path.mkdir(parents=True, exist_ok=True)

            files = list(input_path.glob("*.png"))

            # ⭐ Class balancing (optional)
            factor = 2 if cls == "autism" else 1

            for file in tqdm(files, desc=f"Augmenting {cls}"):

                img = cv2.imread(str(file), 0) / 255.0

                # save original
                self.save_png(img, output_path / file.name)

                for i in range(factor):

                    aug = self.rotate(img)
                    aug = self.zoom(aug)
                    aug = self.elastic(aug)
                    aug = self.add_noise(aug)

                    new_name = file.stem + f"_aug{i}.png"

                    self.save_png(
                        aug,
                        output_path / new_name
                    )

        print("Augmentation Completed")