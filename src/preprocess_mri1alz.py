import os
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom


# ==================================================
# CONFIG — ALZ & HEALTHY
# ==================================================

ALZ_INPUT_DIR = r"C:\quantum_gan_mri\data\alz_raw"
HEALTHY_INPUT_DIR = r"C:\quantum_gan_mri\data\healthyt1mri"

OUTPUT_BASE = r"C:\quantum_gan_mri\data\preprocessed_images"

TARGET_SHAPE = (192, 192, 96)


# ==================================================
# Load NIfTI (RAS+ orientation)
# ==================================================
def load_nifti(path):
    img = nib.load(path)
    img = nib.as_closest_canonical(img)     # FIX ORIENTATION → RAS+
    return img.get_fdata().astype(np.float32)


# ==================================================
# Light brain mask (segmentation-safe)
# ==================================================
def light_brain_mask(volume):
    thresh = np.percentile(volume, 2)
    return (volume > thresh).astype(np.float32)


# ==================================================
# Resample image to target shape
# ==================================================
def resample_image(volume, target_shape):
    factors = (
        target_shape[0] / volume.shape[0],
        target_shape[1] / volume.shape[1],
        target_shape[2] / volume.shape[2],
    )
    return zoom(volume, factors, order=1)    # trilinear


# ==================================================
# Percentile normalization (1–99)
# ==================================================
def percentile_normalize(volume):
    p1, p99 = np.percentile(volume, (1, 99))
    volume = np.clip(volume, p1, p99)
    return (volume - p1) / (p99 - p1 + 1e-8)


# ==================================================
# IMAGE-ONLY PREPROCESSING
# ==================================================
def preprocess_image(image_path):
    vol = load_nifti(image_path)

    # 1. Resample FIRST
    vol = resample_image(vol, TARGET_SHAPE)

    # 2. Light background suppression
    mask = light_brain_mask(vol)
    vol = vol * mask

    # 3. Normalize
    vol = percentile_normalize(vol)

    # 4. Convert to tensor (C, H, W, D)
    vol = torch.tensor(vol, dtype=torch.float32).unsqueeze(0)

    return vol


# ==================================================
# Batch preprocessing
# ==================================================
def preprocess_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                in_path = os.path.join(root, f)
                print(f"Processing: {in_path}")

                try:
                    tensor = preprocess_image(in_path)

                    base = f.replace(".nii.gz", "").replace(".nii", "")
                    out_path = os.path.join(output_dir, base + ".pt")

                    torch.save(tensor, out_path)

                except Exception as e:
                    print(f"❌ Failed {in_path}: {e}")


# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":

    print("\n=== Processing ALZHEIMER DATA ===")
    preprocess_folder(
        input_dir=ALZ_INPUT_DIR,
        output_dir=os.path.join(OUTPUT_BASE, "alz")
    )

    print("\n=== Processing HEALTHY DATA ===")
    preprocess_folder(
        input_dir=HEALTHY_INPUT_DIR,
        output_dir=os.path.join(OUTPUT_BASE, "healthy")
    )

    print("\n✅ ALZ & Healthy preprocessing completed")
