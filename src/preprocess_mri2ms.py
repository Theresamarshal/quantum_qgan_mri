import os
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom


# ==================================================
# CONFIG — MATCHES YOUR REAL DATA
# ==================================================

MS_RAW_DIR = r"C:\quantum_gan_mri\data\raw"
HEALTHY_DIR = r"C:\quantum_gan_mri\data\healthyflairmri"

OUTPUT_BASE = r"C:\quantum_gan_mri\data\preprocessed_ms"

TARGET_SHAPE = (192, 192, 96)


# ==================================================
# Load NIfTI (RAS+ orientation)
# ==================================================
def load_nifti(path):
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    return img.get_fdata().astype(np.float32)


# ==================================================
# Resample volume
# ==================================================
def resample(volume, target_shape, is_mask=False):
    factors = (
        target_shape[0] / volume.shape[0],
        target_shape[1] / volume.shape[1],
        target_shape[2] / volume.shape[2],
    )
    order = 0 if is_mask else 1
    return zoom(volume, factors, order=order)


# ==================================================
# FLAIR normalization
# ==================================================
def normalize_flair(volume):
    p1, p99 = np.percentile(volume, (1, 99))
    volume = np.clip(volume, p1, p99)
    return (volume - p1) / (p99 - p1 + 1e-8)


# ==================================================
# Find .nii or .nii.gz inside a folder
# ==================================================
def find_nifti_in_folder(folder):
    for f in os.listdir(folder):
        if f.endswith(".nii") or f.endswith(".nii.gz"):
            return os.path.join(folder, f)
    raise FileNotFoundError(f"No NIfTI file found in {folder}")


# ==================================================
# MS SUBJECT (IMAGE + MASK)
# ==================================================
def preprocess_ms_subject(flair_path, mask_path):
    img = load_nifti(flair_path)
    mask = load_nifti(mask_path)

    img  = resample(img, TARGET_SHAPE, is_mask=False)
    mask = resample(mask, TARGET_SHAPE, is_mask=True)

    img = normalize_flair(img)
    mask = (mask > 0.5).astype(np.float32)

    img  = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    return img, mask


# ==================================================
# HEALTHY SUBJECT (IMAGE ONLY)
# ==================================================
def preprocess_healthy_subject(flair_path):
    img = load_nifti(flair_path)

    img = resample(img, TARGET_SHAPE, is_mask=False)
    img = normalize_flair(img)

    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img


# ==================================================
# PROCESS MS DATASET
# ==================================================
def preprocess_ms_dataset():
    img_out  = os.path.join(OUTPUT_BASE, "ms", "images")
    mask_out = os.path.join(OUTPUT_BASE, "ms", "masks")

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    for f in os.listdir(MS_RAW_DIR):
        if f.endswith("-Flair.nii") and "LesionSeg" not in f:
            flair_path = os.path.join(MS_RAW_DIR, f)
            mask_name = f.replace("-Flair.nii", "-LesionSeg-Flair.nii")
            mask_path = os.path.join(MS_RAW_DIR, mask_name)

            if not os.path.exists(mask_path):
                print(f"⚠️ Mask missing for {f}")
                continue

            print(f"Processing MS: {f}")

            img, mask = preprocess_ms_subject(flair_path, mask_path)

            base = f.replace("-Flair.nii", "")
            torch.save(img,  os.path.join(img_out,  base + ".pt"))
            torch.save(mask, os.path.join(mask_out, base + "_mask.pt"))


# ==================================================
# PROCESS HEALTHY DATASET (BIDS-STYLE FOLDERS)
# ==================================================
def preprocess_healthy_dataset():
    img_out = os.path.join(OUTPUT_BASE, "healthy", "images")
    os.makedirs(img_out, exist_ok=True)

    for subject in os.listdir(HEALTHY_DIR):
        subject_dir = os.path.join(HEALTHY_DIR, subject)

        if not os.path.isdir(subject_dir):
            continue

        try:
            flair_path = find_nifti_in_folder(subject_dir)
            print(f"Processing healthy: {flair_path}")

            img = preprocess_healthy_subject(flair_path)

            base = os.path.basename(flair_path).replace(".nii.gz", "").replace(".nii", "")
            torch.save(img, os.path.join(img_out, base + ".pt"))

        except Exception as e:
            print(f"❌ Failed healthy subject {subject}: {e}")


# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":

    print("\n=== Processing MS FLAIR images + lesion masks ===")
    preprocess_ms_dataset()

    print("\n=== Processing Healthy FLAIR images (image only) ===")
    preprocess_healthy_dataset()

    print("\n✅ MS & Healthy preprocessing completed successfully")
