# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import nibabel as nib
from merlin import Merlin
from monai.transforms import (
    Compose,
    EnsureTyped,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.data import MetaTensor

from util.util import fix_random_seeds
from util.leavs_utils import get_organ_crop
from util.sliding_window import sliding_window_3d
from util.snakemake_helpers import VALID_ORGANS
import argparse

INFERENCE_BATCH_SIZE = 1
PREPROCESS_WORKERS = 1


def load_model():
    model = Merlin(ImageEmbedding=True)
    model.cuda().eval()
    return model


def apply_spacing_to_crop(crop: np.ndarray, scan_path: str) -> np.ndarray:
    """
    Apply spacing (1.5, 1.5, 3) to the entire organ crop using MONAI.
    This should be done before extracting patches to avoid changing patch dimensionality.

    Args:
        crop: 3D numpy array (Z, Y, X)
        scan_path: Path to original scan (used to get spacing info)

    Returns:
        Resampled 3D numpy array with spacing (1.5, 1.5, 3)
    """
    scan_img = nib.load(scan_path)
    original_affine = scan_img.affine

    transform = Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode="bilinear")

    crop_tensor = torch.from_numpy(crop).unsqueeze(0).float()
    crop_meta = MetaTensor(crop_tensor, affine=original_affine)

    data_dict = {"image": crop_meta}
    transformed = transform(data_dict)
    resampled = transformed["image"].squeeze(0).numpy()

    return resampled


def preprocess_patch(patch: np.ndarray) -> torch.Tensor:
    """
    Preprocess a 3D patch for Merlin.
    Expected input: 160x224x224 (Z, Y, X) after spacing has been applied to the crop.
    """
    transform = Compose([
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])

    patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()
    data_dict = {"image": patch_tensor}
    transformed = transform(data_dict)

    img = transformed["image"].unsqueeze(0)  # (1, C, D, H, W)
    return img


def extract_features_for_organ(
    model,
    organ_crop: np.ndarray,
    window_size: tuple,
    stride: tuple = None
) -> tuple:
    """
    Extract features for an organ crop using sliding windows.

    Returns:
        features: List of feature vectors
        positions: List of (z, y, x) positions
    """
    if stride is None:
        stride = tuple(s // 2 for s in window_size)  # 50% overlap

    patches = []
    positions = []
    for patch, (z, y, x) in sliding_window_3d(organ_crop, window_size, stride):
        patches.append(patch)
        positions.append((z, y, x))

    if not patches:
        return np.array([]), np.array([])

    with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as executor:
        preprocessed_patches = list(executor.map(preprocess_patch, patches))

    features = []
    with torch.no_grad():
        for batch_start in range(0, len(preprocessed_patches), INFERENCE_BATCH_SIZE):
            batch_items = preprocessed_patches[batch_start:batch_start + INFERENCE_BATCH_SIZE]
            batch_tensor = torch.cat(batch_items, dim=0).cuda()
            outputs = model(batch_tensor)
            batch_features = outputs[0].detach().cpu().numpy()
            for feature in batch_features:
                features.append(np.expand_dims(feature, axis=0))

    return np.array(features), np.array(positions)


def is_valid_output_file(output_path: str) -> bool:
    if not os.path.exists(output_path):
        return False
    if not os.path.isfile(output_path):
        return False
    if not os.access(output_path, os.R_OK):
        return False
    try:
        with np.load(output_path, allow_pickle=True) as data:
            required_keys = {"features", "positions", "bbox_origin", "organ_name", "is_placeholder"}
            if not required_keys.issubset(set(data.files)):
                return False
            _ = data["features"]
            _ = data["positions"]
            _ = data["is_placeholder"]
    except Exception:
        return False
    return True


def process_scan_for_organ(
    model,
    scan_path: str,
    seg_path: str,
    organ_name: str,
    window_size: tuple,
    output_path: str
):
    """
    Process a single scan for a specific organ.
    Returns True if features were extracted, False if placeholder was saved.
    """
    result = get_organ_crop(scan_path, seg_path, organ_name, window_size)
    if result is None:
        print(f"Warning: Organ {organ_name} not found in segmentation {seg_path} for scan {scan_path}. Saving placeholder file.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(
            output_path,
            features=np.array([]),
            positions=np.array([]),
            bbox_origin=None,
            organ_name=organ_name,
            is_placeholder=True
        )
        return False

    organ_crop, bbox_origin = result

    organ_crop = apply_spacing_to_crop(organ_crop, scan_path)

    features, positions = extract_features_for_organ(model, organ_crop, window_size)

    if len(features) == 0:
        print(f"Warning: No features extracted from organ {organ_name} in scan {scan_path}. Organ crop may be too small. Saving placeholder file.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(
            output_path,
            features=np.array([]),
            positions=np.array([]),
            bbox_origin=bbox_origin,
            organ_name=organ_name,
            is_placeholder=True
        )
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        features=features,
        positions=positions,
        bbox_origin=bbox_origin,
        organ_name=organ_name,
        is_placeholder=False
    )
    return True


def process_scan_for_all_organs(
    model,
    scan_path: str,
    seg_path: str,
    organ_names: list,
    window_size: tuple,
    output_root: str,
    model_name: str,
    split: str,
    scan_id: str,
):
    """
    Process a scan for all specified organs.
    Saves one file per organ using the standard output path convention.
    """
    processed_count = 0
    for organ_name in organ_names:
        output_path = os.path.join(
            output_root,
            model_name,
            organ_name,
            split,
            "features",
            "raw",
            f"{scan_id}.npz",
        )
        if is_valid_output_file(output_path):
            print(f"Skipping organ {organ_name}: valid output already exists at {output_path}")
            continue
        if os.path.exists(output_path):
            print(f"Recomputing organ {organ_name}: existing output is invalid or unreadable at {output_path}")
        else:
            print(f"Extracting features for organ: {organ_name}")
        if process_scan_for_organ(model, scan_path, seg_path, organ_name, window_size, output_path):
            processed_count += 1

    print(f"Successfully processed {processed_count}/{len(organ_names)} organs for scan")


def _read_paths_file(paths_file: str) -> list:
    with open(paths_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main(args):
    fix_random_seeds(getattr(args, "seed", 0))

    scan_paths = _read_paths_file(args.scan_paths_file)
    seg_paths = _read_paths_file(args.seg_paths_file)

    if len(scan_paths) != len(seg_paths):
        raise ValueError(f"Number of scan paths ({len(scan_paths)}) must match number of seg paths ({len(seg_paths)})")

    for scan_path in scan_paths:
        if not os.path.exists(scan_path):
            raise FileNotFoundError(f"Scan file not found: {scan_path}")
        if not os.path.isfile(scan_path):
            raise ValueError(f"Scan path is not a file: {scan_path}")
        if not os.access(scan_path, os.R_OK):
            raise PermissionError(f"Cannot read scan file: {scan_path}")

    for seg_path in seg_paths:
        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
        if not os.path.isfile(seg_path):
            raise ValueError(f"Segmentation path is not a file: {seg_path}")
        if not os.access(seg_path, os.R_OK):
            raise PermissionError(f"Cannot read segmentation file: {seg_path}")

    organ_names = VALID_ORGANS

    # Merlin patch size: 224 x 224 x 160 (in-plane x in-plane x depth) -> (Z, Y, X)
    window_size = (160, 224, 224)

    print("Loading model...")
    try:
        model = load_model()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    for scan_idx, (scan_path, seg_path) in enumerate(zip(scan_paths, seg_paths)):
        seg_basename = os.path.basename(seg_path)
        if seg_basename.endswith('_segmentation.nii.gz'):
            scan_id = seg_basename[:-20]
        else:
            scan_basename = os.path.basename(scan_path)
            scan_id = scan_basename.replace('.nii.gz', '')

        print(f"Processing scan {scan_idx + 1}/{len(scan_paths)}: {scan_id}")

        try:
            process_scan_for_all_organs(
                model,
                scan_path,
                seg_path,
                organ_names,
                window_size,
                args.output_root,
                args.model_name,
                args.split,
                scan_id,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to process scan {scan_path}: {e}") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merlin Feature Extraction for LEAVS")
    parser.add_argument("--scan-paths-file", type=str, required=True, help="File containing scan file paths (.nii.gz), one per line")
    parser.add_argument("--seg-paths-file", type=str, required=True, help="File containing segmentation file paths (.nii.gz), one per line")
    parser.add_argument("--output-root", type=str, required=True, help="Root output directory following workflow conventions")
    parser.add_argument("--model-name", type=str, required=True, help="Feature model name")
    parser.add_argument("--split", type=str, required=True, choices=["training", "validation", "test"], help="Dataset split")
    args = parser.parse_args()

    sys.exit(main(args))
