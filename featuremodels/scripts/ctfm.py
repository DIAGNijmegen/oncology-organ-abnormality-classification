# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np
import torch
from lighter_zoo import SegResEncoder
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureTyped,
    Orientationd,
    ScaleIntensityRanged,
)
from tqdm import tqdm

from util.util import fix_random_seeds
from util.leavs_utils import get_organ_crop
from util.sliding_window import sliding_window_3d
from util.snakemake_helpers import VALID_ORGANS
import argparse

INFERENCE_BATCH_SIZE = 2
PREPROCESS_WORKERS = 2


def load_model():
    model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")
    model.eval()
    model.cuda()
    return model


def preprocess_patch(patch: np.ndarray) -> torch.Tensor:
    """
    Preprocess a 3D patch for CT-FM model.
    """
    transform = Compose([
        Orientationd(keys=["image"], axcodes="SPL"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1024,
            a_max=2048,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        EnsureTyped(keys=["image"]),
    ])
    
    # Convert to MONAI format: (C, H, W, D)
    patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()
    data_dict = {"image": patch_tensor}
    transformed = transform(data_dict)
    
    # CT-FM expects (B, C, D, H, W)
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
    # get random integer
    import random
    random_int = random.randint(0, 1000000)
    for patch, (z, y, x) in sliding_window_3d(organ_crop, window_size, stride):
        # Debug: save patch as nifti
        patch_nifti = nib.Nifti1Image(patch, np.eye(4))
        nib.save(patch_nifti, f"/tmp/patch_{z}_{y}_{x}_{random_int}.nii.gz")

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
            output = model(batch_tensor)
            batch_features = output[-1].detach().cpu().numpy()
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
    # Get organ crop
    result = get_organ_crop(scan_path, seg_path, organ_name, window_size)
    if result is None:
        # Organ not found - save placeholder file
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

    # Debug: save organ crop as nifti
    organ_crop_nifti = nib.Nifti1Image(organ_crop, np.eye(4))
    nib.save(organ_crop_nifti, f"/tmp/organ_crop_{organ_name}.nii.gz")
    
    # Extract features
    features, positions = extract_features_for_organ(model, organ_crop, window_size)
    
    if len(features) == 0:
        # No features extracted - save placeholder file
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
    
    # Save features with position information
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
    
    # Parse scan paths and seg paths from files
    scan_paths = _read_paths_file(args.scan_paths_file)
    seg_paths = _read_paths_file(args.seg_paths_file)
    
    if len(scan_paths) != len(seg_paths):
        raise ValueError(f"Number of scan paths ({len(scan_paths)}) must match number of seg paths ({len(seg_paths)})")
    
    # Validate inputs early
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
    
    # Window size for CT-FM: 128x128x128
    window_size = (128, 128, 128)
    
    # Load model once for the entire batch
    print("Loading model...")
    try:
        model = load_model()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e
    
    # Process each scan sequentially
    for scan_idx, (scan_path, seg_path) in enumerate(zip(scan_paths, seg_paths)):
        # Extract scan_id from seg_path (format: .../{scan_id}_segmentation.nii.gz)
        seg_basename = os.path.basename(seg_path)
        if seg_basename.endswith('_segmentation.nii.gz'):
            scan_id = seg_basename[:-20]  # Remove '_segmentation.nii.gz'
        else:
            # Fallback: try to extract from scan path
            scan_basename = os.path.basename(scan_path)
            scan_id = scan_basename.replace('.nii.gz', '')
        
        print(f"Processing scan {scan_idx + 1}/{len(scan_paths)}: {scan_id}")
        
        # Process scan for all organs
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
    parser = argparse.ArgumentParser(description="CT-FM Feature Extraction for LEAVS")
    parser.add_argument("--scan-paths-file", type=str, required=True, help="File containing scan file paths (.nii.gz), one per line")
    parser.add_argument("--seg-paths-file", type=str, required=True, help="File containing segmentation file paths (.nii.gz), one per line")
    parser.add_argument("--output-root", type=str, required=True, help="Root output directory following workflow conventions")
    parser.add_argument("--model-name", type=str, required=True, help="Feature model name")
    parser.add_argument("--split", type=str, required=True, choices=["training", "validation", "test"], help="Dataset split")
    args = parser.parse_args()

    sys.exit(main(args))
