# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import nibabel as nib
from monai.transforms import (
    Compose,
    EnsureTyped,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.data import MetaTensor
from tqdm import tqdm

from spectre import SpectreImageFeatureExtractor, MODEL_CONFIGS
from util.util import fix_random_seeds
from util.leavs_utils import get_organ_crop
from util.sliding_window import sliding_window_3d
from util.snakemake_helpers import VALID_ORGANS
import argparse

PREPROCESS_WORKERS = 4


def load_model():
    config = MODEL_CONFIGS['spectre-large-pretrained']
    model = SpectreImageFeatureExtractor.from_config(config)
    model.cuda().eval()
    return model


def apply_spacing_to_crop(crop: np.ndarray, scan_path: str) -> np.ndarray:
    """
    Apply spacing (0.5, 0.5, 1.0) to the entire organ crop using MONAI.
    This should be done before extracting patches to avoid changing patch dimensionality.
    
    Args:
        crop: 3D numpy array (Z, Y, X)
        scan_path: Path to original scan (used to get spacing info)
    
    Returns:
        Resampled 3D numpy array with spacing (0.5, 0.5, 1.0)
    """
    # Load the original scan to get its affine matrix for spacing info
    scan_img = nib.load(scan_path)
    original_affine = scan_img.affine
    
    # Use MONAI's Spacingd transform with proper MetaTensor setup
    transform = Spacingd(keys=["image"], pixdim=(0.5, 0.5, 1.0), mode="bilinear")
    
    # Convert crop to MetaTensor with affine information
    crop_tensor = torch.from_numpy(crop).unsqueeze(0).float()
    crop_meta = MetaTensor(crop_tensor, affine=original_affine)
    
    # Apply spacing transform
    data_dict = {"image": crop_meta}
    transformed = transform(data_dict)
    resampled = transformed["image"].squeeze(0).numpy()
    
    return resampled


def preprocess_patch(patch: np.ndarray) -> torch.Tensor:
    """
    Preprocess a 3D patch for SPECTRE model.
    Expected input: 128x128x64 (after spacing has been applied to the crop)
    
    Returns:
        Preprocessed patch tensor with shape (1, C, H, W, D)
        (MONAI outputs (C, H, W, D), then we add batch dim)
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
    
    # Convert to MONAI format: (C, H, W, D)
    patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()
    data_dict = {"image": patch_tensor}
    transformed = transform(data_dict)
    
    # SPECTRE expects (B, C, D, H, W)
    img = transformed["image"].unsqueeze(0)  # (1, C, D, H, W)
    
    return img


def extract_features_for_organ_grid(
    model,
    organ_crop: np.ndarray,
    window_size: tuple,
) -> np.ndarray:
    """
    Extract features for an organ crop using grid-based processing.
    Extracts patches with no overlap and passes all patches through the model at once.
    
    Args:
        model: SPECTRE model
        organ_crop: 3D numpy array (Z, Y, X) - already resampled with spacing
        window_size: (depth, height, width) of patches
    
    Returns:
        Aggregated feature vector (already aggregated by the model)
    """
    # Extract patches with no overlap (stride = window_size)
    stride = window_size
    
    patches = []
    positions = []
    for patch, (z, y, x) in sliding_window_3d(organ_crop, window_size, stride):
        patches.append(patch)
        positions.append((z, y, x))
    
    if not patches:
        return np.array([])
    
    # Calculate grid size based on number of patches in each dimension
    # We need to determine how many patches fit in each dimension
    d, h, w = organ_crop.shape
    win_d, win_h, win_w = window_size
    
    # Calculate grid dimensions
    grid_d = (d + win_d - 1) // win_d  # Number of patches in depth dimension
    grid_h = (h + win_h - 1) // win_h  # Number of patches in height dimension
    grid_w = (w + win_w - 1) // win_w  # Number of patches in width dimension
    
    # Verify that we have the expected number of patches
    expected_patches = grid_d * grid_h * grid_w
    if len(patches) != expected_patches:
        # This can happen if the crop doesn't divide evenly
        # Recalculate grid size based on actual patches
        # We need to infer the grid from the positions
        if len(positions) > 0:
            z_positions = sorted(set(pos[0] for pos in positions))
            y_positions = sorted(set(pos[1] for pos in positions))
            x_positions = sorted(set(pos[2] for pos in positions))
            grid_d = len(z_positions)
            grid_h = len(y_positions)
            grid_w = len(x_positions)
        else:
            grid_d = grid_h = grid_w = 1
    
    grid_size = (grid_d, grid_h, grid_w)
    
    print(f"  Grid size: {grid_size}, Total patches: {len(patches)}")
    
    # Preprocess all patches
    with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as executor:
        preprocessed_patches = list(executor.map(preprocess_patch, patches))
    
    # Stack all patches into a single batch
    patch_tensors = [p.squeeze(0) for p in preprocessed_patches]  # Each is (C, H, W, D)
    stacked_patches = torch.stack(patch_tensors, dim=0)  # (n_patches, C, H, W, D)
    batch_tensor = stacked_patches.unsqueeze(0).cuda()  # (1, n_patches, C, H, W, D)
    
    # Pass all patches through the model at once with the calculated grid size
    with torch.no_grad():
        output = model(batch_tensor, grid_size=grid_size)
        # Output is already aggregated (one feature vector per organ)
        # If output has batch dimension, squeeze it
        if output.ndim > 1 and output.shape[0] == 1:
            feature = output.squeeze(0).detach().cpu().numpy()
        else:
            feature = output.detach().cpu().numpy()
    
    # Use CLS token as feature
    return np.expand_dims(feature[0], axis=0)


def is_valid_output_file(output_path: str) -> bool:
    if not os.path.exists(output_path):
        return False
    if not os.path.isfile(output_path):
        return False
    if not os.access(output_path, os.R_OK):
        return False
    try:
        with np.load(output_path, allow_pickle=True) as data:
            required_keys = {"features", "bbox_origin", "organ_name", "is_placeholder"}
            if not required_keys.issubset(set(data.files)):
                return False
            _ = data["features"]
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
    Process a single scan for a specific organ using grid-based processing.
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
            bbox_origin=None,
            organ_name=organ_name,
            is_placeholder=True
        )
        return False
    
    organ_crop, bbox_origin = result
    
    # Apply spacing to the entire crop before extracting patches
    organ_crop = apply_spacing_to_crop(organ_crop, scan_path)
    
    # Extract features using grid-based processing
    features = extract_features_for_organ_grid(model, organ_crop, window_size)
    
    if features.size == 0:
        # No features extracted - save placeholder file
        print(f"Warning: No features extracted from organ {organ_name} in scan {scan_path}. Organ crop may be too small. Saving placeholder file.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(
            output_path,
            features=np.array([]),
            bbox_origin=bbox_origin,
            organ_name=organ_name,
            is_placeholder=True
        )
        return False
    
    # Save aggregated features (already aggregated by the model)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        features=features,
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
    Process a scan for all specified organs using grid-based processing.
    Saves one file per organ to the raw path (features are already aggregated by the model).
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
    
    # Window size for SPECTRE: 128x128x64
    window_size = (128, 128, 64)
    
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
    parser = argparse.ArgumentParser(description="SPECTRE ViTG Feature Extraction for LEAVS")
    parser.add_argument("--scan-paths-file", type=str, required=True, help="File containing scan file paths (.nii.gz), one per line")
    parser.add_argument("--seg-paths-file", type=str, required=True, help="File containing segmentation file paths (.nii.gz), one per line")
    parser.add_argument("--output-root", type=str, required=True, help="Root output directory following workflow conventions")
    parser.add_argument("--model-name", type=str, required=True, help="Feature model name")
    parser.add_argument("--split", type=str, required=True, choices=["training", "validation", "test"], help="Dataset split")
    args = parser.parse_args()

    sys.exit(main(args))
