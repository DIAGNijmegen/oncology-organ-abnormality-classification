# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import os
import sys

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
import argparse


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
    Expected input: 256x256x128 (after spacing has been applied to the crop)
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
    
    features = []
    positions = []
    
    for patch, (z, y, x) in sliding_window_3d(organ_crop, window_size, stride):
        # Preprocess patch
        patch_tensor = preprocess_patch(patch)
        
        # Extract features
        with torch.no_grad():
            patch_tensor = patch_tensor.cuda()
            output = model(patch_tensor.unsqueeze(0), grid_size=(1, 1, 1))
            feature = output[-1].detach().cpu().numpy()
        
        features.append(feature)
        positions.append((z, y, x))
    
    return np.array(features), np.array(positions)


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
    
    # Apply spacing to the entire crop before extracting patches
    organ_crop = apply_spacing_to_crop(organ_crop, scan_path)
    
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
    output_paths: dict
):
    """
    Process a scan for all specified organs.
    Saves one file per organ using the provided output paths.
    
    Args:
        output_paths: Dict mapping organ_name to output_path
    """
    processed_count = 0
    for organ_name in organ_names:
        if organ_name in output_paths:
            print(f"Extracting features for organ: {organ_name}")
            output_path = output_paths[organ_name]
            if process_scan_for_organ(model, scan_path, seg_path, organ_name, window_size, output_path):
                processed_count += 1
    
    print(f"Successfully processed {processed_count}/{len(organ_names)} organs for scan")


def extract_scan_id_from_path(path):
    """Extract scan_id from output path like .../features/raw/{scan_id}.npz"""
    basename = os.path.basename(path)
    if basename.endswith('.npz'):
        return basename[:-4]  # Remove .npz extension
    return None


def main(args):
    fix_random_seeds(getattr(args, "seed", 0))
    
    # Parse scan paths and seg paths (comma-separated)
    if not args.scan_paths:
        raise ValueError("--scan-paths is required")
    if not args.seg_paths:
        raise ValueError("--seg-paths is required")
    
    scan_paths = [p.strip() for p in args.scan_paths.split(',')]
    seg_paths = [p.strip() for p in args.seg_paths.split(',')]
    
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
    
    # Parse output paths
    if not args.output_paths:
        raise ValueError("--output-paths is required. Format: 'organ1:path1,organ2:path2,...'")
    
    # Parse all output paths - format is organ:path pairs, organized by organ first, then scan
    all_output_pairs = []
    for pair in args.output_paths.split(','):
        if ':' not in pair:
            raise ValueError(f"Invalid output path format: {pair}. Expected 'organ:path'")
        organ_name, output_path = pair.split(':', 1)
        organ_name = organ_name.strip()
        output_path = output_path.strip()
        scan_id = extract_scan_id_from_path(output_path)
        if scan_id is None:
            raise ValueError(f"Could not extract scan_id from path: {output_path}")
        all_output_pairs.append((scan_id, organ_name, output_path))
    
    # Get unique organ names and scan IDs
    organ_names = sorted(set(organ_name for _, organ_name, _ in all_output_pairs))
    scan_ids = sorted(set(scan_id for scan_id, _, _ in all_output_pairs))
    
    if not organ_names:
        raise ValueError("No organs specified for processing")
    
    if len(scan_ids) != len(scan_paths):
        raise ValueError(f"Number of unique scan IDs in output paths ({len(scan_ids)}) does not match number of scan paths ({len(scan_paths)})")
    
    # Group output paths by scan_id
    scan_outputs = {}
    for scan_id, organ_name, output_path in all_output_pairs:
        if scan_id not in scan_outputs:
            scan_outputs[scan_id] = {}
        scan_outputs[scan_id][organ_name] = output_path
    
    # Ensure output directories can be created
    for scan_id, output_paths_dict in scan_outputs.items():
        for output_path in output_paths_dict.values():
            output_dir = os.path.dirname(output_path)
            if output_dir:
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except OSError as e:
                    raise OSError(f"Cannot create output directory {output_dir}: {e}")
            # Check write permissions
            if os.path.exists(output_path) and not os.access(output_path, os.W_OK):
                raise PermissionError(f"Cannot write to output file: {output_path}")
    
    # Window size for SPECTRE: 256x256x128
    window_size = (256, 256, 128)
    
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
        
        # Get output paths for this scan
        if scan_id not in scan_outputs:
            raise ValueError(f"Could not find output paths for scan {scan_id}. Available scan IDs: {list(scan_outputs.keys())}")
        
        output_paths = scan_outputs[scan_id]
        
        # Process scan for all organs
        try:
            process_scan_for_all_organs(
                model, 
                scan_path, 
                seg_path, 
                organ_names, 
                window_size, 
                output_paths
            )
        except Exception as e:
            raise RuntimeError(f"Failed to process scan {scan_path}: {e}") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPECTRE Feature Extraction for LEAVS")
    parser.add_argument("--scan-paths", type=str, required=True, help="Comma-separated list of scan file paths (.nii.gz)")
    parser.add_argument("--seg-paths", type=str, required=True, help="Comma-separated list of segmentation file paths (.nii.gz)")
    parser.add_argument("--output-paths", type=str, required=True, help="Comma-separated list of 'organ:path' pairs for all scans (e.g., 'spleen:/path/to/spleen_scan1.npz,liver:/path/to/liver_scan1.npz,spleen:/path/to/spleen_scan2.npz,...')")
    args = parser.parse_args()

    sys.exit(main(args))
