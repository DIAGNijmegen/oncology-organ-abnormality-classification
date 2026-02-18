# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import os
import sys

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
import argparse


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
    
    features = []
    positions = []
    
    for patch, (z, y, x) in sliding_window_3d(organ_crop, window_size, stride):
        # Preprocess patch
        patch_tensor = preprocess_patch(patch)
        
        # Extract features
        with torch.no_grad():
            patch_tensor = patch_tensor.cuda()
            output = model(patch_tensor)
            feature = output[-1].detach().cpu().numpy()
        
        features.append(feature)
        positions.append((z, y, x))
    
    return np.array(features), np.array(positions)


def process_scan(
    model,
    scan_path: str,
    seg_path: str,
    organ_name: str,
    window_size: tuple,
    output_path: str
):
    """
    Process a single scan for a specific organ.
    """
    # Get organ crop
    result = get_organ_crop(scan_path, seg_path, organ_name, padding=20)
    if result is None:
        raise ValueError(f"Organ {organ_name} not found in segmentation {seg_path} for scan {scan_path}")
    
    organ_crop, bbox_origin = result
    
    # Extract features
    features, positions = extract_features_for_organ(model, organ_crop, window_size)
    
    if len(features) == 0:
        raise RuntimeError(f"No features extracted from organ {organ_name} in scan {scan_path}. Organ crop may be too small.")
    
    # Save features with position information
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        features=features,
        positions=positions,
        bbox_origin=bbox_origin,
        organ_name=organ_name
    )


def main(args):
    fix_random_seeds(getattr(args, "seed", 0))
    
    # Validate inputs early
    if not os.path.exists(args.scan_path):
        raise FileNotFoundError(f"Scan file not found: {args.scan_path}")
    if not os.path.isfile(args.scan_path):
        raise ValueError(f"Scan path is not a file: {args.scan_path}")
    if not os.access(args.scan_path, os.R_OK):
        raise PermissionError(f"Cannot read scan file: {args.scan_path}")
    
    if not os.path.exists(args.seg_path):
        raise FileNotFoundError(f"Segmentation file not found: {args.seg_path}")
    if not os.path.isfile(args.seg_path):
        raise ValueError(f"Segmentation path is not a file: {args.seg_path}")
    if not os.access(args.seg_path, os.R_OK):
        raise PermissionError(f"Cannot read segmentation file: {args.seg_path}")
    
    # Ensure output directory can be created
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise OSError(f"Cannot create output directory {output_dir}: {e}")
    
    # Check write permissions for output
    if os.path.exists(args.output_path) and not os.access(args.output_path, os.W_OK):
        raise PermissionError(f"Cannot write to output file: {args.output_path}")
    
    if not args.organ_name or not args.organ_name.strip():
        raise ValueError(f"Organ name cannot be empty")
    
    # Window size for CT-FM: 128x128x128
    window_size = (128, 128, 128)
    
    # Load model
    print("Loading model...")
    try:
        model = load_model()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e
    
    # Process scan
    print(f"Processing {args.scan_path} for organ {args.organ_name}...")
    try:
        process_scan(model, args.scan_path, args.seg_path, args.organ_name, window_size, args.output_path)
    except Exception as e:
        raise RuntimeError(f"Failed to process scan {args.scan_path} for organ {args.organ_name}: {e}") from e
    
    # Verify output was created
    if not os.path.exists(args.output_path):
        raise RuntimeError(f"Output file was not created: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT-FM Feature Extraction for LEAVS")
    parser.add_argument("--scan-path", type=str, required=True, help="Path to scan file (.nii.gz)")
    parser.add_argument("--seg-path", type=str, required=True, help="Path to segmentation file (.nii.gz)")
    parser.add_argument("--organ-name", type=str, required=True, help="Organ name (e.g., spleen)")
    parser.add_argument("--output-path", type=str, required=True, help="Output path for features")
    args = parser.parse_args()

    sys.exit(main(args))
