# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from mmm.labelstudio_ext.NativeBlocks import NativeBlocks, MMM_MODELS, DEFAULT_MODEL
from monai.transforms import (
    Compose,
    EnsureTyped,
    ScaleIntensityRanged,
)
from tqdm import tqdm

from util.util import fix_random_seeds
from util.leavs_utils import get_organ_crop
from util.sliding_window import sliding_window_2d_slices
import argparse


def load_model():
    model = NativeBlocks(MMM_MODELS[DEFAULT_MODEL], device_identifier="cuda:0")
    return model


def preprocess_slice(slice_2d: np.ndarray) -> torch.Tensor:
    """
    Preprocess a 2D slice for UMedPT model.
    Expected input: 224x224
    """
    transform = Compose([
        ScaleIntensityRanged(
            keys=["image"],
            a_min=None,
            a_max=None,
            b_min=0.0,
            b_max=1.0,
            clip=False,
        ),
        EnsureTyped(keys=["image"]),
    ])
    
    # Convert to MONAI format: (C, H, W)
    slice_tensor = torch.from_numpy(slice_2d).unsqueeze(0).float()
    data_dict = {"image": slice_tensor}
    transformed = transform(data_dict)
    
    # UMedPT expects (B, C, H, W) format
    img = transformed["image"]  # (C, H, W)
    
    return img


def extract_features_for_organ(
    model,
    organ_crop: np.ndarray,
    window_size: tuple,
    stride: int = 1
) -> tuple:
    """
    Extract features for an organ crop using sliding windows (2D slices).
    
    Returns:
        features: List of feature vectors
        positions: List of slice indices
    """
    features = []
    positions = []
    
    for slice_2d, slice_idx in sliding_window_2d_slices(organ_crop, window_size, stride, axis=0):
        # Preprocess slice
        slice_tensor = preprocess_slice(slice_2d)
        
        # Extract features
        with torch.inference_mode():
            # UMedPT expects (B, C, H, W) format
            model_input = slice_tensor.unsqueeze(0).cuda()  # (1, C, H, W)
            feature_pyramid = model["encoder"](model_input.to(model.device))
            feature = model["squeezer"](feature_pyramid)[1].detach().cpu().numpy()
        
        features.append(feature)
        positions.append(slice_idx)
    
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
    
    # Window size for UMedPT: 224x224
    window_size = (224, 224)
    
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
    parser = argparse.ArgumentParser(description="UMedPT Feature Extraction for LEAVS")
    parser.add_argument("--scan-path", type=str, required=True, help="Path to scan file (.nii.gz)")
    parser.add_argument("--seg-path", type=str, required=True, help="Path to segmentation file (.nii.gz)")
    parser.add_argument("--organ-name", type=str, required=True, help="Organ name (e.g., spleen)")
    parser.add_argument("--output-path", type=str, required=True, help="Output path for features")
    args = parser.parse_args()

    sys.exit(main(args))
