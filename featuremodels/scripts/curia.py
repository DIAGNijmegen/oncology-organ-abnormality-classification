# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import os
import sys

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureTyped,
    Orientationd,
    EnsureChannelFirstd,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor

import argparse
from util.util import fix_random_seeds
from util.leavs_utils import get_organ_crop
from util.sliding_window import sliding_window_2d_slices


def load_model():
    model = AutoModel.from_pretrained("raidium/curia")
    processor = AutoImageProcessor.from_pretrained("raidium/curia", trust_remote_code=True)
    model.cuda().eval()
    return model, processor


def preprocess_slice(slice_2d: np.ndarray, processor) -> dict:
    """
    Preprocess a 2D slice for Curia model.
    Expected input: 256x256
    """
    transform = Compose([
        Orientationd(keys=["image"], axcodes="PLS"),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image"]),
    ])
    
    # Convert to MONAI format: (C, H, W)
    slice_tensor = torch.from_numpy(slice_2d).unsqueeze(0).float()
    data_dict = {"image": slice_tensor}
    transformed = transform(data_dict)
    
    # Curia expects PIL Image or numpy array (H, W)
    img = transformed["image"].squeeze(0).numpy()  # (H, W)
    
    # Use processor to prepare for model
    model_input = processor(img)
    
    return model_input


def extract_features_for_organ(
    model,
    processor,
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
        model_input = preprocess_slice(slice_2d, processor)
        
        # Extract features
        with torch.no_grad():
            # Move inputs to GPU
            model_input_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in model_input.items()}
            output = model(**model_input_gpu)
            feature = output["pooler_output"].detach().cpu().numpy()
        
        features.append(feature)
        positions.append(slice_idx)
    
    return np.array(features), np.array(positions)


def process_scan(
    model,
    processor,
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
    features, positions = extract_features_for_organ(model, processor, organ_crop, window_size)
    
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
    
    # Window size for Curia: 256x256
    window_size = (256, 256)
    
    # Load model
    print("Loading model...")
    try:
        model, processor = load_model()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e
    
    # Process scan
    print(f"Processing {args.scan_path} for organ {args.organ_name}...")
    try:
        process_scan(model, processor, args.scan_path, args.seg_path, args.organ_name, window_size, args.output_path)
    except Exception as e:
        raise RuntimeError(f"Failed to process scan {args.scan_path} for organ {args.organ_name}: {e}") from e
    
    # Verify output was created
    if not os.path.exists(args.output_path):
        raise RuntimeError(f"Output file was not created: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curia Feature Extraction for LEAVS")
    parser.add_argument("--scan-path", type=str, required=True, help="Path to scan file (.nii.gz)")
    parser.add_argument("--seg-path", type=str, required=True, help="Path to segmentation file (.nii.gz)")
    parser.add_argument("--organ-name", type=str, required=True, help="Organ name (e.g., spleen)")
    parser.add_argument("--output-path", type=str, required=True, help="Output path for features")
    args = parser.parse_args()

    sys.exit(main(args))
