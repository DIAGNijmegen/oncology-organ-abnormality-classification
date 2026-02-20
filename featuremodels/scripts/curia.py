# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import os
import sys

import numpy as np
import torch
import nibabel as nib
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor
from monai.transforms import Orientationd
from monai.data import MetaTensor

import argparse
from util.util import fix_random_seeds
from util.leavs_utils import get_organ_crop
from util.sliding_window import sliding_window_2d_slices
from util.snakemake_helpers import VALID_ORGANS


def load_model():
    model = AutoModel.from_pretrained("raidium/curia")
    processor = AutoImageProcessor.from_pretrained("raidium/curia", trust_remote_code=True)
    model.cuda().eval()
    return model, processor


def reorient_crop_to_pl(crop: np.ndarray, scan_path: str) -> np.ndarray:
    """
    Reorient a 3D crop to PL (Posterior-Left) orientation using MONAI.
    This ensures axial slices are in PL orientation as expected by Curia.
    
    Args:
        crop: 3D numpy array (Z, Y, X)
        scan_path: Path to original scan (used to get orientation info)
    
    Returns:
        Reoriented 3D numpy array in PL orientation
    """
    # Load the original scan to get its affine matrix for orientation info
    scan_img = nib.load(scan_path)
    original_affine = scan_img.affine
    
    # Use MONAI's Orientationd transform with proper MetaTensor setup
    transform = Orientationd(keys=["image"], axcodes="PLI")
    
    # Convert crop to MetaTensor with affine information
    crop_tensor = torch.from_numpy(crop).unsqueeze(0).float()
    crop_meta = MetaTensor(crop_tensor, affine=original_affine)
    
    # Apply reorientation transform
    data_dict = {"image": crop_meta}
    transformed = transform(data_dict)
    reoriented = transformed["image"].squeeze(0).numpy()
    
    return reoriented


def preprocess_slice(slice_2d: np.ndarray, processor) -> dict:
    """
    Preprocess a 2D slice for Curia model.
    Expected input: 256x256
    Curia processor expects numpy array (H, W).
    """
    # Normalize HU values (-1000 to 3000) to [0, 1] range for processor
    img_clipped = np.clip(slice_2d, -1000, 3000)
    img_normalized = (img_clipped + 1000) / 4000
    
    # Use processor to prepare for model (processor handles model-specific preprocessing)
    model_input = processor(img_normalized)
    
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
    
    # Extract slices along axis=0 (which should be the I axis after PLI reorientation)
    # This gives us axial slices in PL orientation as expected by Curia
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


def process_scan_for_organ(
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
    
    # Reorient crop to PL orientation before extracting slices
    # This ensures axial slices are in PL orientation as expected by Curia
    organ_crop = reorient_crop_to_pl(organ_crop, scan_path)
    
    # Extract features
    features, positions = extract_features_for_organ(model, processor, organ_crop, window_size)
    
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
    processor,
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
        print(f"Extracting features for organ: {organ_name}")
        output_path = os.path.join(
            output_root,
            model_name,
            organ_name,
            split,
            "features",
            "raw",
            f"{scan_id}.npz",
        )
        if process_scan_for_organ(model, processor, scan_path, seg_path, organ_name, window_size, output_path):
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
    
    # Window size for Curia: 256x256
    window_size = (256, 256)
    
    # Load model once for the entire batch
    print("Loading model...")
    try:
        model, processor = load_model()
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
                processor,
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
    parser = argparse.ArgumentParser(description="Curia Feature Extraction for LEAVS")
    parser.add_argument("--scan-paths-file", type=str, required=True, help="File containing scan file paths (.nii.gz), one per line")
    parser.add_argument("--seg-paths-file", type=str, required=True, help="File containing segmentation file paths (.nii.gz), one per line")
    parser.add_argument("--output-root", type=str, required=True, help="Root output directory following workflow conventions")
    parser.add_argument("--model-name", type=str, required=True, help="Feature model name")
    parser.add_argument("--split", type=str, required=True, choices=["training", "validation", "test"], help="Dataset split")
    args = parser.parse_args()

    sys.exit(main(args))
