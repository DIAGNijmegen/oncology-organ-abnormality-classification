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
from PIL import Image
from monai.transforms import Orientationd, Compose, EnsureTyped
from monai.data import MetaTensor

import argparse
from util.util import fix_random_seeds
from util.leavs_utils import get_organ_crop
from util.sliding_window import sliding_window_2d_slices


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
    Curia processor expects PIL Image or numpy array (H, W).
    """
    # Convert to PIL Image for processor
    # Normalize HU values (-1000 to 3000) to uint8 [0, 255] for PIL
    img_clipped = np.clip(slice_2d, -1000, 3000)
    img_normalized = ((img_clipped + 1000) / 4000 * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_normalized)
    
    # Use processor to prepare for model (processor handles model-specific preprocessing)
    model_input = processor(img_pil)
    
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
            if process_scan_for_organ(model, processor, scan_path, seg_path, organ_name, window_size, output_path):
                processed_count += 1
    
    print(f"Successfully processed {processed_count}/{len(organ_names)} organs for scan")


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
    
    # Parse organ names and output paths
    output_paths = {}
    organ_names = []
    
    if args.output_paths:
        for pair in args.output_paths.split(','):
            if ':' not in pair:
                raise ValueError(f"Invalid output path format: {pair}. Expected 'organ:path'")
            organ_name, output_path = pair.split(':', 1)
            organ_name = organ_name.strip()
            output_path = output_path.strip()
            output_paths[organ_name] = output_path
            organ_names.append(organ_name)
    else:
        raise ValueError("--output-paths is required. Format: 'organ1:path1,organ2:path2'")
    
    if not organ_names:
        raise ValueError("No organs specified for processing")
    
    # Ensure output directories can be created
    for output_path in output_paths.values():
        output_dir = os.path.dirname(output_path)
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                raise OSError(f"Cannot create output directory {output_dir}: {e}")
        # Check write permissions
        if os.path.exists(output_path) and not os.access(output_path, os.W_OK):
            raise PermissionError(f"Cannot write to output file: {output_path}")
    
    # Window size for Curia: 256x256
    window_size = (256, 256)
    
    # Load model
    print("Loading model...")
    try:
        model, processor = load_model()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e
    
    # Process scan for all organs
    try:
        process_scan_for_all_organs(
            model, 
            processor,
            args.scan_path, 
            args.seg_path, 
            organ_names, 
            window_size, 
            output_paths
        )
    except Exception as e:
        raise RuntimeError(f"Failed to process scan {args.scan_path}: {e}") from e
    
    # Verify at least one output was created
    output_created = False
    for output_path in output_paths.values():
        if os.path.exists(output_path):
            output_created = True
            break
    
    if not output_created:
        raise RuntimeError(f"No output files were created for scan")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curia Feature Extraction for LEAVS")
    parser.add_argument("--scan-path", type=str, required=True, help="Path to scan file (.nii.gz)")
    parser.add_argument("--seg-path", type=str, required=True, help="Path to segmentation file (.nii.gz)")
    parser.add_argument("--output-paths", type=str, required=True, help="Comma-separated list of 'organ:path' pairs (e.g., 'spleen:/path/to/spleen.npz,liver:/path/to/liver.npz')")
    args = parser.parse_args()

    sys.exit(main(args))
