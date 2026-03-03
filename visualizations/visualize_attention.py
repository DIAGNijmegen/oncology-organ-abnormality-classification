# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import os
import sys
import argparse
from typing import Tuple

import numpy as np
import torch
import nibabel as nib

from evaluation.scripts.attention import AttentionMIL
from evaluation.scripts.evaluation_utils import (
    get_attention_checkpoint_output_dir,
)
from util.leavs_utils import get_organ_crop


def load_foundation_model(model_name: str):
    """
    Load foundation model dynamically based on model name.
    
    Args:
        model_name: Name of the model (e.g., 'spectre', 'ctfm', 'tapct', 'curia', 'umedpt', 'spectrevitg')
    
    Returns:
        Model object (and processor for curia)
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower == "spectre":
        from featuremodels.scripts.spectre import load_model
        return load_model(), None
    elif model_name_lower == "ctfm":
        from featuremodels.scripts.ctfm import load_model
        return load_model(), None
    elif model_name_lower == "tapct":
        from featuremodels.scripts.tapct import load_model
        return load_model(), None
    elif model_name_lower == "curia":
        from featuremodels.scripts.curia import load_model
        return load_model()  # Returns (model, processor)
    elif model_name_lower == "umedpt":
        from featuremodels.scripts.umedpt import load_model
        return load_model(), None
    elif model_name_lower == "spectrevitg":
        from featuremodels.scripts.spectrevitg import load_model
        return load_model(), None
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported models: spectre, ctfm, tapct, curia, umedpt, spectrevitg")


def extract_features_for_organ_with_positions(
    model,
    processor,
    scan_path: str,
    seg_path: str,
    organ_name: str,
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """
    Extract features from a scan for a specific organ, returning features, positions, and organ crop.
    
    Args:
        model: Foundation model
        processor: Processor (for curia, None otherwise)
        scan_path: Path to scan NIfTI file
        seg_path: Path to segmentation NIfTI file
        organ_name: Name of the organ
        model_name: Name of the model (to determine window size and extraction method)
    
    Returns:
        (features, positions, organ_crop, bbox_origin)
        - features: (n_patches, feature_dim) array
        - positions: (n_patches, 3) array of (z, y, x) positions
        - organ_crop: 3D numpy array of the organ crop
        - bbox_origin: (z, y, x) origin of the bounding box
    """
    model_name_lower = model_name.lower()
    
    # Define window sizes for each model
    window_sizes = {
        "spectre": (128, 128, 64),
        "ctfm": (128, 128, 48),
        "tapct": (96, 96, 96),
        "curia": (512, 512),  # 2D
        "umedpt": (512, 512),  # 2D
        "spectrevitg": (128, 128, 64),
    }
    
    window_size = window_sizes.get(model_name_lower)
    if window_size is None:
        raise ValueError(f"Unknown window size for model: {model_name}")
    
    # Get organ crop
    result = get_organ_crop(scan_path, seg_path, organ_name, window_size)
    if result is None:
        raise ValueError(f"Organ {organ_name} not found in segmentation {seg_path}")
    
    organ_crop, bbox_origin = result
    
    # Extract features based on model type
    if model_name_lower == "spectre":
        from featuremodels.scripts.spectre import extract_features_for_organ, apply_spacing_to_crop
        organ_crop = apply_spacing_to_crop(organ_crop, scan_path)
        features, positions = extract_features_for_organ(model, organ_crop, window_size)
    elif model_name_lower == "ctfm":
        from featuremodels.scripts.ctfm import extract_features_for_organ
        features, positions = extract_features_for_organ(model, organ_crop, window_size)
    elif model_name_lower == "tapct":
        from featuremodels.scripts.tapct import extract_features_for_organ
        features, positions = extract_features_for_organ(model, organ_crop, window_size)
    elif model_name_lower == "curia":
        from featuremodels.scripts.curia import extract_features_for_organ, reorient_crop_to_pl
        organ_crop = reorient_crop_to_pl(organ_crop, scan_path)
        features, positions = extract_features_for_organ(model, processor, organ_crop, window_size)
    elif model_name_lower == "umedpt":
        from featuremodels.scripts.umedpt import extract_features_for_organ
        features, positions = extract_features_for_organ(model, organ_crop, window_size)
    elif model_name_lower == "spectrevitg":
        from featuremodels.scripts.spectrevitg import extract_features_for_organ_grid, apply_spacing_to_crop
        organ_crop = apply_spacing_to_crop(organ_crop, scan_path)
        raise NotImplementedError("spectrevitg uses grid-based processing without positions. Use a different model for attention visualization.")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    if len(features) == 0:
        raise ValueError(f"No features extracted from organ {organ_name}")
    
    # Flatten features to 1D per patch (same as in load_raw_features_and_labels)
    if len(features.shape) == 1:
        features = features.flatten().reshape(1, -1)
    else:
        n_patches = features.shape[0]
        flattened_patches = []
        for i in range(n_patches):
            flattened_patch = features[i].flatten()
            flattened_patches.append(flattened_patch)
        features = np.array(flattened_patches)
    
    return features, positions, organ_crop, bbox_origin


def create_attention_volume(
    attention_weights: np.ndarray,
    positions: np.ndarray,
    organ_crop_shape: Tuple[int, int, int],
    window_size: Tuple[int, ...],
) -> np.ndarray:
    """
    Create a 3D volume with attention weights mapped to spatial positions.
    Uses max pooling for overlapping patches to show the maximum attention at each voxel.
    
    Args:
        attention_weights: (n_patches,) array of attention weights
        positions: (n_patches, 3) array of (z, y, x) patch positions
        organ_crop_shape: (z, y, x) shape of the organ crop
        window_size: (z, y, x) or (y, x) window size for patches
    
    Returns:
        3D numpy array with attention weights at patch locations
    """
    attention_volume = np.zeros(organ_crop_shape, dtype=np.float32)
    count_volume = np.zeros(organ_crop_shape, dtype=np.float32)  # Track overlaps
    
    # Handle 2D vs 3D window sizes
    if len(window_size) == 2:
        window_z = 1  # For 2D models, assume single slice
        window_y, window_x = window_size
    else:
        window_z, window_y, window_x = window_size
    
    # Map attention weights to spatial locations
    for i, (z, y, x) in enumerate(positions):
        weight = attention_weights[i]
        
        # Ensure positions are within bounds
        z_start = max(0, int(z))
        z_end = min(organ_crop_shape[0], int(z) + window_z)
        y_start = max(0, int(y))
        y_end = min(organ_crop_shape[1], int(y) + window_y)
        x_start = max(0, int(x))
        x_end = min(organ_crop_shape[2], int(x) + window_x)
        
        # Use max pooling for overlapping patches (show maximum attention)
        patch_region = attention_volume[z_start:z_end, y_start:y_end, x_start:x_end]
        attention_volume[z_start:z_end, y_start:y_end, x_start:x_end] = np.maximum(patch_region, weight)
    
    return attention_volume


def get_scan_paths(scan_id: str, dataset_root: str) -> Tuple[str, str]:
    """
    Get scan and segmentation paths from scan ID.
    
    Args:
        scan_id: Scan ID
        dataset_root: Root directory of the dataset
    
    Returns:
        (scan_path, seg_path)
    """
    # Try training/validation first
    train_scan_path = os.path.join(dataset_root, "LEAVS", "imagesTr", f"{scan_id}.nii.gz")
    if os.path.exists(train_scan_path):
        scan_path = train_scan_path
    else:
        # Try test
        test_scan_path = os.path.join(dataset_root, "LEAVS", "imagesTs", f"{scan_id}.nii.gz")
        if os.path.exists(test_scan_path):
            scan_path = test_scan_path
        else:
            raise FileNotFoundError(f"Scan not found: {scan_id}. Checked {train_scan_path} and {test_scan_path}")
    
    seg_path = os.path.join(dataset_root, "LEAVS", "AMOS-MM-TotalSegmentator", f"{scan_id}_segmentation.nii.gz")
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Segmentation not found: {seg_path}")
    
    return scan_path, seg_path


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset_root = args.dataset_root
    
    # Get scan and segmentation paths
    scan_id_for_output = args.scan_id
    if os.path.isfile(args.scan_id):
        # If scan_id is actually a path, use it directly
        scan_path = args.scan_id
        # Extract scan_id from path for output filename
        scan_id_for_output = os.path.basename(scan_path).replace(".nii.gz", "")
        # Try to infer seg_path
        scan_basename = scan_id_for_output
        seg_path = os.path.join(os.path.dirname(scan_path).replace("imagesTr", "AMOS-MM-TotalSegmentator").replace("imagesTs", "AMOS-MM-TotalSegmentator"), f"{scan_basename}_segmentation.nii.gz")
        if not os.path.exists(seg_path):
            # Try alternative location
            seg_path = os.path.join(dataset_root, "LEAVS", "AMOS-MM-TotalSegmentator", f"{scan_basename}_segmentation.nii.gz")
    else:
        scan_path, seg_path = get_scan_paths(args.scan_id, dataset_root)
        scan_id_for_output = args.scan_id
    
    print(f"Scan path: {scan_path}")
    print(f"Segmentation path: {seg_path}")
    
    # Load foundation model
    print(f"Loading foundation model: {args.model_name}...")
    model_result = load_foundation_model(args.model_name)
    if isinstance(model_result, tuple) and len(model_result) == 2:
        foundation_model, processor = model_result
    else:
        foundation_model, processor = model_result, None
    
    # Extract features
    print(f"Extracting features for organ: {args.organ_name}...")
    features, positions, organ_crop, bbox_origin = extract_features_for_organ_with_positions(
        foundation_model,
        processor,
        scan_path,
        seg_path,
        args.organ_name,
        args.model_name,
    )
    
    print(f"Extracted {len(features)} patches with feature dimension {features.shape[1]}")
    print(f"Organ crop shape: {organ_crop.shape}")
    
    # Load attention checkpoint
    checkpoint_dir = get_attention_checkpoint_output_dir(
        args.output_root,
        args.model_name,
        args.organ_name,
    )
    checkpoint_path = os.path.join(checkpoint_dir, "best_model_attention.pth")
    
    if not os.path.exists(checkpoint_path):
        # Try exclude_amos22 checkpoint
        checkpoint_dir_exclude = checkpoint_dir + "_exclude_amos22"
        checkpoint_path = os.path.join(checkpoint_dir_exclude, "best_model_attention.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Attention checkpoint not found. Checked:\n  - {os.path.join(checkpoint_dir, 'best_model_attention.pth')}\n  - {checkpoint_path}")
    
    print(f"Loading attention checkpoint from: {checkpoint_path}")
    
    # Initialize attention model
    embedding_dim = features.shape[1]
    attention_model = AttentionMIL(embedding_dim=embedding_dim, hidden_dim=128)
    attention_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    attention_model.to(device)
    attention_model.eval()
    
    # Run inference
    print("Running attention inference...")
    with torch.no_grad():
        # Prepare input
        patches_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # [1, n_patches, embedding_dim]
        mask = torch.ones(1, len(features), dtype=torch.bool).to(device)  # All patches are valid
        
        # Forward pass
        logits, attention_weights = attention_model(patches_tensor, mask=mask, return_attention=True)
        probability = torch.sigmoid(logits).item()
        attention_weights = attention_weights[0].cpu().numpy()  # [n_patches]
    
    print(f"Prediction probability: {probability:.4f}")
    print(f"Attention weights - min: {attention_weights.min():.4f}, max: {attention_weights.max():.4f}, mean: {attention_weights.mean():.4f}")
    
    # Get window size for attention mapping
    window_sizes = {
        "spectre": (128, 128, 64),
        "ctfm": (128, 128, 48),
        "tapct": (96, 96, 96),
        "curia": (512, 512),
        "umedpt": (512, 512),
        "spectrevitg": (128, 128, 64),
    }
    window_size = window_sizes.get(args.model_name.lower(), (128, 128, 64))
    
    # Create attention volume
    print("Creating attention volume...")
    attention_volume = create_attention_volume(
        attention_weights,
        positions,
        organ_crop.shape,
        window_size,
    )
    
    # Load original scan to get affine and header for saving
    scan_img = nib.load(scan_path)
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save organ crop
    crop_output_path = os.path.join(output_dir, f"{scan_id_for_output}_{args.organ_name}_{args.model_name}_crop.nii.gz")
    print(f"Saving organ crop to: {crop_output_path}")
    
    # Create Nifti image for crop
    # We need to adjust the affine to account for bbox_origin
    crop_affine = scan_img.affine.copy()
    # Translate affine to account for bbox origin
    # bbox_origin is (z, y, x) but affine uses (x, y, z)
    bbox_origin_xyz = np.array([bbox_origin[2], bbox_origin[1], bbox_origin[0]])
    crop_affine[:3, 3] += np.dot(crop_affine[:3, :3], bbox_origin_xyz)
    
    crop_nifti = nib.Nifti1Image(organ_crop.astype(np.float32), crop_affine, scan_img.header)
    nib.save(crop_nifti, crop_output_path)
    
    # Save attention map
    attention_output_path = os.path.join(output_dir, f"{scan_id_for_output}_{args.organ_name}_{args.model_name}_attention.nii.gz")
    print(f"Saving attention map to: {attention_output_path}")
    
    attention_nifti = nib.Nifti1Image(attention_volume.astype(np.float32), crop_affine, scan_img.header)
    nib.save(attention_nifti, attention_output_path)
    
    print("Done!")
    print(f"\nOutput files:")
    print(f"  - Organ crop: {crop_output_path}")
    print(f"  - Attention map: {attention_output_path}")
    print(f"\nPrediction probability: {probability:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize attention maps for a single scan and organ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scan-id",
        type=str,
        required=True,
        help="Scan ID (or path to scan file)",
    )
    parser.add_argument(
        "--organ-name",
        type=str,
        required=True,
        help="Organ name",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Foundation model name (spectre, ctfm, tapct, curia, umedpt, spectrevitg)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Workflow output root directory (where checkpoints are stored)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./attention_visualizations",
        help="Output directory for visualization files",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Dataset root directory",
    )
    
    args = parser.parse_args()
    sys.exit(main(args) or 0)
