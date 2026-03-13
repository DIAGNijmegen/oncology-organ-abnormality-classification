# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import os
import sys
import argparse
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm
from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureTyped,
    Orientationd,
    ScaleIntensityRanged,
    ResizeWithPadOrCropd,
    NormalizeIntensityd,
    Spacingd,
)
from monai.data import MetaTensor

from evaluation.scripts.attention import AttentionMIL
from evaluation.scripts.evaluation_utils import (
    get_attention_checkpoint_output_dir,
)
from util.leavs_utils import get_organ_crop
from util.sliding_window import sliding_window_3d, sliding_window_2d_slices

INFERENCE_BATCH_SIZE = 2
PREPROCESS_WORKERS = 2


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


def apply_spacing_to_crop(crop: np.ndarray, scan_path: str) -> np.ndarray:
    """Apply spacing (0.5, 0.5, 1.0) to the entire organ crop using MONAI."""
    scan_img = nib.load(scan_path)
    original_affine = scan_img.affine
    
    transform = Spacingd(keys=["image"], pixdim=(0.5, 0.5, 1.0), mode="bilinear")
    crop_tensor = torch.from_numpy(crop).unsqueeze(0).float()
    crop_meta = MetaTensor(crop_tensor, affine=original_affine)
    
    data_dict = {"image": crop_meta}
    transformed = transform(data_dict)
    resampled = transformed["image"].squeeze(0).numpy()
    
    return resampled


def reorient_crop_to_pl(crop: np.ndarray, scan_path: str) -> np.ndarray:
    """Reorient crop to PL orientation."""
    scan_img = nib.load(scan_path)
    original_affine = scan_img.affine
    
    transform = Orientationd(keys=["image"], axcodes="PLI")
    crop_tensor = torch.from_numpy(crop).unsqueeze(0).float()
    crop_meta = MetaTensor(crop_tensor, affine=original_affine)
    
    data_dict = {"image": crop_meta}
    transformed = transform(data_dict)
    reoriented = transformed["image"].squeeze(0).numpy()
    
    return reoriented


def extract_features_spectre(model, organ_crop: np.ndarray, window_size: tuple, stride: tuple = None):
    """Extract features for SPECTRE model."""
    if stride is None:
        stride = tuple(s // 2 for s in window_size)
    
    def preprocess_patch(patch: np.ndarray) -> torch.Tensor:
        transform = Compose([
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ])
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()
        data_dict = {"image": patch_tensor}
        transformed = transform(data_dict)
        return transformed["image"].unsqueeze(0)
    
    patches = []
    positions = []
    print("Extracting patches...")
    for patch, (z, y, x) in tqdm(sliding_window_3d(organ_crop, window_size, stride), desc="Extracting patches"):
        patches.append(patch)
        positions.append((z, y, x))
    
    if not patches:
        return np.array([]), np.array([]), []
    
    print(f"Preprocessing {len(patches)} patches...")
    with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as executor:
        preprocessed_patches = list(tqdm(executor.map(preprocess_patch, patches), total=len(patches), desc="Preprocessing"))
    
    features = []
    num_batches = (len(preprocessed_patches) + INFERENCE_BATCH_SIZE - 1) // INFERENCE_BATCH_SIZE
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(preprocessed_patches), INFERENCE_BATCH_SIZE), desc="Extracting features", total=num_batches):
            batch_items = preprocessed_patches[batch_start:batch_start + INFERENCE_BATCH_SIZE]
            batch_tensor = torch.cat(batch_items, dim=0).cuda()
            output = model(batch_tensor.unsqueeze(1), grid_size=(1, 1, 1))
            batch_features = output.detach().cpu().numpy()
            for feature in batch_features:
                features.append(np.expand_dims(feature, axis=0))
    
    return np.array(features), np.array(positions), patches


def extract_features_ctfm(model, organ_crop: np.ndarray, window_size: tuple, stride: tuple = None):
    """Extract features for CT-FM model."""
    if stride is None:
        stride = tuple(s // 2 for s in window_size)
    
    def preprocess_patch(patch: np.ndarray) -> torch.Tensor:
        transform = Compose([
            Orientationd(keys=["image"], axcodes="SPL"),
            ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
            EnsureTyped(keys=["image"]),
        ])
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()
        data_dict = {"image": patch_tensor}
        transformed = transform(data_dict)
        return transformed["image"].unsqueeze(0)
    
    patches = []
    positions = []
    print("Extracting patches...")
    for patch, (z, y, x) in tqdm(sliding_window_3d(organ_crop, window_size, stride), desc="Extracting patches"):
        patches.append(patch)
        positions.append((z, y, x))
    
    if not patches:
        return np.array([]), np.array([]), []
    
    print(f"Preprocessing {len(patches)} patches...")
    with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as executor:
        preprocessed_patches = list(tqdm(executor.map(preprocess_patch, patches), total=len(patches), desc="Preprocessing"))
    
    features = []
    num_batches = (len(preprocessed_patches) + INFERENCE_BATCH_SIZE - 1) // INFERENCE_BATCH_SIZE
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(preprocessed_patches), INFERENCE_BATCH_SIZE), desc="Extracting features", total=num_batches):
            batch_items = preprocessed_patches[batch_start:batch_start + INFERENCE_BATCH_SIZE]
            batch_tensor = torch.cat(batch_items, dim=0).cuda()
            output = model(batch_tensor)
            batch_features = output[-1].detach().cpu().numpy()
            for feature in batch_features:
                features.append(np.expand_dims(feature, axis=0))
    
    return np.array(features), np.array(positions), patches


def extract_features_tapct(model, organ_crop: np.ndarray, window_size: tuple, stride: tuple = None):
    """Extract features for TAP-CT model."""
    if stride is None:
        stride = tuple(s // 2 for s in window_size)
    
    def preprocess_patch(patch: np.ndarray) -> torch.Tensor:
        transform = Compose([
            ScaleIntensityRanged(keys=["image"], a_min=-1008, a_max=822, b_min=-1008, b_max=822, clip=True),
            NormalizeIntensityd(keys=["image"], subtrahend=-86.8086, divisor=322.6347),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=(224, 224, -1)),
            DivisiblePadd(keys=["image"], k=(1, 1, 4)),
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ])
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()
        data_dict = {"image": patch_tensor}
        transformed = transform(data_dict)
        img = transformed["image"]
        img = img.permute(0, 3, 1, 2)
        return img.unsqueeze(0)
    
    patches = []
    positions = []
    print("Extracting patches...")
    for patch, (z, y, x) in tqdm(sliding_window_3d(organ_crop, window_size, stride), desc="Extracting patches"):
        patches.append(patch)
        positions.append((z, y, x))
    
    if not patches:
        return np.array([]), np.array([]), []
    
    print(f"Preprocessing {len(patches)} patches...")
    with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as executor:
        preprocessed_patches = list(tqdm(executor.map(preprocess_patch, patches), total=len(patches), desc="Preprocessing"))
    
    features = []
    num_batches = (len(preprocessed_patches) + INFERENCE_BATCH_SIZE - 1) // INFERENCE_BATCH_SIZE
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(preprocessed_patches), INFERENCE_BATCH_SIZE), desc="Extracting features", total=num_batches):
            batch_items = preprocessed_patches[batch_start:batch_start + INFERENCE_BATCH_SIZE]
            batch_tensor = torch.cat(batch_items, dim=0).cuda()
            output = model(batch_tensor)
            batch_features = output["pooler_output"].detach().cpu().numpy()
            for feature in batch_features:
                features.append(np.expand_dims(feature, axis=0))
    
    return np.array(features), np.array(positions), patches


def extract_features_curia(model, processor, organ_crop: np.ndarray, window_size: tuple, stride: int = 1):
    """Extract features for Curia model."""
    def preprocess_slice(slice_2d: np.ndarray, processor) -> dict:
        img_clipped = np.clip(slice_2d, -1000, 3000)
        img_normalized = (img_clipped + 1000) / 4000
        return processor(img_normalized)
    
    features = []
    positions = []
    patches = []
    
    # Count slices first for progress bar
    slice_list = list(sliding_window_2d_slices(organ_crop, window_size, stride, axis=0))
    
    for slice_2d, slice_idx in tqdm(slice_list, desc="Extracting features"):
        model_input = preprocess_slice(slice_2d, processor)
        patches.append(slice_2d)
        
        with torch.no_grad():
            model_input_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in model_input.items()}
            output = model(**model_input_gpu)
            feature = output["pooler_output"].detach().cpu().numpy()
        
        features.append(feature)
        positions.append(slice_idx)
    
    return np.array(features), np.array(positions), patches


def extract_features_umedpt(model, organ_crop: np.ndarray, window_size: tuple, stride: int = 1):
    """Extract features for UMedPT model."""
    def preprocess_slice(slice_2d: np.ndarray) -> torch.Tensor:
        transform = Compose([
            ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ])
        slice_tensor = torch.from_numpy(slice_2d).unsqueeze(0).float()
        data_dict = {"image": slice_tensor}
        transformed = transform(data_dict)
        return transformed["image"]
    
    slices = []
    positions = []
    patches = []
    print("Extracting slices...")
    for slice_2d, slice_idx in tqdm(sliding_window_2d_slices(organ_crop, window_size, stride, axis=0), desc="Extracting slices"):
        slices.append(slice_2d)
        positions.append(slice_idx)
        patches.append(slice_2d)
    
    if not slices:
        return np.array([]), np.array([]), []
    
    print(f"Preprocessing {len(slices)} slices...")
    with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as executor:
        preprocessed_slices = list(tqdm(executor.map(preprocess_slice, slices), total=len(slices), desc="Preprocessing"))
    
    features = []
    num_batches = (len(preprocessed_slices) + INFERENCE_BATCH_SIZE - 1) // INFERENCE_BATCH_SIZE
    with torch.inference_mode():
        for batch_start in tqdm(range(0, len(preprocessed_slices), INFERENCE_BATCH_SIZE), desc="Extracting features", total=num_batches):
            batch_items = preprocessed_slices[batch_start:batch_start + INFERENCE_BATCH_SIZE]
            batch_tensor = torch.stack(batch_items, dim=0)
            model_input = batch_tensor.expand(batch_tensor.shape[0], 3, 224, 224).cuda()
            feature_pyramid = model["encoder"](model_input.to(model.device))
            batch_features = model["squeezer"](feature_pyramid)[1].detach().cpu().numpy()
            for feature in batch_features:
                features.append(np.expand_dims(feature, axis=0))
    
    return np.array(features), np.array(positions), patches


def extract_features_for_organ_with_positions(
    model,
    processor,
    scan_path: str,
    seg_path: str,
    organ_name: str,
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int], List[np.ndarray]]:
    """
    Extract features from a scan for a specific organ, returning features, positions, organ crop, and patches.
    
    Args:
        model: Foundation model
        processor: Processor (for curia, None otherwise)
        scan_path: Path to scan NIfTI file
        seg_path: Path to segmentation NIfTI file
        organ_name: Name of the organ
        model_name: Name of the model (to determine window size and extraction method)
    
    Returns:
        (features, positions, organ_crop, bbox_origin, patches)
        - features: (n_patches, feature_dim) array
        - positions: (n_patches, 3) or (n_patches,) array of positions
        - organ_crop: 3D numpy array of the organ crop
        - bbox_origin: (z, y, x) origin of the bounding box
        - patches: List of patch arrays
    """
    model_name_lower = model_name.lower()
    
    # Define window sizes for each model
    window_sizes = {
        "spectre": (128, 128, 64),
        "ctfm": (128, 128, 48),
        "tapct": (224, 224, 12),
        "curia": (256, 256),  # 2D
        "umedpt": (224, 224),  # 2D
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
        organ_crop = apply_spacing_to_crop(organ_crop, scan_path)
        features, positions, patches = extract_features_spectre(model, organ_crop, window_size)
    elif model_name_lower == "ctfm":
        features, positions, patches = extract_features_ctfm(model, organ_crop, window_size)
    elif model_name_lower == "tapct":
        features, positions, patches = extract_features_tapct(model, organ_crop, window_size)
    elif model_name_lower == "curia":
        organ_crop = reorient_crop_to_pl(organ_crop, scan_path)
        features, positions, patches = extract_features_curia(model, processor, organ_crop, window_size)
    elif model_name_lower == "umedpt":
        features, positions, patches = extract_features_umedpt(model, organ_crop, window_size)
    elif model_name_lower == "spectrevitg":
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
    
    return features, positions, organ_crop, bbox_origin, patches


def create_attention_volume(
    attention_weights: np.ndarray,
    positions: np.ndarray,
    organ_crop_shape: Tuple[int, int, int],
    window_size: Tuple[int, ...],
) -> np.ndarray:
    """
    Create a 3D attention volume by interpolating attention point values.
    
    For each patch, its attention weight is treated as a point value at the patch
    center. These point values are interpolated to obtain an attention value at
    every voxel. At the scan boundaries, the outermost point values are repeated
    (nearest extrapolation).
    
    Args:
        attention_weights: (n_patches,) array of attention weights
        positions: (n_patches, 3) or (n_patches,) array of positions
        organ_crop_shape: (z, y, x) shape of the organ crop
        window_size: (z, y, x) or (y, x) window size for patches
    
    Returns:
        3D numpy array with interpolated attention weights for each voxel.
    """
    Dz, Dy, Dx = organ_crop_shape
    attention_weights = np.asarray(attention_weights, dtype=np.float32)
    positions = np.asarray(positions)

    # 2D models (curia, umedpt): positions are slice indices along z
    if len(window_size) == 2 or positions.ndim == 1:
        # Ensure positions is 1D array of z-indices
        if positions.ndim > 1:
            positions_1d = positions.reshape(-1)
        else:
            positions_1d = positions

        z_points = positions_1d.astype(float)
        att_points = attention_weights.astype(float)

        # Sort by z position
        order = np.argsort(z_points)
        z_sorted = z_points[order]
        att_sorted = att_points[order]

        # Target z grid
        z_grid = np.arange(Dz, dtype=float)

        # 1D linear interpolation along z, with nearest extrapolation at boundaries
        att_z = np.interp(z_grid, z_sorted, att_sorted,
                          left=att_sorted[0], right=att_sorted[-1]).astype(np.float32)

        # Broadcast over y and x
        attention_volume = np.repeat(att_z[:, None, None], Dy, axis=1)
        attention_volume = np.repeat(attention_volume, Dx, axis=2)
        return attention_volume.astype(np.float32)

    # 3D models: positions are (z, y, x) patch origins
    # Compute patch centers in voxel coordinates
    if len(window_size) == 3:
        window_z, window_y, window_x = window_size
    else:
        # Fallback: derive a 3D window from 2D tuple
        window_z = max(window_size)
        window_y, window_x = window_size

    z_centers = positions[:, 0].astype(float) + window_z / 2.0
    y_centers = positions[:, 1].astype(float) + window_y / 2.0
    x_centers = positions[:, 2].astype(float) + window_x / 2.0

    # Get unique sorted center coordinates along each axis
    z_unique = np.unique(z_centers)
    y_unique = np.unique(y_centers)
    x_unique = np.unique(x_centers)

    Dzg, Hyg, Wxg = len(z_unique), len(y_unique), len(x_unique)

    # Map each center to its grid index
    def indices_from_centers(values, unique_vals):
        # Since centers form a regular grid from sliding_window, use searchsorted
        return np.searchsorted(unique_vals, values)

    zi = indices_from_centers(z_centers, z_unique)
    yi = indices_from_centers(y_centers, y_unique)
    xi = indices_from_centers(x_centers, x_unique)

    # Build coarse attention grid
    coarse_grid = np.zeros((Dzg, Hyg, Wxg), dtype=np.float32)
    count_grid = np.zeros((Dzg, Hyg, Wxg), dtype=np.int32)

    for w, z_idx, y_idx, x_idx in zip(attention_weights, zi, yi, xi):
        # If multiple patches map to same grid cell, take the max attention
        if count_grid[z_idx, y_idx, x_idx] == 0:
            coarse_grid[z_idx, y_idx, x_idx] = w
        else:
            coarse_grid[z_idx, y_idx, x_idx] = max(coarse_grid[z_idx, y_idx, x_idx], w)
        count_grid[z_idx, y_idx, x_idx] += 1

    # Pad coarse grid by repeating edge values so interpolation extrapolates
    coarse_padded = np.pad(coarse_grid, pad_width=1, mode="edge")  # (Dzg+2, Hyg+2, Wxg+2)

    # Upsample coarse grid to full crop resolution using trilinear interpolation
    coarse_torch = torch.from_numpy(coarse_padded)[None, None, ...]  # [1,1,D,H,W]
    coarse_torch = coarse_torch.float()

    attention_torch = F.interpolate(
        coarse_torch,
        size=(Dz, Dy, Dx),
        mode="trilinear",
        align_corners=False,
    )

    attention_volume = attention_torch[0, 0].cpu().numpy().astype(np.float32)
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
    features, positions, organ_crop, bbox_origin, patches = extract_features_for_organ_with_positions(
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
    
    # Load original scan to get spacing information
    scan_img = nib.load(scan_path)
    original_affine = scan_img.affine
    original_header = scan_img.header
    
    # Get spacing from header (pixdim)
    original_spacing = original_header.get_zooms()[:3]  # (x, y, z) spacing
    
    # Determine if crop was resampled (SPECTRE applies spacing transform)
    model_name_lower = args.model_name.lower()
    if model_name_lower == "spectre":
        # SPECTRE resamples to (0.5, 0.5, 1.0) spacing
        crop_spacing = (0.5, 0.5, 1.0)  # (x, y, z)
    else:
        # Other models use original spacing
        crop_spacing = original_spacing
    
    # Create affine for crop that preserves spacing correctly
    # Use a simple diagonal affine with proper spacing
    crop_affine = np.eye(4)
    crop_affine[0, 0] = crop_spacing[0]  # x spacing
    crop_affine[1, 1] = crop_spacing[1]  # y spacing
    crop_affine[2, 2] = crop_spacing[2]  # z spacing
    # Set origin to (0, 0, 0) - positioning doesn't matter for visualization
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Optionally save individual patches with attention weights
    if args.save_patches:
        patches_dir = os.path.join(output_dir, f"{scan_id_for_output}_{args.organ_name}_{args.model_name}_patches")
        os.makedirs(patches_dir, exist_ok=True)
        print(f"Saving individual patches to: {patches_dir}")
        
        for i, (patch, pos, attn_weight) in enumerate(zip(patches, positions, attention_weights)):
            # Format attention weight for filename (6 decimal places)
            attn_str = f"{attn_weight:.6f}".replace(".", "p")
            
            # Handle 2D vs 3D positions
            # For 2D models (curia, umedpt), pos is a scalar (slice index)
            # For 3D models (spectre, ctfm, tapct), pos is a tuple/array (z, y, x)
            if np.isscalar(pos) or (isinstance(pos, np.ndarray) and pos.ndim == 0):
                # 2D model (curia, umedpt) - pos is slice index (scalar)
                pos_val = int(pos) if np.isscalar(pos) else int(pos.item())
                pos_str = f"slice_{pos_val:04d}"
            else:
                # 3D model - pos is (z, y, x) tuple or array
                if isinstance(pos, (tuple, list)):
                    pos_z, pos_y, pos_x = int(pos[0]), int(pos[1]), int(pos[2])
                else:
                    # numpy array
                    pos_z, pos_y, pos_x = int(pos[0]), int(pos[1]), int(pos[2])
                pos_str = f"z{pos_z:04d}_y{pos_y:04d}_x{pos_x:04d}"
            
            patch_filename = f"patch_{i:04d}_{pos_str}_attn{attn_str}.nii.gz"
            patch_path = os.path.join(patches_dir, patch_filename)
            
            # Create patch affine with correct spacing
            # Use the same spacing as the crop
            patch_affine = np.eye(4)
            patch_affine[0, 0] = crop_spacing[0]
            patch_affine[1, 1] = crop_spacing[1]
            patch_affine[2, 2] = crop_spacing[2]
            # Origin at (0, 0, 0) - patches don't need to stack correctly
            
            # Save patch
            if len(patch.shape) == 2:
                # 2D patch - add singleton dimension for z
                patch_3d = patch[np.newaxis, :, :]
            else:
                patch_3d = patch
            
            # Create a simple header for patches
            patch_header = original_header.copy()
            patch_header.set_zooms((crop_spacing[0], crop_spacing[1], crop_spacing[2]) + original_header.get_zooms()[3:])
            
            patch_nifti = nib.Nifti1Image(patch_3d.astype(np.float32), patch_affine, patch_header)
            nib.save(patch_nifti, patch_path)
        
        print(f"Saved {len(patches)} patches")
    
    # Get window size for attention mapping
    window_sizes = {
        "spectre": (128, 128, 64),
        "ctfm": (128, 128, 48),
        "tapct": (224, 224, 12),
        "curia": (256, 256),
        "umedpt": (224, 224),
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
    
    # Save organ crop
    crop_output_path = os.path.join(output_dir, f"{scan_id_for_output}_{args.organ_name}_{args.model_name}_crop.nii.gz")
    print(f"Saving organ crop to: {crop_output_path}")
    
    # Create header with correct spacing for crop
    crop_header = original_header.copy()
    crop_header.set_zooms((crop_spacing[0], crop_spacing[1], crop_spacing[2]) + original_header.get_zooms()[3:])
    
    crop_nifti = nib.Nifti1Image(organ_crop.astype(np.float32), crop_affine, crop_header)
    nib.save(crop_nifti, crop_output_path)
    
    # Save attention map (use same affine and header as crop so they align perfectly)
    attention_output_path = os.path.join(output_dir, f"{scan_id_for_output}_{args.organ_name}_{args.model_name}_attention.nii.gz")
    print(f"Saving attention map to: {attention_output_path}")
    
    attention_nifti = nib.Nifti1Image(attention_volume.astype(np.float32), crop_affine, crop_header)
    nib.save(attention_nifti, attention_output_path)
    
    print("Done!")
    print(f"\nOutput files:")
    print(f"  - Organ crop: {crop_output_path}")
    print(f"  - Attention map: {attention_output_path}")
    if args.save_patches:
        print(f"  - Individual patches: {patches_dir} ({len(patches)} patches)")
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
    parser.add_argument(
        "--save-patches",
        action="store_true",
        help="If set, save individual patches as NIfTI files with attention weights in filenames",
    )
    
    args = parser.parse_args()
    sys.exit(main(args) or 0)
