# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import json
import os
import sys

import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureTyped,
    Orientationd,
    ScaleIntensityRanged,
    ResizeWithPadOrCropd,
    NormalizeIntensityd,
)
from tqdm import tqdm

from transformers import AutoConfig
from transformers.models.auto.auto_factory import get_class_from_dynamic_module
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

from util.util import fix_random_seeds
from util.leavs_utils import get_organ_crop, ORGAN_NAME_TO_LABEL
from util.sliding_window import sliding_window_3d
import argparse


def load_model():
    repo_id = "fomofo/tap-ct-b-3d"

    config = AutoConfig.from_pretrained(
        repo_id,
        trust_remote_code=True,
    )

    model_cls = get_class_from_dynamic_module(
        class_reference="modeling_tapct.TAPCTModel",
        pretrained_model_name_or_path=repo_id,
    )
    model = model_cls(config)

    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
    )

    state_dict = load_safetensors(ckpt_path)
    model.load_state_dict(state_dict, strict=True)

    model.cuda().eval()
    return model


def preprocess_patch(patch: np.ndarray) -> torch.Tensor:
    """
    Preprocess a 3D patch for TAP-CT model.
    Expected input: 224x224xZ where Z is divisible by 4.
    """
    transform = Compose([
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1008,
            a_max=822,
            b_min=-1008,
            b_max=822,
            clip=True,
        ),
        NormalizeIntensityd(
            keys=["image"],
            subtrahend=-86.8086,
            divisor=322.6347,
        ),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=(224, 224, -1)),
        DivisiblePadd(keys=["image"], k=(1, 1, 4)),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])
    
    # Convert to MONAI format: (C, H, W, D)
    patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()
    data_dict = {"image": patch_tensor}
    transformed = transform(data_dict)
    
    # TAP-CT expects (B, D, H, W) format
    img = transformed["image"].squeeze(0)  # (H, W, D)
    img = img.permute(2, 0, 1).unsqueeze(0)  # (1, D, H, W)
    
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
            feature = output["pooler_output"].detach().cpu().numpy()
        
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
    
    # Window size for TAP-CT: 224x224x64
    window_size = (224, 224, 64)
    
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
    parser = argparse.ArgumentParser(description="TAP-CT Feature Extraction for LEAVS")
    parser.add_argument("--scan-path", type=str, required=True, help="Path to scan file (.nii.gz)")
    parser.add_argument("--seg-path", type=str, required=True, help="Path to segmentation file (.nii.gz)")
    parser.add_argument("--organ-name", type=str, required=True, help="Organ name (e.g., spleen)")
    parser.add_argument("--output-path", type=str, required=True, help="Output path for features")
    args = parser.parse_args()

    sys.exit(main(args))
