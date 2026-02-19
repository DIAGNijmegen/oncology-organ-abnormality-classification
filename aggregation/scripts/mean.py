# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import argparse
import os
import sys
from typing import Optional

import numpy as np


def get_args_parser(description: Optional[str] = None, add_help: bool = True):
    parser = argparse.ArgumentParser(description=description, add_help=add_help)
    parser.add_argument("--input-features", required=True, help="Comma-separated list of input feature file paths (.npz)")
    parser.add_argument("--output-features", required=True, help="Comma-separated list of output file paths for aggregated features")
    return parser


def _create_output_directory(output_file_path):
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def process_single_file(input_path, output_path):
    """Process a single input file and save aggregated features."""
    _create_output_directory(output_path)
    
    # Load patch features
    data = np.load(input_path)
    
    # Check if this is a placeholder file
    is_placeholder = data.get("is_placeholder", False)
    if is_placeholder:
        print(f"Skipping placeholder file: {input_path}")
        # Save an empty placeholder file for aggregated features as well
        save_dict = {
            "features": np.array([]),
            "is_placeholder": True,
        }
        if "organ_name" in data:
            save_dict["organ_name"] = data["organ_name"]
        np.savez(output_path, **save_dict)
        print(f"Placeholder aggregated features saved to {output_path}")
        return
    
    patch_features = data["features"]  # Shape: (n_patches, ...) - can be any shape
    
    if len(patch_features) == 0:
        raise ValueError(f"No features found in {input_path}")
    
    positions = data.get("positions", None)
    bbox_origin = data.get("bbox_origin", None)
    organ_name = data.get("organ_name", None)
    
    # Aggregate using element-wise mean along first axis (patches)
    # This preserves all other dimensions
    aggregated_features = np.mean(patch_features, axis=0)
    
    # Save aggregated features
    save_dict = {
        "features": aggregated_features,
        "is_placeholder": False,
    }
    if positions is not None:
        save_dict["positions"] = positions
    if bbox_origin is not None:
        save_dict["bbox_origin"] = bbox_origin
    if organ_name is not None:
        save_dict["organ_name"] = organ_name
    
    np.savez(output_path, **save_dict)
    
    print(f"Mean aggregated features saved to {output_path}")
    print(f"Aggregated {len(patch_features)} patches. Input shape: {patch_features.shape}, Output shape: {aggregated_features.shape}")


def main(args):
    # Parse input and output paths (comma-separated)
    if hasattr(args, 'input_features') and args.input_features:
        input_paths = [p.strip() for p in args.input_features.split(',')]
    else:
        raise ValueError("--input-features is required")
    
    if hasattr(args, 'output_features') and args.output_features:
        output_paths = [p.strip() for p in args.output_features.split(',')]
    else:
        raise ValueError("--output-features is required")
    
    if len(input_paths) != len(output_paths):
        raise ValueError(f"Number of input paths ({len(input_paths)}) must match number of output paths ({len(output_paths)})")
    
    # Process each file sequentially
    for idx, (input_path, output_path) in enumerate(zip(input_paths, output_paths)):
        print(f"Processing file {idx + 1}/{len(input_paths)}: {input_path}")
        try:
            process_single_file(input_path, output_path)
        except Exception as e:
            raise RuntimeError(f"Failed to process {input_path}: {e}") from e


if __name__ == "__main__":
    description = "Mean Aggregation for Organ Features"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()

    sys.exit(main(args))
