# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import os
import sys

import numpy as np
import torch
import nibabel as nib
from merlin import Merlin
from monai.transforms import (
    Compose,
    EnsureTyped,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.data import MetaTensor

from util.util import fix_random_seeds
from util.leavs_utils import ORGAN_NAME_TO_LABEL
from util.snakemake_helpers import VALID_ORGANS
import argparse

MERLIN_PATCH_SIZE = (160, 224, 224)
MERLIN_TARGET_SPACING = (1.5, 1.5, 3)
PAD_VALUE = -1024.0


def _debug_save_nifti(volume: np.ndarray, path: str) -> None:
    nib.save(nib.Nifti1Image(np.ascontiguousarray(volume, dtype=np.float32), np.eye(4)), path)


def load_model():
    model = Merlin(ImageEmbedding=True)
    model.cuda().eval()
    return model


def resample_scan_and_seg(scan_path: str, seg_path: str) -> tuple:
    """Orient to RAS and resample scan/seg to Merlin target spacing."""
    scan_img = nib.load(scan_path)
    seg_img = nib.load(seg_path)

    scan_tensor = torch.from_numpy(scan_img.get_fdata()).unsqueeze(0).float()
    scan_meta = MetaTensor(scan_tensor, affine=scan_img.affine)

    seg_tensor = torch.from_numpy(seg_img.get_fdata()).unsqueeze(0).float()
    seg_meta = MetaTensor(seg_tensor, affine=seg_img.affine)

    scan_data = Compose([
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=MERLIN_TARGET_SPACING, mode="bilinear"),
    ])({"image": scan_meta})["image"].squeeze(0).numpy()

    seg_data = Compose([
        Orientationd(keys=["seg"], axcodes="RAS"),
        Spacingd(keys=["seg"], pixdim=MERLIN_TARGET_SPACING, mode="nearest"),
    ])({"seg": seg_meta})["seg"].squeeze(0).numpy().astype(int)

    return scan_data, seg_data


def get_organ_bbox_origin(seg_path: str, organ_name: str):
    """Mask bounding-box origin in original segmentation voxel space."""
    organ_labels = ORGAN_NAME_TO_LABEL.get(organ_name)
    if organ_labels is None:
        return None
    if not isinstance(organ_labels, list):
        organ_labels = [organ_labels]

    seg_data = nib.load(seg_path).get_fdata().astype(int)
    organ_mask = np.isin(seg_data, organ_labels)
    if not np.any(organ_mask):
        return None

    coords = np.where(organ_mask)
    return (int(coords[0].min()), int(coords[1].min()), int(coords[2].min()))


def get_organ_center(seg_volume: np.ndarray, organ_name: str):
    """Organ center in resampled segmentation voxel space."""
    organ_labels = ORGAN_NAME_TO_LABEL.get(organ_name)
    if organ_labels is None:
        return None
    if not isinstance(organ_labels, list):
        organ_labels = [organ_labels]

    organ_mask = np.isin(seg_volume, organ_labels)
    if not np.any(organ_mask):
        return None

    coords = np.where(organ_mask)
    return (
        (int(coords[0].min()) + int(coords[0].max())) // 2,
        (int(coords[1].min()) + int(coords[1].max())) // 2,
        (int(coords[2].min()) + int(coords[2].max())) // 2,
    )


def extract_centered_patch(
    volume: np.ndarray,
    center: tuple,
    patch_size: tuple = MERLIN_PATCH_SIZE,
    fill_value: float = PAD_VALUE,
) -> tuple:
    """Extract a fixed-size patch centered on center, padding with fill_value at edges."""
    win_z, win_y, win_x = patch_size
    cz, cy, cx = center
    z0, y0, x0 = cz - win_z // 2, cy - win_y // 2, cx - win_x // 2
    z1, y1, x1 = z0 + win_z, y0 + win_y, x0 + win_x

    patch = np.full(patch_size, fill_value, dtype=np.float32)
    sz0, sy0, sx0 = max(0, z0), max(0, y0), max(0, x0)
    sz1 = min(volume.shape[0], z1)
    sy1 = min(volume.shape[1], y1)
    sx1 = min(volume.shape[2], x1)

    if sz1 > sz0 and sy1 > sy0 and sx1 > sx0:
        oz0, oy0, ox0 = sz0 - z0, sy0 - y0, sx0 - x0
        oz1, oy1, ox1 = oz0 + (sz1 - sz0), oy0 + (sy1 - sy0), ox0 + (sx1 - sx0)
        patch[oz0:oz1, oy0:oy1, ox0:ox1] = volume[sz0:sz1, sy0:sy1, sx0:sx1]

    return patch, (z0, y0, x0)


def preprocess_patch(patch: np.ndarray) -> torch.Tensor:
    transform = Compose([
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
    patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()
    transformed = transform({"image": patch_tensor})
    return transformed["image"].unsqueeze(0)  # (1, C, D, H, W)


def extract_feature(model, patch: np.ndarray) -> np.ndarray:
    batch_tensor = preprocess_patch(patch).cuda()
    with torch.no_grad():
        outputs = model(batch_tensor)
    return outputs[0].detach().cpu().numpy()


def is_valid_output_file(output_path: str) -> bool:
    if not os.path.exists(output_path):
        return False
    if not os.path.isfile(output_path):
        return False
    if not os.access(output_path, os.R_OK):
        return False
    try:
        with np.load(output_path, allow_pickle=True) as data:
            required_keys = {"features", "positions", "bbox_origin", "organ_name", "is_placeholder"}
            if not required_keys.issubset(set(data.files)):
                return False
            _ = data["features"]
            _ = data["positions"]
            _ = data["is_placeholder"]
    except Exception:
        return False
    return True


def _save_placeholder(output_path: str, organ_name: str, bbox_origin):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        features=np.array([]),
        positions=np.array([]),
        bbox_origin=bbox_origin,
        organ_name=organ_name,
        is_placeholder=True,
    )


def process_scan_for_organ(
    model,
    scan_volume: np.ndarray,
    seg_volume: np.ndarray,
    seg_path: str,
    organ_name: str,
    output_path: str,
    scan_id: str,
):
    """
    Process a single scan for a specific organ.
    Returns True if features were extracted, False if placeholder was saved.
    """
    bbox_origin = get_organ_bbox_origin(seg_path, organ_name)
    if bbox_origin is None:
        print(f"Warning: Organ {organ_name} not found in segmentation {seg_path}. Saving placeholder file.")
        _save_placeholder(output_path, organ_name, None)
        return False

    center = get_organ_center(seg_volume, organ_name)
    if center is None:
        print(f"Warning: Organ {organ_name} not found in resampled segmentation for {seg_path}. Saving placeholder file.")
        _save_placeholder(output_path, organ_name, bbox_origin)
        return False

    patch, position = extract_centered_patch(scan_volume, center)
    _debug_save_nifti(patch, f"/tmp/merlin_{scan_id}_{organ_name}_patch.nii.gz")

    feature = extract_feature(model, patch)
    features = np.array([np.expand_dims(feature[0], axis=0)])
    positions = np.array([position])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        features=features,
        positions=positions,
        bbox_origin=bbox_origin,
        organ_name=organ_name,
        is_placeholder=False,
    )
    return True


def process_scan_for_all_organs(
    model,
    scan_path: str,
    seg_path: str,
    organ_names: list,
    output_root: str,
    model_name: str,
    split: str,
    scan_id: str,
):
    """Process a scan for all specified organs."""
    scan_volume, seg_volume = resample_scan_and_seg(scan_path, seg_path)

    processed_count = 0
    for organ_name in organ_names:
        output_path = os.path.join(
            output_root,
            model_name,
            organ_name,
            split,
            "features",
            "raw",
            f"{scan_id}.npz",
        )
        if is_valid_output_file(output_path):
            print(f"Skipping organ {organ_name}: valid output already exists at {output_path}")
            continue
        if os.path.exists(output_path):
            print(f"Recomputing organ {organ_name}: existing output is invalid or unreadable at {output_path}")
        else:
            print(f"Extracting features for organ: {organ_name}")
        if process_scan_for_organ(
            model, scan_volume, seg_volume, seg_path, organ_name, output_path, scan_id
        ):
            processed_count += 1

    print(f"Successfully processed {processed_count}/{len(organ_names)} organs for scan")


def _read_paths_file(paths_file: str) -> list:
    with open(paths_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main(args):
    fix_random_seeds(getattr(args, "seed", 0))

    scan_paths = _read_paths_file(args.scan_paths_file)
    seg_paths = _read_paths_file(args.seg_paths_file)

    if len(scan_paths) != len(seg_paths):
        raise ValueError(f"Number of scan paths ({len(scan_paths)}) must match number of seg paths ({len(seg_paths)})")

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

    print("Loading model...")
    try:
        model = load_model()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    for scan_idx, (scan_path, seg_path) in enumerate(zip(scan_paths, seg_paths)):
        seg_basename = os.path.basename(seg_path)
        if seg_basename.endswith('_segmentation.nii.gz'):
            scan_id = seg_basename[:-20]
        else:
            scan_basename = os.path.basename(scan_path)
            scan_id = scan_basename.replace('.nii.gz', '')

        print(f"Processing scan {scan_idx + 1}/{len(scan_paths)}: {scan_id}")

        try:
            process_scan_for_all_organs(
                model,
                scan_path,
                seg_path,
                organ_names,
                args.output_root,
                args.model_name,
                args.split,
                scan_id,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to process scan {scan_path}: {e}") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merlin Feature Extraction for LEAVS")
    parser.add_argument("--scan-paths-file", type=str, required=True, help="File containing scan file paths (.nii.gz), one per line")
    parser.add_argument("--seg-paths-file", type=str, required=True, help="File containing segmentation file paths (.nii.gz), one per line")
    parser.add_argument("--output-root", type=str, required=True, help="Root output directory following workflow conventions")
    parser.add_argument("--model-name", type=str, required=True, help="Feature model name")
    parser.add_argument("--split", type=str, required=True, choices=["training", "validation", "test"], help="Dataset split")
    args = parser.parse_args()

    sys.exit(main(args))
