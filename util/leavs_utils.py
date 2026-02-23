# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import csv
import os
import json
import re
import numpy as np
import nibabel as nib
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split

# Mapping from CSV organ names to TotalSegmentator labels
ORGAN_NAME_TO_LABEL = {
    "spleen": [1],
    "kidney_right": [2],
    "kidney_left": [3],
    "gallbladder": [4],
    "liver": [5],
    "stomach": [6],
    "pancreas": [7],
    "small_bowel": [18, 19],  # small bowel and duodenum
    "large_bowel": [20],
}

# Reverse mapping for CSV column names
CSV_ORGAN_TO_STANDARD = {
    "spleen": "spleen",
    "right kidney": "kidney_right",
    "left kidney": "kidney_left",
    "gallbladder": "gallbladder",
    "liver": "liver",
    "stomach": "stomach",
    "pancreas": "pancreas",
    "small bowel": "small_bowel",
    "large bowel": "large_bowel",
}

# Organ names in CSV files
ORGAN_NAMES = [
    'spleen',
    'liver',
    'right kidney',
    'left kidney',
    'stomach',
    'pancreas',
    'gallbladder',
    'small bowel',
    'large bowel',
]


def _extract_scan_id_from_train_subjectid(subjectid: str) -> str:
    """
    Extract scan ID from training CSV subjectid_studyid field.
    Format: ./imagesTr/amos_5478.nii.gz_./imagesTr/amos_5478.nii.gz
    Returns: amos_5478
    """
    # Use regex to extract amos_XXXX pattern (where XXXX is digits)
    match = re.search(r'amos_\d+', subjectid)
    if match:
        return match.group(0)
    
    raise ValueError(f"Could not extract scan ID from subjectid: {subjectid}")


def _extract_scan_id_from_test_image1(image1: str) -> str:
    """
    Extract scan ID from test CSV image1 field.
    Format: amos_0029.nii.gz.txt or amos_0029.nii.gz
    """
    return image1.replace('.nii.gz.txt', '').replace('.txt', '').replace('.nii.gz', '')


def infer_labels_from_subgroups(
    subgroup_annotations: Dict[str, Dict[str, Dict[str, int]]]
) -> Dict[str, Dict[str, int]]:
    """
    Infer normality labels from subgroup annotations.
    
    Logic: For each organ, if ANY subgroup has value 1, then label = 1 (abnormal).
    Otherwise, label = 0 (normal).
    
    Args:
        subgroup_annotations: {scan_id: {organ: {subgroup_name: value}}}
    
    Returns:
        {scan_id: {organ: normality_label}} where 0 = normal, 1 = abnormal
    """
    annotations = {}
    
    for scan_id, organs in subgroup_annotations.items():
        if scan_id not in annotations:
            annotations[scan_id] = {}
        
        for organ_name, subgroups in organs.items():
            # Check if any subgroup has value 1 (abnormality present)
            has_abnormality = any(
                value == 1 for value in subgroups.values()
            )
            
            # Infer label: 1 = abnormal, 0 = normal
            label = 1 if has_abnormality else 0
            annotations[scan_id][organ_name] = label
    
    return annotations


def parse_train_subgroup_annotations(csv_path: str) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Parse training annotations CSV to extract subgroup information.
    Returns: {scan_id: {organ: {subgroup_name: value}}}
    
    Subgroups: postsurgical, enlarged, atrophy, diffuse, focal
    Values: 1 if present, 0 if absent, -2 if not applicable/unknown
    """
    subgroup_annotations = {}
    
    # Subgroup columns to extract
    subgroup_columns = ['postsurgical', 'enlarged', 'atrophy', 'diffuse', 'focal']
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            type_annotation = row.get('type_annotation', '')
            # Only process 'labels' rows, skip 'urgency'
            if type_annotation != 'labels':
                continue
            scan_id = _extract_scan_id_from_train_subjectid(row['subjectid_studyid'])
            organ = row['organ']
            normal = row.get('normal', '').strip()
            
            # Filter based on "normal" column:
            # - If neither 0 nor 1, skip this organ/scan (do not use for training or evaluation)
            if normal not in ['0', '1']:
                continue
            
            # Map organ name to standard name
            standard_organ = CSV_ORGAN_TO_STANDARD.get(organ.lower(), organ.lower().replace(' ', '_'))
            
            if scan_id not in subgroup_annotations:
                subgroup_annotations[scan_id] = {}
            if standard_organ not in subgroup_annotations[scan_id]:
                subgroup_annotations[scan_id][standard_organ] = {}
            
            # Extract subgroup values
            for subgroup in subgroup_columns:
                val = row.get(subgroup, '').strip()
                if val and val != '':
                    try:
                        val_int = int(float(val))
                        # Only store if value is 0 or 1 (skip -2, -3, etc.)
                        if val_int in [0, 1]:
                            subgroup_annotations[scan_id][standard_organ][subgroup] = val_int
                    except (ValueError, KeyError):
                        pass
    
    return subgroup_annotations


def parse_test_subgroup_annotations(csv_path: str) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Parse test annotations CSV to extract subgroup information.
    Returns: {scan_id: {organ: {subgroup_name: value}}}
    
    Subgroups: postsurgical_absent, enlarged_atrophy, diffuse, focal
    Note: postsurgical_absent and enlarged_atrophy are combined columns in test CSV
    
    Uses majority vote across all labelers for each organ and subgroup.
    If there's a tie, defaults to 0 (normal/absent).
    """
    # First pass: collect all labeler votes for each scan, organ, and subgroup
    # Structure: {scan_id: {organ: {subgroup: [list of values from all labelers]}}}
    labeler_votes = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            type_annotation = row.get('type_annotation', '')
            # Only process 'labels' rows, skip 'urgency'
            if type_annotation != 'labels':
                continue

            scan_id = _extract_scan_id_from_test_image1(row['image1'])
            
            if scan_id not in labeler_votes:
                labeler_votes[scan_id] = {}
            
            for organ_name in ORGAN_NAMES:
                # Map organ name to standard name
                standard_organ = CSV_ORGAN_TO_STANDARD.get(organ_name.lower(), organ_name.lower().replace(' ', '_'))
                
                if standard_organ not in labeler_votes[scan_id]:
                    labeler_votes[scan_id][standard_organ] = {}
                
                # Extract subgroup columns for this organ
                # Test CSV has: {organ}_postsurgical_absent, {organ}_enlarged_atrophy, {organ}_diffuse, {organ}_focal
                subgroup_mappings = {
                    'postsurgical_absent': 'postsurgical',
                    'enlarged_atrophy': 'enlarged_atrophy',  # Keep as combined for now
                    'diffuse': 'diffuse',
                    'focal': 'focal',
                }
                
                for csv_suffix, subgroup_key in subgroup_mappings.items():
                    col_name = f"{organ_name}_{csv_suffix}"
                    val = row.get(col_name, '').strip()
                    if val and val != '':
                        try:
                            val_float = float(val)
                            if val_float in [0.0, 1.0]:
                                if subgroup_key not in labeler_votes[scan_id][standard_organ]:
                                    labeler_votes[scan_id][standard_organ][subgroup_key] = []
                                labeler_votes[scan_id][standard_organ][subgroup_key].append(int(val_float))
                        except (ValueError, KeyError):
                            pass
    
    # Second pass: apply majority vote to get final values
    subgroup_annotations = {}
    for scan_id, organs in labeler_votes.items():
        if scan_id not in subgroup_annotations:
            subgroup_annotations[scan_id] = {}
        
        for organ_name, subgroups in organs.items():
            if organ_name not in subgroup_annotations[scan_id]:
                subgroup_annotations[scan_id][organ_name] = {}
            
            for subgroup_key, votes in subgroups.items():
                if len(votes) == 0:
                    continue
                
                # Majority vote: if more 1s than 0s, then 1, else 0
                # In case of tie, default to 0
                count_ones = sum(votes)
                count_zeros = len(votes) - count_ones
                final_value = 1 if count_ones > count_zeros else 0
                subgroup_annotations[scan_id][organ_name][subgroup_key] = final_value
    
    return subgroup_annotations


def get_organ_crop(scan_path: str, seg_path: str, organ_name: str, window_size: Tuple[int, ...]) -> Optional[Tuple[np.ndarray, Tuple[int, int, int]]]:
    """
    Extract organ crop from scan using segmentation mask.
    
    If the organ bounding box is smaller than window_size, returns a crop of exactly
    window_size centered on the organ. The scan is padded with -1024 before cropping
    to handle cases where the crop extends beyond scan boundaries.
    
    Args:
        scan_path: Path to scan NIfTI file
        seg_path: Path to segmentation NIfTI file
        organ_name: Name of the organ to extract
        window_size: Minimum crop size as (z, y, x) tuple for 3D or (y, x) tuple for 2D
    
    Returns:
        (organ_crop, bbox_origin) or None if organ not found
        bbox_origin is the original mask bounding box origin (before padding/centering)
    """
    # Load scan and segmentation
    scan_img = nib.load(scan_path)
    seg_img = nib.load(seg_path)
    
    scan_data = scan_img.get_fdata()
    seg_data = seg_img.get_fdata().astype(int)
    
    # Get organ label(s) - can be a single label or a list of labels
    organ_labels = ORGAN_NAME_TO_LABEL.get(organ_name)
    
    if organ_labels is None:
        return None
    
    # Ensure organ_labels is a list
    if not isinstance(organ_labels, list):
        organ_labels = [organ_labels]
    
    # Create binary mask for this organ (matches any of the labels)
    organ_mask = np.isin(seg_data, organ_labels)
    
    if not np.any(organ_mask):
        return None
    
    # Get bounding box from mask
    coords = np.where(organ_mask)
    if len(coords[0]) == 0:
        return None
    
    z_min_mask, z_max_mask = coords[0].min(), coords[0].max()
    y_min_mask, y_max_mask = coords[1].min(), coords[1].max()
    x_min_mask, x_max_mask = coords[2].min(), coords[2].max()
    
    # Convert window_size to 3D: handle both 2D (y, x) and 3D (z, y, x) inputs
    if len(window_size) == 2:
        # 2D window_size: use max dimension for z
        window_z = max(window_size)
        window_y, window_x = window_size
    else:
        window_z, window_y, window_x = window_size
    
    # Calculate organ center
    z_center = (z_min_mask + z_max_mask) // 2
    y_center = (y_min_mask + y_max_mask) // 2
    x_center = (x_min_mask + x_max_mask) // 2
    
    # Determine crop size: use window_size if mask bbox is smaller, otherwise use mask bbox
    mask_z_size = z_max_mask - z_min_mask + 1
    mask_y_size = y_max_mask - y_min_mask + 1
    mask_x_size = x_max_mask - x_min_mask + 1
    
    crop_z_size = max(mask_z_size, window_z)
    crop_y_size = max(mask_y_size, window_y)
    crop_x_size = max(mask_x_size, window_x)
    
    # Calculate crop bounds centered on organ
    z_min_crop = z_center - crop_z_size // 2
    z_max_crop = z_min_crop + crop_z_size - 1
    y_min_crop = y_center - crop_y_size // 2
    y_max_crop = y_min_crop + crop_y_size - 1
    x_min_crop = x_center - crop_x_size // 2
    x_max_crop = x_min_crop + crop_x_size - 1
    
    # Calculate padding: use max window dimension for all sides
    max_padding = max(window_z, window_y, window_x)
    
    # Pad scan with -1024
    scan_data = np.pad(
        scan_data,
        ((max_padding, max_padding), (max_padding, max_padding), (max_padding, max_padding)),
        mode='constant',
        constant_values=-1024
    )
    
    # Adjust crop coordinates for padding
    z_min_crop += max_padding
    z_max_crop += max_padding
    y_min_crop += max_padding
    y_max_crop += max_padding
    x_min_crop += max_padding
    x_max_crop += max_padding
    
    # Crop
    organ_crop = scan_data[z_min_crop:z_max_crop+1, y_min_crop:y_max_crop+1, x_min_crop:x_max_crop+1]
    bbox_origin = (z_min_mask, y_min_mask, x_min_mask)
    
    return organ_crop, bbox_origin


def create_train_val_split(scan_ids: List[str], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Create train/validation split from scan IDs.
    """
    train_ids, val_ids = train_test_split(scan_ids, test_size=val_ratio, random_state=seed)
    return sorted(train_ids), sorted(val_ids)


def get_leavs_scans(dataset_root: str) -> Tuple[List[str], List[str]]:
    """
    Get list of training and test scan IDs from LEAVS dataset.
    Returns: (train_scan_ids, test_scan_ids)
    """
    train_dir = os.path.join(dataset_root, "LEAVS", "imagesTr")
    test_dir = os.path.join(dataset_root, "LEAVS", "imagesTs")
    
    train_scans = []
    if os.path.exists(train_dir):
        for f in os.listdir(train_dir):
            if f.endswith('.nii.gz') and f.startswith('amos_'):
                scan_id = f.replace('.nii.gz', '')
                train_scans.append(scan_id)
    
    test_scans = []
    if os.path.exists(test_dir):
        for f in os.listdir(test_dir):
            if f.endswith('.nii.gz') and f.startswith('amos_'):
                scan_id = f.replace('.nii.gz', '')
                test_scans.append(scan_id)
    
    return sorted(train_scans), sorted(test_scans)
