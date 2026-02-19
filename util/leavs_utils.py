# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import csv
import os
import json
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


def parse_train_annotations(csv_path: str) -> Dict[str, Dict[str, int]]:
    """
    Parse training annotations CSV.
    Returns: {scan_id: {organ: normality_label}}
    """
    annotations = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract scan ID from subjectid_studyid
            subjectid = row['subjectid_studyid']
            # Format: ./imagesTr/amos_5478.nii.gz_./imagesTr/amos_5478.nii.gz
            # Extract the filename part before the underscore
            if '_' in subjectid:
                # Split by underscore and take the first part
                first_part = subjectid.split('_')[0]
                # Extract filename from path
                scan_id = os.path.basename(first_part).replace('.nii.gz', '')
            else:
                # Fallback: try to extract from path
                scan_id = os.path.basename(subjectid).replace('.nii.gz', '')
            
            organ = row['organ']
            normal = row['normal']
            
            # Filter for valid normality labels (0 or 1)
            if normal in ['0', '1']:
                if scan_id not in annotations:
                    annotations[scan_id] = {}
                
                # Map organ name to standard name
                standard_organ = CSV_ORGAN_TO_STANDARD.get(organ.lower(), organ.lower().replace(' ', '_'))
                annotations[scan_id][standard_organ] = int(normal)
    
    return annotations


def parse_test_annotations(csv_path: str) -> Dict[str, Dict[str, int]]:
    """
    Parse test annotations CSV.
    Returns: {scan_id: {organ: normality_label}}
    
    Logic: For each organ, look at all organ-specific columns (excluding quality).
    - If ANY column has value "1" → abnormal → label = 0
    - If ALL columns have value "0" → normal → label = 1
    """
    annotations = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            scan_id_raw = row['image1']
            # Format: amos_0029.nii.gz.txt or amos_0029.nii.gz
            scan_id = scan_id_raw.replace('.nii.gz.txt', '').replace('.txt', '').replace('.nii.gz', '')
            
            type_annotation = row['type_annotation']
            
            # Only process 'labels' rows, skip 'urgency'
            if type_annotation != 'labels':
                continue
            
            if scan_id not in annotations:
                annotations[scan_id] = {}
            
            # Organ names in the CSV
            organ_names = [
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
            
            # Get all column names
            all_columns = row.keys()
            
            for organ_name in organ_names:
                # Find all columns for this organ (excluding quality)
                organ_columns = [
                    col for col in all_columns 
                    if col.startswith(f"{organ_name}_") and not col.endswith("_quality")
                ]
                
                if len(organ_columns) == 0:
                    continue
                
                # Check values in all organ-specific columns
                has_abnormality = False
                all_zero = True
                
                for col in organ_columns:
                    val = row[col].strip()
                    if val and val != '':
                        try:
                            val_float = float(val)
                            if val_float == 1.0:
                                has_abnormality = True
                                all_zero = False
                                break  # Found abnormality, no need to check further
                            elif val_float != 0.0:
                                all_zero = False
                        except (ValueError, KeyError):
                            pass
                
                # Determine label: 1 = normal, 0 = abnormal
                if has_abnormality:
                    # Any column has 1 → abnormal
                    label = 0
                elif all_zero:
                    # All columns are 0 → normal
                    label = 1
                else:
                    # Mixed or unclear → skip this organ for this scan
                    continue
                
                # Map organ name to standard name
                standard_organ = CSV_ORGAN_TO_STANDARD.get(organ_name.lower(), organ_name.lower().replace(' ', '_'))
                annotations[scan_id][standard_organ] = label
    
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
            # Extract scan ID from subjectid_studyid
            subjectid = row['subjectid_studyid']
            if '_' in subjectid:
                first_part = subjectid.split('_')[0]
                scan_id = os.path.basename(first_part).replace('.nii.gz', '')
            else:
                scan_id = os.path.basename(subjectid).replace('.nii.gz', '')
            
            organ = row['organ']
            normal = row['normal']
            
            # Only process rows with valid normality labels
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
    """
    subgroup_annotations = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            scan_id_raw = row['image1']
            scan_id = scan_id_raw.replace('.nii.gz.txt', '').replace('.txt', '').replace('.nii.gz', '')
            
            type_annotation = row['type_annotation']
            
            # Only process 'labels' rows, skip 'urgency'
            if type_annotation != 'labels':
                continue
            
            if scan_id not in subgroup_annotations:
                subgroup_annotations[scan_id] = {}
            
            # Organ names in the CSV
            organ_names = [
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
            
            for organ_name in organ_names:
                # Map organ name to standard name
                standard_organ = CSV_ORGAN_TO_STANDARD.get(organ_name.lower(), organ_name.lower().replace(' ', '_'))
                
                if standard_organ not in subgroup_annotations[scan_id]:
                    subgroup_annotations[scan_id][standard_organ] = {}
                
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
                            if val_float == 1.0:
                                subgroup_annotations[scan_id][standard_organ][subgroup_key] = 1
                            elif val_float == 0.0:
                                subgroup_annotations[scan_id][standard_organ][subgroup_key] = 0
                        except (ValueError, KeyError):
                            pass
    
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
