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
    "pancreas": [10],
    "small_bowel": [55, 56],  # small bowel and duodenum
    "large_bowel": [57],
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


def get_organ_crop(scan_path: str, seg_path: str, organ_name: str, padding: int = 10) -> Optional[Tuple[np.ndarray, Tuple[int, int, int]]]:
    """
    Extract organ crop from scan using segmentation mask.
    Returns: (organ_crop, bbox_origin) or None if organ not found
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
    
    # Get bounding box
    coords = np.where(organ_mask)
    if len(coords[0]) == 0:
        return None
    
    z_min, z_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    x_min, x_max = coords[2].min(), coords[2].max()
    
    # Add padding
    z_min = max(0, z_min - padding)
    z_max = min(scan_data.shape[0], z_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(scan_data.shape[1], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(scan_data.shape[2], x_max + padding)
    
    # Crop
    organ_crop = scan_data[z_min:z_max, y_min:y_max, x_min:x_max]
    bbox_origin = (z_min, y_min, x_min)
    
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
