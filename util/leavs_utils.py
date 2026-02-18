# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import csv
import os
import json
import numpy as np
import nibabel as nib
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split

# TotalSegmentator label mapping
TOTALSEGMENTATOR_LABELS = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "aorta",
    8: "inferior_vena_cava",
    9: "portal_vein_and_splenic_vein",
    10: "pancreas",
    11: "adrenal_gland_right",
    12: "adrenal_gland_left",
    13: "lung_upper_lobe_left",
    14: "lung_lower_lobe_left",
    15: "lung_upper_lobe_right",
    16: "lung_middle_lobe_right",
    17: "lung_lower_lobe_right",
    18: "vertebrae_L5",
    19: "vertebrae_L4",
    20: "vertebrae_L3",
    21: "vertebrae_L2",
    22: "vertebrae_L1",
    23: "vertebrae_T12",
    24: "vertebrae_T11",
    25: "vertebrae_T10",
    26: "vertebrae_T9",
    27: "vertebrae_T8",
    28: "vertebrae_T7",
    29: "vertebrae_T6",
    30: "vertebrae_T5",
    31: "vertebrae_T4",
    32: "vertebrae_T3",
    33: "vertebrae_T2",
    34: "vertebrae_T1",
    35: "vertebrae_C7",
    36: "vertebrae_C6",
    37: "vertebrae_C5",
    38: "vertebrae_C4",
    39: "vertebrae_C3",
    40: "vertebrae_C2",
    41: "vertebrae_C1",
    42: "esophagus",
    43: "trachea",
    44: "heart_myocardium",
    45: "heart_atrium_left",
    46: "heart_ventricle_left",
    47: "heart_atrium_right",
    48: "heart_ventricle_right",
    49: "pulmonary_artery",
    50: "brain",
    51: "iliac_artery_left",
    52: "iliac_artery_right",
    53: "iliac_vena_left",
    54: "iliac_vena_right",
    55: "small_bowel",
    56: "duodenum",
    57: "colon",
    58: "rib_left_1",
    59: "rib_left_2",
    60: "rib_left_3",
    61: "rib_left_4",
    62: "rib_left_5",
    63: "rib_left_6",
    64: "rib_left_7",
    65: "rib_left_8",
    66: "rib_left_9",
    67: "rib_left_10",
    68: "rib_left_11",
    69: "rib_left_12",
    70: "rib_right_1",
    71: "rib_right_2",
    72: "rib_right_3",
    73: "rib_right_4",
    74: "rib_right_5",
    75: "rib_right_6",
    76: "rib_right_7",
    77: "rib_right_8",
    78: "rib_right_9",
    79: "rib_right_10",
    80: "rib_right_11",
    81: "rib_right_12",
    82: "humerus_left",
    83: "humerus_right",
    84: "scapula_left",
    85: "scapula_right",
    86: "clavicula_left",
    87: "clavicula_right",
    88: "femur_left",
    89: "femur_right",
    90: "hip_left",
    91: "hip_right",
    92: "sacrum",
    93: "face",
    94: "gluteus_maximus_left",
    95: "gluteus_maximus_right",
    96: "gluteus_medius_left",
    97: "gluteus_medius_right",
    98: "gluteus_minimus_left",
    99: "gluteus_minimus_right",
    100: "autochthon_left",
    101: "autochthon_right",
    102: "iliopsoas_left",
    103: "iliopsoas_right",
    104: "urinary_bladder"
}

# Mapping from CSV organ names to TotalSegmentator labels
ORGAN_NAME_TO_LABEL = {
    "spleen": 1,
    "right kidney": 2,
    "left kidney": 3,
    "gallbladder": 4,
    "liver": 5,
    "stomach": 6,
    "pancreas": 10,
    "small bowel": 55,
    "large bowel": 57,
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
    
    # Get organ label
    organ_label = ORGAN_NAME_TO_LABEL.get(organ_name)
    if organ_label is None:
        for k, v in ORGAN_NAME_TO_LABEL.items():
            if k.replace('_', ' ') == organ_name.replace('_', ' '):
                organ_label = v
                break
    
    if organ_label is None:
        return None
    
    # Create binary mask for this organ
    organ_mask = (seg_data == organ_label)
    
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
