# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from util.util import fix_random_seeds
from .evaluation_utils import (
    get_raw_feature_dir,
    get_attention_metrics_output_path,
    get_attention_checkpoint_output_dir,
    load_raw_features_and_labels,
    validate_evaluation_inputs,
    load_and_validate_annotations,
    load_subgroup_annotations,
    save_metrics,
    load_amos22_scan_ids,
    get_dataset_root_from_annotations_path,
    filter_patch_features_by_scan_ids,
    get_predictions_output_path,
    get_subgroup_info,
    save_predictions,
    load_raw_features_and_labels_all_organs,
    get_all_organs_attention_metrics_output_path,
    get_all_organs_attention_checkpoint_output_dir,
    filter_patch_features_all_organs_with_scan_ids,
    get_subgroup_info_all_organs,
)


class AttentionMIL(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128):
        super().__init__()
        
        self.attention_V = nn.Linear(embedding_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)
        
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, patches, mask=None, return_attention=False):
        # patches: [batch_size, n_patches, embedding_dim]
        # mask: [batch_size, n_patches] - True for real patches, False for padding
        batch_size, n_patches, embedding_dim = patches.shape
        
        # Reshape for attention computation
        patches_flat = patches.view(-1, embedding_dim)  # [batch_size * n_patches, embedding_dim]
        
        # Compute attention weights
        A = torch.tanh(self.attention_V(patches_flat))  # [batch_size * n_patches, hidden_dim]
        A = self.attention_w(A)  # [batch_size * n_patches, 1]
        A = A.view(batch_size, n_patches)  # [batch_size, n_patches]
        
        # Apply mask if provided (set padded positions to large negative value)
        if mask is not None:
            A = A.masked_fill(~mask, float('-inf'))
        
        A = torch.softmax(A, dim=1)  # [batch_size, n_patches]
        
        # Weighted sum of patches
        A_expanded = A.unsqueeze(-1)  # [batch_size, n_patches, 1]
        z = torch.sum(A_expanded * patches, dim=1)  # [batch_size, embedding_dim]
        
        # Classification
        out = self.classifier(z)  # [batch_size, 1]
        
        if return_attention:
            return out, A
        return out


class PatchDataset(torch.utils.data.Dataset):
    """Dataset for variable-length patch features."""
    def __init__(self, patch_features_list, labels):
        self.patch_features_list = patch_features_list
        self.labels = labels
    
    def __len__(self):
        return len(self.patch_features_list)
    
    def __getitem__(self, idx):
        return torch.tensor(self.patch_features_list[idx], dtype=torch.float32), self.labels[idx]


def collate_fn(batch):
    """Custom collate function to pad variable-length patch sequences."""
    patch_features, labels = zip(*batch)
    
    # Find max number of patches in this batch
    max_patches = max(p.shape[0] for p in patch_features)
    embedding_dim = patch_features[0].shape[1]
    
    # Pad all sequences to max_patches and create mask
    padded_patches = []
    masks = []
    for patches in patch_features:
        n_patches = patches.shape[0]
        if n_patches < max_patches:
            # Pad with zeros
            padding = torch.zeros(max_patches - n_patches, embedding_dim, dtype=patches.dtype)
            padded = torch.cat([patches, padding], dim=0)
            # Create mask: True for real patches, False for padding
            mask = torch.cat([
                torch.ones(n_patches, dtype=torch.bool),
                torch.zeros(max_patches - n_patches, dtype=torch.bool)
            ])
        else:
            padded = patches
            mask = torch.ones(n_patches, dtype=torch.bool)
        padded_patches.append(padded)
        masks.append(mask)
    
    # Stack into batch tensor
    batch_patches = torch.stack(padded_patches)  # [batch_size, max_patches, embedding_dim]
    batch_masks = torch.stack(masks)  # [batch_size, max_patches]
    batch_labels = torch.tensor(labels, dtype=torch.long)
    
    return batch_patches, batch_masks, batch_labels


def make_data_loaders(
    patch_features_train: list,
    y_train: np.ndarray,
    patch_features_val: list,
    y_val: np.ndarray,
    patch_features_test: list,
    y_test: np.ndarray,
    batch_size: int = 32,
):
    train_dataset = PatchDataset(patch_features_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    val_loader = None
    if len(patch_features_val) > 0:
        val_dataset = PatchDataset(patch_features_val, y_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
    
    test_loader = None
    if len(patch_features_test) > 0:
        test_dataset = PatchDataset(patch_features_test, y_test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    return train_loader, val_loader, test_loader


def evaluate(model, data_loader, device, return_attention=False, return_predictions=False):
    """Evaluate model on a data loader."""
    if data_loader is None:
        if return_predictions:
            return None, None, None, None, None, None
        return None, None, None, None
    
    model.eval()
    all_logits = []
    all_true_labels = []
    all_attention_weights = []
    all_masks = []

    with torch.no_grad():
        for patches_batch, masks_batch, y_batch in data_loader:
            patches_batch = patches_batch.to(device)
            masks_batch = masks_batch.to(device)
            if return_attention:
                logits, attention_weights = model(patches_batch, mask=masks_batch, return_attention=True)
                all_attention_weights.append(attention_weights.cpu())
                all_masks.append(masks_batch.cpu())
            else:
                logits = model(patches_batch, mask=masks_batch)
            all_logits.append(logits.cpu())
            all_true_labels.append(y_batch)

    all_logits = torch.cat(all_logits, dim=0)
    ground_truth_labels = torch.cat(all_true_labels, dim=0).numpy()
    probabilities = torch.sigmoid(all_logits).numpy().flatten()
    predicted_labels = (probabilities > 0.5).astype(int)

    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    num_classes = len(np.unique(ground_truth_labels))
    if num_classes == 2:
        auc_value = roc_auc_score(ground_truth_labels, probabilities)
    else:
        auc_value = roc_auc_score(
            ground_truth_labels,
            probabilities,
            multi_class="ovr",
            average="macro",
        )
    
    attention_weights_list = None
    masks_list = None
    if return_attention and len(all_attention_weights) > 0:
        attention_weights_list = all_attention_weights
        masks_list = all_masks
    
    if return_predictions:
        return accuracy, auc_value, attention_weights_list, masks_list, ground_truth_labels, probabilities
    return accuracy, auc_value, attention_weights_list, masks_list


def compute_attention_statistics(attention_weights_list, masks_list):
    """
    Compute statistics of attention weights.
    
    Args:
        attention_weights_list: List of tensors, each [batch_size, n_patches]
        masks_list: List of tensors, each [batch_size, n_patches] (True for real patches)
    
    Returns:
        Dictionary with statistics: std_dev, entropy, avg_n_patches, avg_mean_weight, avg_max_weight
    """
    if attention_weights_list is None or len(attention_weights_list) == 0:
        return {
            "std_dev": None,
            "entropy": None,
            "avg_n_patches": None,
            "avg_mean_weight": None,
            "avg_max_weight": None,
        }
    
    std_devs = []
    entropies = []
    n_patches_list = []
    mean_weights_list = []
    max_weights_list = []
    
    # Process each batch
    for attention_weights_batch, masks_batch in zip(attention_weights_list, masks_list):
        attention_weights_batch = attention_weights_batch.numpy()
        masks_batch = masks_batch.numpy()
        
        # Process each sample in the batch
        for i in range(attention_weights_batch.shape[0]):
            # Get attention weights for real patches only
            sample_weights = attention_weights_batch[i][masks_batch[i]]
            
            if len(sample_weights) == 0:
                continue
            
            # Number of patches (bag size)
            n_patches_list.append(len(sample_weights))
            
            # Mean attention weight
            mean_weight = float(np.mean(sample_weights))
            mean_weights_list.append(mean_weight)
            
            # Max attention weight
            max_weight = float(np.max(sample_weights))
            max_weights_list.append(max_weight)
            
            # Standard deviation
            std_dev = float(np.std(sample_weights))
            std_devs.append(std_dev)
            
            # Entropy: -sum(p * log(p))
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            sample_weights_safe = sample_weights + epsilon
            entropy = float(-np.sum(sample_weights_safe * np.log(sample_weights_safe)))
            entropies.append(entropy)
    
    if len(std_devs) == 0:
        return {
            "std_dev": None,
            "entropy": None,
            "avg_n_patches": None,
            "avg_mean_weight": None,
            "avg_max_weight": None,
        }
    
    return {
        "std_dev": float(np.mean(std_devs)),
        "entropy": float(np.mean(entropies)),
        "avg_n_patches": float(np.mean(n_patches_list)),
        "avg_mean_weight": float(np.mean(mean_weights_list)),
        "avg_max_weight": float(np.mean(max_weights_list)),
    }


def filter_patch_features_by_subgroup(
    patch_features_list: list,
    y: np.ndarray,
    scan_ids: list,
    subgroup_annotations: dict,
    organ_name: str,
    subgroup_name: str,
) -> tuple:
    """Filter patch features and labels using the same logic as filter_normal_and_subgroup_abnormal."""
    if len(patch_features_list) == 0:
        return [], np.array([])
    
    if len(scan_ids) != len(patch_features_list):
        raise ValueError(f"Mismatch: {len(scan_ids)} scan_ids but {len(patch_features_list)} samples")
    
    filtered_features = []
    filtered_labels = []
    
    for idx, scan_id in enumerate(scan_ids):
        # Include all normal samples (label=0)
        if y[idx] == 0:
            filtered_features.append(patch_features_list[idx])
            filtered_labels.append(y[idx])
        # Include abnormal samples (label=1) with the specified subgroup
        elif y[idx] == 1:
            if scan_id in subgroup_annotations:
                organ_subgroups = subgroup_annotations[scan_id].get(organ_name, {})
                # Check if this abnormal sample has the specified subgroup value=1
                if subgroup_name in organ_subgroups and organ_subgroups[subgroup_name] == 1:
                    filtered_features.append(patch_features_list[idx])
                    filtered_labels.append(y[idx])
    
    return filtered_features, np.array(filtered_labels)


def run_attention_evaluation(
    patch_features_train, y_train, train_scan_ids,
    patch_features_val, y_val, val_scan_ids,
    patch_features_test, y_test, test_scan_ids,
    train_subgroups, val_subgroups, test_subgroups,
    organ_name,
    device,
    checkpoint_dir=None,
    is_all_organs_mode=False,
):
    """
    Run attention MIL evaluation on the provided data.
    
    Args:
        checkpoint_dir: Directory to save checkpoint. If None, checkpoint is not saved.
    
    Returns:
        Dictionary with evaluation_groups and best_model info
    """
    # Validate that we have features
    if len(patch_features_train) == 0:
        raise ValueError(f"No training samples found for organ {organ_name}. Cannot perform evaluation.")
    
    # Get flattened dimension from first sample (each patch is flattened to 1D)
    embedding_dim = patch_features_train[0].shape[1]
    
    # Validate flattened dimension consistency across all samples
    for patches in patch_features_train + patch_features_val + patch_features_test:
        if patches.shape[1] != embedding_dim:
            raise ValueError(f"Flattened dimension mismatch: expected {embedding_dim}, got {patches.shape[1]}")
    
    # Validate labels
    unique_labels = set(y_train)
    if len(unique_labels) < 2:
        raise ValueError(f"Need at least 2 classes for evaluation, found {len(unique_labels)}: {unique_labels}")
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Labels must be 0 or 1, found: {unique_labels}")

    model = AttentionMIL(embedding_dim=embedding_dim, hidden_dim=128)
    model.to(device)

    train_loader, val_loader, test_loader = make_data_loaders(
        patch_features_train, y_train, patch_features_val, y_val, 
        patch_features_test, y_test, batch_size=64
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_checkpoint_path = None
    best_epoch = None
    epochs_without_improvement = 0
    
    for epoch in range(1, 1001):
        model.train()
        epoch_loader = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
        for patches_batch, masks_batch, y_batch in epoch_loader:
            patches_batch = patches_batch.to(device)
            masks_batch = masks_batch.to(device)
            y_batch = y_batch.to(device).float()

            optimizer.zero_grad()
            logits = model(patches_batch, mask=masks_batch).squeeze(-1)  # [batch_size]
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loader.set_postfix(loss=loss.item())
        
        # Evaluate on validation set every epoch and save best model
        if val_loader is not None:
            val_acc, val_auc, _, _ = evaluate(model, val_loader, device, return_attention=False, return_predictions=False)
            if val_auc is not None and val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                epochs_without_improvement = 0
                if checkpoint_dir is not None:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model_attention.pth")
                    torch.save(model.state_dict(), best_checkpoint_path)
            else:
                epochs_without_improvement += 1
            
            # Early stopping: stop if no improvement in last 50 epochs
            if epochs_without_improvement >= 50:
                print(f"Early stopping at epoch {epoch}: no improvement in validation AUC for 50 epochs")
                break

    # Load best model checkpoint if available, otherwise use final model
    if best_checkpoint_path is not None and os.path.exists(best_checkpoint_path):
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        print(f"Loaded best model from {best_checkpoint_path}")
    else:
        print("No best checkpoint found, using final model state")

    metrics = {
        "evaluation_groups": {},
        "best_model": {
            "checkpoint_path": best_checkpoint_path,
            "validation_auc": float(best_val_auc) if best_epoch is not None else None,
            "epoch": best_epoch,
        },
    }
    predictions_dict = {"evaluation_groups": {}}
    
    # Define evaluation groups: all, normal+diffuse, normal+focal
    evaluation_groups = [
        ("all", None),  # All samples
        ("normal_and_diffuse", "diffuse"),  # Normal + diffuse abnormal
        ("normal_and_focal", "focal"),  # Normal + focal abnormal
    ]
    
    for group_name, subgroup_name in evaluation_groups:
        group_metrics = {}
        group_predictions = {}
        
        # Filter data for this group
        if subgroup_name is None:
            # All samples - no filtering
            patch_features_train_group = patch_features_train
            y_train_group = y_train
            train_scan_ids_group = train_scan_ids
            patch_features_val_group = patch_features_val
            y_val_group = y_val
            val_scan_ids_group = val_scan_ids
            patch_features_test_group = patch_features_test
            y_test_group = y_test
            test_scan_ids_group = test_scan_ids
        else:
            # Normal + specific subgroup abnormal
            if is_all_organs_mode:
                # For 'all' mode, scan_ids are actually (scan_id, organ_name) tuples
                patch_features_train_group, y_train_group, train_scan_ids_group = filter_patch_features_all_organs_with_scan_ids(
                    patch_features_train, y_train, train_scan_ids, train_subgroups, subgroup_name
                )
                patch_features_val_group, y_val_group, val_scan_ids_group = filter_patch_features_all_organs_with_scan_ids(
                    patch_features_val, y_val, val_scan_ids, val_subgroups, subgroup_name
                )
                patch_features_test_group, y_test_group, test_scan_ids_group = filter_patch_features_all_organs_with_scan_ids(
                    patch_features_test, y_test, test_scan_ids, test_subgroups, subgroup_name
                )
            else:
                patch_features_train_group, y_train_group, train_scan_ids_group = _filter_patch_features_with_scan_ids(
                    patch_features_train, y_train, train_scan_ids, train_subgroups,
                    organ_name, subgroup_name
                )
                patch_features_val_group, y_val_group, val_scan_ids_group = _filter_patch_features_with_scan_ids(
                    patch_features_val, y_val, val_scan_ids, val_subgroups,
                    organ_name, subgroup_name
                )
                patch_features_test_group, y_test_group, test_scan_ids_group = _filter_patch_features_with_scan_ids(
                    patch_features_test, y_test, test_scan_ids, test_subgroups,
                    organ_name, subgroup_name
                )
        
        # Create data loaders for this group
        train_loader_group, val_loader_group, test_loader_group = make_data_loaders(
            patch_features_train_group, y_train_group, 
            patch_features_val_group, y_val_group, 
            patch_features_test_group, y_test_group, 
            batch_size=64
        )
        
        # Evaluate on each split
        splits = [
            ("train", train_loader_group, y_train_group, train_scan_ids_group, train_subgroups),
            ("validation", val_loader_group, y_val_group, val_scan_ids_group, val_subgroups),
            ("test", test_loader_group, y_test_group, test_scan_ids_group, test_subgroups),
        ]
        
        for split_name, loader, y_split, scan_ids_split, subgroups_split in splits:
            if loader is None or len(y_split) == 0:
                group_metrics[split_name] = {
                    "accuracy": None,
                    "auc": None,
                    "n_normal": 0,
                    "n_abnormal": 0,
                    "attention_weights": {
                        "std_dev": None,
                        "entropy": None,
                        "avg_n_patches": None,
                        "avg_mean_weight": None,
                        "avg_max_weight": None,
                    },
                }
                group_predictions[split_name] = []
                continue
            
            # Evaluate with attention weights and predictions
            acc, auc, attention_weights_list, masks_list, y_true, prob_scores = evaluate(
                model, loader, device, return_attention=True, return_predictions=True
            )
            
            # Compute attention statistics
            attention_stats = compute_attention_statistics(attention_weights_list, masks_list)
            
            # Count normal and abnormal samples
            n_normal = int(np.sum(y_split == 0))
            n_abnormal = int(np.sum(y_split == 1))
            
            group_metrics[split_name] = {
                "accuracy": float(acc) if acc is not None else None,
                "auc": float(auc) if auc is not None else None,
                "n_normal": n_normal,
                "n_abnormal": n_abnormal,
                "attention_weights": attention_stats,
            }
            
            # Collect predictions
            split_predictions = []
            for idx, scan_id_or_tuple in enumerate(scan_ids_split):
                if is_all_organs_mode:
                    scan_id, organ_name_for_pred = scan_id_or_tuple
                    is_focal, is_diffuse = get_subgroup_info_all_organs(
                        scan_id, organ_name_for_pred, subgroups_split
                    )
                    split_predictions.append({
                        "scan_id": scan_id,
                        "organ_name": organ_name_for_pred,
                        "ground_truth": int(y_true[idx]),
                        "is_focal": is_focal,
                        "is_diffuse": is_diffuse,
                        "probability": float(prob_scores[idx]),
                    })
                else:
                    is_focal, is_diffuse = get_subgroup_info(scan_id_or_tuple, subgroups_split, organ_name)
                    split_predictions.append({
                        "scan_id": scan_id_or_tuple,
                        "ground_truth": int(y_true[idx]),
                        "is_focal": is_focal,
                        "is_diffuse": is_diffuse,
                        "probability": float(prob_scores[idx]),
                    })
            group_predictions[split_name] = split_predictions
        
        metrics["evaluation_groups"][group_name] = group_metrics
        predictions_dict["evaluation_groups"][group_name] = group_predictions

    return metrics, predictions_dict


def _filter_patch_features_with_scan_ids(
    patch_features_list: list,
    y: np.ndarray,
    scan_ids: list,
    subgroup_annotations: dict,
    organ_name: str,
    subgroup_name: str,
) -> tuple:
    """
    Filter patch features, labels, and scan IDs using the same logic as filter_patch_features_by_subgroup.
    Returns (patch_features_filtered, y_filtered, scan_ids_filtered).
    """
    if len(patch_features_list) == 0:
        return patch_features_list, y, scan_ids
    
    if len(scan_ids) != len(patch_features_list):
        raise ValueError(f"Mismatch: {len(scan_ids)} scan_ids but {len(patch_features_list)} samples")
    
    filtered_features = []
    filtered_labels = []
    filtered_scan_ids = []
    
    for idx, scan_id in enumerate(scan_ids):
        # Include all normal samples (label=0)
        if y[idx] == 0:
            filtered_features.append(patch_features_list[idx])
            filtered_labels.append(y[idx])
            filtered_scan_ids.append(scan_id)
        # Include abnormal samples (label=1) with the specified subgroup
        elif y[idx] == 1:
            if scan_id in subgroup_annotations:
                organ_subgroups = subgroup_annotations[scan_id].get(organ_name, {})
                # Check if this abnormal sample has the specified subgroup value=1
                if subgroup_name in organ_subgroups and organ_subgroups[subgroup_name] == 1:
                    filtered_features.append(patch_features_list[idx])
                    filtered_labels.append(y[idx])
                    filtered_scan_ids.append(scan_id)
    
    return filtered_features, np.array(filtered_labels), filtered_scan_ids


def main(args):
    fix_random_seeds(args.seed)

    is_all_organs_mode = (args.organ_name == "all")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if is_all_organs_mode:
        output_metrics = get_all_organs_attention_metrics_output_path(
            args.output_root, args.model_name
        )
        output_checkpoint = get_all_organs_attention_checkpoint_output_dir(
            args.output_root, args.model_name
        )
        
        # Load annotations
        train_annotations, test_annotations = load_and_validate_annotations(
            args.annotations_train_csv,
            args.annotations_test_csv,
        )
        
        val_annotations = train_annotations
        
        # Load subgroup annotations
        train_subgroups, test_subgroups = load_subgroup_annotations(
            args.annotations_train_csv,
            args.annotations_test_csv,
        )
        val_subgroups = train_subgroups
        
        # Load raw patch features and labels from all organs
        try:
            patch_features_train, y_train, train_scan_organ_ids = load_raw_features_and_labels_all_organs(
                args.output_root, args.model_name, "training", train_annotations, return_scan_ids=True
            )
            patch_features_val, y_val, val_scan_organ_ids = load_raw_features_and_labels_all_organs(
                args.output_root, args.model_name, "validation", val_annotations, return_scan_ids=True
            )
            patch_features_test, y_test, test_scan_organ_ids = load_raw_features_and_labels_all_organs(
                args.output_root, args.model_name, "test", test_annotations, return_scan_ids=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load features: {e}") from e
        
        # Get dataset root and load AMOS22 scan IDs
        dataset_root = get_dataset_root_from_annotations_path(args.annotations_train_csv)
        amos22_scan_ids = load_amos22_scan_ids(dataset_root)
        
        # Filter out AMOS22 scans
        def filter_all_organs_patch_features_by_scan_ids(patch_features_list, y, scan_organ_ids, exclude_scan_ids):
            """Filter patch features by excluding scan IDs."""
            filtered_features = []
            filtered_labels = []
            filtered_scan_organ_ids = []
            for idx, (scan_id, organ_name) in enumerate(scan_organ_ids):
                if scan_id not in exclude_scan_ids:
                    filtered_features.append(patch_features_list[idx])
                    filtered_labels.append(y[idx])
                    filtered_scan_organ_ids.append((scan_id, organ_name))
            return filtered_features, np.array(filtered_labels), filtered_scan_organ_ids
        
        # Run evaluation on all data
        all_metrics, all_predictions = run_attention_evaluation(
            patch_features_train, y_train, train_scan_organ_ids,
            patch_features_val, y_val, val_scan_organ_ids,
            patch_features_test, y_test, test_scan_organ_ids,
            train_subgroups, val_subgroups, test_subgroups,
            "all",
            device,
            checkpoint_dir=output_checkpoint,
            is_all_organs_mode=True,
        )
        
        # Filter out AMOS22 scans and run evaluation again
        patch_features_train_filtered, y_train_filtered, train_scan_organ_ids_filtered = filter_all_organs_patch_features_by_scan_ids(
            patch_features_train, y_train, train_scan_organ_ids, amos22_scan_ids
        )
        patch_features_val_filtered, y_val_filtered, val_scan_organ_ids_filtered = filter_all_organs_patch_features_by_scan_ids(
            patch_features_val, y_val, val_scan_organ_ids, amos22_scan_ids
        )
        patch_features_test_filtered, y_test_filtered, test_scan_organ_ids_filtered = filter_all_organs_patch_features_by_scan_ids(
            patch_features_test, y_test, test_scan_organ_ids, amos22_scan_ids
        )
        
        # Use a separate checkpoint directory for exclude_amos22
        exclude_checkpoint_dir = output_checkpoint + "_exclude_amos22"
        exclude_amos22_metrics, exclude_amos22_predictions = run_attention_evaluation(
            patch_features_train_filtered, y_train_filtered, train_scan_organ_ids_filtered,
            patch_features_val_filtered, y_val_filtered, val_scan_organ_ids_filtered,
            patch_features_test_filtered, y_test_filtered, test_scan_organ_ids_filtered,
            train_subgroups, val_subgroups, test_subgroups,
            "all",
            device,
            checkpoint_dir=exclude_checkpoint_dir,
            is_all_organs_mode=True,
        )
    else:
        feature_dir_training = get_raw_feature_dir(
            args.output_root, args.model_name, args.organ_name, "training"
        )
        feature_dir_validation = get_raw_feature_dir(
            args.output_root, args.model_name, args.organ_name, "validation"
        )
        feature_dir_test = get_raw_feature_dir(
            args.output_root, args.model_name, args.organ_name, "test"
        )
        output_metrics = get_attention_metrics_output_path(
            args.output_root, args.model_name, args.organ_name
        )
        output_checkpoint = get_attention_checkpoint_output_dir(
            args.output_root, args.model_name, args.organ_name
        )
        
        # Validate inputs early
        validate_evaluation_inputs(
            feature_dir_training,
            feature_dir_validation,
            feature_dir_test,
            args.annotations_train_csv,
            args.annotations_test_csv,
            args.organ_name,
            output_metrics,
            output_checkpoint,
        )
        
        # Load annotations
        train_annotations, test_annotations = load_and_validate_annotations(
            args.annotations_train_csv,
            args.annotations_test_csv,
        )
        
        val_annotations = train_annotations
        
        # Load subgroup annotations
        train_subgroups, test_subgroups = load_subgroup_annotations(
            args.annotations_train_csv,
            args.annotations_test_csv,
        )
        val_subgroups = train_subgroups
        
        # Load raw patch features and labels with scan IDs
        try:
            patch_features_train, y_train, train_scan_ids = load_raw_features_and_labels(
                feature_dir_training, train_annotations, args.organ_name, return_scan_ids=True
            )
            patch_features_val, y_val, val_scan_ids = load_raw_features_and_labels(
                feature_dir_validation, val_annotations, args.organ_name, return_scan_ids=True
            )
            patch_features_test, y_test, test_scan_ids = load_raw_features_and_labels(
                feature_dir_test, test_annotations, args.organ_name, return_scan_ids=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load features: {e}") from e
        
        # Get dataset root and load AMOS22 scan IDs
        dataset_root = get_dataset_root_from_annotations_path(args.annotations_train_csv)
        amos22_scan_ids = load_amos22_scan_ids(dataset_root)
        
        # Run evaluation on all data
        all_metrics, all_predictions = run_attention_evaluation(
            patch_features_train, y_train, train_scan_ids,
            patch_features_val, y_val, val_scan_ids,
            patch_features_test, y_test, test_scan_ids,
            train_subgroups, val_subgroups, test_subgroups,
            args.organ_name,
            device,
            checkpoint_dir=output_checkpoint,
            is_all_organs_mode=False,
        )
        
        # Filter out AMOS22 scans and run evaluation again
        patch_features_train_filtered, y_train_filtered, train_scan_ids_filtered = filter_patch_features_by_scan_ids(
            patch_features_train, y_train, train_scan_ids, amos22_scan_ids
        )
        patch_features_val_filtered, y_val_filtered, val_scan_ids_filtered = filter_patch_features_by_scan_ids(
            patch_features_val, y_val, val_scan_ids, amos22_scan_ids
        )
        patch_features_test_filtered, y_test_filtered, test_scan_ids_filtered = filter_patch_features_by_scan_ids(
            patch_features_test, y_test, test_scan_ids, amos22_scan_ids
        )
        
        # Use a separate checkpoint directory for exclude_amos22
        exclude_checkpoint_dir = output_checkpoint + "_exclude_amos22"
        exclude_amos22_metrics, exclude_amos22_predictions = run_attention_evaluation(
            patch_features_train_filtered, y_train_filtered, train_scan_ids_filtered,
            patch_features_val_filtered, y_val_filtered, val_scan_ids_filtered,
            patch_features_test_filtered, y_test_filtered, test_scan_ids_filtered,
            train_subgroups, val_subgroups, test_subgroups,
            args.organ_name,
            device,
            checkpoint_dir=exclude_checkpoint_dir,
            is_all_organs_mode=False,
        )
    
    # Structure output with two top-level objects
    metrics = {
        "all_data": all_metrics,
        "exclude_amos22": exclude_amos22_metrics,
    }
    
    # Save metrics
    save_metrics(output_metrics, metrics)
    
    # Save predictions
    predictions = {
        "all_data": all_predictions,
        "exclude_amos22": exclude_amos22_predictions,
    }
    output_predictions = get_predictions_output_path(output_metrics)
    save_predictions(output_predictions, predictions)

    return 0


if __name__ == "__main__":
    # Custom parser for attention (doesn't require aggregation_method)
    import argparse
    parser = argparse.ArgumentParser(description="Attention MIL evaluation", add_help=True)
    parser.add_argument("--output-root", type=str, required=True, help="Workflow output root directory")
    parser.add_argument("--model-name", type=str, required=True, help="Feature model name")
    parser.add_argument(
        "--annotations-train-csv",
        type=str,
        required=True,
        help="Path to training annotations CSV",
    )
    parser.add_argument(
        "--annotations-test-csv",
        type=str,
        required=True,
        help="Path to test annotations CSV",
    )
    parser.add_argument(
        "--organ-name",
        type=str,
        required=True,
        help="Organ name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    sys.exit(main(args))
