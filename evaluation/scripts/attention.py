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
)


class AttentionMIL(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128):
        super().__init__()
        
        self.attention_V = nn.Linear(embedding_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)
        
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, patches, mask=None):
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


def evaluate(model, data_loader, device):
    """Evaluate model on a data loader."""
    if data_loader is None:
        return None, None
    
    model.eval()
    all_logits = []
    all_true_labels = []

    with torch.no_grad():
        for patches_batch, masks_batch, y_batch in data_loader:
            patches_batch = patches_batch.to(device)
            masks_batch = masks_batch.to(device)
            logits = model(patches_batch, mask=masks_batch).cpu()
            all_logits.append(logits)
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
    
    return accuracy, auc_value


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


def main(args):
    fix_random_seeds(args.seed)

    feature_dir_training = get_raw_feature_dir(
        args.output_root, args.model_name, args.organ_name, "training"
    )
    feature_dir_validation = get_raw_feature_dir(
        args.output_root, args.model_name, args.organ_name, "validation"
    )
    feature_dir_test = get_raw_feature_dir(
        args.output_root, args.model_name, args.organ_name, "test"
    )
    # For attention, we don't use aggregation_method, but we need to provide a dummy value for path compatibility
    # The actual features come from raw feature directory
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load annotations
    train_annotations, test_annotations = load_and_validate_annotations(
        args.annotations_train_csv,
        args.annotations_test_csv,
    )
    
    val_annotations = train_annotations  # Validation uses train annotations
    
    # Load subgroup annotations
    train_subgroups, test_subgroups = load_subgroup_annotations(
        args.annotations_train_csv,
        args.annotations_test_csv,
    )
    val_subgroups = train_subgroups  # Validation uses train subgroups
    
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
    
    # Validate that we have features
    if len(patch_features_train) == 0:
        raise ValueError(f"No training samples found for organ {args.organ_name}. Cannot perform evaluation.")
    
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
        patch_features_test, y_test, batch_size=128
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
            val_acc, val_auc = evaluate(model, val_loader, device)
            if val_auc is not None and val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                epochs_without_improvement = 0
                best_checkpoint_path = os.path.join(output_checkpoint, "best_model_attention.pth")
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
    
    # Define evaluation groups: all, normal+diffuse, normal+focal
    evaluation_groups = [
        ("all", None),  # All samples
        ("normal_and_diffuse", "diffuse"),  # Normal + diffuse abnormal
        ("normal_and_focal", "focal"),  # Normal + focal abnormal
    ]
    
    for group_name, subgroup_name in evaluation_groups:
        group_metrics = {}
        
        # Filter data for this group
        if subgroup_name is None:
            # All samples - no filtering
            patch_features_train_group = patch_features_train
            y_train_group = y_train
            patch_features_val_group = patch_features_val
            y_val_group = y_val
            patch_features_test_group = patch_features_test
            y_test_group = y_test
        else:
            # Normal + specific subgroup abnormal
            patch_features_train_group, y_train_group = filter_patch_features_by_subgroup(
                patch_features_train, y_train, train_scan_ids, train_subgroups,
                args.organ_name, subgroup_name
            )
            patch_features_val_group, y_val_group = filter_patch_features_by_subgroup(
                patch_features_val, y_val, val_scan_ids, val_subgroups,
                args.organ_name, subgroup_name
            )
            patch_features_test_group, y_test_group = filter_patch_features_by_subgroup(
                patch_features_test, y_test, test_scan_ids, test_subgroups,
                args.organ_name, subgroup_name
            )
        
        # Create data loaders for this group
        train_loader_group, val_loader_group, test_loader_group = make_data_loaders(
            patch_features_train_group, y_train_group, 
            patch_features_val_group, y_val_group, 
            patch_features_test_group, y_test_group, 
            batch_size=32
        )
        
        # Evaluate on each split
        splits = [
            ("train", train_loader_group, y_train_group),
            ("validation", val_loader_group, y_val_group),
            ("test", test_loader_group, y_test_group),
        ]
        
        for split_name, loader, y_split in splits:
            if loader is None or len(y_split) == 0:
                group_metrics[split_name] = {
                    "accuracy": None,
                    "auc": None,
                    "n_normal": 0,
                    "n_abnormal": 0,
                }
                continue
            
            # Evaluate
            acc, auc = evaluate(model, loader, device)
            
            # Count normal and abnormal samples
            n_normal = int(np.sum(y_split == 0))
            n_abnormal = int(np.sum(y_split == 1))
            
            group_metrics[split_name] = {
                "accuracy": float(acc) if acc is not None else None,
                "auc": float(auc) if auc is not None else None,
                "n_normal": n_normal,
                "n_abnormal": n_abnormal,
            }
        
        metrics["evaluation_groups"][group_name] = group_metrics

    # Save metrics
    save_metrics(output_metrics, metrics)

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
