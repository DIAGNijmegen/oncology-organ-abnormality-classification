# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from util.util import fix_random_seeds
from .evaluation_utils import (
    get_base_args_parser,
    get_feature_dir,
    get_metrics_output_path,
    get_checkpoint_output_dir,
    load_features_and_labels,
    validate_evaluation_inputs,
    load_and_validate_annotations,
    load_subgroup_annotations,
    validate_features_and_labels,
    save_metrics,
    filter_normal_and_subgroup_abnormal,
    load_amos22_scan_ids,
    get_dataset_root_from_annotations_path,
    filter_by_scan_ids,
    get_predictions_output_path,
    get_subgroup_info,
    save_predictions,
)


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def make_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
):
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    val_loader = None
    if len(X_val) > 0:
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    
    test_loader = None
    if len(X_test) > 0:
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    return train_loader, val_loader, test_loader


def evaluate(model, data_loader, device, return_predictions=False):
    """Evaluate model on a data loader."""
    if data_loader is None:
        return (None, None) if not return_predictions else (None, None, None, None)
    
    model.eval()
    all_logits = []
    all_true_labels = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch).cpu()
            all_logits.append(logits)
            all_true_labels.append(y_batch)

    all_logits = torch.cat(all_logits, dim=0)
    ground_truth_labels = torch.cat(all_true_labels, dim=0).numpy()
    probabilities = torch.softmax(all_logits, dim=1).numpy()
    predicted_labels = probabilities.argmax(axis=1)

    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    num_classes = len(np.unique(ground_truth_labels))
    if num_classes == 2:
        auc_value = roc_auc_score(ground_truth_labels, probabilities[:, 1])
        prob_scores = probabilities[:, 1]  # Probability of class 1 (abnormal)
    else:
        auc_value = roc_auc_score(
            ground_truth_labels,
            probabilities,
            multi_class="ovr",
            average="macro",
        )
        prob_scores = probabilities.max(axis=1)  # Max probability across classes
    
    if return_predictions:
        return accuracy, auc_value, ground_truth_labels, prob_scores
    return accuracy, auc_value


def run_linear_probing_evaluation(
    X_train, y_train, train_scan_ids,
    X_val, y_val, val_scan_ids,
    X_test, y_test, test_scan_ids,
    train_subgroups, val_subgroups, test_subgroups,
    organ_name,
    device,
    checkpoint_dir=None,
):
    """
    Run linear probing evaluation on the provided data.
    
    Args:
        checkpoint_dir: Directory to save checkpoint. If None, checkpoint is not saved.
    
    Returns:
        Dictionary with evaluation_groups and best_model info
    """
    # Validate features and labels
    validate_features_and_labels(
        X_train, y_train, X_val, y_val, X_test, y_test, organ_name
    )
    
    feature_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    model = LinearClassifier(input_dim=feature_dim, num_classes=num_classes)
    model.to(device)

    train_loader, val_loader, test_loader = make_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_auc = 0.0
    best_checkpoint_path = None
    best_epoch = None
    epochs_without_improvement = 0
    
    for epoch in range(1, 1001):
        model.train()
        epoch_loader = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
        for x_batch, y_batch in epoch_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
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
                if checkpoint_dir is not None:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
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
            X_train_group, y_train_group = X_train, y_train
            train_scan_ids_group = train_scan_ids
            X_val_group, y_val_group = X_val, y_val
            val_scan_ids_group = val_scan_ids
            X_test_group, y_test_group = X_test, y_test
            test_scan_ids_group = test_scan_ids
        else:
            # Normal + specific subgroup abnormal
            X_train_group, y_train_group, train_scan_ids_group = _filter_with_scan_ids(
                X_train, y_train, train_scan_ids, train_subgroups,
                organ_name, subgroup_name
            )
            X_val_group, y_val_group, val_scan_ids_group = _filter_with_scan_ids(
                X_val, y_val, val_scan_ids, val_subgroups,
                organ_name, subgroup_name
            )
            X_test_group, y_test_group, test_scan_ids_group = _filter_with_scan_ids(
                X_test, y_test, test_scan_ids, test_subgroups,
                organ_name, subgroup_name
            )
        
        # Create data loaders for this group
        train_loader_group, val_loader_group, test_loader_group = make_data_loaders(
            X_train_group, y_train_group, X_val_group, y_val_group, 
            X_test_group, y_test_group, batch_size=128
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
                }
                group_predictions[split_name] = []
                continue
            
            # Evaluate
            acc, auc, y_true, prob_scores = evaluate(model, loader, device, return_predictions=True)
            
            # Count normal and abnormal samples
            n_normal = int(np.sum(y_split == 0))
            n_abnormal = int(np.sum(y_split == 1))
            
            group_metrics[split_name] = {
                "accuracy": float(acc) if acc is not None else None,
                "auc": float(auc) if auc is not None else None,
                "n_normal": n_normal,
                "n_abnormal": n_abnormal,
            }
            
            # Collect predictions
            split_predictions = []
            for idx, scan_id in enumerate(scan_ids_split):
                is_focal, is_diffuse = get_subgroup_info(scan_id, subgroups_split, organ_name)
                split_predictions.append({
                    "scan_id": scan_id,
                    "ground_truth": int(y_true[idx]),
                    "is_focal": is_focal,
                    "is_diffuse": is_diffuse,
                    "probability": float(prob_scores[idx]),
                })
            group_predictions[split_name] = split_predictions
        
        metrics["evaluation_groups"][group_name] = group_metrics
        predictions_dict["evaluation_groups"][group_name] = group_predictions

    return metrics, predictions_dict


def _filter_with_scan_ids(
    X: np.ndarray,
    y: np.ndarray,
    scan_ids: list,
    subgroup_annotations: dict,
    organ_name: str,
    subgroup_name: str,
) -> tuple:
    """
    Filter features, labels, and scan IDs using the same logic as filter_normal_and_subgroup_abnormal.
    Returns (X_filtered, y_filtered, scan_ids_filtered).
    """
    if len(X) == 0:
        return X, y, scan_ids
    
    if len(scan_ids) != len(X):
        raise ValueError(f"Mismatch: {len(scan_ids)} scan_ids but {len(X)} samples")
    
    filtered_indices = []
    for idx, scan_id in enumerate(scan_ids):
        # Include all normal samples (label=0)
        if y[idx] == 0:
            filtered_indices.append(idx)
        # Include abnormal samples (label=1) with the specified subgroup
        elif y[idx] == 1:
            if scan_id in subgroup_annotations:
                organ_subgroups = subgroup_annotations[scan_id].get(organ_name, {})
                # Check if this abnormal sample has the specified subgroup value=1
                if subgroup_name in organ_subgroups and organ_subgroups[subgroup_name] == 1:
                    filtered_indices.append(idx)
    
    if len(filtered_indices) == 0:
        return np.array([]), np.array([]), []
    
    filtered_indices = np.array(filtered_indices)
    return X[filtered_indices], y[filtered_indices], [scan_ids[i] for i in filtered_indices]


def main(args):
    fix_random_seeds(args.seed)

    feature_dir_training = get_feature_dir(
        args.output_root, args.model_name, args.organ_name, "training", args.aggregation_method
    )
    feature_dir_validation = get_feature_dir(
        args.output_root, args.model_name, args.organ_name, "validation", args.aggregation_method
    )
    feature_dir_test = get_feature_dir(
        args.output_root, args.model_name, args.organ_name, "test", args.aggregation_method
    )
    output_metrics = get_metrics_output_path(
        args.output_root, args.model_name, args.organ_name, args.aggregation_method, "linear"
    )
    output_checkpoint = get_checkpoint_output_dir(
        args.output_root, args.model_name, args.organ_name, args.aggregation_method
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
    
    # Load features and labels with scan IDs for subgroup filtering
    try:
        X_train, y_train, train_scan_ids = load_features_and_labels(
            feature_dir_training, train_annotations, args.organ_name, return_scan_ids=True
        )
        X_val, y_val, val_scan_ids = load_features_and_labels(
            feature_dir_validation, val_annotations, args.organ_name, return_scan_ids=True
        )
        X_test, y_test, test_scan_ids = load_features_and_labels(
            feature_dir_test, test_annotations, args.organ_name, return_scan_ids=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load features: {e}") from e
    
    # Get dataset root and load AMOS22 scan IDs
    dataset_root = get_dataset_root_from_annotations_path(args.annotations_train_csv)
    amos22_scan_ids = load_amos22_scan_ids(dataset_root)
    
    # Run evaluation on all data
    all_metrics, all_predictions = run_linear_probing_evaluation(
        X_train, y_train, train_scan_ids,
        X_val, y_val, val_scan_ids,
        X_test, y_test, test_scan_ids,
        train_subgroups, val_subgroups, test_subgroups,
        args.organ_name,
        device,
        checkpoint_dir=output_checkpoint,
    )
    
    # Filter out AMOS22 scans and run evaluation again
    X_train_filtered, y_train_filtered, train_scan_ids_filtered = filter_by_scan_ids(
        X_train, y_train, train_scan_ids, amos22_scan_ids
    )
    X_val_filtered, y_val_filtered, val_scan_ids_filtered = filter_by_scan_ids(
        X_val, y_val, val_scan_ids, amos22_scan_ids
    )
    X_test_filtered, y_test_filtered, test_scan_ids_filtered = filter_by_scan_ids(
        X_test, y_test, test_scan_ids, amos22_scan_ids
    )
    
    # Use a separate checkpoint directory for exclude_amos22
    exclude_checkpoint_dir = output_checkpoint + "_exclude_amos22"
    exclude_amos22_metrics, exclude_amos22_predictions = run_linear_probing_evaluation(
        X_train_filtered, y_train_filtered, train_scan_ids_filtered,
        X_val_filtered, y_val_filtered, val_scan_ids_filtered,
        X_test_filtered, y_test_filtered, test_scan_ids_filtered,
        train_subgroups, val_subgroups, test_subgroups,
        args.organ_name,
        device,
        checkpoint_dir=exclude_checkpoint_dir,
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
    parser = get_base_args_parser(description="Linear probing evaluation")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    sys.exit(main(args))
