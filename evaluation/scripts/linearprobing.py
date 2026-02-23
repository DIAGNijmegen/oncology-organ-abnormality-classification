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
    filter_by_subgroup,
    get_available_subgroups,
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


def evaluate(model, data_loader, device):
    """Evaluate model on a data loader."""
    if data_loader is None:
        return None, None
    
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
    else:
        auc_value = roc_auc_score(
            ground_truth_labels,
            probabilities,
            multi_class="ovr",
            average="macro",
        )
    
    return accuracy, auc_value


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
    
    # Validate features and labels
    validate_features_and_labels(
        X_train, y_train, X_val, y_val, X_test, y_test, args.organ_name
    )
    
    feature_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    model = LinearClassifier(input_dim=feature_dim, num_classes=num_classes)
    model.to(device)

    train_loader, val_loader, test_loader = make_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_checkpoint_path = None
    
    for epoch in range(1, 10001):
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

        checkpoint_path = os.path.join(output_checkpoint, f"linear_probing_epoch{epoch}.pth")
        if epoch % 1000 == 0:
            torch.save(model.state_dict(), checkpoint_path)
        
        # Evaluate on validation set and save best model
        if val_loader is not None and epoch % 100 == 0:
            val_acc, _ = evaluate(model, val_loader, device)
            if val_acc is not None and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_checkpoint_path = os.path.join(output_checkpoint, "best_model.pth")
                torch.save(model.state_dict(), best_checkpoint_path)

    # Evaluate on all splits (overall metrics)
    train_acc, train_auc = evaluate(model, train_loader, device)
    val_acc, val_auc = evaluate(model, val_loader, device)
    test_acc, test_auc = evaluate(model, test_loader, device)

    metrics = {
        "overall": {
            "train": {
                "accuracy": float(train_acc) if train_acc is not None else None,
                "auc": float(train_auc) if train_auc is not None else None,
            },
            "validation": {
                "accuracy": float(val_acc) if val_acc is not None else None,
                "auc": float(val_auc) if val_auc is not None else None,
            },
            "test": {
                "accuracy": float(test_acc) if test_acc is not None else None,
                "auc": float(test_auc) if test_auc is not None else None,
            },
        },
        "subgroups": {},
        "best_checkpoint": best_checkpoint_path,
    }
    
    # Get available subgroups for train and test
    train_available_subgroups = get_available_subgroups(train_subgroups, args.organ_name)
    test_available_subgroups = get_available_subgroups(test_subgroups, args.organ_name)
    
    # Combine and deduplicate subgroups (evaluate on all subgroups present in either split)
    all_subgroups = sorted(set(train_available_subgroups + test_available_subgroups))
    
    # Evaluate on each subgroup
    for subgroup_name in all_subgroups:
        subgroup_metrics = {}
        
        # Filter train split
        X_train_sub, y_train_sub = filter_by_subgroup(
            X_train, y_train, train_scan_ids, train_subgroups,
            args.organ_name, subgroup_name, subgroup_value=1
        )
        
        # Filter validation split
        X_val_sub, y_val_sub = filter_by_subgroup(
            X_val, y_val, val_scan_ids, val_subgroups,
            args.organ_name, subgroup_name, subgroup_value=1
        )
        
        # Filter test split
        X_test_sub, y_test_sub = filter_by_subgroup(
            X_test, y_test, test_scan_ids, test_subgroups,
            args.organ_name, subgroup_name, subgroup_value=1
        )
        
        # Only evaluate if we have samples in all splits
        if len(X_train_sub) > 0 and len(X_val_sub) > 0 and len(X_test_sub) > 0:
            train_loader_sub, val_loader_sub, test_loader_sub = make_data_loaders(
                X_train_sub, y_train_sub, X_val_sub, y_val_sub, X_test_sub, y_test_sub, batch_size=128
            )
            
            train_acc_sub, train_auc_sub = evaluate(model, train_loader_sub, device)
            val_acc_sub, val_auc_sub = evaluate(model, val_loader_sub, device)
            test_acc_sub, test_auc_sub = evaluate(model, test_loader_sub, device)
            
            subgroup_metrics = {
                "train": {
                    "accuracy": float(train_acc_sub) if train_acc_sub is not None else None,
                    "auc": float(train_auc_sub) if train_auc_sub is not None else None,
                    "n_samples": int(len(X_train_sub)),
                },
                "validation": {
                    "accuracy": float(val_acc_sub) if val_acc_sub is not None else None,
                    "auc": float(val_auc_sub) if val_auc_sub is not None else None,
                    "n_samples": int(len(X_val_sub)),
                },
                "test": {
                    "accuracy": float(test_acc_sub) if test_acc_sub is not None else None,
                    "auc": float(test_auc_sub) if test_auc_sub is not None else None,
                    "n_samples": int(len(X_test_sub)),
                },
            }
        else:
            # No samples for this subgroup
            subgroup_metrics = {
                "train": {"accuracy": None, "auc": None, "n_samples": 0},
                "validation": {"accuracy": None, "auc": None, "n_samples": 0},
                "test": {"accuracy": None, "auc": None, "n_samples": 0},
            }
        
        metrics["subgroups"][subgroup_name] = subgroup_metrics

    # Save metrics
    save_metrics(output_metrics, metrics)

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
