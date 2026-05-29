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
    load_features_and_labels,
    validate_evaluation_inputs,
    load_and_validate_annotations,
    load_subgroup_annotations,
    validate_features_and_labels,
    save_metrics,
    load_amos22_scan_ids,
    get_dataset_root_from_annotations_path,
    filter_by_scan_ids,
    get_predictions_output_path,
    get_subgroup_info,
    save_predictions,
    load_features_and_labels_all_organs,
    get_all_organs_metrics_output_path,
    get_all_organs_checkpoint_output_dir,
    filter_all_organs_with_scan_ids,
    get_subgroup_info_all_organs,
)


class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, num_classes: int):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, num_classes))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def get_hidden_dims(mlp_variant: str) -> list:
    if mlp_variant == "mlp1":
        return [256]
    if mlp_variant == "mlp2":
        return [256, 64]
    raise ValueError(f"Unknown MLP variant: {mlp_variant}")


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
        prob_scores = probabilities[:, 1]
    else:
        auc_value = roc_auc_score(
            ground_truth_labels,
            probabilities,
            multi_class="ovr",
            average="macro",
        )
        prob_scores = probabilities.max(axis=1)

    if return_predictions:
        return accuracy, auc_value, ground_truth_labels, prob_scores
    return accuracy, auc_value


def _filter_with_scan_ids(
    X: np.ndarray,
    y: np.ndarray,
    scan_ids: list,
    subgroup_annotations: dict,
    organ_name: str,
    subgroup_name: str,
) -> tuple:
    if len(X) == 0:
        return X, y, scan_ids

    if len(scan_ids) != len(X):
        raise ValueError(f"Mismatch: {len(scan_ids)} scan_ids but {len(X)} samples")

    filtered_indices = []
    for idx, scan_id in enumerate(scan_ids):
        if y[idx] == 0:
            filtered_indices.append(idx)
        elif y[idx] == 1:
            if scan_id in subgroup_annotations:
                organ_subgroups = subgroup_annotations[scan_id].get(organ_name, {})
                if subgroup_name in organ_subgroups and organ_subgroups[subgroup_name] == 1:
                    filtered_indices.append(idx)

    if len(filtered_indices) == 0:
        return np.array([]), np.array([]), []

    filtered_indices = np.array(filtered_indices)
    return X[filtered_indices], y[filtered_indices], [scan_ids[i] for i in filtered_indices]


def run_mlp_evaluation(
    X_train,
    y_train,
    train_scan_ids,
    X_val,
    y_val,
    val_scan_ids,
    X_test,
    y_test,
    test_scan_ids,
    train_subgroups,
    val_subgroups,
    test_subgroups,
    organ_name,
    device,
    mlp_variant,
    checkpoint_dir=None,
    is_all_organs_mode=False,
):
    validate_features_and_labels(
        X_train, y_train, X_val, y_val, X_test, y_test, organ_name
    )

    feature_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    hidden_dims = get_hidden_dims(mlp_variant)

    model = MLPClassifier(input_dim=feature_dim, hidden_dims=hidden_dims, num_classes=num_classes)
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

        if val_loader is not None:
            _, val_auc = evaluate(model, val_loader, device)
            if val_auc is not None and val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                epochs_without_improvement = 0
                if checkpoint_dir is not None:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    best_checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{mlp_variant}.pth")
                    torch.save(model.state_dict(), best_checkpoint_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= 50:
                print(f"Early stopping at epoch {epoch}: no improvement in validation AUC for 50 epochs")
                break

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
            "mlp_variant": mlp_variant,
            "hidden_dims": hidden_dims,
        },
    }
    predictions_dict = {"evaluation_groups": {}}

    evaluation_groups = [
        ("all", None),
        ("normal_and_diffuse", "diffuse"),
        ("normal_and_focal", "focal"),
    ]

    for group_name, subgroup_name in evaluation_groups:
        group_metrics = {}
        group_predictions = {}

        if subgroup_name is None:
            X_train_group, y_train_group = X_train, y_train
            train_scan_ids_group = train_scan_ids
            X_val_group, y_val_group = X_val, y_val
            val_scan_ids_group = val_scan_ids
            X_test_group, y_test_group = X_test, y_test
            test_scan_ids_group = test_scan_ids
        else:
            if is_all_organs_mode:
                X_train_group, y_train_group, train_scan_ids_group = filter_all_organs_with_scan_ids(
                    X_train, y_train, train_scan_ids, train_subgroups, subgroup_name
                )
                X_val_group, y_val_group, val_scan_ids_group = filter_all_organs_with_scan_ids(
                    X_val, y_val, val_scan_ids, val_subgroups, subgroup_name
                )
                X_test_group, y_test_group, test_scan_ids_group = filter_all_organs_with_scan_ids(
                    X_test, y_test, test_scan_ids, test_subgroups, subgroup_name
                )
            else:
                X_train_group, y_train_group, train_scan_ids_group = _filter_with_scan_ids(
                    X_train, y_train, train_scan_ids, train_subgroups, organ_name, subgroup_name
                )
                X_val_group, y_val_group, val_scan_ids_group = _filter_with_scan_ids(
                    X_val, y_val, val_scan_ids, val_subgroups, organ_name, subgroup_name
                )
                X_test_group, y_test_group, test_scan_ids_group = _filter_with_scan_ids(
                    X_test, y_test, test_scan_ids, test_subgroups, organ_name, subgroup_name
                )

        train_loader_group, val_loader_group, test_loader_group = make_data_loaders(
            X_train_group, y_train_group, X_val_group, y_val_group, X_test_group, y_test_group, batch_size=128
        )

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

            acc, auc, y_true, prob_scores = evaluate(model, loader, device, return_predictions=True)
            n_normal = int(np.sum(y_split == 0))
            n_abnormal = int(np.sum(y_split == 1))

            group_metrics[split_name] = {
                "accuracy": float(acc) if acc is not None else None,
                "auc": float(auc) if auc is not None else None,
                "n_normal": n_normal,
                "n_abnormal": n_abnormal,
            }

            split_predictions = []
            for idx, scan_id_or_tuple in enumerate(scan_ids_split):
                if is_all_organs_mode:
                    scan_id, organ_name_for_pred = scan_id_or_tuple
                    is_focal, is_diffuse = get_subgroup_info_all_organs(
                        scan_id, organ_name_for_pred, subgroups_split
                    )
                    split_predictions.append(
                        {
                            "scan_id": scan_id,
                            "organ_name": organ_name_for_pred,
                            "ground_truth": int(y_true[idx]),
                            "is_focal": is_focal,
                            "is_diffuse": is_diffuse,
                            "probability": float(prob_scores[idx]),
                        }
                    )
                else:
                    is_focal, is_diffuse = get_subgroup_info(scan_id_or_tuple, subgroups_split, organ_name)
                    split_predictions.append(
                        {
                            "scan_id": scan_id_or_tuple,
                            "ground_truth": int(y_true[idx]),
                            "is_focal": is_focal,
                            "is_diffuse": is_diffuse,
                            "probability": float(prob_scores[idx]),
                        }
                    )
            group_predictions[split_name] = split_predictions

        metrics["evaluation_groups"][group_name] = group_metrics
        predictions_dict["evaluation_groups"][group_name] = group_predictions

    return metrics, predictions_dict


def main(args):
    fix_random_seeds(args.seed)

    is_all_organs_mode = args.organ_name == "all"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dims = get_hidden_dims(args.mlp_variant)

    if is_all_organs_mode:
        output_metrics = get_all_organs_metrics_output_path(
            args.output_root, args.model_name, args.aggregation_method, args.mlp_variant
        )
        output_checkpoint = os.path.join(
            get_all_organs_checkpoint_output_dir(args.output_root, args.model_name, args.aggregation_method),
            args.mlp_variant,
        )

        train_annotations, test_annotations = load_and_validate_annotations(
            args.annotations_train_csv, args.annotations_test_csv
        )
        val_annotations = train_annotations
        train_subgroups, test_subgroups = load_subgroup_annotations(
            args.annotations_train_csv, args.annotations_test_csv
        )
        val_subgroups = train_subgroups

        X_train, y_train, train_scan_organ_ids = load_features_and_labels_all_organs(
            args.output_root, args.model_name, "training", args.aggregation_method, train_annotations, return_scan_ids=True
        )
        X_val, y_val, val_scan_organ_ids = load_features_and_labels_all_organs(
            args.output_root, args.model_name, "validation", args.aggregation_method, val_annotations, return_scan_ids=True
        )
        X_test, y_test, test_scan_organ_ids = load_features_and_labels_all_organs(
            args.output_root, args.model_name, "test", args.aggregation_method, test_annotations, return_scan_ids=True
        )

        dataset_root = get_dataset_root_from_annotations_path(args.annotations_train_csv)
        amos22_scan_ids = load_amos22_scan_ids(dataset_root)

        def filter_all_organs_by_scan_ids(X, y, scan_organ_ids, exclude_scan_ids):
            filtered_indices = []
            for idx, (scan_id, _) in enumerate(scan_organ_ids):
                if scan_id not in exclude_scan_ids:
                    filtered_indices.append(idx)
            if len(filtered_indices) == 0:
                return np.array([]), np.array([]), []
            filtered_indices = np.array(filtered_indices)
            return X[filtered_indices], y[filtered_indices], [scan_organ_ids[i] for i in filtered_indices]

        all_metrics, all_predictions = run_mlp_evaluation(
            X_train,
            y_train,
            train_scan_organ_ids,
            X_val,
            y_val,
            val_scan_organ_ids,
            X_test,
            y_test,
            test_scan_organ_ids,
            train_subgroups,
            val_subgroups,
            test_subgroups,
            "all",
            device,
            mlp_variant=args.mlp_variant,
            checkpoint_dir=output_checkpoint,
            is_all_organs_mode=True,
        )

        X_train_filtered, y_train_filtered, train_scan_organ_ids_filtered = filter_all_organs_by_scan_ids(
            X_train, y_train, train_scan_organ_ids, amos22_scan_ids
        )
        X_val_filtered, y_val_filtered, val_scan_organ_ids_filtered = filter_all_organs_by_scan_ids(
            X_val, y_val, val_scan_organ_ids, amos22_scan_ids
        )
        X_test_filtered, y_test_filtered, test_scan_organ_ids_filtered = filter_all_organs_by_scan_ids(
            X_test, y_test, test_scan_organ_ids, amos22_scan_ids
        )

        exclude_checkpoint_dir = output_checkpoint + "_exclude_amos22"
        exclude_amos22_metrics, exclude_amos22_predictions = run_mlp_evaluation(
            X_train_filtered,
            y_train_filtered,
            train_scan_organ_ids_filtered,
            X_val_filtered,
            y_val_filtered,
            val_scan_organ_ids_filtered,
            X_test_filtered,
            y_test_filtered,
            test_scan_organ_ids_filtered,
            train_subgroups,
            val_subgroups,
            test_subgroups,
            "all",
            device,
            mlp_variant=args.mlp_variant,
            checkpoint_dir=exclude_checkpoint_dir,
            is_all_organs_mode=True,
        )
    else:
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
            args.output_root, args.model_name, args.organ_name, args.aggregation_method, args.mlp_variant
        )
        output_checkpoint = os.path.join(
            args.output_root,
            args.model_name,
            args.organ_name,
            "checkpoints",
            "aggregated",
            args.aggregation_method,
            args.mlp_variant,
        )

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

        train_annotations, test_annotations = load_and_validate_annotations(
            args.annotations_train_csv, args.annotations_test_csv
        )
        val_annotations = train_annotations
        train_subgroups, test_subgroups = load_subgroup_annotations(
            args.annotations_train_csv, args.annotations_test_csv
        )
        val_subgroups = train_subgroups

        X_train, y_train, train_scan_ids = load_features_and_labels(
            feature_dir_training, train_annotations, args.organ_name, return_scan_ids=True
        )
        X_val, y_val, val_scan_ids = load_features_and_labels(
            feature_dir_validation, val_annotations, args.organ_name, return_scan_ids=True
        )
        X_test, y_test, test_scan_ids = load_features_and_labels(
            feature_dir_test, test_annotations, args.organ_name, return_scan_ids=True
        )

        dataset_root = get_dataset_root_from_annotations_path(args.annotations_train_csv)
        amos22_scan_ids = load_amos22_scan_ids(dataset_root)

        all_metrics, all_predictions = run_mlp_evaluation(
            X_train,
            y_train,
            train_scan_ids,
            X_val,
            y_val,
            val_scan_ids,
            X_test,
            y_test,
            test_scan_ids,
            train_subgroups,
            val_subgroups,
            test_subgroups,
            args.organ_name,
            device,
            mlp_variant=args.mlp_variant,
            checkpoint_dir=output_checkpoint,
            is_all_organs_mode=False,
        )

        X_train_filtered, y_train_filtered, train_scan_ids_filtered = filter_by_scan_ids(
            X_train, y_train, train_scan_ids, amos22_scan_ids
        )
        X_val_filtered, y_val_filtered, val_scan_ids_filtered = filter_by_scan_ids(
            X_val, y_val, val_scan_ids, amos22_scan_ids
        )
        X_test_filtered, y_test_filtered, test_scan_ids_filtered = filter_by_scan_ids(
            X_test, y_test, test_scan_ids, amos22_scan_ids
        )

        exclude_checkpoint_dir = output_checkpoint + "_exclude_amos22"
        exclude_amos22_metrics, exclude_amos22_predictions = run_mlp_evaluation(
            X_train_filtered,
            y_train_filtered,
            train_scan_ids_filtered,
            X_val_filtered,
            y_val_filtered,
            val_scan_ids_filtered,
            X_test_filtered,
            y_test_filtered,
            test_scan_ids_filtered,
            train_subgroups,
            val_subgroups,
            test_subgroups,
            args.organ_name,
            device,
            mlp_variant=args.mlp_variant,
            checkpoint_dir=exclude_checkpoint_dir,
            is_all_organs_mode=False,
        )

    metrics = {
        "all_data": all_metrics,
        "exclude_amos22": exclude_amos22_metrics,
    }
    save_metrics(output_metrics, metrics)

    predictions = {
        "all_data": all_predictions,
        "exclude_amos22": exclude_amos22_predictions,
    }
    output_predictions = get_predictions_output_path(output_metrics)
    save_predictions(output_predictions, predictions)

    print(f"Finished {args.mlp_variant} with hidden_dims={hidden_dims}")
    return 0


if __name__ == "__main__":
    parser = get_base_args_parser(description="MLP evaluation")
    parser.add_argument(
        "--mlp-variant",
        type=str,
        required=True,
        choices=["mlp1", "mlp2"],
        help="MLP variant: mlp1 (input->256->1) or mlp2 (input->256->64->1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    sys.exit(main(args))
