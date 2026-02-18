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
    load_features_and_labels,
    validate_evaluation_inputs,
    load_and_validate_annotations,
    validate_features_and_labels,
    save_metrics,
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
    
    # Validate inputs early
    validate_evaluation_inputs(
        args.feature_dir_training,
        args.feature_dir_validation,
        args.feature_dir_test,
        args.annotations_train_csv,
        args.annotations_test_csv,
        args.organ_name,
        args.output_metrics,
        args.output_checkpoint,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load annotations
    train_annotations, test_annotations = load_and_validate_annotations(
        args.annotations_train_csv,
        args.annotations_test_csv,
    )
    
    val_annotations = train_annotations  # Validation uses train annotations
    
    # Load features and labels
    try:
        X_train, y_train = load_features_and_labels(args.feature_dir_training, train_annotations, args.organ_name)
        X_val, y_val = load_features_and_labels(args.feature_dir_validation, val_annotations, args.organ_name)
        X_test, y_test = load_features_and_labels(args.feature_dir_test, test_annotations, args.organ_name)
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

        checkpoint_path = os.path.join(args.output_checkpoint, f"linear_probing_epoch{epoch}.pth")
        if epoch % 1000 == 0:
            torch.save(model.state_dict(), checkpoint_path)
        
        # Evaluate on validation set and save best model
        if val_loader is not None and epoch % 100 == 0:
            val_acc, _ = evaluate(model, val_loader, device)
            if val_acc is not None and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_checkpoint_path = os.path.join(args.output_checkpoint, "best_model.pth")
                torch.save(model.state_dict(), best_checkpoint_path)

    # Evaluate on all splits
    train_acc, train_auc = evaluate(model, train_loader, device)
    val_acc, val_auc = evaluate(model, val_loader, device)
    test_acc, test_auc = evaluate(model, test_loader, device)

    metrics = {
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
        "best_checkpoint": best_checkpoint_path,
    }

    # Save metrics
    save_metrics(args.output_metrics, metrics)

    return 0


if __name__ == "__main__":
    parser = get_base_args_parser(description="Linear probing evaluation")
    parser.add_argument(
        "--output-checkpoint",
        type=str,
        required=True,
        help="Directory in which to save per-epoch checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    sys.exit(main(args))
