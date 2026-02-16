# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from util.util import fix_random_seeds


def get_args_parser(
    description: Optional[str] = None,
    add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        add_help=add_help,
    )
    parser.add_argument(
        "--feature-file-training",
        type=str,
        required=True,
        help="Path to the `.npz` containing training features and labels",
    )
    parser.add_argument(
        "--feature-file-test",
        type=str,
        required=True,
        help="Path to the `.npz` containing test features and labels",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        required=True,
        help="File path to write the JSON metrics",
    )
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
    return parser


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def make_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
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

    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def main(args):
    fix_random_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_train = np.load(args.feature_file_training)
    X_train, y_train = data_train["features"], data_train["labels"].astype(np.int64)

    data_test = np.load(args.feature_file_test)
    X_test, y_test = data_test["features"], data_test["labels"].astype(np.int64)

    feature_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    model = LinearClassifier(input_dim=feature_dim, num_classes=num_classes)
    model.to(device)

    train_loader, test_loader = make_data_loaders(X_train, y_train, X_test, y_test, batch_size=128)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(args.output_checkpoint, exist_ok=True)
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

    model.eval()
    all_logits = []
    all_true_labels = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch).cpu()
            all_logits.append(logits)
            all_true_labels.append(y_batch)

    all_logits = torch.cat(all_logits, dim=0)
    ground_truth_labels = torch.cat(all_true_labels, dim=0).numpy()
    probabilities = torch.softmax(all_logits, dim=1).numpy()
    predicted_labels = probabilities.argmax(axis=1)

    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    if num_classes == 2:
        auc_value = roc_auc_score(ground_truth_labels, probabilities[:, 1])
    else:
        auc_value = roc_auc_score(
            ground_truth_labels,
            probabilities,
            multi_class="ovr",
            average="macro",
        )

    metrics = {
        "accuracy": float(accuracy),
        "auc": float(auc_value),
    }

    # ensure output dir exists
    output_dir = os.path.dirname(args.output_metrics)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_metrics, "w") as f:
        json.dump(metrics, f)

    return 0


if __name__ == "__main__":
    parser = get_args_parser(description="Linear probing evaluation")
    args = parser.parse_args()
    sys.exit(main(args))
