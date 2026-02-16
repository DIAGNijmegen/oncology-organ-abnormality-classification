# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

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
        help="Path to the file containing features of the training split",
    )
    parser.add_argument(
        "--feature-file-test",
        type=str,
        help="Path to the file containing features of the test split",
    )
    parser.add_argument(
        "--output-metrics",
        default="",
        type=str,
        help="Output file to write metrics",
    )
    return parser

def main(args):
    fix_random_seeds(getattr(args, "seed", 0))

    scaler = StandardScaler()

    data_train = np.load(args.feature_file_training)
    X_train, y_train = data_train['features'], data_train['labels']
    X_train = scaler.fit_transform(X_train)

    data_test = np.load(args.feature_file_test)
    X_test, y_test = data_test['features'], data_test['labels']
    X_test = scaler.fit_transform(X_test)

    k_values = [10, 20, 100, 200]
    results = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        y_prob = knn.predict_proba(X_test)[:, 1] if len(set(y_train)) == 2 else None

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
        results.append({"k": k, "accuracy": acc, "auc": auc})

    output_dir = os.path.dirname(args.output_metrics)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_metrics, "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    description = "kNN Evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()

    sys.exit(main(args))
