# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import argparse
import os
import sys
from typing import Optional

import numpy as np


def get_args_parser(description: Optional[str] = None, add_help: bool = True):
    parser = argparse.ArgumentParser(description=description, add_help=add_help)
    parser.add_argument("--input-features-training", nargs='+', required=True, help="Path to training vector file")
    parser.add_argument("--input-features-test", nargs='+', required=True, help="Path to test vector file")
    parser.add_argument("--output-features-training", required=True, help="Output file for averaged training vector")
    parser.add_argument("--output-features-test", required=True, help="Output file for averaged test vector")
    return parser

def _create_output_directory(output_file_path):
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def _load_features(input_files):
    return [np.load(file)["features"] for file in input_files]

def _load_labels(input_files):
    return np.load(input_files[0])["labels"]

def _save_features(features, labels, output_file):
    np.savez(output_file, features=np.array(features), labels=np.array(labels))

def _average_vectors(vectors):
    return np.mean(vectors, axis=0)

def main(args):
    _create_output_directory(args.output_features_training)
    _create_output_directory(args.output_features_test)

    train_vectors = _load_features(args.input_features_training)
    test_vectors = _load_features(args.input_features_test)

    train_labels = _load_labels(args.input_features_training)
    test_labels = _load_labels(args.input_features_test)

    avg_train_vector = _average_vectors(train_vectors)
    avg_test_vector = _average_vectors(test_vectors)

    _save_features(avg_train_vector, train_labels, args.output_features_training)
    _save_features(avg_test_vector, test_labels, args.output_features_test)

    print(f"Mean training vector saved to {args.output_features_training}")
    print(f"Mean test vector saved to {args.output_features_test}")

if __name__ == "__main__":
    description = "Modality Mean Aggregation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()

    sys.exit(main(args))
