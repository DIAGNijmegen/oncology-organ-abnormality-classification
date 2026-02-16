# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import argparse
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

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
        "--output-lda",
        default="",
        type=str,
        help="Output file to store the LDA visualization",
    )
    parser.add_argument(
        "--output-pca",
        default="",
        type=str,
        help="Output file to store the PCA visualization",
    )
    parser.add_argument(
        "--output-tsne",
        default="",
        type=str,
        help="Output file to store the tSNE visualization",
    )
    return parser

def _save_plot(x_label, y_label, title, output_name):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_name, dpi=300)

def main(args):
    fix_random_seeds(getattr(args, "seed", 0))

    train_data = np.load(args.feature_file_training)
    test_data = np.load(args.feature_file_test)

    X_train, y_train = train_data['features'], train_data['labels'].astype(int)
    X_test, y_test = test_data['features'], test_data['labels'].astype(int)

    X = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))

    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(X, y).flatten()
    plt.figure(figsize=(8, 6))
    sns.kdeplot(X_lda[y == 0], fill=True, color='blue', alpha=0.6, label='Class 0')
    sns.kdeplot(X_lda[y == 1], fill=True, color='red', alpha=0.6, label='Class 1')
    _save_plot("LDA Component", "Density", "LDA Density Plot of Data Separability", args.output_lda)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    for label, color in zip([0, 1], ['blue', 'red']):
        plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1],
                    label=f'Class {label}', alpha=0.6, edgecolors='k')
    _save_plot("PCA Component 1", "PCA Component 2", "PCA Visualization", args.output_pca)

    tsne = TSNE(n_components=2, perplexity=80, random_state=42)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(8, 6))
    for label, color in zip([0, 1], ['blue', 'red']):
        plt.scatter(X_tsne[y == label, 0], X_tsne[y == label, 1],
                    label=f'Class {label}', alpha=0.6, edgecolors='k')
    _save_plot("t-SNE Component 1", "t-SNE Component 2", "t-SNE Visualization", args.output_tsne)


if __name__ == "__main__":
    description = "Visualization Evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()

    sys.exit(main(args))
