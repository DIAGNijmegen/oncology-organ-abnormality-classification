# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import os

REPOSITORY_ROOT = os.getenv("REPOSITORY_ROOT")
if REPOSITORY_ROOT is None:
    raise ValueError("REPOSITORY_ROOT environment variable is not set. Please set it to the root of the repository.")

DATASET_ROOT = os.getenv("DATASET_ROOT")
if DATASET_ROOT is None:
    raise ValueError("DATASET_ROOT environment variable is not set. Please set it to the root of your datasets.")

HF_HOME = os.getenv("HF_HOME")
if HF_HOME is None:
    raise ValueError("HF_HOME environment variable is not set. Please set it to the desired home folder of Huggingface.")

OUTPUT_ROOT = os.getenv("OUTPUT_ROOT")
if OUTPUT_ROOT is None:
    raise ValueError("OUTPUT_ROOT environment variable is not set. Please set it to the desired root of your outputs.")

# Batch size configuration - harmonized for feature models and aggregation
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))

import json

with open(f"{REPOSITORY_ROOT}/experiments.json","r") as f:
    EXPERIMENTS = json.load(f)

from util.snakemake_helpers import setup_leavs_dataset, VALID_ORGANS

# Set up LEAVS dataset information
leavs_data = setup_leavs_dataset(DATASET_ROOT, val_ratio=0.2, seed=42, filter_valid_labels=True)
train_annotations = leavs_data["train_annotations"]
test_annotations = leavs_data["test_annotations"]
train_scan_ids_split = leavs_data["train_scan_ids_split"]
val_scan_ids_split = leavs_data["val_scan_ids_split"]
get_scans_for_split_and_organ = leavs_data["get_scans_for_split_and_organ"]


def create_batches(items, batch_size):
    """Create batches from a list of items."""
    # Enforce deterministic batching independent of incoming list order.
    ordered_items = sorted(items)
    batches = []
    for i in range(0, len(ordered_items), batch_size):
        batch = ordered_items[i:i + batch_size]
        batches.append(batch)
    return batches


def get_batch_id(batch_idx, total_batches):
    """Generate a batch ID string."""
    # Use zero-padded batch index for consistent sorting
    max_digits = len(str(total_batches - 1))
    return f"batch_{str(batch_idx).zfill(max_digits)}"


output_files = []
for experiment_name, experiment in EXPERIMENTS.items():
    for organ_name in VALID_ORGANS:
        for evaluation_mode in experiment['evaluation_modes']:
            if evaluation_mode == "attention":
                # Attention doesn't use aggregation, outputs to metrics/attention/attention.json
                output_files.append(
                    OUTPUT_ROOT + f"/{experiment_name}/{organ_name}/metrics/attention/attention.json"
                )
            else:
                # Other evaluation modes use aggregation
                for aggregation_method in experiment['aggregation_methods']:
                    output_files.append(
                        OUTPUT_ROOT + f"/{experiment_name}/{organ_name}/metrics/aggregated/{aggregation_method}/{evaluation_mode}.json"
                    )

rule all:
    input: output_files

# Include all specialized blocks
include: "featuremodels/Snakefile"
include: "aggregation/Snakefile"
include: "evaluation/Snakefile"
