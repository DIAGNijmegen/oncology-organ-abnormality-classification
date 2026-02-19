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
    batches = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batches.append(batch)
    return batches


def get_batch_id(batch_idx, total_batches):
    """Generate a batch ID string."""
    # Use zero-padded batch index for consistent sorting
    max_digits = len(str(total_batches - 1))
    return f"batch_{str(batch_idx).zfill(max_digits)}"


def get_batch_ids_for_model_split_organ(model_name, split, organ_name):
    """Get all batch IDs for a given model/split/organ combination - harmonized with feature models and aggregation."""
    scans = get_scans_for_split_and_organ(split, organ_name)
    scan_batches = create_batches(scans, BATCH_SIZE)
    batch_ids = []
    for batch_idx in range(len(scan_batches)):
        batch_id = get_batch_id(batch_idx, len(scan_batches))
        batch_ids.append(batch_id)
    return batch_ids


output_files = []
# Add evaluation metrics
for experiment_name, experiment in EXPERIMENTS.items():
    for organ_name in VALID_ORGANS:
        for evaluation_mode in experiment['evaluation_modes']:
            for aggregation_method in experiment['aggregation_methods']:
                output_files.append(
                    OUTPUT_ROOT + f"/{experiment_name}/{organ_name}/metrics/aggregated/{aggregation_method}/{evaluation_mode}.json"
                )

# Add aggregation progress files
for experiment_name, experiment in EXPERIMENTS.items():
    for aggregation_method in experiment['aggregation_methods']:
        for organ_name in VALID_ORGANS:
            for split in ["training", "validation", "test"]:
                batch_ids = get_batch_ids_for_model_split_organ(experiment_name, split, organ_name)
                for batch_id in batch_ids:
                    output_files.append(
                        OUTPUT_ROOT + f"/{experiment_name}/progress/{split}/{organ_name}/{batch_id}_{aggregation_method}.done"
                    )

rule all:
    input: output_files

# Include all specialized blocks
include: "featuremodels/Snakefile"
include: "aggregation/Snakefile"
include: "evaluation/Snakefile"
