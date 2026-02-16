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

import json

with open(f"{REPOSITORY_ROOT}/experiments.json","r") as f:
    EXPERIMENTS = json.load(f)


def get_info_from_dataset(experiment_name, dataset_name):
    import os

    experiment = EXPERIMENTS[experiment_name]
    dataset = next((d for d in experiment['datasets'] if d['name'] == dataset_name),None)
    entries_file = dataset['entries_file']

    # Get the folds from the dataset
    folds = []
    for file in os.listdir(f"{DATASET_ROOT}/{dataset_name}"):
        if file.startswith(entries_file) and file.endswith(".json"):
            folds.append(int(file.split("_fold")[1].split(".json")[0]))

    # Get the modalities from the JSON file for the 0'th fold
    with open(f"{DATASET_ROOT}/{dataset_name}/{entries_file}_fold0.json","r") as f:
        dataset_json = json.load(f)
        modalities = dataset_json['modality']

        # In CT datasets, the modality can be the string "CT" instead of a list ["CT"]
        if not isinstance(modalities,list):
            modalities = [modalities]

    return sorted(folds), sorted(modalities)


def add_experiment_to_output_files(experiment_name, dataset_name, fold, feature_path, epoch, evaluation_mode, output_files):
    if evaluation_mode == 'visualizations':
        for visualization in ['lda', 'pca', 'tsne']:
            output_files.append(f"{OUTPUT_ROOT}/{experiment_name}/{epoch}/{dataset_name}/fold{fold}/visualizations/{feature_path}/{visualization}.png")
    else:
        output_files.append(f"{OUTPUT_ROOT}/{experiment_name}/{epoch}/{dataset_name}/fold{fold}/metrics/{feature_path}/{evaluation_mode}.json")


# This loop enumerates all possible combinations of datasets, folds, modalities, epochs, evaluation modes, and aggregation methods
# to create a comprehensive list of expected output files for the Snakemake workflow.
output_files = []
for experiment_name, experiment in EXPERIMENTS.items():
    for dataset in experiment['datasets']:
        folds, modalities = get_info_from_dataset(experiment_name,dataset['name'])

        for fold in folds:
            for modality in modalities:
                for evaluation_mode in experiment['evaluation_modes']:
                    for epoch in experiment['epochs']:
                        add_experiment_to_output_files(experiment_name,
                            dataset['name'],fold,f"modalities/{modality}",epoch,evaluation_mode,output_files)

                        for aggregation_method in experiment['aggregation_methods']:
                            add_experiment_to_output_files(experiment_name,dataset[
                                'name'],fold,f"aggregated/{aggregation_method}",epoch,evaluation_mode,output_files)


rule all:
    input: output_files

# Include all specialized blocks
include: "featuremodels/Snakefile"
include: "aggregation/Snakefile"
include: "evaluation/Snakefile"
