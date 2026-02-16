# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import json
import os
import sys

import monai
import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    CenterSpatialCropd,
)
from tqdm import tqdm

from spectre import SpectreImageFeatureExtractor, MODEL_CONFIGS
from util.util import fix_random_seeds
from .feature_model_util import get_args_parser


def load_model():
    config = MODEL_CONFIGS['spectre-large-pretrained']
    model = SpectreImageFeatureExtractor.from_config(config)
    model.cuda().eval()
    return model


def load_dataset(args, dataset_type):
    preprocess = Compose(
        [
            LoadImaged(keys=["image"], ensure_channel_first=True),
            EnsureTyped(keys=["image"], dtype=torch.float32),
            EnsureTyped(keys=["label"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000,
                a_max=1000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Spacingd(keys=["image"], pixdim=(0.5, 0.5, 1.0), mode=("bilinear")),
            CenterSpatialCropd(keys=["image"], roi_size=(512, 512, 384)),
        ]
    )
    json_path = args.entries_file
    with open(json_path, "r") as f:
        json_data = json.load(f)

    images = json_data[dataset_type]
    data_files = []
    for image in images:
        data_files.extend(image["patches"])

    data = [{"image": sample["patch_path"], "label": sample["label"]} for sample in data_files]
    dataset = monai.data.Dataset(data=data, transform=preprocess)

    print(f"Using {len(data)} samples for {dataset_type} split")

    return dataset


def compute_features(args, model, dataset_type):
    dataset = load_dataset(args, dataset_type)

    features = []
    labels = []
    for i in tqdm(range(len(dataset))):
        with torch.no_grad():
            features.append(
                model(dataset[i]['image'].unsqueeze(0).unsqueeze(0).cuda(), grid_size=(1, 1, 1))[-1].mean(dim=0)
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )
            labels.append(dataset[i]['label'])

    return np.array(features), np.array(labels)


def _create_output_directory(output_file_path):
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def main(args):
    fix_random_seeds(getattr(args, "seed", 0))

    _create_output_directory(args.output_features_training)
    _create_output_directory(args.output_features_test)

    print("Loading model...")
    model = load_model()

    for dataset_type in ["training", "test"]:
        print(f"Computing features for {dataset_type} split...")
        features, labels = compute_features(args, model, dataset_type)
        np.savez(vars(args)[f"output_features_{dataset_type}"], features=features, labels=labels)


if __name__ == "__main__":
    description = "SPECTRE Feature Extraction"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()

    sys.exit(main(args))
