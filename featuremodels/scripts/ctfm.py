# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import argparse
import json
import os
import sys
from typing import Optional

import monai
import numpy as np
import torch
from lighter_zoo import SegResEncoder
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
)
from tqdm import tqdm

from .feature_model_util import get_args_parser
from util.util import fix_random_seeds


def load_model():
    model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")
    model.eval()
    model.cuda()
    return model


def load_dataset(args, dataset_type):
    preprocess = Compose(
        [
            LoadImaged(keys=["image"], ensure_channel_first=True),
            EnsureTyped(keys=["image"]),
            EnsureTyped(keys=["label"]),
            Orientationd(keys=["image"], axcodes="SPL"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1024,
                a_max=2048,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
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
        features.append(
            torch.nn.functional.adaptive_avg_pool3d(model(dataset[i]['image'].unsqueeze(0).cuda())[-1], 1)
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
    description = "CT-FM Feature Extraction"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()

    sys.exit(main(args))
