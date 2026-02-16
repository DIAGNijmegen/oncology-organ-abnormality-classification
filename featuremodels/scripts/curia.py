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
    EnsureChannelFirstd,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor

from .feature_model_util import get_args_parser
from util.util import fix_random_seeds


def load_dataset(args, dataset_type):
    preprocess = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="PLS"),
            EnsureTyped(keys=["image"]),
        ]
    )
    json_path = args.entries_file
    with open(json_path, "r") as f:
        json_data = json.load(f)

    images = json_data[dataset_type]
    data_files = []
    for image in images:
        data_files.extend(image["patches"])

    if args.modality == "CT":
        data = [{"image": sample["patch_path"], "label": sample["label"]} for sample in data_files]
    elif args.modality in ["t2w", "adc", "hbv"]:
        data = [
            {"image": sample["patch_path"].replace(".mha", f"_{args.modality}.mha"), "label": sample["label"]}
            for sample in data_files
        ]
    else:
        raise ValueError(f"Unknown modality: {args.modality}")

    dataset = monai.data.Dataset(data=data, transform=preprocess)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Using {len(data)} samples for {dataset_type} split")

    return data_loader


def compute_features(args, model, processor, dataset_type):
    dataloader = load_dataset(args, dataset_type)

    features = []
    labels = []

    for idx, batch_data in tqdm(enumerate(dataloader)):
        model_input = processor(batch_data["image"].squeeze().cuda())
        features.append(model(**model_input)["pooler_output"].squeeze().detach().cpu().numpy())
        labels.extend(batch_data["label"])

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
    model = AutoModel.from_pretrained("raidium/curia")
    processor = AutoImageProcessor.from_pretrained("raidium/curia", trust_remote_code=True)

    for dataset_type in ["training", "test"]:
        print(f"Computing features for {dataset_type} split...")
        features, labels = compute_features(args, model, processor, dataset_type)
        np.savez(vars(args)[f"output_features_{dataset_type}"], features=features, labels=labels)


if __name__ == "__main__":
    description = "Curia Feature Extraction"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()

    sys.exit(main(args))
