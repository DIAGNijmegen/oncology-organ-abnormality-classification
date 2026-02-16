# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import json
import os
import sys

import monai
import numpy as np
import torch
import torch.nn as nn
from mmm.labelstudio_ext.NativeBlocks import NativeBlocks, MMM_MODELS, DEFAULT_MODEL
from monai.transforms import (
    Compose,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    EnsureChannelFirstd,
    Spacingd,
    ResizeWithPadOrCropd,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.util import fix_random_seeds
from .feature_model_util import get_args_parser


def load_dataset(args, dataset_type):
    preprocess = Compose(
        [
            LoadImaged(keys=["image"], reader="itkreader", image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=(224, 224, 3)),
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

    print(f"Using {len(data)} samples for {dataset_type} split")
    dataset = monai.data.Dataset(data=data, transform=preprocess)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
    )

    return data_loader


def compute_features(args, model, dataset_type):
    dataloader = load_dataset(args, dataset_type)

    features = []
    labels = []

    with torch.inference_mode():
        for idx, batch_data in tqdm(enumerate(dataloader)):
            model_input = batch_data["image"].squeeze().unsqueeze(0).permute(0, 3, 1, 2).cuda()
            feature_pyramid = model["encoder"](model_input.to(model.device))
            hidden_vector = nn.Flatten(1)(model["squeezer"](feature_pyramid)[1])
            features.append(hidden_vector.squeeze().detach().cpu().numpy())
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
    model = NativeBlocks(MMM_MODELS[DEFAULT_MODEL], device_identifier="cuda:0")

    for dataset_type in ["training", "test"]:
        print(f"Computing features for {dataset_type} split...")
        features, labels = compute_features(args, model, dataset_type)
        np.savez(vars(args)[f"output_features_{dataset_type}"], features=features, labels=labels)


if __name__ == "__main__":
    description = "UMedPT Feature Extraction"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()

    sys.exit(main(args))
