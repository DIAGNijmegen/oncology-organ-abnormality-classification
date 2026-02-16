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
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Transform,
    ResizeWithPadOrCropd,
    NormalizeIntensityd,
)
from tqdm import tqdm

from transformers import AutoConfig
from transformers.models.auto.auto_factory import get_class_from_dynamic_module
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

from util.util import fix_random_seeds
from .feature_model_util import get_args_parser


def load_model():
    repo_id = "fomofo/tap-ct-b-3d"

    config = AutoConfig.from_pretrained(
        repo_id,
        trust_remote_code=True,
    )

    model_cls = get_class_from_dynamic_module(
        class_reference="modeling_tapct.TAPCTModel",
        pretrained_model_name_or_path=repo_id,
    )
    model = model_cls(config)

    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
    )

    state_dict = load_safetensors(ckpt_path)
    model.load_state_dict(state_dict, strict=True)

    model.cuda().eval()
    return model


def load_dataset(args, dataset_type):
    preprocess = Compose(
        [
            LoadImaged(
                keys=["image"], ensure_channel_first=True, image_only=True
            ),
            EnsureTyped(keys=["image"]),
            EnsureTyped(keys=["label"]),
            Orientationd(keys=["image"], axcodes="LPS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1008,
                a_max=822,
                b_min=-1008,
                b_max=822,
                clip=True,
            ),
            NormalizeIntensityd(
                keys=["image"],
                subtrahend=-86.8086,
                divisor=322.6347,
            ),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=(224, 224, 16)),
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
                model(dataset[i]['image'].permute(0, 3, 1, 2).unsqueeze(0).cuda())["pooler_output"]
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
