# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import argparse
from typing import Optional


def get_args_parser(
        description: Optional[str] = None,
        add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        add_help=add_help,
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    parser.add_argument(
        "--output-features-training",
        default="",
        type=str,
        help="Output file to write training features",
    )
    parser.add_argument(
        "--output-features-test",
        default="",
        type=str,
        help="Output file to write test features",
    )
    parser.add_argument(
        "--entries-file",
        type=str,
        help="JSON file with entries located under the dataset root",
    )
    parser.add_argument(
        "--modality",
        default="",
        type=str,
        help="Modality of the input patches",
    )
    return parser
