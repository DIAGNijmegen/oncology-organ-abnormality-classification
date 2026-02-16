# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

# Dockerfile for the CT-FM foundation model.

FROM dockerdex.umcn.nl:5005/diag/base-images:base-pt2.7.1

ENV PIP_NO_CACHE_DIR=1

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir numpy scipy tqdm torch monai[all] transformers

RUN python3 -m pip install --no-cache-dir lighter-zoo

ENTRYPOINT ["/bin/bash"]
