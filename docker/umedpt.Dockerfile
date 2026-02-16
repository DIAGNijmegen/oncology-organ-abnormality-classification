# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

# Dockerfile for the UMedPT foundation model.

FROM dockerdex.umcn.nl:5005/diag/base-images:base-pt2.7.1

ENV PIP_NO_CACHE_DIR=1

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir numpy scipy tqdm monai[all] transformers

RUN python3 -m pip install --no-cache-dir medicalmultitaskmodeling
RUN python3 -m pip install --no-cache-dir torch==2.2

ENV MMM_LICENSE_ACCEPTED='i accept'

ENTRYPOINT ["/bin/bash"]
