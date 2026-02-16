#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <repository_root> <command>"
    exit 1
fi

REPO_ROOT="$1"
COMMAND="$2"

cd "$REPO_ROOT" || { echo "Failed to cd into repository root"; exit 1; }

pip3 install h5py

eval "$COMMAND"
