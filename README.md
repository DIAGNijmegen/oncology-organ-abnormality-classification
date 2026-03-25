# Systematic Evaluation of Foundation Models for Organ-Level Classification on CT Scans

![Overview Image](assets/overview.png?raw=true)

This EvalBlocks pipeline evaluates five state-of-the-art foundation models in medical imaging on the task of organ abnormality classification. It uses several aggregation methods to learn an organ-level downstream classification task from patch-level embeddings.

## Running the pipeline

To run the evaluation pipeline using EvalBlocks, follow these steps:

### Install Snakemake

Refer to the [Snakemake installation guide](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html) for detailed instructions on how to install Snakemake in your environment.

### Define a cluster configuration

Create a YAML file specifying your cluster settings. This file defines how the pipeline should interact with your compute environment.
An example file for a Slurm cluster that will run up to 12 jobs in parallel could look like this:

```yaml
executor: cluster-generic
cluster-generic-submit-cmd: "sbatch --qos={resources.qos} --cpus-per-task={resources.cpus} --gpus-per-task={resources.gpus} --ntasks=1 --mem={resources.mem_mb} --time={resources.time} --nodes=1 --container-image='{resources.image}' --container-mounts=/data:/data -o ./slurm-logs/slurm-%j.out"
jobs: 12
default-resources:
  - qos=low
  - cpus=1
  - gpus=0
  - mem_mb=4000
  - time=1-00:00:00
```

Place this file in a folder of your choice. You will reference it when running the pipeline.

### Export environment variables

Before running the pipeline, ensure that the necessary environment variables are set.

```bash
export REPOSITORY_ROOT="{Path to where you have cloned this repository}"
export DATASET_ROOT="{Path to where the datasets are located}"
export HF_HOME="{Path to where your Hugging Face home directory is located}"
export OUTPUT_ROOT="{Path to where the pipeline should store its outputs}"
```

If necessary, the pipeline will download model checkpoints. For gated models, make sure that you are authenticated and have been granted read permissions.

### Run the evaluation pipeline

From the root of the repository, run the following command:

```bash
snakemake --profile ./cluster-config-folder/
```

That's it! The pipeline will orchestrate all computation steps, leveraging your cluster resources as specified in the configuration file.
