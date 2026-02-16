# EvalBlocks: A Modular Pipeline for Rapidly Evaluating Foundation Models in Medical Imaging

![Overview Image](assets/overview.png?raw=true)

Training foundation models in medical imaging requires continuous monitoring of downstream performance. Researchers must track numerous experiments, design choices, and their effects on performance, often relying on ad-hoc, manual workflows that are slow and error-prone.

EvalBlocks streamlines this process by enabling fast, modular, and transparent evaluation workflows of medical imaging foundation models.

## Running the pipeline

To run the demonstration evaluation pipeline using EvalBlocks, follow these steps:

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

In EvalBlocks, batteries are included: Foundation models are shipped with inference code with the author's recommended preprocessing steps, running via provided Docker images that include all dependencies.

### Export environment variables

Before running the pipeline, ensure that the necessary environment variables are set.

```bash
export REPOSITORY_ROOT="{Path to where you have cloned the EvalBlocks repository}"
export DATASET_ROOT="{Path to where your datasets are located}"
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

## Customizing the pipeline

The evaluation pipeline is designed to be modular and easily customizable. You can modify various components to suit your specific needs.

### Changing experiments

You can customize the experiments by modifying the `experiments.json` file located in the root directory. This file defines which foundation models, aggregation models, and evaluation strategies to use. You can add or remove entries as needed to tailor the experiments to your requirements.

### Adding new blocks

You can add a new foundation model, aggregation model or evaluation strategy by adding a new block in the associated directory. For example, to add a new foundation model, add a new rule to `featuremodels/Snakefile` and create a new Python file in the `featuremodels/scripts` folder that transforms the given input files into feature embeddings in the required format.

## Citation

If you use EvalBlocks for your research, please cite the [arXiv preprint](https://arxiv.org/abs/2601.03811):
```
@article{tagscherer2026evalblocksmodularpipelinerapidly,
  title={EvalBlocks: A Modular Pipeline for Rapidly Evaluating Foundation Models in Medical Imaging}, 
  author={Jan Tagscherer and Sarah de Boer and Lena Philipp and Fennie van der Graaf and Dré Peeters and Joeran Bosma and Lars Leijten and Bogdan Obreja and Ewoud Smit and Alessa Hering},
  year={2026},
  eprint={2601.03811},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2601.03811}, 
  note={Accepted at BVM 2026},
}
```
