# SLURM Environment Setup

For running jobs on clusters using SLURM's environment modules, e.g. the CSCS clariden cluster, run the Dockerfile. A script to do so is provided in `scripts/cscs/build_image.sbatch`.

Once the image is built, copy the `finetune.toml` file from `scratch/finetune/slurm/` to your `.edf/` root directory. Make sure to adjust any paths in the `finetune.toml` file as necessary for your environment. Then, in your SLURM job submission scripts, specify the environment using the following directive:

```bash
#SBATCH --environment=finetune
```