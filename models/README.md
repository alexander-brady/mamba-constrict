# Models Directory

This directory contains all models for evaluation. The evaluation scripts automatically discover models from this directory.

## Structure

```
models/
├── README.md                      # This file
├── huggingface_models.txt        # List of HuggingFace model paths
├── my_finetuned_model_1/         # Local finetuned model
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
├── my_finetuned_model_2/         # Another local model
│   ├── config.json
│   ├── model.safetensors
│   └── ...
└── ...
```

## Adding Models

### HuggingFace Models

Add HuggingFace model paths to `huggingface_models.txt`:

```bash
# Edit the file
nano models/huggingface_models.txt

# Add one model path per line (lines starting with # are comments)
state-spaces/mamba-130m-hf
state-spaces/mamba-370m-hf
state-spaces/mamba-790m-hf
```

### Local Finetuned Models

Place your finetuned models directly in this directory:

```bash
# Copy or symlink your finetuned model
cp -r /path/to/my_finetuned_model models/
# Or symlink
ln -s /path/to/my_finetuned_model models/my_finetuned_model

# The model directory must contain:
# - config.json (HuggingFace model config)
# - pytorch_model.bin or *.safetensors (model weights)
```

## Listing Available Models

To see all discovered models:

```bash
python3 eval/model_utils.py
```

Example output:
```
Found 3 model(s):

HuggingFace Models:
  mamba-130m-hf                  -> state-spaces/mamba-130m-hf
  mamba-370m-hf                  -> state-spaces/mamba-370m-hf

Local Models:
  my_finetuned_model_1          -> /path/to/models/my_finetuned_model_1
```

## How Evaluation Works

All evaluation scripts automatically discover models from this directory:

```bash
# Run evaluation on all models
sbatch scripts/eval/babilong.sh
sbatch scripts/eval/eval_passkey.sh
sbatch scripts/eval/eval_pg19.sh
sbatch scripts/eval/longbench.sh
```

The scripts will:
1. Discover all models in this directory
2. Run evaluation on each model
3. Save results to `results/{benchmark_name}/{model_name}/`

## Notes

- Model names are derived from:
  - HuggingFace: Last part of path (e.g., `state-spaces/mamba-130m-hf` → `mamba-130m-hf`)
  - Local: Directory name (e.g., `models/my_model/` → `my_model`)
- Ensure model names are unique to avoid result collisions
- Hidden directories (starting with `.`) are ignored
