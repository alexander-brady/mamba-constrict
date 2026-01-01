# Finetune Mamba

This repository contains configurations and scripts to finetune large language models, specifically Mamba models, using a novel hidden-state regularization auxiliary loss term.

## Installation

To install the necessary dependencies, we recommend using [uv](https://docs.astral.sh/uv/installation).

```bash
uv sync --all-extras --dev
```

However, if you prefer not to use `uv`, you can manually install the dependencies listed in `pyproject.toml` using pip:

```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv/Scripts/activate 
pip install -e .[dev]
```

Once you have installed the dependencies, you can prepare the Pile dataset using the provided script:

```bash
uv run -m finetune.cli.prepare_data
```


## Quick Start

To finetune a Mamba model on the Pile dataset, you can use the provided script. Here is an example command to start finetuning:

```bash
uv run -m finetune
```

## Configuration

The finetuning process can be customized using the configuration files located in the `configs` directory. You can modify parameters such as learning rate, batch size, and even the model used for finetuning. This repository uses [Hydra](https://hydra.cc/) for configuration management, allowing for easy experimentation with different settings. Thus, you can override any configuration parameter directly from the command line. For example:

```bash
uv run -m finetune trainer.lr=3e-5 model.name=moonshotai/Kimi-K2-Thinking
```

## Logging and Monitoring

We use [Weights & Biases](https://wandb.ai/) for logging and monitoring the finetuning process. Make sure to set up your W&B account and configure the API key before starting the finetuning.

The project name for W&B logging can be set in the configuration file or overridden from the command line:

```bash
uv run -m finetune wandb.project=finetune-mamba
```

## Defining a Custom Loss

Auxiliary losses can be defined in the `finetune/criterion/` directory. To use a custom loss function, specify it in the configuration file by setting the `criterion._target_` parameter to point to your custom loss class, which is instantiated with any required arguments.

The custom loss class should inherit from `finetune.criterion.Criterion` and implement the `compute_loss` method.

For example, if you have a custom loss class `MyCustomLoss` defined in `finetune/criterion/my_custom_loss.py`, you can set it up in the configuration as follows:

```yaml
_target_: finetune.criterion.my_custom_loss.MyCustomLoss
weight: 0.5  # Defined for all criterion classes, default is 1.0
other_param: value
```

## Contributing

Contributions are welcome! Open an issue to discuss ideas or submit a PR directly.

**Guidelines**

Before submitting a commit or pull request, ensure:
```
ruff check . --fix && ruff format .   # Lint & format
mypy .                                # Type check
```

Keep commits reasonably small and clear. We recommend using Conventional Commits.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
