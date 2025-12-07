# How to run benchmarks

First, add the model paths and maximum context lengths to the configuration files located at `eval/config/model2path.json` and `eval/config/model2maxlen.json`, respectively.

Then, you can submit the evaluation job (which will run all benchmarks) using the following command:

```bash
sbatch scripts/eval/run_all.sh
```

## Available Benchmarks

- **LongBench**: Long-context understanding tasks (uses vLLM)
- **Babilong**: Long-context question answering (uses vLLM)
- **Passkey Retrieval**: Information retrieval from long contexts (uses vLLM)
- **PG19 Perplexity**: Perplexity evaluation on PG19 test set (uses PyTorch)

## Running Individual Benchmarks

You can also run individual benchmarks:

```bash
# LongBench
sbatch scripts/eval/longbench.sh

# Babilong
sbatch scripts/eval/babilong.sh

# Passkey Retrieval
sbatch scripts/eval/eval_passkey.sh

# PG19 Perplexity
sbatch scripts/eval/eval_pg19.sh
```

See each benchmark's directory for more detailed documentation.

Happy evaluating!