# How to run benchmarks

First, add the model paths and maximum context lengths to the configuration files located at `eval/config/model2path.json` and `eval/config/model2maxlen.json`, respectively.

Then, you can submit the evaluation job (which will run all benchmarks) using the following command:

```bash
sbatch scripts/run_all_evals.sh
```

Happy evaluating!