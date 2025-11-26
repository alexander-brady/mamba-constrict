# How to run LongBench Evaluation

First, add the model paths and maximum context lengths to the configuration files located at `eval/LongBench/config/model2path.json` and `eval/LongBench/config/model2maxlen.json`, respectively.

Then, you can submit the evaluation job using the following command:

```bash
sbatch scripts/eval_longbench.sh
```

Happy evaluating!