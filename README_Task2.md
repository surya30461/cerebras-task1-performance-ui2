# Task 2 — Benchmark Pruning and MMMU Probe

Run pruning:

```bash
python task2/run_pruning.py --evals_dir data/Evals --keep_ratio 0.3 --output_dir task2/results
```

Run MMMU probe:

```bash
python task2/mmmu_probe/run_mmmu_probe.py --evals_dir data/Evals --output_dir task2/results
```

The main pruning method is difficulty-aware stratified pruning. The universal adapter and registry pattern live under `task2/pruning/`.
