# Cerebras AI Engineer — Model Quality & Performance Challenge

**Live Streamlit URL:** `<add Streamlit Community Cloud URL here>`

This repo contains one Streamlit-deployed project for both required challenge tasks:

1. **Task 1 — Performance Explorer:** customer-facing and internal engineering dashboards for Cerebras performance projection `.xlsx` sweeps.
2. **Task 2 — Benchmark Pruning:** reproducible benchmark-pruning pipeline, validation metrics, universal pruned adapter, AA-LCR pruned registration pattern, and a working MMMU multimodal probe.

## Project structure

```text
cerebras-ai-engineer-challenge/
├── README.md
├── requirements.txt
├── streamlit_app.py
├── app/
│   ├── shared_ui.py
│   ├── task1_dashboard.py
│   └── task2_dashboard.py
├── src/
│   ├── parser.py
│   ├── metrics.py
│   ├── views_customer.py
│   ├── views_engineer.py
│   └── utils.py
├── task2/
│   ├── run_pruning.py
│   ├── load_evals.py
│   ├── pruning_methods.py
│   ├── evaluate_subset.py
│   ├── reporting.py
│   ├── pruning/
│   │   ├── universal_pruned_adapter.py
│   │   ├── sample_selector.py
│   │   ├── difficulty_estimator.py
│   │   └── registry.py
│   ├── mmmu_probe/
│   │   ├── run_mmmu_probe.py
│   │   ├── load_mmmu_samples.py
│   │   ├── image_text_probe.py
│   │   ├── score_probe.py
│   │   └── README_MMMU_Probe.md
│   ├── results/
│   └── docs/task2_methodology.md
├── docs/task1_design.md
└── data/
    ├── perf_data/
    └── Evals/
```

## Setup from clean clone

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Community Cloud deployment

1. Push this repo to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app.
4. Select the GitHub repo.
5. Set the main file path to `streamlit_app.py`.
6. Deploy.
7. Copy the live app URL and place it at the top of this README.

## Task 1 — Performance Explorer

The app preloads the provided default performance sweeps from `data/perf_data/`. Reviewers can open the app and immediately see Task 1 results without uploading anything.

Key behavior:

- Recursively loads all `.xlsx` files from `data/perf_data`.
- Preloads the first 11 model families represented in the provided data, covering Models A–K.
- Upload remains a first-class workflow through the Streamlit uploader.
- Uploaded sweeps are parsed live and merged with the preloaded dataframe.
- Default data is not replaced unless the code is intentionally changed to do so.
- The parser infers `model_id`, `model_name`, and `profile_id` from paths like `Model_A_profile_1/Model A profile 1.xlsx`.
- It is not hardcoded to Models A–K. A future `Model_L_profile_3/Model L profile 3.xlsx` works without code edits.

Customer/PM view focuses on go/caution/no-go recommendations, best model/profile, throughput, TTFT, latency, context length, and readable comparison charts.

Internal engineer view exposes diagnostics: full parsed dataframe, missing values, duplicate rows, outlier detection, distribution charts, column selector, filters, and raw CSV download.

## Task 2 — Benchmark pruning

The command-line pruning pipeline answers whether fewer benchmark samples can preserve model ranking, score accuracy, and benchmark signal.

Run:

```bash
python task2/run_pruning.py --evals_dir data/Evals --keep_ratio 0.3 --output_dir task2/results
```

Try a different keep ratio:

```bash
python task2/run_pruning.py --evals_dir data/Evals --keep_ratio 0.5 --output_dir task2/results
```

Outputs:

- `task2/results/full_scores.csv`
- `task2/results/pruned_scores.csv`
- `task2/results/comparison.csv`
- `task2/results/summary_metrics.csv`
- `task2/results/selected_samples.csv`
- `task2/results/live_code_bench_selected_samples.json`
- `task2/results/aa_lcr_selected_samples.json`
- `task2/results/mmmu_selected_samples.json`
- `task2/results/leave_one_model_out.csv`
- `task2/results/plots/*.png`

## Pruning methodology

The main method is **difficulty-aware stratified pruning**:

1. Compute each sample's average score across models.
2. Define difficulty as `1 - average_score`.
3. Bucket samples into easy, medium, and hard strata.
4. Preserve benchmark/category/difficulty balance when selecting samples.
5. Compare against random pruning and uniform stratified pruning.

Validation metrics include Spearman rank correlation, Kendall tau, mean absolute error, max absolute error, top-1 preservation, pairwise ranking agreement, cost reduction percentage, and leave-one-model-out validation.

## Universal pruned adapter and AA-LCR registration

The pruned adapter is intentionally universal. It wraps a base benchmark, loads selected sample IDs, and filters the original dataset while preserving sample structure.

```python
PrunedBenchmarkAdapter(
    base_benchmark="live_code_bench",
    pruned_name="live_code_bench_pruned",
    selected_sample_ids="task2/results/live_code_bench_selected_samples.json",
)

PrunedBenchmarkAdapter(
    base_benchmark="aa_lcr",
    pruned_name="aa_lcr_pruned",
    selected_sample_ids="task2/results/aa_lcr_selected_samples.json",
)
```

The registration records are defined in `task2/pruning/registry.py`. If integrating into the original evalscope repo, map these records into the repo's existing benchmark registry/decorator/config pattern. The uploaded bundle contained evaluation data, not the full evalscope source tree, so the registration code is isolated and ready to wire into the original registry.

Conceptual commands after repo integration:

```bash
python -m evalscope.run --benchmark live_code_bench_pruned
python -m evalscope.run --benchmark aa_lcr_pruned
```

Use the exact command style from the original evalscope repo once this adapter is placed in that codebase.

## Task 2 Part B — MMMU multimodal probe

This repo includes working MMMU probe code. It is not only a design proposal.

Run:

```bash
python task2/mmmu_probe/run_mmmu_probe.py --evals_dir data/Evals --output_dir task2/results
```

Fast smoke test:

```bash
python task2/mmmu_probe/run_mmmu_probe.py --evals_dir data/Evals --output_dir task2/results --max_samples 200
```

The deterministic probe:

- Loads MMMU review JSONL files.
- Extracts embedded base64 images.
- Computes image size, mode, aspect ratio, brightness, and channel statistics.
- Parses answer choices from the text.
- Produces deterministic predictions from image + text features.
- Scores against labels when available.
- Saves `mmmu_probe_results.csv` and `mmmu_probe_summary.json`.

## Results interpretation

High Spearman/Kendall means the pruned benchmark preserves model ranking. Low MAE means the pruned score remains close to the full benchmark score. Pairwise agreement measures how often pairwise model ordering is unchanged. Cost reduction shows the percentage of benchmark samples removed.

## Known limitations

- The uploaded data bundle did not include the full evalscope source tree, so the adapter and registry are provided as clean integration code rather than a direct patch into an unavailable registry file.
- The MMMU probe is a deterministic baseline, not a full vision-language model inference pipeline. It is designed to prove the multimodal data path works under local and Streamlit constraints.
- Streamlit Community Cloud may not be suitable for committing the full raw `Evals/` dataset if it is very large. Use Git LFS or run Task 2 locally and commit the generated `task2/results` files.

## Interview talking points

### Task 1

**Why Streamlit?** It gives reviewers an immediately usable app with upload, filtering, charts, and deployment in one lightweight Python stack.

**How does preload work?** The app recursively loads default `.xlsx` files from `data/perf_data` on startup and caches the parsed dataframe.

**How does upload work?** Uploaded `.xlsx` files are parsed with the same parser and concatenated with default data.

**How does Model L work?** Model/profile metadata is inferred with regex from path/file names; there is no model-letter whitelist.

**Customer vs engineer view?** Customer view gives decision-ready metrics and recommendations; engineer view exposes raw diagnostics, outliers, missing values, and downloadable data.

### Task 2

**Why difficulty-aware pruning?** It preserves easy, medium, and hard samples, reducing evaluation cost without collapsing the benchmark into only trivial or impossible examples.

**How is difficulty defined?** Difficulty is `1 - average model score` for each sample.

**How did you avoid overfitting?** Leave-one-model-out validation builds the subset without the held-out model and checks held-out score error.

**How do you know the pruned benchmark preserves signal?** Spearman/Kendall, pairwise ranking agreement, top-1 preservation, and MAE compare full and pruned evaluations.

**Why universal adapter?** It prevents benchmark-specific branching and lets LiveCodeBench, AA-LCR, and MMMU share the same filter-by-sample-ID mechanism.

**What does the MMMU probe do?** It runs a real multimodal data path: extracts images, computes image features, combines them with text choices, predicts deterministically, and scores the outputs.
