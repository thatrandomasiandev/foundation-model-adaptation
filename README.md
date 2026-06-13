# Foundation Model Adaptation

PhD-level research suite for **fine-tuning**, **parameter-efficient adaptation (PEFT)**, **domain adaptation**, and **rigorous evaluation** — all on synthetic data with known ground truth.

## Modules

| Module | Purpose |
|--------|---------|
| **Data** | Domain-shift and counterfactual DGPs with causal vs spurious token structure |
| **Models** | Small transformer encoder (controlled "foundation model") |
| **Adaptation** | Full fine-tune, linear probe, LoRA, DANN |
| **Eval** | Accuracy/F1/AUROC, ECE, temperature scaling, bootstrap CIs, risk-coverage, counterfactual consistency |
| **Attribution** | TracIn-style gradient influence scores |

## Quick start

```bash
cd 02-foundation-model-adaptation
pip install -e ".[dev]"
pytest
python scripts/run_benchmark.py --config configs/adaptation_benchmark.yaml --module adaptation
python scripts/run_benchmark.py --config configs/eval_benchmark.yaml --module eval
```

## Synthetic DGP assumptions

**Domain shift:** Labels follow a causal token rule (`label = causal_token mod n_classes`). Source domain injects spurious correlations; target domain removes them while shifting covariate markers.

**Counterfactual eval:** Training data exploits spurious shortcuts. Counterfactual test swaps spurious tokens while holding causal tokens fixed — models that memorize shortcuts show a large `shortcut_gap`.

## Benchmark outputs

Results land in `results/{timestamp}/`:
- `metrics.json` — seed-averaged metrics
- `summary.md` — markdown table report

## Identification & limitations (v1)

- Assumes token-level causal/spurious structure is known in synthetic data (for oracle eval only)
- Small transformer stand-in for large FMs — architecture is swappable
- DANN uses gradient reversal on pooled representations
- TracIn attribution is approximate (few checkpoints, small data)

## Future extensions

- HuggingFace backbone integration (BERT, Llama + LoRA)
- Real domain adaptation datasets (Amazon reviews, WILDS)
- Data contamination detection for eval splits
- Influence functions with Hessian approximation
