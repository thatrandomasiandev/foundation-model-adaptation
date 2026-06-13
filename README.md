# Foundation Model Adaptation

A research benchmark suite for studying **how foundation models adapt to new domains** — covering parameter-efficient fine-tuning (PEFT), adversarial domain adaptation, post-hoc calibration, selective prediction, and training-data attribution. All experiments run on synthetic token-sequence data where causal and spurious feature structure is known by construction, enabling oracle evaluation of shortcut learning and domain shift robustness.

The motivating research question: *when a pretrained model is adapted to a new domain, which adaptation strategies preserve causal features, calibrate uncertainty, and remain consistent under counterfactual perturbations?*

---

## Research scope

| Module | Problem | Methods | Primary metrics |
|--------|---------|---------|-----------------|
| **Adaptation** | Transfer under covariate and correlation shift | Linear probe, full fine-tune, LoRA, DANN | Accuracy, macro-F1, AUROC, domain gap |
| **Evaluation** | Rigorous post-adaptation assessment | Temperature scaling, bootstrap CIs, risk-coverage | ECE, Brier, AURC, counterfactual consistency |
| **Attribution** | Which training examples influenced predictions? | TracIn-style gradient tracing | Influence scores, shortcut attribution |

---

## Synthetic data design

### Domain-shift DGP (`data/domain_shift_dgp.py`)

Labels follow a **causal token rule**:

$$\text{label} = \text{causal\_token} \bmod K$$

The **source domain** injects spurious correlations between label and non-causal tokens. The **target domain** removes spurious shortcuts while shifting covariate markers — mirroring the covariate shift + correlation shift setting studied in Ben-David et al. (2010) and Arjovsky et al. (2019).

### Counterfactual evaluation DGP (`data/counterfactual_dgp.py`)

Training data exploits spurious shortcuts. At test time, **counterfactual examples** swap spurious tokens while holding causal tokens fixed. Models that memorize shortcuts exhibit a large **shortcut gap** — the accuracy drop under counterfactual perturbation. This follows the causal invariance testing framework of Veitch et al. (2021) and the spurious correlation literature (Geirhos et al., 2020).

---

## Module 1: Adaptation methods

### Model backbone

A small `TransformerEncoder` (`models/transformer.py`) serves as a controlled foundation-model stand-in. The architecture is intentionally compact so adaptation dynamics can be studied without confounding from scale, while remaining structurally faithful to the pretrain-then-adapt paradigm (Devlin et al., 2019; Brown et al., 2020).

### Implemented methods

| Method | Description | Reference |
|--------|-------------|-----------|
| **Linear probe** | Freeze encoder; train linear head on target labels | Standard transfer learning baseline |
| **Full fine-tune** | Update all parameters on target domain | Yosinski et al. (2014) |
| **LoRA** | Low-rank adaptation of attention projections | Hu et al. (2022) |
| **DANN** | Domain-adversarial training with gradient reversal | Ganin et al. (2016) |

LoRA injects trainable low-rank matrices into attention layers (`models/peft.py`), reducing trainable parameters by orders of magnitude while preserving representational capacity — the core insight of Hu et al. (2022).

DANN minimizes task loss while an adversarial domain classifier cannot distinguish source from target representations, encouraging **domain-invariant features** (Ganin et al., 2016).

---

## Module 2: Rigorous evaluation

### Calibration

**Temperature scaling** (Guo et al., 2017) fits a single scalar T on a held-out validation set to minimize NLL, correcting overconfident softmax outputs without retraining.

Metrics: **ECE** (expected calibration error), **NLL**, **Brier score**.

### Selective prediction

**Risk-coverage curves** (Geifman & El-Yaniv, 2017) measure accuracy as a function of the fraction of examples the model abstains on (low-confidence predictions). **AURC** (area under risk-coverage) summarizes selective classification quality.

### Counterfactual consistency

Accuracy on counterfactual test sets where spurious features are perturbed. A large gap between standard and counterfactual accuracy indicates reliance on non-causal features.

### Bootstrap confidence intervals

Nonparametric bootstrap over test-set predictions provides uncertainty bounds on accuracy and F1 (Efron & Tibshirani, 1993).

---

## Module 3: Training-data attribution

**TracIn** (Pruthi et al., 2020) estimates the influence of training example z on test prediction ŷ by tracing gradient dot-products across training checkpoints:

$$\text{TracIn}(z, \hat{y}) = \sum_{t} \eta_t \nabla_\theta \ell(z, \theta_t) \cdot \nabla_\theta \ell(\hat{x}, \hat{y}, \theta_t)$$

Implemented approximately in `attribution/influence.py` with a small number of checkpoints for computational tractability.

---

## Benchmark protocol

```bash
pip install -e ".[dev]"
pytest

python scripts/run_benchmark.py --config configs/adaptation_benchmark.yaml --module adaptation
python scripts/run_benchmark.py --config configs/eval_benchmark.yaml --module eval
python scripts/run_benchmark.py --config configs/attribution_benchmark.yaml --module attribution
```

Configs sweep adaptation methods, domain shift levels, and calibration settings. Results: `results/{timestamp}/metrics.json` + `summary.md`.

---

## Project layout

```
src/fm_adapt/
├── data/           # Domain-shift and counterfactual DGPs
├── models/         # Transformer encoder, LoRA injection
├── adaptation/     # Full fine-tune, linear probe, LoRA, DANN
├── eval/           # Calibration, bootstrap, risk-coverage, counterfactual
├── attribution/    # TracIn-style influence tracing
└── evaluation/     # Benchmark runner and reporting
```

---

## Implementation notes

- The transformer is a **stand-in** for large FMs; conclusions about scale-specific phenomena (emergent abilities, in-context learning) require HuggingFace backbone integration
- DANN applies gradient reversal on **pooled** sequence representations, not per-token
- TracIn uses few checkpoints and small batches — suitable for relative ranking, not exact influence recovery

---

## References

- Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2019). Invariant risk minimization. [arXiv](https://arxiv.org/abs/1907.02893)
- Ben-David, S., Blitzer, J., Crammer, K., & Pereira, F. (2010). Analysis of representations for domain adaptation. *NeurIPS*. [Proceedings](https://papers.nips.cc/paper/2980-analysis-of-representations-for-domain-adaptation)
- Brown, T., et al. (2020). Language models are few-shot learners. *NeurIPS*. [arXiv](https://arxiv.org/abs/2005.14165)
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL*. [arXiv](https://arxiv.org/abs/1810.04805)
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. [DOI](https://doi.org/10.1201/9780429246593)
- Ganin, Y., et al. (2016). Domain-adversarial training of neural networks. *JMLR*, 17(59), 1–35. [Paper](https://jmlr.org/papers/v17/15-239.html)
- Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS*. [arXiv](https://arxiv.org/abs/1705.08500)
- Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2, 665–673. [DOI](https://doi.org/10.1038/s42256-020-00257-z)
- Guo, C., et al. (2017). On calibration of modern neural networks. *ICML*. [arXiv](https://arxiv.org/abs/1706.04599)
- Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR*. [arXiv](https://arxiv.org/abs/2106.09685)
- Pruthi, G., et al. (2020). Estimating training data influence by tracing gradient descent. *NeurIPS*. [arXiv](https://arxiv.org/abs/2002.08484)
- Veitch, V., D'Amour, A., Yadlowsky, S., & Eisenstein, J. (2021). Counterfactual invariance to spurious correlations. [arXiv](https://arxiv.org/abs/1911.08714)
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? *NeurIPS*. [arXiv](https://arxiv.org/abs/1411.1792)

---

## Future work

- HuggingFace integration (BERT, Llama + LoRA/QLoRA)
- WILDS benchmark adapters (Koh et al., 2021)
- Influence functions with Hessian approximation (Koh & Liang, 2017)
