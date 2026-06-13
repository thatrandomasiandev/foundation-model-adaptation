"""Model components: transformers, PEFT layers, and ensembles."""

from fm_adapt.models.ensemble import (
    ModelEnsemble,
    UncertaintyDecomposition,
    ensemble_predict,
    ensemble_uncertainty,
)
from fm_adapt.models.peft import (
    AdapterLayer,
    LoRAConfig,
    LoRALinear,
    PrefixLayer,
    count_trainable_params,
    freeze_backbone,
    inject_lora,
    trainable_parameter_count,
    unfreeze_all,
)
from fm_adapt.models.transformer import TransformerEncoder

__all__ = [
    "AdapterLayer",
    "LoRAConfig",
    "LoRALinear",
    "ModelEnsemble",
    "PrefixLayer",
    "TransformerEncoder",
    "UncertaintyDecomposition",
    "count_trainable_params",
    "ensemble_predict",
    "ensemble_uncertainty",
    "freeze_backbone",
    "inject_lora",
    "trainable_parameter_count",
    "unfreeze_all",
]
