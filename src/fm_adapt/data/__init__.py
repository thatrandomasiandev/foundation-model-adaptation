"""Synthetic data generators for adaptation experiments."""

from fm_adapt.data.base import AdaptationDataset, CounterfactualBundle, DomainShiftBundle
from fm_adapt.data.counterfactual_dgp import CounterfactualDGPConfig, generate_counterfactual_data
from fm_adapt.data.domain_shift_dgp import DomainShiftDGPConfig, generate_domain_shift_data
from fm_adapt.data.tokenizer import TokenVocab

__all__ = [
    "AdaptationDataset",
    "CounterfactualBundle",
    "CounterfactualDGPConfig",
    "DomainShiftBundle",
    "DomainShiftDGPConfig",
    "TokenVocab",
    "generate_counterfactual_data",
    "generate_domain_shift_data",
]
