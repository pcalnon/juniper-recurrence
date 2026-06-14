"""Recurrent / continuous-time memory units for juniper-recurrence.

Currently exposes the Δt-native Legendre Memory Unit (Approach-C). Additional unit
kinds (e.g. a self-recurrent RCC candidate, P1) may be added as the framework grows.
"""

from juniper_recurrence_model.units.lmu_varstep import VariableStepLMUMemory, lmu_matrices

__all__ = ["VariableStepLMUMemory", "lmu_matrices"]
