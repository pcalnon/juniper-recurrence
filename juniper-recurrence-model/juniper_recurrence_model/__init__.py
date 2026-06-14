"""juniper-recurrence-model — the model-specific core for the juniper-recurrence app.

The selected model is **P3-C (LMU + Approach-C)**: a closed-form, variable-Δt Legendre
Memory Unit discretisation (C1-clean, irregular-Δt-native). This package currently ships
the Δt-native memory unit; the recurrent model implementing juniper-model-core's
``TrainableModel`` interface is added when that shared package lands.

See the design of record:
``notes/JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md`` (juniper-ml).
"""

from juniper_recurrence_model._version import __version__
from juniper_recurrence_model.units import VariableStepLMUMemory, lmu_matrices

__all__ = ["__version__", "VariableStepLMUMemory", "lmu_matrices"]
