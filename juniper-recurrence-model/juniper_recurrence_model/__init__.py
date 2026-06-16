"""juniper-recurrence-model — the model-specific core for the juniper-recurrence app.

The selected model is **P3-C (LMU + Approach-C)**: a closed-form, variable-Δt Legendre
Memory Unit discretisation (C1-clean, irregular-Δt-native). This package ships the Δt-native
memory unit (:class:`VariableStepLMUMemory`), the fixed-order LMU regressor
(:class:`LMURegressor`) implementing juniper-model-core's ``TrainableModel`` interface, and a
lean loader (:func:`load_sequence_npz`) for the WS-1 3-D sequence NPZ contract.

See the design of record ``notes/JUNIPER_RECURRENCE_MODEL_DETAILED_DESIGN_2026-06-14.md`` and
the WS-4 build plan ``notes/JUNIPER_RECURRENCE_WS4_MODEL_BUILD_PLAN_2026-06-15.md`` (juniper-ml).
"""

from juniper_recurrence_model._version import __version__
from juniper_recurrence_model.data import SequenceData, load_sequence_npz, sequence_data_from_arrays
from juniper_recurrence_model.model import LMURegressor, LMUSerializer
from juniper_recurrence_model.units import VariableStepLMUMemory, lmu_matrices

__all__ = [
    "__version__",
    "LMURegressor",
    "LMUSerializer",
    "SequenceData",
    "load_sequence_npz",
    "sequence_data_from_arrays",
    "VariableStepLMUMemory",
    "lmu_matrices",
]
