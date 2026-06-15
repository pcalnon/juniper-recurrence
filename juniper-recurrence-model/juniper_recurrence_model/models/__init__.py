"""Trainable recurrent models for juniper-recurrence.

Exposes the fixed-order Δt-native LMU regressor (P3-C / Approach-C) that implements
juniper-model-core's :class:`~juniper_model_core.interfaces.TrainableModel` contract, plus
its :class:`~juniper_model_core.serialization.ModelSerializer` strategy. The LMU *memory* is
fixed (closed-form LegT matrices, never trained); only the linear readout is fit, in closed
form via least squares — fully deterministic, no BPTT.
"""

from juniper_recurrence_model.models.lmu_regressor import FixedOrderLMURegressor, LMURegressorSerializer

__all__ = ["FixedOrderLMURegressor", "LMURegressorSerializer"]
