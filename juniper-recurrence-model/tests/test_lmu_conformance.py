"""Run juniper-model-core's conformance kit against :class:`FixedOrderLMURegressor`.

This proves the WS-4 refactor template: a non-cascor model (the fixed-order Δt-native LMU
regressor) plugs into the shared ``TrainableModel`` seam and passes every contract check
unchanged. The kit's base class is deliberately not named ``Test*``, so pytest collects only
this concrete subclass; it runs ~10 contract assertions (isinstance, task_type, fit-returns-
TrainResult, fit/predict/metrics round-trip, predict output shape, metric keys, the RK-6
no-classification-assumptions guard, renderable topology, legal event ordering, and a lossless
serialization round-trip).
"""

from __future__ import annotations

from juniper_model_core.conformance import TrainableModelConformance, tiny_regression_3d

from juniper_recurrence_model.models.lmu_regressor import FixedOrderLMURegressor, LMURegressorSerializer


class TestLMUConformance(TrainableModelConformance):
    def make_model(self):
        return FixedOrderLMURegressor(d=6)

    def make_dataset(self):
        return tiny_regression_3d()

    def make_serializer(self):
        return LMURegressorSerializer()
