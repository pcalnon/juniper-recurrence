"""juniper-model-core conformance kit, run against the LMU regressor (WS-4, PR-2).

Subclasses the kit's :class:`TrainableModelConformance` and supplies the three factories;
pytest then runs every contract assertion against ``LMURegressor`` — ``isinstance``, the
fit/predict/metrics round-trip, predict output shape, regression-only metric keys + the RK-6
no-``accuracy`` guard, a renderable topology, legal training-event order, and a lossless
serializer round-trip. This is the model-core "installable kit + thin per-repo wrapper"
resolution (OQ-12): the kit is imported code, not a pytest plugin.
"""

from __future__ import annotations

from juniper_model_core.conformance import TrainableModelConformance, tiny_regression_3d

from juniper_recurrence_model import LMURegressor, LMUSerializer, RFFReadoutSpec


class TestLMUConformance(TrainableModelConformance):
    """Drive ``LMURegressor`` (default linear readout) through the full ``TrainableModel`` contract."""

    def make_model(self):
        return LMURegressor(d=16, theta=30.0)

    def make_dataset(self):
        return tiny_regression_3d()

    def make_serializer(self):
        return LMUSerializer()


class TestRFFLMUConformance(TrainableModelConformance):
    """Drive the LMU regressor with the **RFF nonlinear readout** through the full contract (P2).

    The check that matters here is the **bit-exact lossless serialization round-trip**: ``φ = cos`` of
    a recomputed-from-``d``/θ memory matmul is *not* trivially lossless, so this subclass gates it
    (and the finite-prediction property — a NaN would fail ``np.array_equal``) rather than assuming it.
    """

    def make_model(self):
        return LMURegressor(d=16, theta=30.0, readout=RFFReadoutSpec(n_features_out=32, ridge="gcv"))

    def make_dataset(self):
        return tiny_regression_3d()

    def make_serializer(self):
        return LMUSerializer()
