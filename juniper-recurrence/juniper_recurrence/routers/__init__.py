"""API routers for the juniper-recurrence service (plan §6).

Each router is mounted by :func:`juniper_recurrence.app.build_app` via
``create_app(routers=...)``. All routes are regression-generic (RK-6) and protected by
the app's ``SecurityMiddleware`` (health / docs stay exempt) — no per-route auth needed.
"""

from juniper_recurrence.routers.dataset import router as dataset_router
from juniper_recurrence.routers.model import router as model_router
from juniper_recurrence.routers.predict import router as predict_router
from juniper_recurrence.routers.training import router as training_router

__all__ = ["training_router", "predict_router", "model_router", "dataset_router"]
