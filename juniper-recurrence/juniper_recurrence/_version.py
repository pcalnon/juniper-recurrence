"""Single source of truth for the juniper-recurrence application version.

Kept import-free so setuptools can parse ``__version__`` statically at build time
(``[tool.setuptools.dynamic]`` in pyproject.toml) without importing fastapi /
pydantic-settings. This also lets the TestPyPI publish-verify run a clean
``import juniper_recurrence`` (top-level package re-exports only this value).
"""

__version__ = "0.1.0"
