"""Single source of truth for the juniper-recurrence-model version.

Kept import-free so setuptools can parse ``__version__`` statically at build
time (``[tool.setuptools.dynamic]`` in pyproject.toml) without importing numpy.
"""

__version__ = "0.1.0a0"
