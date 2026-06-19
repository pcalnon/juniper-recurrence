"""Single source of truth for the juniper-recurrence-client version.

Kept import-free so setuptools can parse ``__version__`` statically at build time
(``[tool.setuptools.dynamic]`` in pyproject.toml) without importing requests.
"""

__version__ = "0.1.0"
