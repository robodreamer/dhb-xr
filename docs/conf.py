# DHB-XR documentation configuration
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

project = "dhb_xr"
copyright = "2025, Andy Park"
author = "Andy Park"
release = "0.1.0"
version = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]
try:
    import sphinx_autodoc_typehints  # noqa: F401
    extensions.append("sphinx_autodoc_typehints")
except ImportError:
    pass

templates_path = ["_templates"]
exclude_patterns = []
html_theme = "alabaster"
html_static_path = ["_static"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
autodoc_default_options = {"members": True, "undoc-members": True, "show-inheritance": True}
