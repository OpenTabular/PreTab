# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from sphinxawesome_theme.postprocess import Icons

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pretab"
project_copyright = "2024-2026, OpenTabular"
author = "OpenTabular"

try:
    version = _version("pretab")
except PackageNotFoundError:
    version = "0+unknown"
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "myst_parser",
    "sphinx_design",
    # "sphinx_copybutton",
]

# Optional dependency imported lazily by the embeddings transformer.
autodoc_mock_imports = ["sentence_transformers"]

# Paths that contain custom templates, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
source_suffix = [".rst", ".md"]

# The root toctree document.
root_doc = "index"

language = "en"

# ``homepage.md`` is included directly into ``index.rst`` and must not be
# treated as a standalone document.
exclude_patterns = ["_build", "_templates", "homepage.md"]

# The reST default role (single back ticks ``dict``) cross links to any code
# object (including Python, but others as well).
default_role = "literal"

add_function_parentheses = True
add_module_names = True

# The name of the Pygments (syntax highlighting) style to use. These are
# provided by the ``accessible-pygments`` package; the bundled ``custom.css``
# fine-tunes the exact colours on top of them.
pygments_style = "github-light"
pygments_style_dark = "github-dark"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "pretab"
html_logo = "logo/pretab-favicon.png"
html_favicon = "logo/pretab-favicon.png"
html_last_updated_fmt = "%Y-%m-%d"
html_show_sourcelink = False

html_theme_options = {
    "show_breadcrumbs": True,
    "show_prev_next": True,
    "show_scrolltop": True,
    "awesome_headerlinks": True,
    "awesome_external_links": False,
    "main_nav_links": {
        "GitHub": "https://github.com/OpenTabular/PreTab",
        "PyPI": "https://pypi.org/project/pretab/",
    },
}

# Use the theme's own permalink icon.
html_permalinks_icon = Icons.permalinks_icon

# -- Options for autodoc -----------------------------------------------------

# Rely on the numpy-style docstrings rather than type annotations for the
# rendered signatures. This keeps the API pages clean and avoids unresolved
# cross-references to third-party types.
autodoc_typehints = "none"

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "inherited-members": False,
    # Exclude scikit-learn metadata-routing boilerplate that is inherited from
    # BaseEstimator / TransformerMixin and is not part of pretab's public API.
    "exclude-members": (
        "set_output,get_metadata_routing,set_fit_request,set_transform_request,set_inverse_transform_request,"
    ),
}

# Generate autosummary stub pages automatically.
autosummary_generate = True

# -- Options for napoleon ----------------------------------------------------

# The codebase mixes Google-style ("Attributes:") and NumPy-style ("Attributes\n
# ----------") docstrings, so enable both parsers.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True

# Render any "Methods:" docstring section as a simple field list instead of
# emitting ``.. method::`` directives, which would duplicate the members that
# autodoc already documents.
napoleon_custom_sections = [("Methods", "params_style")]

# -- Options for MyST parser -------------------------------------------------

myst_enable_extensions = [
    "colon_fence",  # Enable ```{note}, ```{tip}, etc.
    "deflist",  # Definition lists
    "dollarmath",  # LaTeX math with $...$
    "fieldlist",  # Field lists
    "html_admonition",  # HTML admonitions
    "html_image",  # HTML images
    "replacements",  # Text replacements
    "smartquotes",  # Smart quotes
    "strikethrough",  # ~~strikethrough~~
    "substitution",  # Variable substitution
    "tasklist",  # Task lists [ ]
]

# Render fenced ```{note} blocks as sphinx-design admonitions.
myst_fence_as_directive = [
    "note",
    "warning",
    "tip",
    "important",
    "caution",
    "seealso",
]

# Values available to ``{{ ... }}`` substitutions in Markdown pages.
myst_substitutions = {
    "version": release,
}

# Auto-generate anchor slugs for headings (h1-h3) so in-page and cross-page
# links such as ``choosing_a_method.md#when-basis-expansion-does-not-help``
# resolve under the strict (-W) build.
myst_heading_anchors = 3

# -- Options for todo --------------------------------------------------------

todo_include_todos = False
