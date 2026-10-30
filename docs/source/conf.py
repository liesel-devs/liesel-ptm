# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "liesel-ptm"
copyright = "2025, Johannes Brachem"
author = "Johannes Brachem"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",  # parse NumPy and Google style docstrings
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",  # for automatic API doc tables
    # "sphinx.ext.linkcode",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]


intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "tensorflow_probability": (  # we can use "tfp" as a prefix, too
        "https://www.tensorflow.org/probability/api_docs/python",
        "https://raw.githubusercontent.com/GPflow/tensorflow-intersphinx/master/tfp_py_objects.inv",
    ),
}
intersphinx_timeout = 10

# Napoleon options
napoleon_use_param = True

# For compatibility with sphinx_autodoc_typehints:
# If True, the return text will be rendered as literals.
napoleon_preprocess_types = False

# For compatibility with sphinx_autodoc_typehints:
# If True, Napoleon will add a :rtype: role, causing sphinx_autodoc_typehints
# to not add its own role from the type annotations.
napoleon_use_rtype = False

# sphinx_autodoc_typehints options
typehints_defaults = "braces-after"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to the source directory, that match files and
# directories to ignore when looking for source files. These patterns also
# affect html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

pygments_style = "sphinx"

# The theme to use for HTML and HTML help pages. See the documentation for
# a list of builtin themes.
html_theme = "sphinx_book_theme"
# html_theme = "pydata_sphinx_theme"
html_title = ""
html_logo = "_static/logo-light.png"
html_theme_options = {
    "repository_url": "https://github.com/liesel-devs/liesel-ptm",
    "use_repository_button": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]


# Mock / ignore the following modules.
autodoc_mock_imports = []
autodoc_default_options = {
    "inherited-members": False,  # <- turns inheritance on everywhere
}

# Map functions and classes with the same lowercase names to other filenames.
autosummary_filename_map = {}

# Only document module members that are listed in __all__ (if defined).
autosummary_ignore_module_all = False
