# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath('..'))
import matplotlib
import orbit

# -- Project information -----------------------------------------------------

project = 'orbit'
copyright = '2020, Uber Technologies, Inc'
author = 'Edwin Ng, Steve Yang, Zhishi Wang, Yifeng Wu, Jing Pan'

# The short X.Y version.
version = orbit.__version__
# The full version, including alpha/beta/rc tags
release = orbit.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    # 'sphinx.ext.autosummary',
    # 'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    # 'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'nbsphinx',
]

# autodoc_mock_imports = [
#     'pandas', 'torch', 'pystan', 'tqdm', 'matplotlib.pyplot'
#     'pyro', 'matplotlib', 'seaborn', 'scipy', 'sklearn']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_context = {
  'display_github': True,
  'github_user': 'uber',
  'github_repo': 'orbit',
  'github_version': 'master/docs/',
}

# logo
html_logo = 'img/orbit-logo-black.png'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    # 'logo_only': False,
    'navigation_depth': 1,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_style = 'css/orbit.css'


html_favicon = 'img/favicon/favicon.ico'

napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

nbsphinx_kernel_name = 'python3'
