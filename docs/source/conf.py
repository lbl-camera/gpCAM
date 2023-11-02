from gpcam import _version

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'gpCAM'

copyright = '2021, Marcus Michael Noack'
author = 'Marcus Michael Noack'
version = _version.get_versions()['version']


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_nb',
    'sphinx_panels',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'hoverxref.extension'
]
#if notebooks should not be executed:
nb_execution_mode='off'

# MyST extensions
myst_enable_extensions = ['colon_fence']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
exclude_patterns = []

# hoverxref
hoverxref_role_types = {
    'hoverxref': 'modal',
    'ref': 'modal',  # for hoverxref_auto_ref config
    'confval': 'tooltip',  # for custom object
    'mod': 'tooltip',  # for Python Sphinx Domain
    'class': 'tooltip',  # for Python Sphinx Domain
}
hoverxref_auto_ref = True
hoverxref_intersphinx = ['fvgp', 'hgdl']

# Support links to partnered pacakges
intersphinx_mapping = {
    "fvgp": ("https://fvgp.readthedocs.io/en/latest/", None),
    "hgdl": ("https://hgdl.readthedocs.io/en/latest/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
}



# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_style = 'custom.css'

html_static_path = ['_static']

html_logo = '_static/gpCAM_dark_bg.png'

html_theme_display_version = True

html_theme_options = dict(
    logo_only=True,
    display_version=True,
    collapse_navigation=False,
    titles_only=False
)

autoclass_content = 'both'
