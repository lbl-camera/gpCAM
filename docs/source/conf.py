import shutil
from pathlib import Path

from gpcam import __version__


def _sync_example_notebooks(app):
    """Copy every *.ipynb from the repo ``examples/`` dir into
    ``docs/source/examples/`` so Sphinx can find them.

    ``docs/source/examples/*.ipynb`` is gitignored — the authoritative copies
    live in ``examples/``.  Running this at every Sphinx build keeps the docs
    in sync, both locally (``make html``) and on ReadTheDocs (in addition to
    the pre_build ``cp`` step in ``.readthedocs.yml``).

    We use ``shutil.copy`` (not ``copy2``) so the destination mtime is updated
    to ``now`` — this guarantees Sphinx's incremental build sees the file as
    changed and re-renders it instead of using a cached ``.doctree``.
    """
    src = Path(app.srcdir).parent.parent / "examples"
    dst = Path(app.srcdir) / "examples"
    if not src.is_dir():
        print(f"[conf.py] WARNING: examples source dir not found: {src}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for nb in sorted(src.glob("*.ipynb")):
        shutil.copy(nb, dst / nb.name)
        copied.append(nb.name)
    print(f"[conf.py] Synced {len(copied)} notebook(s) from {src} -> {dst}: {copied}")


def setup(app):
    app.connect("builder-inited", _sync_example_notebooks)


# -- Project information -----------------------------------------------------
project = 'gpCAM'
copyright = '2021, Marcus Michael Noack'
author = 'Marcus Michael Noack'
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_nb',
    'sphinx_design',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
]

nb_execution_mode = 'off'

myst_enable_extensions = ['colon_fence', 'dollarmath', 'amsmath']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'fvgp': ('https://fvgp.readthedocs.io/en/latest/', None),
    'hgdl': ('https://hgdl.readthedocs.io/en/latest/', None),
    'dask': ('https://docs.dask.org/en/stable/', None),
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'

# Force light mode regardless of the visitor's OS preference. pydata-sphinx-theme
# sets data-theme="light" on <html> when default_mode == "light", and the navbar
# theme-switcher is excluded via navbar_end so visitors can't toggle to dark.
html_context = {'default_mode': 'light'}

html_theme_options = {
    'logo': {
        'image_light': '_static/gpCAM_bright_bg.png',
        'image_dark': '_static/gpCAM_dark_bg.png',
        'alt_text': 'gpCAM',
    },
    'github_url': 'https://github.com/lbl-camera/gpcam',
    'navbar_start': ['navbar-logo'],
    'navbar_center': ['navbar-nav'],
    'navbar_end': ['navbar-icon-links'],
    'secondary_sidebar_items': ['page-toc'],
    'footer_start': ['copyright'],
    'footer_end': [],
    'announcement': (
        'gpCAM 8.4.0 is a <strong>beta release</strong> with a few kwarg renames '
        'from 8.3.x. If you hit issues, pin <code>gpcam==8.3.9</code> &mdash; see '
        'the <a href="https://github.com/lbl-camera/gpCAM/blob/master/HISTORY.rst">'
        'migration notes</a>.'
    ),
}

html_static_path = ['_static']
html_css_files = ['custom.css']

autodoc_member_order = 'bysource'
autoclass_content = 'both'
