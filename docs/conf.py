# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Spine Image Registration'
copyright = '2025, Andrew Bennecke'
author = 'Andrew Bennecke and Daniel Tward'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinxcontrib.htmlhelp'
]

# Jupyter Notebook configurations
nbsphinx_execute = 'never'
nbsphinx_execute_arguments = ["--InlineBackend.figure_formats={'svg', 'pdf'}",]

# Configuration for LaTeX output
latex_documents = [
    ('index', 'spine_registration.tex', project, author, 'manual'),
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# import sys
# import os
# from pathlib import Path

# sys.path.append('/home/runner/work/spine_registration/spine_registration/docs')
# sys.path.append('/home/runner/work/spine_registration/spine_registration/docs/notebooks')
# sys.path.append('/home/runner/work/spine_registration/spine_registration/docs/scripts')
# sys.path.insert(0, os.path.abspath("../source/"))
# sys.path.insert(0, os.path.abspath("../source/notebooks"))
# sys.path.insert(0, os.path.abspath("../source/scripts"))

# sys.path.insert(0, str(Path('..', 'source').resolve()))
# sys.path.insert(0, str(Path('..', 'source', 'notebooks').resolve()))
# sys.path.insert(0, str(Path('..', 'source', 'scripts').resolve()))


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'nature'

html_static_path = ['docs/_static']
