from setuptools import setup, find_packages

import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='dragonfly-sim',
    version=get_version('dragonfly_sim/__init__.py'),
    author='Sebastiaan van Schie',
    author_email='svanschie@ucsd.edu',
    license='MIT',
    keywords='aerodynamic shape optimization discontinuous Galerkin compressible Euler FEniCSx CSDL',
    url='https://github.com/LSDOlab/DRAGonFLy',
    description='Aerodynamic shape optimization with a discontinuous Galerkin compressible Euler solver',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(include=['dragonfly_sim', 'dragonfly_sim.*']),
    python_requires='>=3.12',
    platforms=['any'],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'jax',
        'pyvista',
        'networkx',
        # LSDOlab packages, installed from the `main` branch of their public repositories
        'csdl_alpha @ git+https://github.com/LSDOlab/CSDL_alpha.git@main',
        'lsdo_function_spaces @ git+https://github.com/LSDOlab/lsdo_function_spaces.git@main',
        'lsdo_geo @ git+https://github.com/LSDOlab/lsdo_geo.git@main',
        'modopt @ git+https://github.com/LSDOlab/modopt.git@main',
        'idwarp_jax @ git+https://github.com/LSDOlab/IDWarp-JAX.git@main',
        # fenics-dolfinx (with mpi4py, petsc4py, ufl and basix) and mpich are not on PyPI;
        # install them with conda (see environment.yml)
    ],
    extras_require={
        'test': ['pytest', 'pytest-cov'],
        'docs': [
            'sphinx',
            'myst-nb',
            'sphinx_rtd_theme',
            'sphinx-copybutton',
            'sphinx-autoapi',
            'numpydoc',
            'gitpython',
            'sphinx-collections',
            'sphinxcontrib-bibtex',
        ],
    },
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)
