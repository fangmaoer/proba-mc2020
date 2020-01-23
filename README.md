# A set of Jupyter notebooks as part of the [Master Class in Probability 2020](http://irma.math.unistra.fr/stochastique/mc2020/)

**Authors:** Matthieu Boileau, Sébastien Martineau, Marielle Simon, Dario Trevisan, Xiaolin Zeng

## Installation

### Using Conda

#### Install Anaconda

Follow instructions for installing [Anaconda](https://www.anaconda.com/distribution/#download-section) for python 3.

#### Install dependencies

From project root directory, run:

```bash
conda env create -f environment.yml
```

> **Note:** the `Solving environment` step may be long, be patient...

## Usage

### Activate environment

Before first execution, run:

```bash
conda activate mc2020
```

### Run within Jupyter notebooks

From project root directory, run:

```bash
jupyter-notebook
```

### Launch python scripts from command line

From project root directory, run:

```bash
python srw.py
```

## Test with Binder

[![Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fgitlab.math.unistra.fr%2Fproba-mc2020%2Fproba-mc2020/master)
