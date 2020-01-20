# A set of Jupyter notebooks as part of the [Master Class in Probability 2020](http://irma.math.unistra.fr/stochastique/mc2020/)

**Authors:** Matthieu Boileau, Sébastien Martineau, Marielle Simon, Dario Trevisan, Xiaolin Zeng

## Installation

### Install pip

Follow the [instructions](https://pip.pypa.io/en/stable/installing/).

### install `virtualenv`

```bash
pip install --user virtualenv
```

### Install dependencies

Create virtual environment:

```bash
virtualenv .env
```

Activate virtualenv:

```bash
source .env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Activate virtualenv

From project root directory, run:

```bash
source .env/bin/activate
```

### Run within Jupyter notebooks

From project root directory, run:

```bash
jupyter-notebook
```

### Launch python scripts from command line

```bash
python srw.py
```

## Test with Binder

[![Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fgitlab.math.unistra.fr%2Fproba-mc2020%2Fproba-mc2020/master)
