Application of Deep Knockoffs for fMRI to Generate Surrogate Data
======================================================================

This repository provides a pipeline, and the corresponding code to generate surrogate data using knockoffs for fMRI data.

Accompanying report: 

Framework of knockoffs: https://web.stanford.edu/group/candes/knockoffs/.

## Software dependencies

The code contained in this repository was tested on the following configuration of Python:

- python=3.6.5
- numpy=1.14.0
- scipy=1.0.0
- pytorch=0.4.1
- cvxpy=1.0.10
- cvxopt=1.2.0
- pandas=0.23.4
- seaborn
- torch-two-sample
- FANOK

## Installation Guide

```bash
cd DeepKnockoffs
python setup.py install --user
```

## Example pipeline

 - [pipeline.ipynb](pipeline.ipynb) a usage example on how to use our approach to generate knockoffs from fMRI data and perform non-parametric tests for thresholding.


## File Structure
Here is the file structure of the project: 
```bash
Project
|
|-- data
|   |-- input
|   |-- output
|       |-- beta
|       |-- img
|       |-- knockoffs
|
|-- deepknockoffs/
|    |-- torch-two-sample-master/
|
|-- implementation
|    |-- __init__.py
|    |-- create_activations.py
|    |-- glm.py
|    |-- knockoff_class.py
|    |-- knockoff_classes_test.py
|    |-- load.py
|    |-- non_parametric.py
|    |-- params.py
|    |-- utils.py 
|
|-- PlotGraph/
|
|-- tests
|    |-- __init__.py
|    |-- test_glm.py
|    |-- test_load.py
|
|-- .gitignore
|-- .gitlab-ci.yml.deactivate
|-- __init__.py
|-- pipeline.ipynb
|-- README.md
```

- Authors: Alec Flowers, Alexander Glavackij and Janet van der Graaf
- Supervision: Giulia Preti and Younes Farouj

