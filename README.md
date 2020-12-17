Application of Deep Knockoffs for fMRI to Generate Surrogate Data
======================================================================

This repository provides a pipeline, and the corresponding code to generate surrogate data using knockoffs for fMRI data.

Accompanying report: https://arxiv.org/abs/1811.06687

Framework of knockoffs: https://web.stanford.edu/group/candes/knockoffs/.

## Software dependencies

The code contained in this repository was tested on the following configuration of Python:

- python==3.8.5
- numpy==1.19.2
- scipy==1.5.2
- torch==1.7.0
- cvxopt==1.2.5
- cvxpy==1.1.7
- pandas==1.1.3
- matplotlib==3.3.2
- seaborn==0.11.0
- statsmodels==0.12.1
- sklearn
- jupyter
- Cython

## Installation Guide

```bash
pip3 install -r requirements.txt
pip3 install fanok
cd deepknockoffs/
pip3 install DeepKnockoffs/
pip3 install torch-two-sample-master/
```

Some files need to be downloaded [here](https://drive.google.com/drive/folders/1Xziw8kmfp0j8DV3clFZr_7Hh9Bac5X51?usp=sharing) and put into the correct folders manually, since the github
file limit forbids us from uploading them.
`X_tfMRI_WM_LR_Glasser360.mat` is not required to run the project, the checkpoints are necessary to run the jupyter notebook.
However, the jupyter notebook is already pre-run.

These files are:
- X_tfMRI_WM_LR_Glasser360.mat: data/input/
- DeepKO_tWM_s0_checkpoint.pth.tar_prev.pth.tar: data/output/knockoffs/
- DeepKO_tWM_s0_checkpoint.pth.tar:  data/output/knockoffs/
- DeepKO_tMOTOR_s1_checkpoint.pth.tar_prev.pth.tar: data/output/knockoffs/
- DeepKO_tMOTOR_s1_checkpoint.pth.tar: data/output/knockoffs/

## Example pipeline

 - [pipeline.ipynb](pipeline.ipynb) a usage example on how to use our approach to generate knockoffs from fMRI data and perform non-parametric tests for thresholding.

## Brain Plots

In order to obtain the brain plots, a Matlab script is used: `our_plots.m`
This file is located in `PlotGraph`, which also contains the necessary functions.

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

