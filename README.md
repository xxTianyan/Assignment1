# ME700 Course Code

![Python Version](https://img.shields.io/badge/python-3.12-blue)
![OS](https://img.shields.io/badge/os-ubuntu%20%7C%20macos%20%7C%20windows-blue)

[![codecov](https://codecov.io/gh/xxTianyan/ME700/branch/main/graph/badge.svg)](https://codecov.io/gh/xxTianyan/ME700)
![Tests](https://github.com/xxTianyan/ME700/actions/workflows/ci.yml/badge.svg)

## Description
This repository contains all the numerical algorithms that learned in the BU course ME700. 

## Installation
Firt clone or down the repository to your local environment:
```sh
# Clone the repository
git clone https://github.com/xxTianyan/ME700.git
cd ME700
```
Create a virtual environment, here we use conda:
```sh
# Create a new conda environment and activate it
conda create -n numerical python=3.12
conda activate numerical
```
Ensure that pip is using the most up to date version of setuptools:
```sh
pip install --upgrade pip setuptools wheel
```
Create an editable install of the bisection method code (note: you must be in the correct directory):
```sh
pip install -e .
```
Test that the code is working with pytest:
```sh
pytest
```
In order to see the tutorial, you need to install jupyter notebook:
```sh
pip install jupyter notebook
```

## Usage























