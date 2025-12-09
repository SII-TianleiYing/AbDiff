# AbDiff
A novel latent diffusion–based framework for antibody conformational generation.

## Installation for Developers

This is a professional installation version, which is highly beneficial for researchers to fine-tune it for their own studies.

We plan to further organize and upload a more user-friendly image version later. Stay tuned for future updates to this project!

### Installation

Because it involves multiple modules, AbDiff requires several independent conda environments to run. All Python dependencies are specified in ./environmets. We recommend using conda environment, to install dependencies, run:

```python
$ conda env create -f ./environmets/abdiff_environment
$ conda env create -f ./environmets/abfold_environment
$ conda env create -f ./environmets/colabfold_environment
$ conda env create -f ./environmets/igfold_environment
```

Next, we will configure AbFold in an engineering-oriented manner:

```python
$ conda activate abfold
$ pip install -e ./abfold
```

This will install AbFold into your conda environment in development mode. Please check to ensure that AbFold has been installed correctly.

Next, run the following code to check whether the weight files exist. If not, the weights will be downloaded automatically when an internet connection is available:

```python
$ python check.py
```

### Test

We provide example inputs and outputs in `./example`.

After confirming that all four environments above have been fully installed, run:

```python
$ python pipeline.py
```

Please note that when ColabFold is used for the first time, it will automatically download configuration files and requires template files to be available locally on your computer. If you do not need high-accuracy modeling, you can disable templates. For details, please refer to the official ColabFold documentation.
