# 🧠 SSM Learning Project

This repository contains implementations of various **State Space Models**:  
**S4**, **S4DSS**, and **S6**.

The primary goal of this project is to **build intuition** around these models. To that end, the implementations prioritize **clarity and alignment with theoretical derivations** over raw performance.

---

## 📂 Code Structure

- **Models**  
  The implementations of `S4`, `S4DSS`, and `S6` can be found in:  
  [`src/ssm/model.py`](src/ssm/model.py)

- **Parallel Scan**  
  Illustrative code for the **forward and backward passes** of `S6` is provided in:  
  [`src/ssm/parallel_scan.py`](src/ssm/parallel_scan.py)  
  ⚠️ This code does **not** run on appropiately on CUDA. While it omits kernel fusion, the core theoretical structure is accurate but sequential due to the nature of python.

- **Notebook Example**  
  For a working demonstration of how to run the custom autograd function (including forward and backward passes), see:  
  [`notebooks/parallel_scan.ipynb`](notebooks/parallel_scan.ipynb)

---

## 📝 Notes

- The focus is on **didactic clarity**, which means:
  - Code is intentionally written to closely follow mathematical derivations
  - Some performance optimizations are sacrificed for readability


## Prerequisites

Make sure you have the following installed on your system:

Conda (either Anaconda or Miniconda)

Python 3.8 or higher (managed via Conda)

## Step-by-Step Instructions

1. Clone the Repository

First, clone the repository to your local machine:

- ```git clone https://github.com/MarieJuhlJ/StateSpaceModels.git```

- ```cd StateSpaceModels```

2. Install Invoke

Install invoke using Conda:

- ```conda install -c conda-forge invoke```

You can verify the installation by running:

- ```invoke --version```

3. Create the Environment

This repository includes a custom create-environment function defined in the tasks.py file. Use invoke to create the environment by running:

- ```invoke create-environment```

This function will:

Create a Conda environment with the appropriate name "ssm" (specified in the script).

4. Install Requirements

Once the environment is set up, install additional Python dependencies using the requirements function. Activate the new environment and run the following command:

- ```conda activate ssm```

- ```invoke requirements```

This function will:

Install dependencies listed in the requirements.txt file (if applicable).

Note that windows users have to manually run the following commands:
- ```pip install -r requirements.txt```
- ```pip install -e .```

### For developers:
There are extra packages you need if you want to make changes to the project. You can install them using the `requirements_dev.txt` by invoking the task `dev_requirements"`:

```invoke dev_requirements```

or installing `requirements_dev.txt` directly with pip:

```pip install -r requirements_dev.txt```

To use pre-commit checks run the following line:
- ```pre-commit install```


## Project structure

The directory structure of the project looks like this:
```txt
├── configs/                  # Configuration files
│   └── experiment/
├── data/                     # Data directory
│   ├── processed/
│   └── raw
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── hippo.py
│   │   ├── kernel.py
│   │   ├── model.py
│   │   └── train.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
