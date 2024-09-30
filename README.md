# DISCo-Net Package

## Contents

- [Features](#features)
- [Usage](#usage)
  - [Initial setup](#initial-setup)
- [Scalable dependencies](#DISCo-Net-extensively-utilizes-following-for-scalability)
- [Contributing](#contributing)

## Features

- Scalable interpretable model
- Distributed computing
- BERT based inference

## Usage

### Initial setup

1. Create a Python 3.8 or newer virtual environment.

    *If you're not sure how to create a suitable Python environment, the easiest way is using [Anaconda](https://docs.conda.io/en/latest/miniconda.html)*

    ```
    brew install miniconda
    ```

    *Then you can create and activate a new Python environment by running:*

    ```
    conda create -n disconet python=3.8
    source activate disconet
    ```

2. Now that you have a suitable Python environment, you're ready to personalize this repository. Just run:

    ```
    pip install pydisconet
    ```

## DISCo-Net extensively utilizes following for scalability

- [dask](https://www.dask.org/)
- [duckDB](https://duckdb.org/)
- [multiprocessing](https://docs.python.org/3/library/multiprocessing.html)

## Contributing

If you find a bug :bug:, please open a [bug report](https://github.com/swapnilkeshari/disconet/issues/new?assignees=&labels=bug&template=bug_report.md&title=).
If you have an idea for an improvement or new feature :rocket:, please open a [feature request](https://github.com/swapnilkeshari/disconet/issues/new?assignees=&labels=Feature+request&template=feature_request.md&title=). 