# DISCo-Net : **D**istributed, **I**nterpretable, and **S**calable computing for **Co-**authorship **Net**works

![DISCO-Net overview](<Screenshot 2024-09-30 at 12.12.52 PM.png>)

An exponential growth in scientific literature necessitates the development of highly scalable computational tools that can effectively analyze and distill insights from complex, interconnected research landscapes. We introduce **D**istributed, **I**nterpretable, and **S**calable computing for **Co-**authorship **Net**works (DISCo-Net), a robust and scalable tool engineered to curate and examine large-scale co-authorship networks by harnessing the power of distributed computing and advanced relational database queries.

## Contents

- [Features](#features)
- [Usage](#usage)
  - [Initial setup](#initial-setup)
- [Scalable dependencies](#DISCo-Net-extensively-utilizes-following-for-scalability)
- [Contributing](#Contributing)
- [Author information](#Author-information)

## Features

- Scalable interpretable model
- Distributed computing
- BERT based interpretable inference

## Usage

### Initial setup

1. Create a Python 3.8 or newer virtual environment.

    ```
    conda create -n disconet python=3.8
    source activate disconet
    ```

2. Install disconet by running,

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

## Author information

In case of any questions, please reach out to us at:
- [Swapnil-Keshari] (swk25@pitt.edu)
- [Zarifeh-Heidari-Rarani]
- [Akash-Kishore]
- [Jishnu-Das] (jishnu@pitt.edu)
