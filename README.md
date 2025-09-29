# How to setup this repository

## Basic Setup

1. Update the following values `config.yaml` file as per you:
    - Paths to datasets (Can add new datasets as well)
    > Note: If you are adding new datasets make sure to update the code accordingly for handling those datasets

2. Setup Environment:

    ```shell
    # Create environment
    python -m venv env

    # Activate environment
    source ./env/bin/activate

    # Install dependencies
    pip install -r requirements.txt
    ```

## Assignment 1 - Indexing and Retrieval

### Activity 1 - Preprocess data

```shell
# Move into the `indexing_and_retrieval` folder
cd indexing_and_retrieval

# Download the data
python download_data.py

# Preprocess the data
python preprocess_data.py
```
> Note: Output will be stored at the output folder location specified in `config.yaml` file