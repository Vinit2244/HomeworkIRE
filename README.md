# How to setup this repository

## Basic Setup

1. Update the following values in `config.yaml` file as per your specifications:
    - Path to datasets (you can add new datasets as well)
    > Note: If you are adding new datasets make sure to update the code accordingly for handling those datasets

    - Output folder path

2. Setup Environment:

    ```shell
    # Create environment
    python -m venv env

    # Activate environment
    source ./env/bin/activate

    # Install dependencies
    pip install -r requirements.txt
    ```

---

## Assignment 1 - Indexing and Retrieval

### Activity 1 - Preprocess data

```shell
# Move into the `indexing_and_retrieval` folder
cd indexing_and_retrieval

# Download the data
python download_data.py

# Generate the word frequency plots (before preprocessing)
python generate_frequency_plot.py --data_state raw # raw denotes the data has not yet been preprocessed

# Preprocess the data
python preprocess_data.py

# Generate the word frequency plots again
python generate_frequency_plot.py --data_state preprocessed # now the data has been preprocessed
```

#### Notes

- Wikipedia dataset:
  - Need to manually downaload the .parquet files as downloading via code was leading to corrupt files
- Frequency Plots:
  - Only considers the `text` section of a document
  - Omits punctuations
  - Case insensitive

### Activity 2 - Own simple indexing
