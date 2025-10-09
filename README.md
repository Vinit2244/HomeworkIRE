# How to setup this repository

## My System Configuration

- **OS**: macOS 15.5 (24F74)
- **Architecture**: arm64 (Apple Silicon)
- **Processor**: Apple M2
- **RAM**: 8 GB
- **Kernel**: Darwin 24.5.0
- **Python**: Python 3.12.10

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- [Python 3.8+](https://www.python.org/downloads/) installed

## Basic Setup

1. Update the following values in `config.yaml` file as per your specifications:
    - Path to datasets (you can add new datasets as well)
      > Note: If you are adding new datasets make sure to update the code accordingly for handling those datasets

    - Output folder path

2. Setup Environment:

    ```shell
    # Create environment
    python3 -m venv env

    # Activate environment
    source ./env/bin/activate

    # Download the requirements
    pip install -r requirements.txt
    ```

3. Start docker deamon:

    ```shell
    sudo dockerd # Or open using desktop app
    ```

4. Create a local development installation of elasticsearch using docker:

    ```shell
    curl -fsSL https://elastic.co/start-local | sh

    # Please copy the generated username, password and API key and store them in .env file.

    # To stop the container once done: docker stop kibana-local-dev && docker stop es-local-dev
    ```

---

## Assignment 1 - Indexing and Retrieval

### Activity 1 - Preprocess data

```shell
# Move into the `indexing_and_retrieval` folder
cd code/indexing_and_retrieval

# Download the data
python3 download_data.py

# Generate the word frequency plots (before preprocessing)
python3 generate_frequency_plots.py --data_state raw # raw denotes the data has not yet been preprocessed

# Update the preprocessing settings in the config.yaml file before applying preprocessing

# Preprocess the data
python3 preprocess_data.py

# Generate the word frequency plots again
python3 generate_frequency_plots.py --data_state preprocessed # now the data has been preprocessed
```

#### Notes

- Wikipedia dataset:
  - Need to manually downaload the .parquet files as downloading via code was leading to corrupt files
- Frequency Plots:
  - Only considers the `text` section of a document
  - Omits punctuations
  - Case insensitive

### Activity 2 - Own simple indexing
