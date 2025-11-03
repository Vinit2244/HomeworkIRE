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
    sudo dockerd # or open using desktop app
    ```

4. Download a local development installation of elasticsearch using docker:

    ```shell
    curl -fsSL https://elastic.co/start-local | sh

    # Please copy the generated username, password and API key and store them in .env file.
    ```

    To start the container if already exists:

    ```shell
    docker start es-local-dev && docker start kibana-local-settings && docker start kibana-local-dev
    ```

    To stop the container once done:

    ```shell
    docker stop kibana-local-dev && docker stop es-local-dev
    ```

5. Run a local server of Redis using docker:

    ```shell
    docker run -d -p 6379:6379 --name my-redis-server redis:latest
    ```

    To start the container if already exists:

    ```shell
    docker start my-redis-server
    ```

    To stop the container once done:

    ```shell
    docker stop my-redis-server
    ```

6. Install the C++ RocksDB Library (MacOS):

    ```shell
    brew install rocksdb
    ```

---

## Assignment 1 - Indexing and Retrieval

### Activity 1 - Preprocess data & Index News/Wikipedia Dataset in Elasticsearch

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

# Index News & Wikipedia Data in elasticsearch
python3 main.py # To run in auto mode add the flag "--mode auto". But manual mode is recommended as it gives you more control over indexing your data
```

### Activity 2 - Own simple indexing

```shell
# Manipulate the indices
python3 main.py # To run in auto mode (from config file) add the flag "--mode config". But manual mode is recommended as it gives you more control over indexing your data

# Generate queries
python3 generate_queries.py --num_queries <N_QUERIES> --index_id <INDEX_ID> --output_file <PATH_TO_QUERIES_JSON>

# Generate the query outputs from elasticsearch to compare against custom index
python3 setup_queries.py --path <PATH_TO_QUERIES_JSON>

# Performance Testing
python3 performance_metrics.py
```

## About Datasets

### News Dataset

- Source: <https://github.com/Webhose/free-news-datasets>  
- Approx. Size: 1.02 GB  
- File Type(s): JSON (compressed .json / .json.gz in repository)  
- Attributes:  
  - hread (thread id; appears as given)  
  - uuid  
  - url  
  - ord_in_thread  
  - author  
  - published  
  - title  
  - text  
  - highlightText  
  - highlightTitle  
  - highlightThreadTitle  
  - language  
  - sentiment  
  - categories  
  - topics  
  - ai_allow  
  - has_canonical  
  - breaking  
  - webz_reporter  
  - external_links  
  - external_images  
  - internal_images  
  - entities  
  - syndication  
  - trust  
  - rating  
  - crawled  
  - updated  

### Wikipedia Dataset (English)

- Source: <https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.en>  
- Approx. Size: 11.6 GB  
- File Type(s): Parquet (.parquet)  
- Attributes:  
  - id  
  - url  
  - title  
  - text

## Notes

- Wikipedia dataset:
  - Need to manually downaload the .parquet files as downloading via code was leading to corrupt files
- Frequency Plots:
  - Only considers the `text` section of a document
  - Omits punctuations
  - Case insensitive
- Indexes cannot have same names (even when extension is different)
- Queries run will be preprocessed according to how the data is preprocessed (will implement preprocessing queries internally later)
