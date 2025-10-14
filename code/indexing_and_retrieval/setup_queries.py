# ======================== IMPORTS ========================
import os
import json
import argparse
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, ConnectionError
from utils import load_config, Style, ask_es_query, StatusCode


# ======================== GLOBALS ========================
load_dotenv()
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")


# ======================= FUNCTIONS =======================
def query_and_update(file_path: str) -> None:
    """
    Description:
        Executes queries from a JSON file against an Elasticsearch index and updates the file with the results.

    Args:
        file_path: Path to the JSON file containing queries.
    Returns:
        None
    """

    config = load_config()
    ES_HOST     : str = config['elasticsearch']['host']
    ES_PORT     : int = config['elasticsearch']['port']
    ES_SCHEME   : str = config['elasticsearch']['scheme']
    MAX_RESULTS : int = config.get("max_results", 50)
    SEARCH_FIELD: str = config.get("search_field", "text")

    MAX_NUM_DOCUMENTS     : int = config.get("max_num_documents", -1)
    PREPROCESSING_SETTINGS: dict = config.get("preprocessing", {})
    ATTRIBUTES_INDEXED    : list = config["index"].get("attributes", [])

    # Connect to Elasticsearch
    print(f"Connecting to Elasticsearch at {ES_SCHEME}://{ES_HOST}:{ES_PORT}...")
    try:
        es_client = Elasticsearch(
            [{'host': ES_HOST, 'port': ES_PORT, 'scheme': ES_SCHEME}],
            basic_auth=(USERNAME, PASSWORD)
        )
        if not es_client.ping():
            raise ConnectionError("Could not connect to Elasticsearch.")
        else:
            print(f"{Style.FG_GREEN}Connection successful!{Style.RESET}")
    except ConnectionError as e:
        print(f"{Style.FG_RED}Connection Error: {e}{Style.RESET}")
        return

    # Read and load the JSON file
    data = None
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"{Style.FG_GREEN}Successfully loaded {len(data['queries'])} queries from '{file_path}'.{Style.RESET}")
    except Exception as e:
        print(f"{Style.FG_RED}Error reading JSON file '{file_path}'.{Style.RESET}")
        return

    # Store the number of documents that were indexed while producing the results
    data["n_docs_indexed"] = MAX_NUM_DOCUMENTS

    # Store the preprocessing settings used while indexing the documents
    data["preprocessing_settings"] = PREPROCESSING_SETTINGS

    # Store the search field used
    data["search_field"] = SEARCH_FIELD

    # Store the attributes that were indexed
    data["attributes_indexed"] = ATTRIBUTES_INDEXED

    # Store the max results setting used
    data["max_results"] = MAX_RESULTS

    index_id = data.get("index_id")
    if not index_id:
        print(f"{Style.FG_RED}Error: 'index_id' not found in the JSON file.{Style.RESET}")
        return

    # Iterate and execute queries and update the JSON structure
    print("\nProcessing queries...")
    for i, item in enumerate(data["queries"]):
        query_text = item["query"]
        print(f"  ({i+1}/{len(data['queries'])}) Searching for: '{query_text[:70]}...'")

        res = ask_es_query(es_client, index_id, query_text, SEARCH_FIELD, MAX_RESULTS, False)

        if isinstance(res, StatusCode):
            print(f"{Style.FG_RED}Failed to execute query '{query_text}'.{Style.RESET}")
            item['docs'] = [] # If query fails, store empty list
            continue
        else:
            doc_ids = [hit['_id'] for hit in res['hits']['hits']] # Sorted by relevance score in descending order by default
            item['docs'] = doc_ids # Update the 'docs' list in our data structure

    # Write the updated data back to the file
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"{Style.FG_GREEN}Successfully updated '{file_path}' with the document IDs.{Style.RESET}")
    except Exception as e:
        print(f"{Style.FG_RED}Error writing updated data to file.{Style.RESET}")
        return


# ======================= MAIN =======================
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Query Elasticsearch and update JSON file with results.")
    argparser.add_argument('--path', type=str, required=True, help="Path to the JSON file containing queries.")
    args = argparser.parse_args()
    file_path = args.path

    query_and_update(file_path)
