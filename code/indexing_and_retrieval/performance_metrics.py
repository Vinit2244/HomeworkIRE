# ======================== IMPORTS ========================
import os
import json
import time
import psutil
import argparse
import threading
import seaborn as sns
from enum import Enum
from typing import List
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from indexes import ESIndex, CustomIndex
from utils import Style, IndexType, load_config
from dataset_managers import get_news_dataset_handler, get_wikipedia_dataset_handler


# ====================== CONSTANTS ========================
mem_usage_diff_info_args_list = [
    # Order: index_id, index_type, dataset, info, dstore, qproc, compr, optim, attributes
    # ES Index - News Dataset
    ["es_news",         "ESIndex",     "News",      "NONE",      "NONE", "NONE", "NONE", "NONE", ["uuid", "text"]],
    # ES Index - Wikipedia Dataset
    ["es_wiki",         "ESIndex",     "Wikipedia", "NONE",      "NONE", "NONE", "NONE", "NONE", ["id", "text"]],
    # Custom Index - News Dataset
    ["cust_news_bool",  "CustomIndex", "News",      "BOOLEAN",   "NONE", "NONE", "NONE", "NONE", ["uuid", "text"]],
    ["cust_news_wc",    "CustomIndex", "News",      "WORDCOUNT", "NONE", "NONE", "NONE", "NONE", ["uuid", "text"]],
    ["cust_news_tfidf", "CustomIndex", "News",      "TFIDF",     "NONE", "NONE", "NONE", "NONE", ["uuid", "text"]],
    # Custom Index - Wikipedia Dataset
    ["cust_wiki_boo",   "CustomIndex", "Wikipedia", "BOOLEAN",   "NONE", "NONE", "NONE", "NONE", ["id", "text"]],
    ["cust_wiki_wc",    "CustomIndex", "Wikipedia", "WORDCOUNT", "NONE", "NONE", "NONE", "NONE", ["id", "text"]],
    ["cust_wiki_tfidf", "CustomIndex", "Wikipedia", "TFIDF",     "NONE", "NONE", "NONE", "NONE", ["id", "text"]]
]


# ======================= GLOBALS =========================
memory_usage = []
monitoring = True
INTERVAL = 0.01  # seconds


# ======================= THREADS =========================
def monitor_memory(interval:int=1):
    """
    Continuously record memory usage every `interval` seconds.
    """
    while monitoring:
        memory_usage.append(psutil.virtual_memory().percent)
        time.sleep(interval)


# =================== HELPER FUNCTIONS ====================
def clear_folder(folder_path: str) -> None:
    for file in os.listdir(folder_path):
        os.remove(os.path.join(folder_path, file))


# ====================== FUNCTIONS ========================
def calc_memory_usage(index_id: str, index_type: str, dataset: str, info: str="NONE", dstore: str="NONE", qproc: str="NONE", compr: str="NONE", optim: str="NONE", attributes: List[str]=["text"]) -> None:
    # Reset the variables
    global monitoring, memory_usage
    monitoring = True
    memory_usage.clear()

    # Start monitoring thread
    t = threading.Thread(target=monitor_memory, args=(INTERVAL,))
    t.start()

    # Select Dataset
    if dataset == "News":
        dataset_handler = get_news_dataset_handler(verbose=False)
    elif dataset == "Wikipedia":
        dataset_handler = get_wikipedia_dataset_handler(verbose=False)
    else:
        print(f"{Style.FG_RED}Invalid dataset for memory usage calculation.{Style.RESET}")
        return
    
    # Select Index
    if index_type == "ESIndex":
        index = ESIndex(ES_HOST, ES_PORT, ES_SCHEME, index_type, verbose=False)
    elif index_type == "CustomIndex":
        index = CustomIndex(index_type, info, dstore, qproc, compr, optim)       
    else:
        print(f"{Style.FG_RED}Invalid index type for memory usage calculation.{Style.RESET}")
        return
    
    # Create Index
    index.create_index(index_id, dataset_handler.get_files(attributes))

    # Stop monitoring thread
    monitoring = False
    t.join()

    # Delete the created index to free up memory
    index.delete_index(index_id)


def plot_memory_usage_comparison(output_folder: str) -> None:
    datasets = set([args[2] for args in mem_usage_diff_info_args_list])
    for dataset in datasets:
        plt.figure(figsize=(10, 6))
        for args in mem_usage_diff_info_args_list:
            if args[2] != dataset:
                continue
            # Load the memory usage data from the corresponding JSON file
            input_file_path: str = f"./temp/memory_usage_{args[1]}_{args[2]}_{args[3]}_{args[4]}_{args[5]}_{args[6]}_{args[7]}.json"
            with open(input_file_path, 'r') as f:
                mem_usage_data = json.load(f)
            label = f"{args[1]}-{args[3]}" if args[1] == "CustomIndex" else args[1]
            plt.plot(mem_usage_data, label=label)
        
        plt.title(f'Memory Usage Comparison on {dataset} Dataset')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (%)')
        plt.legend()
        
        output_file_path: str = os.path.join(output_folder, f"diff_info_memory_usage_comparison_{dataset}.png")
        plt.savefig(output_file_path)
        print(f"{Style.FG_GREEN}Memory usage comparison plot saved at '{output_file_path}'.{Style.RESET}")



# ========================= MAIN ==========================
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Calculates various metrics on ES and Custom Index and plots graphs.")
    argparser.add_argument('--query_file', type=str, required=False, help="Path to the JSON file containing queries.")
    
    args = argparser.parse_args()
    query_file_path = args.query_file

    # Create a temp folder to store intermediate outputs
    os.makedirs("temp", exist_ok=True)

    config = load_config()
    output_dir = config.get("output_folder_path", ".")

    global ES_HOST, ES_PORT, ES_SCHEME
    ES_HOST = config["elasticsearch"].get("host", "localhost")
    ES_PORT = config["elasticsearch"].get("port", 9200)
    ES_SCHEME = config["elasticsearch"].get("scheme", "http")

    # ========================================================================
    # Memory footprint comparison for different IndexInfo configurations
    # ========================================================================
    print(f"{Style.FG_MAGENTA}Calculating Memory Usage Comparison for Different Index Information Types...{Style.RESET}")
    for args in mem_usage_diff_info_args_list:
        print(f"{Style.FG_CYAN}Calculating memory usage for IndexType: {args[1]}, Dataset: {args[2]}, Info: {args[3]}...{Style.RESET}")
        calc_memory_usage(*args)

        # Store the memory usage data to a JSON file
        output_file_path: str = f"./temp/memory_usage_{args[1]}_{args[2]}_{args[3]}_{args[4]}_{args[5]}_{args[6]}_{args[7]}.json"
        with open(output_file_path, 'w') as f:
            json.dump(memory_usage, f)

    print(f"{Style.FG_MAGENTA}Plotting Memory Usage Comparison for Different Index Information Types...{Style.RESET}")
    plot_memory_usage_comparison(output_dir)
    clear_folder("./temp")
