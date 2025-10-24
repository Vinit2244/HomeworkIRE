# ======================== IMPORTS ========================
import os
import json
import time
import psutil
import threading
import seaborn as sns
from typing import List
import matplotlib.pyplot as plt
from indexes import ESIndex, CustomIndex
from utils import Style, load_config
from dataset_managers import get_news_dataset_handler, get_wikipedia_dataset_handler


# ====================== CONSTANTS ========================
mem_usage_diff_info_args_list = [
    # Order: index_id, index_type, dataset, info, dstore, qproc, compr, optim, attributes
    # ES Index - News Dataset
    ["es_news",         "ESIndex",     "News",      "NONE",      "NONE", "NONE", "NONE", "NONE", ["uuid", "text"]],
    # ES Index - Wikipedia Dataset
    ["es_wiki",         "ESIndex",     "Wikipedia", "NONE",      "NONE", "NONE", "NONE", "NONE", ["id", "text"]],
    # Custom Index - News Dataset
    ["cust_news_bool",  "CustomIndex", "News",      "BOOLEAN",   "CUSTOM", "NONE", "NONE", "NONE", ["uuid", "text"]],
    ["cust_news_wc",    "CustomIndex", "News",      "WORDCOUNT", "CUSTOM", "NONE", "NONE", "NONE", ["uuid", "text"]],
    ["cust_news_tfidf", "CustomIndex", "News",      "TFIDF",     "CUSTOM", "NONE", "NONE", "NONE", ["uuid", "text"]],
    # Custom Index - Wikipedia Dataset
    ["cust_wiki_boo",   "CustomIndex", "Wikipedia", "BOOLEAN",   "CUSTOM", "NONE", "NONE", "NONE", ["id", "text"]],
    ["cust_wiki_wc",    "CustomIndex", "Wikipedia", "WORDCOUNT", "CUSTOM", "NONE", "NONE", "NONE", ["id", "text"]],
    ["cust_wiki_tfidf", "CustomIndex", "Wikipedia", "TFIDF",     "CUSTOM", "NONE", "NONE", "NONE", ["id", "text"]]
]

latency_diff_dstore_args_list = [
    # Order: query_file_path, index_id, index_type, dataset, info, dstore, qproc, compr, optim, attributes
    # ES Index - News Dataset
    ["../../query_sets/news_queries.json",      "es_news",         "ESIndex",     "News",      "NONE",    "NONE", "NONE", "NONE", "NONE", ["uuid", "text"]],
    # ES Index - Wikipedia Dataset
    ["../../query_sets/wikipedia_queries.json", "es_wiki",         "ESIndex",     "Wikipedia", "NONE",    "NONE", "NONE", "NONE", "NONE", ["id", "text"]],
    # Custom Index - News Dataset
    ["../../query_sets/news_queries.json",      "cust_news_cust",  "CustomIndex", "News",      "BOOLEAN", "CUSTOM",  "NONE", "NONE", "NONE", ["uuid", "text"]],
    ["../../query_sets/news_queries.json",      "cust_news_rocks", "CustomIndex", "News",      "BOOLEAN", "ROCKSDB", "NONE", "NONE", "NONE", ["uuid", "text"]],
    ["../../query_sets/news_queries.json",      "cust_news_redis", "CustomIndex", "News",      "BOOLEAN", "REDIS",   "NONE", "NONE", "NONE", ["uuid", "text"]],
    # Custom Index - Wikipedia Dataset
    ["../../query_sets/wikipedia_queries.json", "cust_wiki_cust",  "CustomIndex", "Wikipedia", "BOOLEAN", "CUSTOM",  "NONE", "NONE", "NONE", ["id", "text"]],
    ["../../query_sets/wikipedia_queries.json", "cust_wiki_rocks", "CustomIndex", "Wikipedia", "BOOLEAN", "ROCKSDB", "NONE", "NONE", "NONE", ["id", "text"]],
    ["../../query_sets/wikipedia_queries.json", "cust_wiki_redis", "CustomIndex", "Wikipedia", "BOOLEAN", "REDIS",   "NONE", "NONE", "NONE", ["id", "text"]]
]

throughput_diff_dstore_args_list = []


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
sns.set_theme(style="whitegrid", palette="muted")
sns.set_context("notebook", font_scale=1.1)

def calc_memory_usage(index_id: str, index_type: str, dataset: str, info: str, dstore: str, qproc: str, compr: str, optim: str, attributes: List[str]) -> None:
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
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Use a color palette
        colors = sns.color_palette("husl", len([args for args in mem_usage_diff_info_args_list if args[2] == dataset]))
        color_idx = 0
        
        for args in mem_usage_diff_info_args_list:
            if args[2] != dataset:
                continue
            # Load the memory usage data from the corresponding JSON file
            input_file_path: str = f"./temp/memory_usage_{args[1]}_{args[2]}_{args[3]}_{args[4]}_{args[5]}_{args[6]}_{args[7]}.json"
            with open(input_file_path, 'r') as f:
                mem_usage_data = json.load(f)
            label = f"{args[1]}-{args[3]}" if args[1] == "CustomIndex" else args[1]
            
            # Plot with seaborn style
            ax.plot(mem_usage_data, label=label, linewidth=2.5, color=colors[color_idx], alpha=0.8)
            color_idx += 1
        
        ax.set_title(f'Memory Usage Comparison on {dataset} Dataset', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time (seconds)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Memory Usage (%)', fontsize=13, fontweight='bold')
        ax.legend(frameon=True, shadow=True, fancybox=True, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add subtle background
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        output_file_path: str = os.path.join(output_folder, f"diff_info_memory_usage_comparison_{dataset}.png")
        plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{Style.FG_GREEN}Memory usage comparison plot saved at '{output_file_path}'.{Style.RESET}")


def calc_latency(query_file_path: str, index_id: str, index_type: str, dataset: str, info: str, dstore: str, qproc: str, compr: str, optim: str, attributes: List[str]) -> None:
    # Select Dataset
    if dataset == "News":
        dataset_handler = get_news_dataset_handler(verbose=False)
    elif dataset == "Wikipedia":
        dataset_handler = get_wikipedia_dataset_handler(verbose=False)
    else:
        print(f"{Style.FG_RED}Invalid dataset for latency calculation.{Style.RESET}")
        return
    
    # Select Index
    if index_type == "ESIndex":
        index = ESIndex(ES_HOST, ES_PORT, ES_SCHEME, index_type, verbose=False)
    elif index_type == "CustomIndex":
        index = CustomIndex(index_type, info, dstore, qproc, compr, optim)       
    else:
        print(f"{Style.FG_RED}Invalid index type for latency calculation.{Style.RESET}")
        return
    
    # Create Index
    index.create_index(index_id, dataset_handler.get_files(attributes))

    # Load queries from the query file
    with open(query_file_path, 'r') as f:
        data = json.load(f)
    
    queries = data.get("queries", [])
    if not queries:
        print(f"{Style.FG_ORANGE}No queries found in the file.{Style.RESET}")
        return

    # Execute each query and measure latency
    latency = []
    for query in queries:
        start_time = time.time()
        index.query(query["query"], index_id)
        end_time = time.time()
        latency.append(end_time - start_time)
    
    # Delete the created index to free up memory
    index.delete_index(index_id)

    return latency

def plot_latency_comparison(output_folder: str, plot_es: bool=True) -> None:
    datasets = list(set([args[3] for args in latency_diff_dstore_args_list]))
    
    fig, axes = plt.subplots(len(datasets), 3, figsize=(20, 7 * len(datasets)))
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)
    
    for dataset_idx, dataset in enumerate(datasets):
        latency_data = {}
        for args in latency_diff_dstore_args_list:

            # Skipping plotting ES for better looking plot
            if not plot_es:
                if args[2] == "ESIndex":
                    continue
            
            if args[3] != dataset:
                continue
            # Load the latency data from the corresponding JSON file
            input_file_path: str = f"./temp/latency_{args[2]}_{args[3]}_{args[4]}_{args[5]}_{args[6]}_{args[7]}_{args[8]}.json"
            with open(input_file_path, 'r') as f:
                data = json.load(f)
            latencies = data.get("latencies", [])
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                p95_latency = sorted(latencies)[int(0.95 * len(latencies)) - 1]
                p99_latency = sorted(latencies)[int(0.99 * len(latencies)) - 1]
                
                label = args[5] if args[2] == "CustomIndex" else args[2]
                latency_data[label] = (avg_latency, p95_latency, p99_latency)
        
        # Color palette
        colors = sns.color_palette("Set2", len(latency_data))
        
        # Plot Average Latency
        x_labels = list(latency_data.keys())
        avg_latencies = [latency_data[label][0] for label in x_labels]
        bars1 = axes[dataset_idx, 0].bar(x_labels, avg_latencies, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
        axes[dataset_idx, 0].set_title(f'Average Latency - {dataset} Dataset', fontsize=14, fontweight='bold', pad=15)
        axes[dataset_idx, 0].set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
        axes[dataset_idx, 0].tick_params(axis='x', rotation=45)
        axes[dataset_idx, 0].grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[dataset_idx, 0].text(bar.get_x() + bar.get_width()/2., height,
                                      f'{height:.3f}s', ha='center', va='bottom', fontsize=9)
        
        # Plot P95 Latency
        p95_latencies = [latency_data[label][1] for label in x_labels]
        bars2 = axes[dataset_idx, 1].bar(x_labels, p95_latencies, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
        axes[dataset_idx, 1].set_title(f'P95 Latency - {dataset} Dataset', fontsize=14, fontweight='bold', pad=15)
        axes[dataset_idx, 1].set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
        axes[dataset_idx, 1].tick_params(axis='x', rotation=45)
        axes[dataset_idx, 1].grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            axes[dataset_idx, 1].text(bar.get_x() + bar.get_width()/2., height,
                                      f'{height:.3f}s', ha='center', va='bottom', fontsize=9)
        
        # Plot P99 Latency
        p99_latencies = [latency_data[label][2] for label in x_labels]
        bars3 = axes[dataset_idx, 2].bar(x_labels, p99_latencies, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
        axes[dataset_idx, 2].set_title(f'P99 Latency - {dataset} Dataset', fontsize=14, fontweight='bold', pad=15)
        axes[dataset_idx, 2].set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
        axes[dataset_idx, 2].tick_params(axis='x', rotation=45)
        axes[dataset_idx, 2].grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            axes[dataset_idx, 2].text(bar.get_x() + bar.get_width()/2., height,
                                      f'{height:.3f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file_path: str = os.path.join(output_folder, f"diff_dstore_latency_comparison_all_datasets.png")
    plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{Style.FG_GREEN}Latency comparison plot saved at '{output_file_path}'.{Style.RESET}")


def calc_throughput(query_file_path: str, index_id: str, index_type: str, dataset: str, info: str, dstore: str, qproc: str, compr: str, optim: str, attributes: List[str]) -> None:
    # Select Dataset
    if dataset == "News":
        dataset_handler = get_news_dataset_handler(verbose=False)
    elif dataset == "Wikipedia":
        dataset_handler = get_wikipedia_dataset_handler(verbose=False)
    else:
        print(f"{Style.FG_RED}Invalid dataset for throughput calculation.{Style.RESET}")
        return
    
    # Select Index
    if index_type == "ESIndex":
        index = ESIndex(ES_HOST, ES_PORT, ES_SCHEME, index_type, verbose=False)
    elif index_type == "CustomIndex":
        index = CustomIndex(index_type, info, dstore, qproc, compr, optim)       
    else:
        print(f"{Style.FG_RED}Invalid index type for throughput calculation.{Style.RESET}")
        return
    
    # Create Index
    index.create_index(index_id, dataset_handler.get_files(attributes))

    # Load queries from the query file
    with open(query_file_path, 'r') as f:
        data = json.load(f)
    
    queries = data.get("queries", [])
    if not queries:
        print(f"{Style.FG_ORANGE}No queries found in the file.{Style.RESET}")
        return

    # Execute all queries and measure throughput
    start_time = time.time()
    for query in queries:
        index.query(query["query"], index_id)
    end_time = time.time()
    
    total_time = end_time - start_time
    throughput = len(queries) / total_time if total_time > 0 else 0

    # Delete the created index to free up memory
    index.delete_index(index_id)

    return throughput

def plot_throughput_comparison(output_folder: str) -> None:
    datasets = list(set([args[3] for args in throughput_diff_dstore_args_list]))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data for grouped bar chart
    x_labels = set()
    for args in throughput_diff_dstore_args_list:
        label = args[5] if args[2] == "CustomIndex" else args[2]
        x_labels.add(label)
    x_labels = sorted(list(x_labels))
    
    bar_width = 0.8 / len(datasets)
    x_pos = range(len(x_labels))
    colors = sns.color_palette("viridis", len(datasets))
    
    for dataset_idx, dataset in enumerate(datasets):
        throughput_data = {}
        for args in throughput_diff_dstore_args_list:
            if args[3] != dataset:
                continue
            # Load the throughput data from the corresponding JSON file
            input_file_path: str = f"./temp/throughput_{args[2]}_{args[3]}_{args[4]}_{args[5]}_{args[6]}_{args[7]}_{args[8]}.json"
            with open(input_file_path, 'r') as f:
                data = json.load(f)
            throughput = data.get("throughput", 0)
            label = args[5] if args[2] == "CustomIndex" else args[2]
            throughput_data[label] = throughput
        
        # Plot grouped bars
        throughputs = [throughput_data.get(label, 0) for label in x_labels]
        offset = bar_width * dataset_idx
        bars = ax.bar([p + offset for p in x_pos], throughputs, bar_width, 
                      label=dataset, color=colors[dataset_idx], edgecolor='black', 
                      linewidth=1.2, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_title('Throughput Comparison Across Datasets', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Data Store / Index Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Throughput (queries/second)', fontsize=13, fontweight='bold')
    ax.set_xticks([p + bar_width * (len(datasets) - 1) / 2 for p in x_pos])
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend(title='Dataset', frameon=True, shadow=True, fancybox=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    output_file_path: str = os.path.join(output_folder, f"diff_dstore_throughput_comparison_all_datasets.png")
    plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{Style.FG_GREEN}Throughput comparison plot saved at '{output_file_path}'.{Style.RESET}")


# ========================= MAIN ==========================
if __name__ == "__main__":
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
    print("\n"+"="*50+"\n")


    # ========================================================================
    # Latency comparison for different Data stores
    # ========================================================================
    print(f"{Style.FG_MAGENTA}Calculating Latency of Queries Execution for Different Data Stores...{Style.RESET}")
    for args in latency_diff_dstore_args_list:
        print(f"{Style.FG_CYAN}Calculating latency for DataStore: {args[5]}, Dataset: {args[3]}...{Style.RESET}")
        latencies = calc_latency(*args)

        # Store the latency data to a JSON file
        output_file_path: str = f"./temp/latency_{args[2]}_{args[3]}_{args[4]}_{args[5]}_{args[6]}_{args[7]}_{args[8]}.json"
        with open(output_file_path, 'w') as f:
            json.dump({"latencies": latencies}, f)

    print(f"{Style.FG_MAGENTA}Plotting Latency Comparison for Different Data Stores...{Style.RESET}")
    plot_latency_comparison(output_dir, plot_es=False)
    clear_folder("./temp")
    print("\n"+"="*50+"\n")
