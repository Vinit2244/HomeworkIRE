# ======================== IMPORTS ========================
import os
import json
import time
import psutil
import threading
import seaborn as sns
from typing import List
import matplotlib.pyplot as plt
from utils import Style, load_config
from indexes import ESIndex, CustomIndex
from dataset_managers import get_news_dataset_handler, get_wikipedia_dataset_handler
from constants import TEMP_FOLDER_PATH


# ======================= GLOBALS =========================
memory_usage = []
monitoring = True

INTERVAL = 0.01 # seconds
PLOT_ES = False # Whether to plot ES index in latency comparison


# ======================= THREADS =========================
def monitor_memory(interval: int=1):
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

def reset() -> None:
    clear_folder(TEMP_FOLDER_PATH)
    print("\n"+"="*50+"\n")

def get_temp_file_path(metric: str, args: dict) -> str:
    return f"{TEMP_FOLDER_PATH}/{metric}_{args.get("index_type")}_{args.get("dataset")}_{args.get("info")}_{args.get("dstore")}_{args.get("qproc")}_{args.get("compr")}_{args.get("optim")}.json"


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

def plot_memory_usage_comparison(output_file_path: str, args_list: List[str]) -> None:
    datasets = set([args.get("dataset") for args in args_list])
    
    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Use a color palette
        colors = sns.color_palette("husl", len([args for args in args_list if args.get("dataset") == dataset]))
        color_idx = 0
        
        for args in args_list:
            if args.get("dataset") != dataset:
                continue
            # Load the memory usage data from the corresponding JSON file
            input_file_path: str = get_temp_file_path("memory_usage", args)
            with open(input_file_path, 'r') as f:
                mem_usage_data = json.load(f)
            
            # Create a more descriptive label
            if args.get("index_type") == "ESIndex":
                label = "Elasticsearch"
            else:
                # Create a more descriptive label for custom indexes
                label_parts = []
                if args.get("info") and args.get("info") != "NONE":
                    label_parts.append(args.get("info"))
                if args.get("dstore") and args.get("dstore") != "NONE":
                    label_parts.append(args.get("dstore"))
                if args.get("compr") and args.get("compr") != "NONE":
                    label_parts.append(args.get("compr"))
                label = "-".join(label_parts) if label_parts else "Custom"
            
            # Plot with seaborn style
            ax.plot(mem_usage_data, label=label, linewidth=2.5, color=colors[color_idx], alpha=0.8)
            color_idx += 1
        
        ax.set_title(f'Memory Usage Comparison on {dataset} Dataset', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time (seconds)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Memory Usage (%)', fontsize=13, fontweight='bold')
        ax.legend(frameon=True, shadow=True, fancybox=True, fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add subtle background
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        output_file_path_db = output_file_path.split(".")[0] + f"_{dataset.lower()}.png"
        plt.savefig(output_file_path_db, dpi=300, bbox_inches='tight')
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

def plot_latency_comparison(output_file_path: str, args_list: List[str], plot_es: bool=True) -> None:
    datasets = list(set([args.get("dataset") for args in args_list]))
    
    for dataset in datasets:
        # Collect latency data for this dataset
        latency_data = {}
        for args in args_list:
            # Skip ES plotting if specified
            if not plot_es and args.get("index_type") == "ESIndex":
                continue
            
            if args.get("dataset") != dataset:
                continue
                
            # Load the latency data from the corresponding JSON file
            input_file_path: str = get_temp_file_path("latency", args)
            with open(input_file_path, 'r') as f:
                data = json.load(f)
            latencies = data.get("latencies", [])
            
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                p95_latency = sorted(latencies)[int(0.95 * len(latencies)) - 1]
                p99_latency = sorted(latencies)[int(0.99 * len(latencies)) - 1]
                
                # Generate label based on configuration
                if args.get("index_type") == "ESIndex":
                    label = "Elasticsearch"
                else:
                    # Create a more descriptive label for custom indexes
                    label_parts = []
                    if args.get("info") and args.get("info") != "NONE":
                        label_parts.append(args.get("info"))
                    if args.get("dstore") and args.get("dstore") != "NONE":
                        label_parts.append(args.get("dstore"))
                    if args.get("compr") and args.get("compr") != "NONE":
                        label_parts.append(args.get("compr"))
                    label = "-".join(label_parts) if label_parts else "Custom"
                
                latency_data[label] = (avg_latency, p95_latency, p99_latency)
        
        if not latency_data:
            continue
            
        # Create subplots for the three metrics
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Color palette
        colors = sns.color_palette("Set2", len(latency_data))
        x_labels = list(latency_data.keys())
        
        # Plot Average Latency
        avg_latencies = [latency_data[label][0] for label in x_labels]
        bars1 = axes[0].bar(x_labels, avg_latencies, color=colors, edgecolor='black', 
                           linewidth=1.2, alpha=0.8)
        axes[0].set_title('Average Latency', fontsize=14, fontweight='bold', pad=15)
        axes[0].set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45, labelsize=10)
        axes[0].grid(True, alpha=0.3, linestyle='--', axis='y')
        axes[0].set_facecolor('#f8f9fa')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}s', ha='center', va='bottom', fontsize=9)
        
        # Plot P95 Latency
        p95_latencies = [latency_data[label][1] for label in x_labels]
        bars2 = axes[1].bar(x_labels, p95_latencies, color=colors, edgecolor='black', 
                           linewidth=1.2, alpha=0.8)
        axes[1].set_title('95th Percentile Latency', fontsize=14, fontweight='bold', pad=15)
        axes[1].set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45, labelsize=10)
        axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')
        axes[1].set_facecolor('#f8f9fa')
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}s', ha='center', va='bottom', fontsize=9)
        
        # Plot P99 Latency
        p99_latencies = [latency_data[label][2] for label in x_labels]
        bars3 = axes[2].bar(x_labels, p99_latencies, color=colors, edgecolor='black', 
                           linewidth=1.2, alpha=0.8)
        axes[2].set_title('99th Percentile Latency', fontsize=14, fontweight='bold', pad=15)
        axes[2].set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
        axes[2].tick_params(axis='x', rotation=45, labelsize=10)
        axes[2].grid(True, alpha=0.3, linestyle='--', axis='y')
        axes[2].set_facecolor('#f8f9fa')
        
        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}s', ha='center', va='bottom', fontsize=9)
        
        # Set overall title and adjust layout
        fig.suptitle(f'Latency Comparison - {dataset} Dataset', fontsize=16, fontweight='bold', y=1.02)
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        
        # Save plot with dataset-specific filename
        dataset_output_path = output_file_path.replace('.png', f'_{dataset.lower()}.png')
        plt.savefig(dataset_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"{Style.FG_GREEN}Latency comparison plots saved at '{output_file_path}'.{Style.RESET}")


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

def plot_throughput_comparison(output_file_path: str, args_list: List[str]) -> None:
    datasets = list(set([args.get("dataset") for args in args_list]))
    for dataset in datasets:
        # Collect throughput data for this dataset
        throughput_data = {}
        for args in args_list:
            if args.get("dataset") != dataset:
                continue
                
            # Load the throughput data from the corresponding JSON file
            input_file_path: str = get_temp_file_path("throughput", args)
            with open(input_file_path, 'r') as f:
                data = json.load(f)
            throughput = data.get("throughput", 0)
            
            if throughput:
                # Generate label based on configuration
                if args.get("index_type") == "ESIndex":
                    label = "Elasticsearch"
                else:
                    # Create a more descriptive label for custom indexes
                    label_parts = []
                    if args.get("info") and args.get("info") != "NONE":
                        label_parts.append(args.get("info"))
                    if args.get("dstore") and args.get("dstore") != "NONE":
                        label_parts.append(args.get("dstore"))
                    if args.get("compr") and args.get("compr") != "NONE":
                        label_parts.append(args.get("compr"))
                    label = "-".join(label_parts) if label_parts else "Custom"
                
                throughput_data[label] = throughput
        
        if not throughput_data:
            continue
            
        # Plot Throughput Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color palette
        colors = sns.color_palette("Set3", len(throughput_data))
        x_labels = list(throughput_data.keys())
        throughputs = [throughput_data[label] for label in x_labels]
        
        bars = ax.bar(x_labels, throughputs, color=colors, edgecolor='black', 
                      linewidth=1.2, alpha=0.8)
        ax.set_title(f'Throughput Comparison - {dataset} Dataset', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Throughput (queries/second)', fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            
        plt.tight_layout()
        dataset_output_path = output_file_path.replace('.png', f'_{dataset.lower()}.png')
        plt.savefig(dataset_output_path, dpi=300, bbox_inches='tight')
        plt.close()
    print(f"{Style.FG_GREEN}Throughput comparison plots saved at '{output_file_path}'.{Style.RESET}")


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
    mem_usage_diff_info_args_list = [
        # Order: index_id, index_type, dataset, info, dstore, qproc, compr, optim, attributes
        # ES Index - News Dataset
        {"id": "es_news", "type": "ESIndex", "dataset": "News", "info": "NONE", "dstore": "NONE", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        # ES Index - Wikipedia Dataset
        {"id": "es_wiki", "type": "ESIndex", "dataset": "Wikipedia", "info": "NONE", "dstore": "NONE", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["id", "text"]},
        # Custom Index - News Dataset
        {"id": "cust_news_bool",  "type": "CustomIndex", "dataset": "News",      "info": "BOOLEAN",   "dstore": "CUSTOM", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        {"id": "cust_news_wc",    "type": "CustomIndex", "dataset": "News",      "info": "WORDCOUNT", "dstore": "CUSTOM", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        {"id": "cust_news_tfidf", "type": "CustomIndex", "dataset": "News",      "info": "TFIDF",     "dstore": "CUSTOM", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        # Custom Index - Wikipedia Dataset
        {"id": "cust_wiki_bool",  "type": "CustomIndex", "dataset": "Wikipedia", "info": "BOOLEAN",   "dstore": "CUSTOM", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["id", "text"]},
        {"id": "cust_wiki_wc",    "type": "CustomIndex", "dataset": "Wikipedia", "info": "WORDCOUNT", "dstore": "CUSTOM", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["id", "text"]},
        {"id": "cust_wiki_tfidf", "type": "CustomIndex", "dataset": "Wikipedia", "info": "TFIDF",     "dstore": "CUSTOM", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["id", "text"]}
    ]

    print(f"{Style.FG_MAGENTA}Calculating Memory Usage Comparison for Different Index Information Types...{Style.RESET}")
    for args in mem_usage_diff_info_args_list:
        print(f"{Style.FG_CYAN}Calculating memory usage for IndexType: {args.get("index_type")}, Dataset: {args.get("dataset")}, Info: {args.get("info")}...{Style.RESET}")
        calc_memory_usage(args.get("id"), args.get("type"), args.get("dataset"), args.get("info"), args.get("dstore"), args.get("qproc"), args.get("compr"), args.get("optim"), args.get("attributes"))

        # Store the memory usage data to a JSON file
        temp_output_file_path: str = get_temp_file_path("memory_usage", args)
        with open(temp_output_file_path, 'w') as f:
            json.dump(memory_usage, f)

    print(f"{Style.FG_MAGENTA}Plotting Memory Usage Comparison for Different Index Information Types...{Style.RESET}")

    output_file_path: str = os.path.join(output_dir, "diff_info_memory_usage.png")
    plot_memory_usage_comparison(output_file_path, mem_usage_diff_info_args_list)
    reset()


    # ========================================================================
    # Latency comparison for different Data stores
    # ========================================================================
    latency_diff_dstore_args_list = [
        # Order: query_file_path, index_id, index_type, dataset, info, dstore, qproc, compr, optim, attributes
        # ES Index - News Dataset
        {"query_file_path": "../../query_sets/news_queries.json",      "index_id": "es_news",         "index_type": "ESIndex",     "dataset": "News",      "info": "NONE",    "dstore": "NONE", "qproc": "NONE", "compr": "NONE", "optim": "NONE",    "attributes": ["uuid", "text"]},
        # ES Index - Wikipedia Dataset
        {"query_file_path": "../../query_sets/wikipedia_queries.json", "index_id": "es_wiki",         "index_type": "ESIndex",     "dataset": "Wikipedia", "info": "NONE",    "dstore": "NONE", "qproc": "NONE", "compr": "NONE", "optim": "NONE",    "attributes": ["id", "text"]},
        # Custom Index - News Dataset
        {"query_file_path": "../../query_sets/news_queries.json",      "index_id": "cust_news_cust",  "index_type": "CustomIndex", "dataset": "News",      "info": "BOOLEAN", "dstore": "CUSTOM",  "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        {"query_file_path": "../../query_sets/news_queries.json",      "index_id": "cust_news_rocks", "index_type": "CustomIndex", "dataset": "News",      "info": "BOOLEAN", "dstore": "ROCKSDB", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        {"query_file_path": "../../query_sets/news_queries.json",      "index_id": "cust_news_redis", "index_type": "CustomIndex", "dataset": "News",      "info": "BOOLEAN", "dstore": "REDIS",   "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        # Custom Index - Wikipedia Dataset
        {"query_file_path": "../../query_sets/wikipedia_queries.json", "index_id": "cust_wiki_cust",  "index_type": "CustomIndex", "dataset": "Wikipedia", "info": "BOOLEAN", "dstore": "CUSTOM",  "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["id", "text"]},
        {"query_file_path": "../../query_sets/wikipedia_queries.json", "index_id": "cust_wiki_rocks", "index_type": "CustomIndex", "dataset": "Wikipedia", "info": "BOOLEAN", "dstore": "ROCKSDB", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["id", "text"]},
        {"query_file_path": "../../query_sets/wikipedia_queries.json", "index_id": "cust_wiki_redis", "index_type": "CustomIndex", "dataset": "Wikipedia", "info": "BOOLEAN", "dstore": "REDIS",   "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["id", "text"]}
    ]

    print(f"{Style.FG_MAGENTA}Calculating Latency of Queries Execution for Different Data Stores...{Style.RESET}")
    for args in latency_diff_dstore_args_list:
        print(f"{Style.FG_CYAN}Calculating latency for DataStore: {args.get("dstore")}, Dataset: {args.get("dataset")}...{Style.RESET}")
        latencies = calc_latency(args.get("query_file_path"), args.get("index_id"), args.get("index_type"), args.get("dataset"), args.get("info"), args.get("dstore"), args.get("qproc"), args.get("compr"), args.get("optim"), args.get("attributes"))

        # Store the latency data to a JSON file
        temp_output_file_path: str = get_temp_file_path("latency", args)
        with open(temp_output_file_path, 'w') as f:
            json.dump({"latencies": latencies}, f)

    print(f"{Style.FG_MAGENTA}Plotting Latency Comparison for Different Data Stores...{Style.RESET}")

    output_file_path: str = os.path.join(output_dir, "diff_dstore_latency_comparison.png")
    plot_latency_comparison(output_dir, latency_diff_dstore_args_list ,plot_es=PLOT_ES)
    reset()


    # ========================================================================
    # Latency & Throughput comparison for different compression techniques
    # ========================================================================
    latency_throughput_diff_compression_args_list = [
        # Order: query_file_path, index_id, index_type, dataset, info, dstore, qproc, compr, optim, attributes
        # ES Index - News Dataset
        {"query_file_path": "../../query_sets/news_queries.json",      "index_id": "es_news",         "index_type": "ESIndex",     "dataset": "News",      "info": "NONE",    "dstore": "NONE", "qproc": "NONE", "compr": "NONE", "optim": "NONE",    "attributes": ["uuid", "text"]},
        # ES Index - Wikipedia Dataset
        {"query_file_path": "../../query_sets/wikipedia_queries.json", "index_id": "es_wiki",         "index_type": "ESIndex",     "dataset": "Wikipedia", "info": "NONE",    "dstore": "NONE", "qproc": "NONE", "compr": "NONE", "optim": "NONE",    "attributes": ["id", "text"]},
        # Custom Index - News Dataset
        {"query_file_path": "../../query_sets/news_queries.json",      "index_id": "cust_news_none",  "index_type": "CustomIndex", "dataset": "News",      "info": "BOOLEAN", "dstore": "CUSTOM",  "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        {"query_file_path": "../../query_sets/news_queries.json",      "index_id": "cust_news_code",  "index_type": "CustomIndex", "dataset": "News",      "info": "BOOLEAN", "dstore": "CUSTOM",  "qproc": "TERM", "compr": "CODE", "optim": "NONE", "attributes": ["uuid", "text"]},
        {"query_file_path": "../../query_sets/news_queries.json",      "index_id": "cust_news_clib",  "index_type": "CustomIndex", "dataset": "News",      "info": "BOOLEAN", "dstore": "CUSTOM",  "qproc": "TERM", "compr": "CLIB", "optim": "NONE", "attributes": ["uuid", "text"]},
        # Custom Index - Wikipedia Dataset
        {"query_file_path": "../../query_sets/wikipedia_queries.json", "index_id": "cust_wiki_none",  "index_type": "CustomIndex", "dataset": "Wikipedia", "info": "BOOLEAN", "dstore": "CUSTOM",  "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["id", "text"]},
        {"query_file_path": "../../query_sets/wikipedia_queries.json", "index_id": "cust_wiki_code",  "index_type": "CustomIndex", "dataset": "Wikipedia", "info": "BOOLEAN", "dstore": "CUSTOM",  "qproc": "TERM", "compr": "CODE", "optim": "NONE", "attributes": ["id", "text"]},
        {"query_file_path": "../../query_sets/wikipedia_queries.json", "index_id": "cust_wiki_clib",  "index_type": "CustomIndex", "dataset": "Wikipedia", "info": "BOOLEAN", "dstore": "CUSTOM",  "qproc": "TERM", "compr": "CLIB", "optim": "NONE", "attributes": ["id", "text"]}
    ]

    print(f"{Style.FG_MAGENTA}Calculating Latency and Throughput of Queries Execution for Different Compression Techniques...{Style.RESET}")
    for args in latency_throughput_diff_compression_args_list:
        print(f"{Style.FG_CYAN}Calculating latency and throughput for Compression: {args.get("compr")}, Dataset: {args.get("dataset")}...{Style.RESET}")
        latencies = calc_latency(args.get("query_file_path"), args.get("index_id"), args.get("index_type"), args.get("dataset"), args.get("info"), args.get("dstore"), args.get("qproc"), args.get("compr"), args.get("optim"), args.get("attributes"))
        throughput = calc_throughput(args.get("query_file_path"), args.get("index_id"), args.get("index_type"), args.get("dataset"), args.get("info"), args.get("dstore"), args.get("qproc"), args.get("compr"), args.get("optim"), args.get("attributes"))

        # Store the latency data to a JSON file
        temp_latency_output_file_path: str = get_temp_file_path("latency", args)
        with open(temp_latency_output_file_path, 'w') as f:
            json.dump({"latencies": latencies}, f)

        # Store the throughput data to a JSON file
        temp_throughput_output_file_path: str = get_temp_file_path("throughput", args)
        with open(temp_throughput_output_file_path, 'w') as f:
            json.dump({"throughput": throughput}, f)

    print(f"{Style.FG_MAGENTA}Plotting Latency and Throughput Comparison for Different Compression Techniques...{Style.RESET}")

    output_file_path: str = os.path.join(output_dir, "diff_compression_latency_comparison.png")
    plot_latency_comparison(output_file_path, latency_throughput_diff_compression_args_list, plot_es=PLOT_ES)

    output_file_path: str = os.path.join(output_dir, "diff_compression_throughput_comparison.png")
    plot_throughput_comparison(output_file_path, latency_throughput_diff_compression_args_list)
    reset()

