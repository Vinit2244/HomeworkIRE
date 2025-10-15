# ======================== IMPORTS ========================
import os
import json
import time
import argparse
import seaborn as sns
from enum import Enum
from typing import List
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from indexes import ESIndex, CustomIndex
from utils import Style, IndexType, load_config


# ======================= CLASSES =========================
class PerformanceMetrics:
    Latency: str = "latency"
    Throughput: str = "throughput"
    MemoryUsage: str = "memory_usage"
    FunctionalMetrics: str = "functional_metrics"


# =================== HELPER FUNCTIONS ====================
def comma_separated_list(arg):
    return arg.split(',')


def print_summary(outputs: dict) -> None:
    
    print("\n================== Performance Metrics Summary ==================")
    
    table = PrettyTable()
    table.field_names = ["Metric", "Key", "Value"]
    table.align = "l"
    
    for metric, result in outputs.items():
        if isinstance(result, dict):
            for key, value in result.items():
                if key == "all_latencies_seconds":
                    continue  # Skip printing all latencies to avoid clutter
                
                # Round numeric values to 3 decimal places
                if isinstance(value, (int, float)):
                    formatted_value = round(value, 3)
                else:
                    formatted_value = value
                
                table.add_row([metric.upper(), key, formatted_value])
        else:
            # Round numeric values to 3 decimal places
            if isinstance(result, (int, float)):
                formatted_result = round(result, 3)
            else:
                formatted_result = result
            table.add_row([metric.upper(), "", formatted_result])
    
    print(table)
    print("=================================================================\n")

# ======================= FUNCTIONS =======================
def calc_latency(index, file_path: str, output_folder: str=None, suffix: str=None) -> None:
    with open(file_path, 'r') as f:
        data = json.load(f)

    index_id: str = data.get("index_id", "")
    
    queries = data.get("queries", [])
    if not queries:
        print(f"{Style.FG_ORANGE}No queries found in the file.{Style.RESET}")
        return
    
    latencies: List[int] = list()
    
    for query in queries:
        start_time = time.time()
        _ = index.query(query.get("query", ""), index_id)
        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)
        print(f"Query: {query.get('query', '')} | Latency: {round(latency, 4)} seconds")

    # Print average, p95 and p99 latencies and plot a graph to visualise
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        latencies.sort()
        p95_latency = latencies[int(0.95 * len(latencies)) - 1]
        p99_latency = latencies[int(0.99 * len(latencies)) - 1]
        print(f"\n{Style.FG_CYAN}Average Latency: {round(avg_latency, 4)} seconds{Style.RESET}")
        print(f"{Style.FG_CYAN}95th Percentile Latency: {round(p95_latency, 4)} seconds{Style.RESET}")
        print(f"{Style.FG_CYAN}99th Percentile Latency: {round(p99_latency, 4)} seconds{Style.RESET}")

        plt.figure(figsize=(10, 6))
        sns.histplot(latencies, bins=20, kde=True)
        plt.title(f'Query Latency Distribution for {index_type}')
        plt.xlabel('Latency (seconds)')
        plt.ylabel('Frequency')
        plt.axvline(avg_latency, color='r', linestyle='--', label='Average Latency')
        plt.axvline(p95_latency, color='g', linestyle='--', label='95th Percentile Latency')
        plt.axvline(p99_latency, color='b', linestyle='--', label='99th Percentile Latency')
        plt.legend()
        
        if output_folder is None:
            plt.show()
        else:
            output_file_path: str = os.path.join(output_folder, f"latency_hist_{index.core}_{index_id}_{str(suffix)}.png")
            plt.savefig(output_file_path)
            print(f"{Style.FG_GREEN}Latency distribution plot saved at '{output_file_path}'.{Style.RESET}")
    else:
        print(f"{Style.FG_ORANGE}No latencies recorded.{Style.RESET}")
    
    return {
        "average_latency_seconds": avg_latency,
        "p95_latency_seconds": p95_latency,
        "p99_latency_seconds": p99_latency,
        "all_latencies_seconds": latencies
    }


def calc_throughput(index, file_path: str) -> None:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    index_id: str = data.get("index_id", "")
    queries = data.get("queries", [])
    if not queries:
        print(f"{Style.FG_ORANGE}No queries found in the file.{Style.RESET}")
        return
    
    start_time = time.time()
    for query in queries:
        _ = index.query(query.get("query", ""), index_id)
    end_time = time.time()
    total_time = end_time - start_time
    throughput = len(queries) / total_time if total_time > 0 else 0
    print(f"\n{Style.FG_CYAN}Throughput: {round(throughput, 2)} queries/second over {len(queries)} queries in {round(total_time, 4)} seconds.{Style.RESET}")

    return {
        "total_queries": len(queries),
        "total_time_seconds": total_time,
        "throughput_qps": throughput
    }


def calc_memory_usage():
    # TODO: Implement memory usage calculation
    ...


def calc_functional_metrics():
    # TODO: Implement functional metrics calculation (precision, recall, ranking measure)
    ...


# ========================= MAIN ==========================
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Runs the queries and reports the system response time for each query.")
    argparser.add_argument('--metrics', type=comma_separated_list, required=True, help="Performance metric to calculate. Comma separated list of all the metrics to calculate (latency, throughput, memory_usage, functional_metrics).")
    argparser.add_argument('--index_type', type=str, required=True, choices=['CustomIndex', 'ESIndex'], help="Type of index: 'custom' or 'es' (Elasticsearch).")
    argparser.add_argument('--dataset', type=str, required=True, choices=['News', 'Wikipedia'], help="Dataset type: 'news' or 'wikipedia'.")
    argparser.add_argument('--query_file', type=str, required=True, help="Path to the JSON file containing queries.")
    
    args = argparser.parse_args()
    metrics = args.metrics
    file_path = args.query_file
    index_type = args.index_type
    dataset = args.dataset

    config = load_config()
    output_dir = config.get("output_folder_path", ".")

    global ES_HOST, ES_PORT, ES_SCHEME
    ES_HOST = config["elasticsearch"].get("host", "localhost")
    ES_PORT = config["elasticsearch"].get("port", 9200)
    ES_SCHEME = config["elasticsearch"].get("scheme", "http")

    if index_type == IndexType.ESIndex.name:
        idx = ESIndex(ES_HOST, ES_PORT, ES_SCHEME, index_type)
    elif index_type == IndexType.CustomIndex.name:
        # TODO: After implementing CustomIndex, update the initialization here
        idx = CustomIndex()
    else:
        print(f"Invalid index type: {index_type}")
    
    outputs: dict = dict()

    if PerformanceMetrics.Latency in metrics:
        latency_output: dict = calc_latency(idx, file_path, output_dir, dataset)
        outputs["latency"] = latency_output
    
    if PerformanceMetrics.Throughput in metrics:
        throughput_output: dict = calc_throughput(idx, file_path)
        outputs["throughput"] = throughput_output
    
    if PerformanceMetrics.MemoryUsage in metrics:
        # TODO: After implementing memory usage calculation, update this
        memory_usage_output: dict = calc_memory_usage()
        outputs["memory_usage"] = memory_usage_output

    if PerformanceMetrics.FunctionalMetrics in metrics:
        # TODO: After implementing functional metrics calculation, update this
        functional_metrics_output: dict = calc_functional_metrics()
        outputs["functional_metrics"] = functional_metrics_output

    print_summary(outputs)