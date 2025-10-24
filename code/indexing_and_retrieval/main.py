# ======================== IMPORTS ========================
import json
import pprint
import argparse
from tqdm import tqdm
from typing import List
from rich.table import Table
from rich.console import Console
from rich.progress import Progress
from indexes import ESIndex, CustomIndex, BaseIndex
from dataset_managers import get_news_dataset_handler, get_wikipedia_dataset_handler
from utils import Style, StatusCode, load_config, clear_screen, wait_for_enter, IndexType, DatasetType


# ======================= GLOBALS ========================
config: dict = load_config()
host: str   = config["elasticsearch"]["host"]
port: int   = config["elasticsearch"]["port"]
scheme: str = config["elasticsearch"]["scheme"]
settings: List[str] = []


# =================== HELPER FUNCTIONS ====================
def handle_create_index(idx: BaseIndex, _type: IndexType) -> None:
    settings.append("Operation: Create Index")
    print_settings()

    # Select dataset to index
    print(f"{Style.FG_YELLOW}Select dataset{Style.RESET}")
    for dataset in DatasetType:
        print(f"{Style.FG_CYAN}  {dataset.value}. {dataset.name}{Style.RESET}")
    dataset: int = int(input(f"{Style.FG_YELLOW}Enter choice: {Style.RESET}").strip())

    # Convert to DatasetType enum and validate
    dataset_handler = None
    for dataset_type in DatasetType:
        if dataset_type.value == dataset:
            dataset = dataset_type
            if dataset == DatasetType.News:
                dataset_handler = get_news_dataset_handler()
            else:
                dataset_handler = get_wikipedia_dataset_handler()
            break

    if dataset not in DatasetType:
        print(f"{Style.FG_RED}Invalid dataset. Please try again.{Style.RESET}")
        return

    # Get index name and attributes to index
    index_id: str = input(f"{Style.FG_YELLOW}Index name (Do not include '.' in name): {Style.RESET}").strip().lower()
    if index_id == "":
        print(f"{Style.FG_RED}Index name cannot be empty.{Style.RESET}")
        return
    
    if '.' in index_id:
        print(f"{Style.FG_RED}Index name should not contain '.'. Please try again.{Style.RESET}")
        return
    
    available_attributes: List[str] = dataset_handler.get_attributes()
    print(f"\n{Style.FG_CYAN}[Make sure the first attribute is the uuid for identification]{Style.RESET}\n{Style.FG_GREEN}Available attributes:{Style.RESET} {', '.join(available_attributes)}")
    attributes: str = input(f"{Style.FG_YELLOW}Enter attributes to index (comma-separated, e.g., uuid,text): {Style.RESET}").strip().lower()
    attributes: List[str] = [attr.strip() for attr in attributes.split(",")]
    
    status: int = idx.create_index(index_id, dataset_handler.get_files(attributes))

    if status == StatusCode.SUCCESS:
        print(f"{Style.FG_GREEN}Index '{index_id}' created successfully.{Style.RESET}")
    elif status == StatusCode.INDEXING_FAILED:
        print(f"{Style.FG_RED}Indexing failed for index '{index_id}'.{Style.RESET}")
    elif status == StatusCode.INDEX_ALREADY_EXISTS:
        print(f"{Style.FG_RED}Index '{index_id}' already exists.{Style.RESET}")


def generate_index_id(core: str, dataset: str, version: str="v1.0") -> str:
    return (core + "-" + version + "-" + dataset).lower()  # Elasticsearch index names must be lowercase


def pretty_print_query_results(results_json: str) -> None:
    try:
        results = json.loads(results_json)
        hits = results.get("hits", {}).get("hits", [])
        total = results.get("hits", {}).get("total", {}).get("value", len(hits))

        print(f"\n{Style.FG_CYAN}{Style.BOLD}Search Results{Style.RESET}")
        print(f"{Style.FG_CYAN}{'='*60}{Style.RESET}")
        print(f"{Style.FG_YELLOW}Total hits:{Style.RESET} {total}\n")

        if not hits:
            print(f"{Style.FG_RED}No matching documents found.{Style.RESET}")
            return

        for i, hit in enumerate(hits, start=1):
            print(f"{Style.FG_MAGENTA}{Style.BOLD}Result #{i}{Style.RESET}")
            print(f"{Style.FG_CYAN}{'-'*60}{Style.RESET}")
            print(f"{Style.FG_BLUE}Index :{Style.RESET} {hit.get('_index', 'N/A')}")
            print(f"{Style.FG_BLUE}ID    :{Style.RESET} {hit.get('_id', 'N/A')}")
            print(f"{Style.FG_BLUE}Score :{Style.RESET} {hit.get('_score', 'N/A')}")

            source = hit.get("_source", {})
            for key, value in source.items():
                print(f"{Style.FG_GREEN}{key.capitalize():<6}:{Style.RESET} {value}")
            
            print(f"{Style.FG_CYAN}{'-'*60}{Style.RESET}\n")

    except Exception as e:
        print(f"{Style.FG_RED}Error parsing results.{Style.RESET}")


# ======================= FUNCTIONS =======================
def menu() -> None:
    global settings

    while True:
        clear_screen()
        settings.clear()

        # Select index type
        print(f"{Style.FG_YELLOW}Select Index Type: {Style.RESET}")
        for index_type in IndexType:
            print(f"{Style.FG_CYAN}  {index_type.value}. {index_type.name}{Style.RESET}")
        print()
        _type: int = int(input(f"{Style.FG_YELLOW}Enter choice: {Style.RESET}").strip())

        # Convert to IndexType enum and validate
        for index_type in IndexType:
            if index_type.value == _type:
                _type = index_type
                break
        print()

        match _type:
            case IndexType.ESIndex:
                print(f"{Style.FG_YELLOW}Using Elasticsearch Index at {host}:{port} with scheme {scheme}{Style.RESET}. To change, modify config.yaml file.\n")
                idx = ESIndex(host, port, scheme, _type.name)
                settings.append("Index Type: ESIndex")
            case IndexType.CustomIndex:
                index_settings: dict = config.get("index")
                info: str = index_settings.get("info", "NONE")
                dstore: str = index_settings.get("dstore", "NONE")
                qproc: str = index_settings.get("qproc", "NONE")
                compr: str = index_settings.get("compr", "NONE")
                optim: str = index_settings.get("optim", "NONE")
                print(f"{Style.FG_YELLOW}Using Custom Index with settings \n\tInfo: {info}, \n\tData Store: {dstore}, \n\tQuery Processor: {qproc}, \n\tCompression: {compr}, \n\tOptimization: {optim}.{Style.RESET} \nTo change, modify config.yaml file.\n")
                idx = CustomIndex(_type.name, info, dstore, qproc, compr, optim)
                settings.append("Index Type: CustomIndex")
            case _:
                print(f"{Style.FG_RED}Invalid index type. Please try again.{Style.RESET}")
                wait_for_enter()
                continue

        wait_for_enter()

        # Main menu loop
        while True:
            settings = settings[:1]  # Retain only the index type in settings
            print_settings()

            print(f"{Style.FG_MAGENTA}Please select an option:{Style.RESET}")
            print(f"{Style.FG_CYAN}1. Create Index{Style.RESET}")
            print(f"{Style.FG_CYAN}2. Delete Index{Style.RESET}")
            print(f"{Style.FG_CYAN}3. List Indices{Style.RESET}")
            print(f"{Style.FG_CYAN}4. Get Index Info{Style.RESET}")
            print(f"{Style.FG_CYAN}5. Change Index Type{Style.RESET}")
            print(f"{Style.FG_CYAN}6. Ask a Query{Style.RESET}")
            print(f"{Style.FG_CYAN}7. List files in Index{Style.RESET}")
            print(f"{Style.FG_CYAN}8. Exit{Style.RESET}")

            print()
            opt: int = int(input(f"{Style.FG_YELLOW}Enter your choice: {Style.RESET}").strip().lower())

            match opt:
                # Create Index
                case 1:
                    handle_create_index(idx, _type)
                    wait_for_enter()
                    continue
                
                # Delete Index
                case 2:
                    settings.append("Operation: Delete Index")
                    print_settings()

                    index_id: str = input(f"{Style.FG_YELLOW}Enter index name to delete (Enter full index name with '.'): {Style.RESET}").strip()
                    
                    if index_id == "":
                        print(f"{Style.FG_RED}Index name cannot be empty.{Style.RESET}")
                        wait_for_enter()
                        continue

                    info = idx.delete_index(index_id)

                    if info == StatusCode.ERROR_ACCESSING_INDEX:
                        print(f"{Style.FG_RED}Error accessing index '{index_id}'.{Style.RESET}")
                        wait_for_enter()
                        continue

                    print(f"{Style.FG_GREEN}Index '{index_id}' deleted successfully.{Style.RESET}")
                    wait_for_enter()
                
                # List Indices
                case 3:
                    settings.append("Operation: List Indices")
                    print_settings()

                    indices: list = idx.list_indices()
                    print(f"{Style.FG_CYAN}Available indices:{Style.RESET}")
                    for index in indices:
                        print(f"{Style.FG_GREEN}{index['index']}{Style.RESET}")
                    print()
                    wait_for_enter()

                # Get Index Info
                case 4:
                    settings.append("Operation: Get Index Info")
                    print_settings()

                    index_id: str = input(f"{Style.FG_YELLOW}Enter index name to get info: {Style.RESET}").strip()
                    
                    if index_id == "":
                        print(f"{Style.FG_RED}Index name cannot be empty.{Style.RESET}\n")
                        wait_for_enter()
                        continue

                    info = idx.get_index_info(index_id)

                    if info == StatusCode.ERROR_ACCESSING_INDEX:
                        print(f"{Style.FG_RED}Error accessing index '{index_id}'.{Style.RESET}\n")
                        wait_for_enter()
                        continue
                    
                    print(f"{Style.FG_CYAN}Index Info for '{index_id}':{Style.RESET}")
                    pprint.pprint(info[index_id])
                    wait_for_enter()
                
                # Change Index Type
                case 5:
                    break  # Break to outer loop to change index type
            
                # Ask a Query
                case 6:
                    settings.append("Operation: Ask a Query")
                    print_settings()

                    index_id = input(f"{Style.FG_MAGENTA}Input index name to query (leave it empty to search across all indexes): {Style.RESET}").strip().lower()
                    success = idx.load_index(index_id)
                    
                    if success != StatusCode.SUCCESS:
                        print(f"{Style.FG_RED}Failed to load index '{index_id}'.{Style.RESET}\n")
                        wait_for_enter()
                        continue
                    
                    settings.append(f"Index: {index_id}" if index_id != "" else "Index: All")

                    while True:
                        print_settings()
                        query = input(f"{Style.FG_MAGENTA}Input your query (type 'EXIT' to go back to previous menu): {Style.RESET}").strip()
                        
                        if query.upper() == "EXIT":
                            break
                        
                        results = idx.query(query, index_id)
                        pretty_print_query_results(results)
                        wait_for_enter()

                # List files in Index
                case 7:
                    settings.append("Operation: List files in Index")
                    print_settings()

                    index_id: str = input(f"{Style.FG_YELLOW}Enter index name to list files: {Style.RESET}").strip()
                    indexed_files = idx.list_indexed_files(index_id)
                    
                    console = Console()
                    table = Table(title=f"Indexed Files in '{index_id}'", show_lines=False)
                    table.add_column("S.No", justify="right", style="cyan", no_wrap=True)
                    table.add_column("File Name", style="green")

                    count = 0
                    for doc in tqdm(indexed_files, desc="Fetching files", unit="files"):
                        count += 1
                        table.add_row(str(count), doc)
                        if count % 5000 == 0:  # print in batches for memory safety
                            console.print(table)
                            table = Table(show_lines=False)
                            table.add_column("S.No", justify="right", style="cyan", no_wrap=True)
                            table.add_column("File Name", style="green")

                    if count % 5000 != 0:  # print remaining rows
                        console.print(table)
                    
                    console.print(f"[bold green]Total files indexed: {count}")

                    wait_for_enter()

                # Exit
                case 8:
                    print(f"{Style.FG_GREEN}Exiting...{Style.RESET}")
                    exit(0)

                # Invalid Option
                case _:
                    print(f"{Style.FG_RED}Invalid option. Please try again.{Style.RESET}")
                    wait_for_enter()


def print_settings() -> None:
    clear_screen()
    for setting in settings:
        print(f"{Style.FG_ORANGE}[!] {setting}{Style.RESET}")
    print()


def create_es_index(args: dict) -> None:
    print(f"{Style.FG_YELLOW}Initializing Elasticsearch Index...{Style.RESET}")
    idx = ESIndex(host, port, scheme, args["core"])

    # Generating index ID
    print(f"{Style.FG_YELLOW}Creating index...{Style.RESET}")
    

    # Selecting dataset handler and validating
    if args["dataset"] == DatasetType.News.name:
        dataset_handler = get_news_dataset_handler()
    elif args["dataset"] == DatasetType.Wikipedia.name:
        dataset_handler = get_wikipedia_dataset_handler()
    else:
        raise ValueError("Invalid dataset specified in the configuration.")

    # Creating the index
    index_id: str = generate_index_id(args["core"], args["dataset"], args.get("version", "v1.0"))
    status: int = idx.create_index(index_id, dataset_handler.get_files(args["attributes"]))
        
    if status == StatusCode.SUCCESS:
        print(f"{Style.FG_GREEN}Index '{index_id}' created successfully.{Style.RESET}")
    elif status == StatusCode.INDEXING_FAILED:
        print(f"{Style.FG_RED}Indexing failed for index '{index_id}'.{Style.RESET}")
    elif status == StatusCode.INDEX_ALREADY_EXISTS:
        print(f"{Style.FG_RED}Index '{index_id}' already exists.{Style.RESET}")


def create_custom_index(args: dict) -> None:
    idx = CustomIndex(args["core"], args["info"], args["dstore"], args["qproc"], args["compr"], args["optim"])

    if args["dataset"] == "news":
        ...
    elif args["dataset"] == "wikipedia":
        ...
    else:
        raise ValueError("Invalid dataset specified in the configuration.")


# ========================= MAIN ==========================
def main(params, mode: str) -> None:
    if mode == "config":
        if params["core"] == "ESIndex":
            create_es_index(params)
        elif params["core"] == "CustomIndex":
            # TODO: After implementing CustomIndex, uncomment the line below
            # create_custom_index(params)
            ...
        else:
            raise ValueError("Invalid index core/type specified in the configuration.")
    elif mode == "manual":
        menu()
    else:
        raise ValueError("Invalid mode specified. Use 'manual' or 'config'.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mode", type=str, default="manual", choices=['manual', 'config'], help="Mode of operation. 'manual' for interactive menu, 'config' for config file.")
    args = argparser.parse_args()

    # TODO: Remove this block after implementing config mode
    if args.mode == "config":
        print(f"{Style.FG_YELLOW}Yet to be implemented, please run in manual mode...{Style.RESET}")
        exit(0)

    main(config["index"], args.mode)
