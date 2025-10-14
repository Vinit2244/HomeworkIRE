# ======================== IMPORTS ========================
import json
import pprint
import argparse
from typing import List
from indexes import ESIndex, CustomIndex, BaseIndex
from dataset_managers import get_news_dataset_handler, get_wikipedia_dataset_handler
from utils import Style, StatusCode, load_config, clear_screen, wait_for_enter, IndexType, DatasetType


# ======================= GLOBALS ========================
config: dict = None
host: str | None = None
port: int | None = None
scheme: str | None = None
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
    index_id: str = input(f"{Style.FG_YELLOW}Index name: {Style.RESET}").strip().lower()
    if index_id == "":
        print(f"{Style.FG_RED}Index name cannot be empty.{Style.RESET}")
        return
    
    available_attributes: List[str] = dataset_handler.get_attributes()
    print(f"\n{Style.FG_CYAN}[Make sure the first attribute is the uuid for identification]{Style.RESET}\n{Style.FG_GREEN}Available attributes:{Style.RESET} {', '.join(available_attributes)}")
    attributes: str = input(f"{Style.FG_YELLOW}Enter attributes to index (comma-separated, e.g., uuid,text): {Style.RESET}").strip().lower()
    attributes: List[str] = [attr.strip() for attr in attributes.split(",")]
    
    # ESIndex
    if _type == IndexType.ESIndex:
        status: int = idx.create_index(index_id, dataset_handler.get_files(attributes))

        if status == StatusCode.SUCCESS:
            print(f"{Style.FG_GREEN}Index '{index_id}' created successfully.{Style.RESET}")
        elif status == StatusCode.INDEXING_FAILED:
            print(f"{Style.FG_RED}Indexing failed for index '{index_id}'.{Style.RESET}")

    # CustomIndex
    # TODO: Implement CustomIndex class and create_index logic
    else:

        # News Dataset
        if dataset == DatasetType.News:
            ...

        # Wikipedia Dataset
        else:
            ...


def generate_index_id(core: str, dataset: str, version: str="v1.0") -> str:
    return (core + "-" + version + "-" + dataset).lower()  # Elasticsearch index names must be lowercase


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
                idx = ESIndex(host, port, scheme, _type.name)
                settings.append("Index Type: ESIndex")
            case IndexType.CustomIndex:
                # TODO: Implement CustomIndex class and its initialization
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
            print(f"{Style.FG_CYAN}6. Exit{Style.RESET}")

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

                    index_id: str = input(f"{Style.FG_YELLOW}Enter index name to delete: {Style.RESET}").strip()
                    
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
            
                # Exit
                case 6:
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
            create_custom_index(params)
        else:
            raise ValueError("Invalid index core/type specified in the configuration.")
    elif mode == "manual":
        menu()
    else:
        raise ValueError("Invalid mode specified. Use 'manual' or 'config'.")


if __name__ == "__main__":
    config = load_config()
    host = config["elasticsearch"]["host"]
    port = config["elasticsearch"]["port"]
    scheme = config["elasticsearch"]["scheme"]

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mode", type=str, default="manual", choices=['manual', 'config'], help="Mode of operation. 'manual' for interactive menu, 'config' for config file.")
    args = argparser.parse_args()

    main(config["index"], args.mode)
