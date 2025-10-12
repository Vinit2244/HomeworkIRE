# ======================== IMPORTS ========================
import json
import pprint
import argparse
from enum import Enum
from typing import List
from indexes import ESIndex, CustomIndex
from utils import Style, StatusCode, load_config, clear_screen, wait_for_enter
from dataset_managers import get_news_dataset_handler, get_wikipedia_dataset_handler


# ======================= GLOBALS ========================
config: dict = None
host: str | None = None
port: int | None = None
scheme: str | None = None
settings: List[str] = []


# ======================= CLASSES ========================
class IndexType(Enum):
    ESIndex = 1
    CustomIndex = 2


# ======================= FUNCTIONS =======================
def menu() -> None:
    global settings

    while True:
        clear_screen()
        settings.clear()

        # Ask for index type
        print(f"{Style.FG_YELLOW}Select Index Type: {Style.RESET}")
        for index_type in IndexType:
            print(f"{Style.FG_CYAN}  {index_type.value}. {index_type.name}{Style.RESET}")
        print()
        type_selected: int = int(input(f"{Style.FG_YELLOW}Enter choice: {Style.RESET}").strip())
        _type = IndexType(type_selected).name
        print()
        if _type == "ESIndex":
            idx = ESIndex(host, port, scheme, _type)
            settings.append("Index Type: ESIndex")
        elif _type == "CustomIndex":
            # TODO: Implement CustomIndex class and its initialization
            settings.append("Index Type: CustomIndex")
        else:
            print(f"{Style.FG_RED}Invalid index type. Please try again.{Style.RESET}")
            wait_for_enter()
            continue

        wait_for_enter()

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

            if opt == 1:
                settings.append("Operation: Create Index")
                print_settings()

                # Ask for dataset to index
                dataset: str = input(f"{Style.FG_YELLOW}Enter dataset (news/wikipedia): {Style.RESET}").strip().lower()
                if dataset not in ["news", "wikipedia"]:
                    print(f"{Style.FG_RED}Invalid dataset. Please try again.{Style.RESET}")
                    wait_for_enter()
                    continue
                
                if _type == "ESIndex":
                    index_id: str = input(f"{Style.FG_YELLOW}Index name: {Style.RESET}").strip().lower()
                    attributes: str = input(f"{Style.FG_CYAN}[Make sure the first attribute is the uuid for identification]{Style.RESET}\n{Style.FG_YELLOW}Enter attributes to index (comma-separated, e.g., uuid,text): {Style.RESET}").strip().lower()
                    attributes: List[str] = [attr.strip() for attr in attributes.split(",")]
                    
                    if dataset == "news":
                        news_dataset_handler = get_news_dataset_handler()
                        status: int = idx.create_index(index_id, news_dataset_handler.get_files(attributes))
                        if status == StatusCode.SUCCESS:
                            print(f"{Style.FG_GREEN}Index '{index_id}' created successfully.{Style.RESET}")
                        elif status == StatusCode.INDEXING_FAILED:
                            print(f"{Style.FG_RED}Indexing failed for index '{index_id}'.{Style.RESET}")
                    else:
                        ...
                    wait_for_enter()
                        
            elif opt == 2:
                settings.append("Operation: Delete Index")
                print_settings()

                index_id: str = input(f"{Style.FG_YELLOW}Enter index name to delete: {Style.RESET}").strip()
                info = idx.delete_index(index_id)

                if info == StatusCode.ERROR_ACCESSING_INDEX:
                    print(f"{Style.FG_RED}Error accessing index '{index_id}'.{Style.RESET}")
                    wait_for_enter()
                    continue

                print(f"{Style.FG_GREEN}Index '{index_id}' deleted successfully.{Style.RESET}")
                wait_for_enter()

            elif opt == 3:
                settings.append("Operation: List Indices")
                print_settings()

                indices: list = idx.list_indices()
                print(f"{Style.FG_CYAN}Available indices:{Style.RESET}")
                for index in indices:
                    print(f"{Style.FG_GREEN}{index['index']}{Style.RESET}")
                print()
                wait_for_enter()

            elif opt == 4:
                settings.append("Operation: Get Index Info")
                print_settings()

                index_id: str = input(f"{Style.FG_YELLOW}Enter index name to get info: {Style.RESET}").strip()
                info = idx.get_index_info(index_id)

                if info == StatusCode.ERROR_ACCESSING_INDEX:
                    print(f"{Style.FG_RED}Error accessing index '{index_id}'.{Style.RESET}\n")
                    wait_for_enter()
                    continue
                
                print(f"{Style.FG_CYAN}Index Info for '{index_id}':{Style.RESET}")
                pprint.pprint(info[index_id])
                wait_for_enter()

            elif opt == 5:
                break  # Break to outer loop to change index type

            elif opt == 6:
                print(f"{Style.FG_GREEN}Exiting...{Style.RESET}")
                exit(0)
            
            else:
                print(f"{Style.FG_RED}Invalid option. Please try again.{Style.RESET}")
                wait_for_enter()


def print_settings() -> None:
    clear_screen()
    for setting in settings:
        print(f"{Style.FG_ORANGE}[!] {setting}{Style.RESET}")
    print()


def create_es_index(args: dict) -> None:
    print(f"{Style.FG_YELLOW}Initializing Elasticsearch Index...{Style.RESET}")
    idx = ESIndex(host, port, scheme, args["core"], args["info"], args["dstore"], args["qproc"], args["compr"], args["optim"])
    
    # Create new index
    print(f"{Style.FG_YELLOW}Creating index...{Style.RESET}")
    index_id: str = args["core"]+"-v1.0"+"-"+args["dataset"]
    index_id = index_id.lower()  # Elasticsearch index names must be lowercase

    if args["dataset"] == "news":
        news_dataset_handler = get_news_dataset_handler()
        status: int = idx.create_index(index_id, news_dataset_handler.get_files(args["attributes"]))
        
        if status == StatusCode.SUCCESS:
            print(f"{Style.FG_GREEN}Index '{index_id}' created successfully.{Style.RESET}")
        elif status == StatusCode.INDEXING_FAILED:
            print(f"{Style.FG_RED}Indexing failed for index '{index_id}'.{Style.RESET}")
    
    elif args["dataset"] == "wikipedia":
        ...
    else:
        raise ValueError("Invalid dataset specified in the configuration.")
    
    print(f"{Style.FG_GREEN}Index creation completed. Index ID: {index_id}{Style.RESET}\n")


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
    argparser.add_argument("--mode", type=str, default="manual", help="Mode of operation: 'manual' or 'config'")
    args = argparser.parse_args()

    main(config["index"], args.mode)
