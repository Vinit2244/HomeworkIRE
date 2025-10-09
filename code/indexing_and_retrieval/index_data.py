# ======================== IMPORTS ========================
from utils import style
from utils import load_config
from indexes import ESIndex, CustomIndex
from dataset_managers import NewsDataset, WikipediaDataset


# ======================= GLOBALS ========================
config: dict = None
host: str | None = None
port: int | None = None
scheme: str | None = None


# ======================= FUNCTIONS =======================
def create_es_index(args: dict) -> None:
    print(f"{style.FG_YELLOW}Initializing Elasticsearch Index...{style.RESET}")
    idx = ESIndex(host, port, scheme, args["core"], args["info"], args["dstore"], args["qproc"], args["compr"], args["optim"])
    
    # List all available indices
    indices = idx.list_indices()
    print(f"{style.FG_CYAN}Available indices:{style.RESET}")
    for index in indices:
        print(f"{style.FG_GREEN}{index["index"]}{style.RESET}")
    print()
    
    # Create new index
    print(f"{style.FG_YELLOW}Creating index...{style.RESET}")
    index_id: str = args["core"]+"-v1.0"+"-"+args["dataset"]
    index_id = index_id.lower()  # Elasticsearch index names must be lowercase
    max_num_documents: int = config["max_num_documents"] if config["max_num_documents"] is not None else -1

    if args['dataset'] == 'news':
        dataset_path: str = config['data'][args['dataset']]['path']
        unzipped: bool = config['data'][args['dataset']]['unzip']

        news_dataset_handler = NewsDataset(dataset_path, max_num_documents, unzipped)
        idx.create_index(index_id, news_dataset_handler.get_files(args["attributes"]))
        print(idx.es_client.count(index=index_id))  # Print document count in the index
    
    elif args['dataset'] == 'wikipedia':
        ...
    else:
        raise ValueError("Invalid dataset specified in the configuration.")
    
    print(f"{style.FG_GREEN}Index creation completed. Index ID: {index_id}{style.RESET}\n")


def create_custom_index(args: dict) -> None:
    idx = CustomIndex(args["core"], args["info"], args["dstore"], args["qproc"], args["compr"], args["optim"])
    if args['dataset'] == 'news':
        ...
    elif args['dataset'] == 'wikipedia':
        ...
    else:
        raise ValueError("Invalid dataset specified in the configuration.")


# ========================= MAIN ==========================
def main(args) -> None:
    if args["core"] == 'ESIndex':
        create_es_index(args)
    elif args["core"] == 'CustomIndex':
        create_custom_index(args)
    else:
        raise ValueError("Invalid index core/type specified in the configuration.")


if __name__ == "__main__":
    config = load_config()
    host = config["elasticsearch"]["host"]
    port = config["elasticsearch"]["port"]
    scheme = config["elasticsearch"]["scheme"]

    main(config["index"])
