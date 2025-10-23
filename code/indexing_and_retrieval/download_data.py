# ======================== IMPORTS ========================
from utils import Style, load_config
from dataset_managers import NewsDataset, WikipediaDataset


# ========================= MAIN ==========================
def main() -> None:
    # Load configuration
    config: dict = load_config()

    # Download and unzip news dataset
    path_to_news_dataset: str = config["data"]["news"]["path"]
    unzip: bool = config["data"]["news"]["unzip"]
    print(f"{Style.FG_YELLOW}Using \n\tPath to News: {path_to_news_dataset}{Style.RESET}, \n\tUnzip: {unzip}. \nTo change, modify config.yaml file.\n")

    news_dataset_handler = NewsDataset(path_to_news_dataset, -1, unzip)
    news_dataset_handler.download_dataset()

    # Download wikipedia dataset
    path_to_wikipedia_dataset: str = config["data"]["wikipedia"]["path"]
    print(f"{Style.FG_YELLOW}Using \n\tPath to Wiki: {path_to_news_dataset}{Style.RESET}. \nTo change, modify config.yaml file.\n")

    wikipedia_dataset_handler = WikipediaDataset(path_to_wikipedia_dataset, -1)
    wikipedia_dataset_handler.download_dataset()


if __name__ == "__main__":
    main()
