# ======================== IMPORTS ========================
import utils
from dataset_handlers import NewsDataset, WikipediaDataset


# ========================= MAIN ==========================
def main() -> None:
    # Create instances of dataset handler classes
    news_dataset_handler = NewsDataset()
    wikipedia_dataset_handler = WikipediaDataset()

    # Load configuration
    config: dict = utils.load_config()

    # Download and unzip news dataset
    path_to_news_dataset: str = config["data"]["news"]["path"]
    unzip: bool = config["data"]["news"]["unzip"]

    news_dataset_handler.download_news_dataset(path_to_news_dataset)
    if unzip:
        news_dataset_handler.unzip_and_clean_news_dataset(path_to_news_dataset)

    # Download wikipedia dataset
    path_to_wikipedia_dataset: str = config["data"]["wikipedia"]["path"]

    wikipedia_dataset_handler.download_wikipedia_dataset(path_to_wikipedia_dataset)


if __name__ == "__main__":
    main()
