# ======================== IMPORTS ========================
import utils
from dataset_handlers import NewsDataset, WikipediaDataset


# ========================= MAIN ==========================
def main() -> None:
    # Load configuration
    config: dict = utils.load_config()

    # Download and unzip news dataset
    path_to_news_dataset: str = config["data"]["news"]["path"]
    unzip: bool = config["data"]["news"]["unzip"]

    news_dataset_handler = NewsDataset(path_to_news_dataset, unzip)
    news_dataset_handler.download_dataset()

    # Download wikipedia dataset
    path_to_wikipedia_dataset: str = config["data"]["wikipedia"]["path"]

    wikipedia_dataset_handler = WikipediaDataset(path_to_wikipedia_dataset)
    wikipedia_dataset_handler.download_dataset()


if __name__ == "__main__":
    main()
