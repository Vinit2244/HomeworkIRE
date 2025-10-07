# ======================== IMPORTS ========================
import utils
from utils import style
from preprocessor import Preprocessor
from dataset_handlers import NewsDataset, WikipediaDataset
from typing import List


# ======================= FUNCTIONS =======================
def preprocess_data(config: dict) -> None:
    # Lower
    lowercase         : bool             = config["preprocessing"]["lowercase"]
    # Remove
    stopword_langs    : List[str] | None = config["preprocessing"]["stopwords"]["languages"]
    rem_stop          : bool             = True if stopword_langs else False
    rem_punc          : bool             = config["preprocessing"]["remove_punctuation"]
    rem_num           : bool             = config["preprocessing"]["remove_numbers"]
    rem_special       : bool             = config["preprocessing"]["remove_special_characters"]
    # Stemming
    stemming_algo     : str | None       = config["preprocessing"]["stemming"]["algorithm"]
    stem              : bool             = True if stemming_algo else False
    # Lemmatization
    lemmatization_algo: str | None       = config["preprocessing"]["lemmatization"]["algorithm"]
    lemmatize         : bool             = True if lemmatization_algo else False

    print(f"{style.FG_CYAN}Preprocessing news dataset...{style.RESET}")
    path_to_news_dataset: str = config["data"]["news"]["path"]
    unzip: bool = config["data"]["news"]["unzip"]
    NewsDataset(path_to_news_dataset, unzip).preprocess(lowercase, rem_stop, stopword_langs, rem_punc,
                             rem_num, rem_special, stem, stemming_algo,
                             lemmatize, lemmatization_algo)
    print(f"{style.FG_GREEN}Preprocessing of news dataset completed.\n{style.RESET}")

    print(f"{style.FG_CYAN}Preprocessing wikipedia dataset...{style.RESET}")
    path_to_wikipedia_dataset: str = config["data"]["wikipedia"]["path"]
    WikipediaDataset(path_to_wikipedia_dataset).preprocess(lowercase, rem_stop, stopword_langs, rem_punc,
                             rem_num, rem_special, stem, stemming_algo,
                             lemmatize, lemmatization_algo)
    print(f"{style.FG_GREEN}Preprocessing of wikipedia dataset completed.\n{style.RESET}")


# ========================= MAIN ==========================
def main():
    config = utils.load_config()
    preprocess_data(config)


if __name__ == "__main__":
    main()
