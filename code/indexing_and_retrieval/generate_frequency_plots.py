# ======================== IMPORTS ========================
import os
import utils
import argparse
from utils import Style
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from dataset_managers import NewsDataset, WikipediaDataset


# ======================= FUNCTIONS =======================
def plot_frequency_distribution(freq_dict: Dict[str, int], k: int, title: str, xlabel: str, ylabel: str, output_file_path: str) -> None:
    
    def get_x_y(freq_dist: dict) -> Tuple[List[str], List[int]]:
        freqs: List[Tuple[str, int]] = [(word, count) for word, count in freq_dist.items()]
        freqs.sort(key=lambda x: x[1], reverse=True)
        x: List[str] = list()
        y: List[int] = list()
        for word, count in freqs:
            x.append(word)
            y.append(count)
        return x, y
    
    x, y = get_x_y(freq_dict)

    plt.figure(figsize=(10, 6))
    plt.bar(x[:k], y[:k])  # Plot only the top k for better visibility
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file_path)
    plt.close()


# ========================= MAIN =========================
def main(args) -> None:
    # Load configuration
    config: dict = utils.load_config()
    top_k_words_threshold: int = config["top_k_words_threshold"]
    max_num_documents: int = config["max_num_documents"] if config["max_num_documents"] is not None else -1
    output_folder_path: str = config["output_folder_path"]
    os.makedirs(output_folder_path, exist_ok=True)


    # News Dataset
    print(f"{Style.FG_CYAN}Calculating word frequency for news dataset...{Style.RESET}")
    news_data_path: str = config["data"]["news"]["path"]
    unzipped: bool = config["data"]["news"]["unzip"]

    news_dataset_handler = NewsDataset(news_data_path, max_num_documents, unzipped)
    news_data_freq_dist: dict = news_dataset_handler.calculate_word_frequency()
    plot_frequency_distribution(news_data_freq_dist,
                                top_k_words_threshold,
                                "Word Frequency Distribution for News Dataset",
                                "Words",
                                "Frequencies",
                                os.path.join(output_folder_path, f"news_word_frequency_{args.data_state}_top_{top_k_words_threshold}.png"))
    print(f"{Style.FG_GREEN}Frequency plot for news dataset saved at {output_folder_path}\n{Style.RESET}")


    # Wikipedia Dataset
    print(f"{Style.FG_CYAN}Calculating word frequency for wikipedia dataset...{Style.RESET}")
    wikipedia_data_path: str = config["data"]["wikipedia"]["path"]

    wikipedia_dataset_handler = WikipediaDataset(wikipedia_data_path, max_num_documents)
    wikipedia_data_freq_dist: dict = wikipedia_dataset_handler.calculate_word_frequency()
    plot_frequency_distribution(wikipedia_data_freq_dist,
                                top_k_words_threshold,
                                "Word Frequency Distribution for Wikipedia Dataset",
                                "Words",
                                "Frequencies",
                                os.path.join(output_folder_path, f"wikipedia_word_frequency_{args.data_state}_top_{top_k_words_threshold}.png"))
    print(f"{Style.FG_GREEN}Frequency plot for wikipedia dataset saved at {output_folder_path}\n{Style.RESET}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_state", type=str, help="State of the data - preprocessed / raw")
    args = argparser.parse_args()
    
    main(args)
