# ======================== IMPORTS ========================
import os
import utils
import argparse
from utils import style
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from dataset_handlers import NewsDataset, WikipediaDataset


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
    # Create instances of dataset handler classes
    news_dataset_handler = NewsDataset()
    wikipedia_dataset_handler = WikipediaDataset()

    # Load configuration
    config: dict = utils.load_config()
    top_k_threshold: int = config["top_k_threshold"]
    output_folder_path: str = config["output_folder_path"]
    os.makedirs(output_folder_path, exist_ok=True)

    # News Dataset
    print(f"{style.FG_CYAN}Calculating word frequency for news dataset...{style.RESET}")
    news_data_path: str = config["data"]["news"]["path"]
    unzipped: bool = config["data"]["news"]["unzip"]
    news_data_freq_dist: dict = news_dataset_handler.calculate_word_frequency(news_data_path, unzipped)
    plot_frequency_distribution(news_data_freq_dist,
                                top_k_threshold,
                                "Word Frequency Distribution for News Dataset",
                                "Words",
                                "Frequencies",
                                os.path.join(output_folder_path, f"news_word_frequency_{args.data_state}_top_{top_k_threshold}.png"))
    print(f"{style.FG_GREEN}Frequency plot for news dataset saved at {output_folder_path}\n{style.RESET}")

    # Wikipedia Dataset
    print(f"{style.FG_CYAN}Calculating word frequency for wikipedia dataset...{style.RESET}")
    wikipedia_data_path: str = config["data"]["wikipedia"]["path"]
    wikipedia_data_freq_dist: dict = wikipedia_dataset_handler.calculate_word_frequency(wikipedia_data_path)
    plot_frequency_distribution(wikipedia_data_freq_dist,
                                top_k_threshold,
                                "Word Frequency Distribution for Wikipedia Dataset",
                                "Words",
                                "Frequencies",
                                os.path.join(output_folder_path, f"wikipedia_word_frequency_{args.data_state}_top_{top_k_threshold}.png"))
    print(f"{style.FG_GREEN}Frequency plot for wikipedia dataset saved at {output_folder_path}\n{style.RESET}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_state", type=str, help="State of the data - preprocessed / raw")
    args = argparser.parse_args()
    
    main(args)
