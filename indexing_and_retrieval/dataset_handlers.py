# ======================== IMPORTS ========================
import os
import json
import utils
import zipfile
import subprocess
import pandas as pd
from tqdm import tqdm
from utils import style
from typing import List
from collections import defaultdict


# ======================== CLASSES ========================
class NewsDataset:
    def __init__(self) -> None:
        pass

    def download_news_dataset(self, destination: str) -> None:
        print(f"{style.FG_CYAN}Downloading news dataset...{style.RESET}")

        # Make sure the destination directory exists
        os.makedirs(destination, exist_ok=True)

        # Clone the repository
        repo_url = "https://github.com/Webhose/free-news-datasets.git"
        subprocess.run(["git", "clone", repo_url, destination], check=True)

        print(f"{style.FG_GREEN}News dataset downloaded at {destination}\n{style.RESET}")

    def unzip_and_clean_news_dataset(self, path_to_news_dataset: str) -> None:
        print(f"{style.FG_CYAN}Unzipping and cleaning news dataset...{style.RESET}")

        # Remove all the files except the "News_Datasets" folder
        all_items: list = os.listdir(path_to_news_dataset)
        for item in all_items:
            if item != "News_Datasets":
                item_path: str = os.path.join(path_to_news_dataset, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    subprocess.run(["rm", "-rf", item_path], check=True)
        
        # Unzip all the zipped folders in "News_Datasets" and remove the zipped files
        path_to_zipped_folders: str = os.path.join(path_to_news_dataset, "News_Datasets")
        all_zipped_folders: list = os.listdir(path_to_zipped_folders)
        unzipped_count = 0
        for folder in all_zipped_folders:
            folder_path: str = os.path.join(path_to_zipped_folders, folder)
            if folder_path.endswith(".zip"):
                try:
                    subprocess.run(["unzip", "-o", folder_path, "-d", path_to_news_dataset], check=True)
                    unzipped_count += 1
                    os.remove(folder_path)
                except:
                    subprocess.run(["rm", "-rf", folder_path], check=True)
        print(f"{style.FG_GREEN}Unzipped {unzipped_count}/{len(all_zipped_folders)}{style.RESET}")

        print(f"{style.FG_GREEN}News dataset unzipped and cleaned at {path_to_news_dataset}\n{style.RESET}")

    def calculate_word_frequency(self, news_data_path: str, unzipped: bool) -> dict:
        freq: dict = defaultdict(int)

        # Updates the overall frequency dictionary with the frequency from a single json file
        def update_freq_dict(f):
            text: str = json.load(f)["text"]
            file_freq: dict = utils.get_word_freq_dist(text)
            for word, count in file_freq.items():
                freq[word] += count

        # The data has been downloaded and unzipped (unzip flag = true in config.yaml for news dataset)
        if unzipped:
            all_folders: List[str] = os.listdir(news_data_path)

            # Flatten list of all JSON files across all folders
            all_json_files_paths: List[str] = []
            for folder in all_folders:
                folder_path = os.path.join(news_data_path, folder)
                # Make sure it's a directory
                if os.path.isdir(folder_path):
                    json_files = os.listdir(folder_path)
                    for json_file in json_files:
                        all_json_files_paths.append(os.path.join(folder_path, json_file))

            # Iterate over all files with tqdm
            for json_file_path in tqdm(all_json_files_paths, desc="Calculating word frequencies"):
                with open(json_file_path, 'r') as f:
                    update_freq_dict(f)
        
        else:
            zipped_folders_path: str = os.path.join(news_data_path, "News_Datasets")
            all_zipped_folders: List[str] = os.listdir(zipped_folders_path)

            # Flatten all JSON files inside all zip folders
            all_json_entries = []

            for zipped_folder in all_zipped_folders:
                if zipped_folder.endswith(".zip"):
                    zip_path = os.path.join(zipped_folders_path, zipped_folder)
                    with zipfile.ZipFile(zip_path, "r") as z:
                        # store both zip path and internal JSON file name
                        for json_file in z.namelist():
                            all_json_entries.append((zip_path, json_file))

            # Iterate over all JSON files with tqdm
            for zip_path, json_file in tqdm(all_json_entries, desc="Calculating word frequencies"):
                with zipfile.ZipFile(zip_path, "r") as z:
                    with z.open(json_file) as f:
                        update_freq_dict(f)
        
        return freq


class WikipediaDataset:
    def __init__(self) -> None:
        pass

    def download_wikipedia_dataset(self, destination: str) -> None:
        os.makedirs(destination, exist_ok=True)
        print(f"{style.FG_RED + style.BG_YELLOW + style.BOLD}Manually download .parquet files of wikipedia dataset from https://huggingface.co/datasets/wikimedia/wikipedia/tree/main and save the files at {destination}{style.RESET}")

    def calculate_word_frequency(self, wikipedia_data_path: str) -> dict:
        # Find all the .parquet files in the given directory
        wikipedia_parquet_files: List[str] = [f for f in os.listdir(wikipedia_data_path) if f.endswith('.parquet')]
        freq: dict = defaultdict(int)

        # Fet total number of rows across all files (to use in tqdm progress bar)
        total_rows = sum(pd.read_parquet(os.path.join(wikipedia_data_path, f), engine='pyarrow').shape[0] for f in wikipedia_parquet_files)
        
        with tqdm(total=total_rows, desc="Calculating word frequencies") as pbar:
            for parquet_file in wikipedia_parquet_files:
                parquet_file_path: str = os.path.join(wikipedia_data_path, parquet_file)
                df = pd.read_parquet(parquet_file_path, engine='pyarrow')
                for text in df["text"]:
                    item_freq = utils.get_word_freq_dist(text)
                    for word, count in item_freq.items():
                        freq[word] += count
                    pbar.update(1)  # update progress bar for each row
        return freq
