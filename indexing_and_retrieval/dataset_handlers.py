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
from abc import ABC, abstractmethod
from preprocessor import Preprocessor


# =================== DATASET CLASSES ====================
class Dataset(ABC):
    # Public functions
    @abstractmethod
    def download_dataset(self) -> None:
        pass

    @abstractmethod
    def calculate_word_frequency(self) -> dict:
        pass

    @abstractmethod
    def preprocess(self, lowercase: bool, rem_stop: bool, stopword_langs: List[str], rem_punc: bool, rem_num: bool, rem_special: bool, stem: bool, stemming_algo: str, lemmatize: bool, lemmatization_algo: str) -> None:
        pass


class NewsDataset(Dataset):
    def __init__(self, data_path: str, max_num_docs: int, unzipped: bool) -> None:
        self.data_path = data_path
        self.max_num_docs = max_num_docs
        self.unzipped = unzipped
        pass

    # Private functions (First returns the total length of the iterator, second is the actual iterator)
    def _file_iterator(self):
        if self.unzipped:
            all_folders: List[str] = os.listdir(self.data_path)

            # Flatten list of all JSON files across all folders
            all_json_files_paths: List[str] = []
            for folder in all_folders:
                folder_path = os.path.join(self.data_path, folder)
                # Make sure it's a directory
                if os.path.isdir(folder_path):
                    json_files = os.listdir(folder_path)
                    for json_file in json_files:
                        all_json_files_paths.append(os.path.join(folder_path, json_file))
            
            # Fet total number of files (to use in tqdm progress bar)
            if self.max_num_docs != -1:
                all_json_files_paths = all_json_files_paths[:self.max_num_docs]
            total_files = len(all_json_files_paths)
            yield total_files

            # Iterate over all files with tqdm
            for json_file_path in all_json_files_paths:
                with open(json_file_path, 'r') as f:
                    yield f
        
        else:
            zipped_folders_path: str = os.path.join(self.data_path, "News_Datasets")
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

            # Fet total number of files (to use in tqdm progress bar)
            if self.max_num_docs != -1:
                all_json_entries = all_json_entries[:self.max_num_docs]
            total_files = len(all_json_entries)
            yield total_files

            # Iterate over all JSON files with tqdm
            for zip_path, json_file in all_json_entries:
                with zipfile.ZipFile(zip_path, "r") as z:
                    with z.open(json_file) as f:
                        yield f

    def _unzip_and_clean_dataset(self) -> None:
        print(f"{style.FG_CYAN}Unzipping and cleaning news dataset...{style.RESET}")

        # Remove all the files except the "News_Datasets" folder
        all_items: list = os.listdir(self.data_path)
        for item in all_items:
            if item != "News_Datasets":
                item_path: str = os.path.join(self.data_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    subprocess.run(["rm", "-rf", item_path], check=True)
        
        # Unzip all the zipped folders in "News_Datasets" and remove the zipped files
        path_to_zipped_folders: str = os.path.join(self.data_path, "News_Datasets")
        all_zipped_folders: list = os.listdir(path_to_zipped_folders)
        unzipped_count = 0
        for folder in all_zipped_folders:
            folder_path: str = os.path.join(path_to_zipped_folders, folder)
            if folder_path.endswith(".zip"):
                try:
                    subprocess.run(["unzip", "-o", folder_path, "-d", self.data_path], check=True)
                    unzipped_count += 1
                    os.remove(folder_path)
                except:
                    subprocess.run(["rm", "-rf", folder_path], check=True)
        print(f"{style.FG_GREEN}Unzipped {unzipped_count}/{len(all_zipped_folders)}{style.RESET}")

        print(f"{style.FG_GREEN}News dataset unzipped and cleaned at {self.data_path}\n{style.RESET}")

    # Public functions
    def download_dataset(self) -> None:
        print(f"{style.FG_CYAN}Downloading news dataset...{style.RESET}")

        # Make sure the destination directory exists
        os.makedirs(self.data_path, exist_ok=True)

        # Clone the repository
        repo_url = "https://github.com/Webhose/free-news-datasets.git"
        subprocess.run(["git", "clone", repo_url, self.data_path], check=True)

        if self.unzipped:
            print(f"{style.FG_CYAN}Unzipping and cleaning news dataset...{style.RESET}")
            self._unzip_and_clean_dataset()

        print(f"{style.FG_GREEN}News dataset downloaded at {self.data_path}\n{style.RESET}")

    def calculate_word_frequency(self) -> dict:
        freq: dict = defaultdict(int)

        # Updates the overall frequency dictionary with the frequency from a single json file
        def update_freq_dict(f):
            text: str = json.load(f)["text"]
            file_freq: dict = utils.get_word_freq_dist(text)
            for word, count in file_freq.items():
                freq[word] += count
        
        # Fet total number of files (to use in tqdm progress bar)
        total_files = next(self._file_iterator())

        for f in tqdm(self._file_iterator(), total=total_files, desc="Calculating word frequencies"):
            # Ignoring the first value which is the total count
            if type(f) is int:
                continue
            update_freq_dict(f)
        return freq

    def preprocess(self, lowercase: bool, rem_stop: bool, stopword_langs: List[str], rem_punc: bool, rem_num: bool, rem_special: bool, stem: bool, stemming_algo: str, lemmatize: bool, lemmatization_algo: str) -> None:
        preprocessor = Preprocessor()
        total_files = next(self._file_iterator())

        for f in tqdm(self._file_iterator(), total=total_files, desc="Preprocessing files"):
            # Ignoring the first value which is the total count
            if type(f) is int:
                continue

            data = json.load(f)
            text: str = data["text"]
            if lowercase:
                text = preprocessor.lowercase(text)
            if rem_stop:
                if "auto" in stopword_langs:
                    try:
                        lang: str = data["language"]
                        text = preprocessor.remove_stopwords(text, lang)
                    except:
                        # If language tag not found, skip "auto" and use other specified languages
                        for lang in stopword_langs:
                            if lang.strip().lower() == "auto":
                                continue
                            text = preprocessor.remove_stopwords(text, lang)
                else:
                    # Use all specified languages (no "auto")
                    for lang in stopword_langs:
                        if lang.strip().lower() == "auto":
                            continue
                        text = preprocessor.remove_stopwords(text, lang)
            if rem_punc or rem_num or rem_special:
                text = preprocessor.remove(text, rem_punc, rem_num, rem_special)
            if stem:
                text = preprocessor.stem(text, stemming_algo)
            if lemmatize:
                text = preprocessor.lemmatize(text, lemmatization_algo)
            data["text"] = text

            # Write back to file (overwriting original content)
            with open(f.name, 'w') as out_f:
                json.dump(data, out_f)


class WikipediaDataset(Dataset):
    def __init__(self, data_path: str, max_num_docs: int) -> None:
        self.data_path = data_path
        self.max_num_docs = max_num_docs
        pass

    # Public functions
    def download_dataset(self) -> None:
        os.makedirs(self.data_path, exist_ok=True)
        print(f"{style.FG_RED + style.BG_YELLOW + style.BOLD}Manually download .parquet files of wikipedia dataset from https://huggingface.co/datasets/wikimedia/wikipedia/tree/main and save the files at {self.data_path}{style.RESET}")

    def calculate_word_frequency(self) -> dict:
        # Find all the .parquet files in the given directory
        wikipedia_parquet_files: List[str] = [f for f in os.listdir(self.data_path) if f.endswith('.parquet')]
        freq: dict = defaultdict(int)

        # Fet total number of rows across all files (to use in tqdm progress bar)
        total_rows = sum(pd.read_parquet(os.path.join(self.data_path, f), engine='pyarrow').shape[0] for f in wikipedia_parquet_files)

        if self.max_num_docs != -1:
            total_rows = min(total_rows, self.max_num_docs)
        
        curr_row_count = 0
        with tqdm(total=total_rows, desc="Calculating word frequencies") as pbar:
            for parquet_file in wikipedia_parquet_files:
                break_flag = False
                parquet_file_path: str = os.path.join(self.data_path, parquet_file)
                df = pd.read_parquet(parquet_file_path, engine='pyarrow')
                for text in df["text"]:
                    if curr_row_count == self.max_num_docs:
                        break_flag = True
                        break
                    item_freq = utils.get_word_freq_dist(text)
                    for word, count in item_freq.items():
                        freq[word] += count
                    pbar.update(1)  # update progress bar for each row
                    curr_row_count += 1
                if break_flag:
                    break
        return freq

    def preprocess(self, lowercase: bool, rem_stop: bool, stopword_langs: List[str], rem_punc: bool, rem_num: bool, rem_special: bool, stem: bool, stemming_algo: str, lemmatize: bool, lemmatization_algo: str) -> None:
        # Find all the .parquet files in the given directory
        wikipedia_parquet_files: List[str] = [f for f in os.listdir(self.data_path) if f.endswith('.parquet')]
        freq: dict = defaultdict(int)

        # Fet total number of rows across all files (to use in tqdm progress bar)
        total_rows = sum(pd.read_parquet(os.path.join(self.data_path, f), engine='pyarrow').shape[0] for f in wikipedia_parquet_files)
        
        if self.max_num_docs != -1:
            total_rows = min(total_rows, self.max_num_docs)

        curr_row_count = 0
        preprocessor = Preprocessor()
        with tqdm(total=total_rows, desc="Preprocessing text") as pbar:
            for parquet_file in wikipedia_parquet_files:
                break_flag = False
                parquet_file_path: str = os.path.join(self.data_path, parquet_file)
                df = pd.read_parquet(parquet_file_path, engine='pyarrow')
                processed_texts = []
                for text in df["text"]:
                    if curr_row_count == self.max_num_docs:
                        # Append remaining unprocessed texts as is and break out
                        break_flag = True
                        remaining_texts = df["text"].iloc[len(processed_texts):].tolist()
                        processed_texts.extend(remaining_texts)
                        break
                    if lowercase:
                        text = preprocessor.lowercase(text)
                    if rem_stop:
                        for lang in stopword_langs:
                            if lang.strip().lower() == "auto":
                                continue
                            text = preprocessor.remove_stopwords(text, lang)
                    if rem_punc or rem_num or rem_special:
                        text = preprocessor.remove(text, rem_punc, rem_num, rem_special)
                    if stem:
                        text = preprocessor.stem(text, stemming_algo)
                    if lemmatize:
                        text = preprocessor.lemmatize(text, lemmatization_algo)

                    processed_texts.append(text)
                    pbar.update(1)
                    curr_row_count += 1

                # Update dataframe with processed text
                df["text"] = processed_texts

                # Overwrite same parquet file (or change filename if you want to keep original)
                df.to_parquet(parquet_file_path, index=False, engine='pyarrow')

                if break_flag:
                    break
        return freq
