# ======================== IMPORTS ========================
import os
import utils
import pandas as pd
from tqdm import tqdm
from typing import List
from .dataset_base import Dataset
from collections import defaultdict
from utils import Style, load_config


# ======================== CLASSES ========================
class WikipediaDataset(Dataset):
    def __init__(self, data_path: str, max_num_docs: int) -> None:
        self.data_path = data_path
        self.max_num_docs = max_num_docs
        self.wikipedia_parquet_files: List[str] = [f for f in os.listdir(self.data_path) if f.endswith('.parquet')]
        
        # Fet total number of rows across all files (to use in tqdm progress bar)
        self.total_rows = sum(pd.read_parquet(os.path.join(self.data_path, f), engine='pyarrow').shape[0] for f in self.wikipedia_parquet_files)

        if self.max_num_docs != -1:
            self.total_rows = min(self.total_rows, self.max_num_docs)

    # Public functions
    def download_dataset(self) -> None:
        os.makedirs(self.data_path, exist_ok=True)
        print(f"{Style.FG_RED + Style.BG_YELLOW + Style.BOLD}Manually download .parquet files of wikipedia dataset from https://huggingface.co/datasets/wikimedia/wikipedia/tree/main and save the files at {self.data_path}{Style.RESET}")

    def get_attributes(self) -> List[str]:
        # Get attributes from the first parquet file in the dataset
        if not self.wikipedia_parquet_files:
            return []
        
        first_file_path: str = os.path.join(self.data_path, self.wikipedia_parquet_files[0])
        df = pd.read_parquet(first_file_path, engine='pyarrow')
        return list(df.columns)

    def calculate_word_frequency(self) -> dict:
        freq: dict = defaultdict(int)

        curr_row_count = 0
        with tqdm(total=self.total_rows, desc="Calculating word frequencies") as pbar:
            for parquet_file in self.wikipedia_parquet_files:
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
        from preprocessing import Preprocessor
        
        freq: dict = defaultdict(int)

        curr_row_count = 0
        preprocessor = Preprocessor()
        with tqdm(total=self.total_rows, desc="Preprocessing text") as pbar:
            for parquet_file in self.wikipedia_parquet_files:
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

    def get_files(self, attributes: List[str]) -> List[tuple[str, dict]]:
        files: List[tuple[str, dict]] = []

        curr_row_count = 0
        for parquet_file in self.wikipedia_parquet_files:
            break_flag = False
            parquet_file_path: str = os.path.join(self.data_path, parquet_file)
            df = pd.read_parquet(parquet_file_path, engine='pyarrow')
            for _, row in df.iterrows():
                if curr_row_count == self.max_num_docs:
                    break_flag = True
                    break
                
                file_id: str = str(row[attributes[0]])  # First attribute is unique id
                content: dict = {attr: row[attr] for attr in attributes[1:]}  # Rest are content attributes
                
                files.append((file_id, content))
                curr_row_count += 1
            
            if break_flag:
                break
        
        return files


# ================== HELPER FUNCTIONS ====================
def get_wikipedia_dataset_handler() -> WikipediaDataset:
    config = load_config()
    
    data_path: str = config["data"]["wikipedia"]["path"]
    max_num_docs: int = config["max_num_documents"] if config["max_num_documents"] is not None else -1
    print(f"{Style.FG_YELLOW}Using Max docs: {max_num_docs}{Style.RESET}. To change, modify config.yaml file.\n")
    
    return WikipediaDataset(data_path, max_num_docs)
