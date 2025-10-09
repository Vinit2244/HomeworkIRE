# ======================== IMPORTS ========================
from typing import List
from abc import ABC, abstractmethod


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
