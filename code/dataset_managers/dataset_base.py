# ======================== IMPORTS ========================
from typing import List
from abc import ABC, abstractmethod


# =================== DATASET CLASSES ====================
class Dataset(ABC):
    """
    Abstract base class for dataset handlers.
    Defines the interface for downloading datasets, calculating word frequencies, and preprocessing text data.
    """

    # Public functions
    @abstractmethod
    def download_dataset(self) -> None:
        """
        About:
        -----
            Downloads the dataset to the specified path.

        Args:
        -----
            None

        Returns:
        --------
            None
        """
        pass

    @abstractmethod
    def calculate_word_frequency(self) -> dict:
        """
        About:
        -----
            Calculates the word frequency distribution across the dataset.
        
        Args:
        -----
            None

        Returns:
        --------
            A dictionary with words as keys and their corresponding frequencies as values.
        """
        pass

    @abstractmethod
    def preprocess(self, lowercase: bool, rem_stop: bool, stopword_langs: List[str], rem_punc: bool, rem_num: bool, rem_special: bool, stem: bool, stemming_algo: str, lemmatize: bool, lemmatization_algo: str) -> None:
        """
        About:
        -----
            Preprocesses the dataset based on the specified parameters.

        Args:
        -----
            lowercase: Whether to convert text to lowercase.
            rem_stop: Whether to remove stopwords.
            stopword_langs: List of languages for stopword removal.
            rem_punc: Whether to remove punctuation.
            rem_num: Whether to remove numbers.
            rem_special: Whether to remove special characters.
            stem: Whether to apply stemming.
            stemming_algo: The stemming algorithm to use.
            lemmatize: Whether to apply lemmatization.
            lemmatization_algo: The lemmatization algorithm to use.

        Returns:
        --------
            None
        """
        pass
