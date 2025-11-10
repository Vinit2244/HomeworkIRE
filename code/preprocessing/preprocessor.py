# ======================== IMPORTS ========================
import nltk


# ======================== CLASSES ========================
class Preprocessor:
    """
    Preprocessor class for text data.
    Provides methods for various preprocessing tasks such as lowercasing, stopword removal,
    punctuation removal, stemming, and lemmatization."""

    def __init__(self) -> None:
        pass

    def lowercase(self, text: str) -> str:
        """
        About:
        ------
            Converts the input text to lowercase.

        Args:
        -----
            text: The input text string.

        Returns:
        --------
            The lowercase version of the input text.
        """

        return text.lower()

    def remove_stopwords(self, text: str, lang: str) -> str:
        """
        About:
        ------
            Removes stopwords from the input text for the specified language.

        Args:
        -----
            text: The input text string.
            lang: The language for stopword removal.

        Returns:
        --------
            The text with stopwords removed.
        """

        stop_words = set(nltk.corpus.stopwords.words(lang))
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    
    def remove(self, text: str, remove_punc: bool, remove_num: bool, remove_special: bool) -> str:
        """
        About:
        ------
            Removes punctuation, numbers, and/or special characters from the input text based on the specified flags.

        Args:
        -----
            text: The input text string.
            remove_punc: Whether to remove punctuation.
            remove_num: Whether to remove numbers.
            remove_special: Whether to remove special characters.

        Returns:
        --------
            The cleaned text with specified characters removed.
        """

        new_text = ""
        for char in text:
            if (remove_punc and not char.isalnum() and not char.isspace()) or \
               (remove_num and char.isdigit()) or \
               (remove_special and not char.isalnum() and not char.isspace()):
                continue
            new_text += char
        return new_text
    
    def stem(self, text: str, algo: str) -> str:
        """
        About:
        ------
            Applies stemming to the input text using the specified algorithm.

        Args:
        -----
            text: The input text string.
            algo: The stemming algorithm to use.

        Returns:
        --------
            The stemmed version of the input text.
        """

        if algo == 'porter':
            stemmer = nltk.stem.PorterStemmer()
        elif algo == 'lancaster':
            stemmer = nltk.stem.LancasterStemmer()
        elif algo == 'snowball':
            stemmer = nltk.stem.SnowballStemmer("english")
        else:
            raise ValueError(f"Unsupported stemming algorithm: {algo}")
        words = text.split()
        stemmed_words = [stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    
    def lemmatize(self, text: str, algo: str) -> str:
        """
        About:
        ------
            Applies lemmatization to the input text using the specified algorithm.

        Args:
        -----
            text: The input text string.
            algo: The lemmatization algorithm to use.

        Returns:
        --------
            The lemmatized version of the input text.
        """

        if algo == 'wordnet':
            lemmatizer = nltk.stem.WordNetLemmatizer()
        else:
            raise ValueError(f"Unsupported lemmatization algorithm: {algo}")
        words = text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
