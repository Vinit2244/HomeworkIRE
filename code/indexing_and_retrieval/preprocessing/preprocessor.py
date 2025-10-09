# ======================== IMPORTS ========================
import nltk


# ======================== CLASSES ========================
class Preprocessor:
    def __init__(self) -> None:
        pass

    def lowercase(self, text: str) -> str:
        return text.lower()

    def remove_stopwords(self, text: str, lang: str) -> str:
        stop_words = set(nltk.corpus.stopwords.words(lang))
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    
    def remove(self, text: str, remove_punc: bool, remove_num: bool, remove_special: bool) -> str:
        new_text = ""
        for char in text:
            if (remove_punc and not char.isalnum() and not char.isspace()) or \
               (remove_num and char.isdigit()) or \
               (remove_special and not char.isalnum() and not char.isspace()):
                continue
            new_text += char
        return new_text
    
    def stem(self, text: str, algo: str) -> str:
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
        if algo == 'wordnet':
            lemmatizer = nltk.stem.WordNetLemmatizer()
        else:
            raise ValueError(f"Unsupported lemmatization algorithm: {algo}")
        words = text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
