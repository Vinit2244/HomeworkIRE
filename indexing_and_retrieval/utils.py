# ======================== IMPORTS ========================
import yaml
from typing import List
from collections import defaultdict


# ======================== CLASSES ========================
class Style:
    '''
    Class for styling the terminal output using ANSI escape codes.
    '''
    def __init__(self) -> None:
        # Styling other than colors
        self.RESET      = "\033[0m"
        self.BOLD       = "\033[01m"
        self.UNDERLINE  = "\033[4m"

        # Foreground colors
        self.FG_RED     = "\033[31m"
        self.FG_GREEN   = "\033[32m"
        self.FG_YELLOW  = "\033[33m"
        self.FG_BLUE    = "\033[34m"
        self.FG_MAGENTA = "\033[35m"
        self.FG_CYAN    = "\033[36m"
        self.FG_WHITE   = "\033[37m"
        self.FG_ORANGE  = "\033[38;5;208m"

        # Background colors
        self.BG_RED     = "\033[41m"
        self.BG_GREEN   = "\033[42m"
        self.BG_YELLOW  = "\033[43m"
        self.BG_BLUE    = "\033[44m"
        self.BG_MAGENTA = "\033[45m"
        self.BG_CYAN    = "\033[46m"
        self.BG_WHITE   = "\033[47m"
        self.BG_ORANGE  = "\033[48;5;208m"

style = Style()


# =================== UTILITY FUNCTIONS ===================
def load_config() -> dict:
    '''
    Returns the configuration from config.yaml file as a dictionary.
    '''
    with open("../config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_word_freq_dist(text: str) -> dict:
    '''
    Returns the frequency distribution of words in the given string. (Omits punctuations and is case insensitive)
    '''
    freq: dict = defaultdict(int)

    text = ''.join(char for char in text if char.isalnum() or char.isspace()) # Remove all the punctuations from the text
    words: List[str] = text.split()
    for word in words:
        freq[word.lower()] += 1 # Case insenitive
    return freq
