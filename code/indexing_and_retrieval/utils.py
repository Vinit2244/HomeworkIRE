# ======================== IMPORTS ========================
import os
import yaml
from typing import List
from collections import defaultdict

CONFIG_FILE_PATH: str = "../../config.yaml" # Relative to this file


# ======================== CLASSES ========================
class Style:
    RESET     : str = "\033[0m"
    BOLD      : str = "\033[01m"
    UNDERLINE : str = "\033[4m"

    # Foreground colors
    FG_RED    : str = "\033[31m"
    FG_GREEN  : str = "\033[32m"
    FG_YELLOW : str = "\033[33m"
    FG_BLUE   : str = "\033[34m"
    FG_MAGENTA: str = "\033[35m"
    FG_CYAN   : str = "\033[36m"
    FG_WHITE  : str = "\033[37m"
    FG_ORANGE : str = "\033[38;5;208m"

    # Background colors
    BG_RED    : str = "\033[41m"
    BG_GREEN  : str = "\033[42m"
    BG_YELLOW : str = "\033[43m"
    BG_BLUE   : str = "\033[44m"
    BG_MAGENTA: str = "\033[45m"
    BG_CYAN   : str = "\033[46m"
    BG_WHITE  : str = "\033[47m"
    BG_ORANGE : str = "\033[48;5;208m"


# =================== UTILITY FUNCTIONS ===================
def load_config() -> dict:
    '''
    Returns the configuration from config.yaml file as a dictionary.
    '''
    with open(CONFIG_FILE_PATH, 'r') as f:
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


def clear_screen():
    '''
    Clears the terminal screen.
    '''
    os.system('cls' if os.name == 'nt' else 'clear')


def wait_for_enter():
    '''
    Waits for the user to press Enter.
    '''
    input(f"\n{Style.FG_BLUE}Press Enter to continue...{Style.RESET}")
