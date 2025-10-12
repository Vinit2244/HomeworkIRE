# ======================== IMPORTS ========================
import os
import yaml
from enum import Enum
from typing import List
from collections import defaultdict

CONFIG_FILE_PATH: str = "../../config.yaml"


# ======================== CLASSES ========================
class Style:
    RESET      = "\033[0m"
    BOLD       = "\033[01m"
    UNDERLINE  = "\033[4m"

    # Foreground colors
    FG_RED     = "\033[31m"
    FG_GREEN   = "\033[32m"
    FG_YELLOW  = "\033[33m"
    FG_BLUE    = "\033[34m"
    FG_MAGENTA = "\033[35m"
    FG_CYAN    = "\033[36m"
    FG_WHITE   = "\033[37m"
    FG_ORANGE  = "\033[38;5;208m"

    # Background colors
    BG_RED     = "\033[41m"
    BG_GREEN   = "\033[42m"
    BG_YELLOW  = "\033[43m"
    BG_BLUE    = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN    = "\033[46m"
    BG_WHITE   = "\033[47m"
    BG_ORANGE  = "\033[48;5;208m"

class StatusCode(Enum):
    SUCCESS = 0
    CONNECTION_FAILED = 1000
    ERROR_ACCESSING_INDEX = 1001
    INVALID_INPUT = 1002
    INDEXING_FAILED = 1003
    
    UNKNOWN_ERROR = 9999

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
