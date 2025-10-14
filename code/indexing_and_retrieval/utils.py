# ======================== IMPORTS ========================
import os
import yaml
from enum import Enum
from typing import List
from collections import defaultdict

CONFIG_FILE_PATH: str = "../../config.yaml"


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


class StatusCode(Enum):
    SUCCESS              : int = 0
    CONNECTION_FAILED    : int = 1000
    ERROR_ACCESSING_INDEX: int = 1001
    INVALID_INPUT        : int = 1002
    INDEXING_FAILED      : int = 1003
    FAILED_TO_REMOVE_FILE: int = 1004
    QUERY_FAILED         : int = 1005
    
    UNKNOWN_ERROR        : int = 9999


class IndexType(Enum):
    ESIndex    : int = 1
    CustomIndex: int = 2


class DatasetType(Enum):
    News     : int = 1
    Wikipedia: int = 2


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


def ask_es_query(es_client, index_id: str, query: str, search_field: str, max_results: int, source: bool=True):
    '''
    Executes the given query on the specified Elasticsearch index and returns the response.
    '''
    query_clause = {
        "match": {
            search_field: query
        }
    }

    try:
        response = es_client.search(
            index=index_id,
            query=query_clause,
            size=max_results,
            _source=source
        )
    except Exception as e:
        return StatusCode.QUERY_FAILED

    return response
