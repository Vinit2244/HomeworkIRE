# ======================== IMPORTS ========================
import os
from enum import Enum
from utils import load_config
from dotenv import load_dotenv


# ======================= CONSTANTS =======================
# Configurations from config.yaml
config = load_config()
ES_HOST               : str  = config.get("elasticsearch", {}).get("host", "localhost")
ES_PORT               : int  = config.get("elasticsearch", {}).get("port", 9200)
ES_SCHEME             : str  = config.get("elasticsearch", {}).get("scheme", "http")
REDIS_HOST            : str  = config.get("redis", {}).get("host", "localhost")
REDIS_PORT            : int  = config.get("redis", {}).get("port", 6379)
REDIS_DB              : int  = config.get("redis", {}).get("db", 0)

DATA_SETTINGS         : dict = config.get("data", {})
INDEX_SETTINGS        : dict = config.get("index", {})
PREPROCESSING_SETTINGS: dict = config.get("preprocessing", {})

MAX_RESULTS           : int  = config.get("max_results", 50)
SEARCH_FIELDS         : str  = config.get("search_fields", ["text"])
MAX_NUM_DOCUMENTS     : int  = config.get("max_num_documents", -1)
TOP_K_WORDS_THRESHOLD : int  = config.get("top_k_words_threshold", 50)
CHUNK_SIZE            : int  = config.get("elasticsearch", {}).get("chunk_size", 500)

STORAGE_DIR           : str  = config.get("storage_folder_path", "./storage")
TEMP_FOLDER_PATH      : str  = config.get("temp_folder_path", "./temp")
OUTPUT_FOLDER_PATH    : str  = config.get("output_folder_path", "./output")
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(TEMP_FOLDER_PATH, exist_ok=True)
os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)

# Environment variables
load_dotenv()
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
API_KEY  = os.getenv("API_KEY")


# ========================= ENUMS =========================
class IndexInfo(Enum):
    NONE     : int = 0 # Just a placeholder
    BOOLEAN  : int = 1
    WORDCOUNT: int = 2
    TFIDF    : int = 3


class DataStore(Enum):
    NONE   : int = 0 # Just a placeholder
    CUSTOM : int = 1
    ROCKSDB: int = 2
    REDIS  : int = 3


class Compression(Enum):
    NONE: int = 0
    CODE: int = 1
    CLIB: int = 2


class Optimizations(Enum):
    NONE         : str = '0'
    SKIPPING     : str = 'sp'
    THRESHOLDING : str = 'th'
    EARLYSTOPPING: str = 'es'


class QueryProc(Enum):
    NONE: str = '0' # Just a placeholder
    TERM: str = 'T'
    DOC : str = 'D'


class StatusCode(Enum):
    SUCCESS              : int = 0
    CONNECTION_FAILED    : int = 1000
    ERROR_ACCESSING_INDEX: int = 1001
    INVALID_INPUT        : int = 1002
    INDEXING_FAILED      : int = 1003
    FAILED_TO_REMOVE_FILE: int = 1004
    QUERY_FAILED         : int = 1005
    INDEX_NOT_FOUND      : int = 1006
    INDEX_ALREADY_EXISTS : int = 1007
    
    UNKNOWN_ERROR        : int = 9999


class IndexType(Enum):
    ESIndex    : int = 1
    CustomIndex: int = 2


class DatasetType(Enum):
    News     : int = 1
    Wikipedia: int = 2
