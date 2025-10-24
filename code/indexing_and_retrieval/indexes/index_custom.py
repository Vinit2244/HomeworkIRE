# ======================== IMPORTS ========================
import os
import yaml
import json
import math
import shutil
from pathlib import Path
from typing import Iterable
from utils import StatusCode, load_config
from .index_base import BaseIndex, IndexInfo, DataStore, Compression, QueryProc, Optimizations


# ======================= GLOBALS ========================
config = load_config()
STORAGE_DIR: str = config.get("storage_folder_path", "./storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

def load_metadata() -> dict:
    file_path = Path(os.path.join(STORAGE_DIR, "metadata.yaml"))

    if not file_path.exists():
        # Create the file with initial content if it doesn't exist
        initial_metadata = {
            "indices": []
        }
        with open(file_path, "w") as f:
            yaml.dump(initial_metadata, f, default_flow_style=False)
        return initial_metadata
    else:
        # Open the existing file for reading and potentially modification
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

METADATA: dict = load_metadata()
SEARCH_FIELDS: list = config.get("search_fields", ["text"])


# ======================== CLASSES ========================
class CustomIndex(BaseIndex):
    def __init__(self, core: str, info: str="NONE", dstore: str="NONE", qproc: str="NONE", compr: str="NONE", optim: str="NONE"):
        super().__init__(core, info, dstore, qproc, compr, optim)
        self.core = core
        self.info = info
        self.dstore = dstore
        self.qproc = qproc
        self.compr = compr
        self.optim = optim
        self.loaded_index = None

        self.name_ext = f"{IndexInfo[info].value}{DataStore[dstore].value}{Compression[compr].value}{Optimizations[optim].value}{QueryProc[qproc].value}"

    # Private Methods
    def _add_ext_to_index_id(self, index_id: str) -> str | StatusCode:
        if "." in index_id:
            return index_id
        
        all_indices: list = METADATA.get("indices", [])
        for idx in all_indices:
            if idx.startswith(index_id + "."):
                return idx
        return StatusCode.INDEX_NOT_FOUND

    def _check_index_exists(self, index_id: str) -> bool:
        all_indices: list = METADATA.get("indices", [])
        return index_id in all_indices

    def _build_inverted_index(self, files: Iterable[tuple[str, dict]], index_data_path: str) -> dict:
        # Create place for storing documents
        if self.dstore == DataStore.CUSTOM.name:
            os.mkdir(os.path.join(index_data_path, "documents")) # Folder for storing all the documents
        elif self.dstore == DataStore.MONGODB.name:
            ...
        elif self.dstore == DataStore.REDIS.name:
            ...

        inverted_index: dict = {}
        file_count: int = 0

        for uuid, payload in files:
            # Store document in the data store
            if self.dstore == DataStore.CUSTOM.name:
                with open(os.path.join(index_data_path, "documents", f"{uuid}.json"), "w") as f:
                    json.dump(payload, f)
            elif self.dstore == DataStore.MONGODB.name:
                # TODO: Implement MongoDB storage of files
                ...
            elif self.dstore == DataStore.REDIS.name:
                # TODO: Implement Redis storage of files
                ...

            # Update inverted index
            for search_field in SEARCH_FIELDS:
                content: str = payload.get(search_field, "")
                words: list = content.split()
                for position, word in enumerate(words):
                    if word not in inverted_index:
                        inverted_index[word] = {}
                    if uuid not in inverted_index[word]:
                        inverted_index[word][uuid] = {"positions": []}
                    inverted_index[word][uuid]["positions"].append(position)
                
            file_count += 1

        # TODO: Implement optimisations
        # TODO: Implement compression
        # TODO: Implement storage

        # Process based on IndexInfo type
        if self.info == IndexInfo.BOOLEAN.name:
            pass
        
        elif self.info == IndexInfo.WORDCOUNT.name:
            # Store position lists and word counts
            for word in inverted_index:
                for doc_id in inverted_index[word]:
                    positions = inverted_index[word][doc_id]["positions"]
                    inverted_index[word][doc_id]["count"] = len(positions)
        
        elif self.info == IndexInfo.TFIDF.name:
            # Calculate TF-IDF scores
            for word in inverted_index:
                doc_freq = len(inverted_index[word])
                idf = math.log(file_count / doc_freq) if doc_freq > 0 else 0
                
                for doc_id in inverted_index[word]:
                    positions = inverted_index[word][doc_id]["positions"]
                    tf = len(positions)
                    tfidf = tf * idf
                    inverted_index[word][doc_id]["tfidf"] = tfidf

        return inverted_index, file_count

    def _update_index_metadata(self, index_id: str, items: dict) -> dict | StatusCode:
        index_id = self._add_ext_to_index_id(index_id)
        index_data_path: str = os.path.join(STORAGE_DIR, index_id)

        if self.dstore == DataStore.CUSTOM.name:
            index_metadata_path: str = os.path.join(index_data_path, "metadata.yaml")

            # If metadata file exists, load and update it
            if os.path.exists(index_metadata_path):
                with open(index_metadata_path, "r") as f:
                    index_info: dict = yaml.safe_load(f)
                
                # Update the metadata with new items
                index_info.update(items)

                with open(index_metadata_path, "w") as f:
                    yaml.dump(index_info, f, default_flow_style=False)
            
            # If metadata file doesn't exist, create it
            else:
                with open(index_metadata_path, "w") as f:
                    yaml.dump(items, f, default_flow_style=False)
    
        elif self.dstore == DataStore.MONGODB.name:
            # TODO: Implement MongoDB metadata update
            ...
        
        elif self.dstore == DataStore.REDIS.name:
            # TODO: Implement Redis metadata update
            ...
    
    def _update_global_metadata(self, action: str, index_id: str) -> None:
        index_id = self._add_ext_to_index_id(index_id)
        all_indices: list = METADATA.get("indices", [])
        
        if action == "add":
            all_indices.append(index_id)
        elif action == "remove":
            all_indices.remove(index_id)
        
        METADATA["indices"] = all_indices
        with open(os.path.join(STORAGE_DIR, "metadata.yaml"), "w") as f:
            yaml.dump(METADATA, f, default_flow_style=False)

    # Public Methods
    def get_index_info(self, index_id: str) -> dict | StatusCode:
        index_id = self._add_ext_to_index_id(index_id)
        if self._check_index_exists(index_id):
            index_data_path: str = os.path.join(STORAGE_DIR, index_id)
            index_metadata_path: str = os.path.join(index_data_path, "metadata.yaml")

            if os.path.exists(index_metadata_path):
                with open(index_metadata_path, "r") as f:
                    index_info: dict = yaml.safe_load(f)
                    final_info = {index_id.split(".")[0] : index_info}
                return final_info
        
        else:
            return StatusCode.ERROR_ACCESSING_INDEX

    def list_indices(self) -> Iterable[str]:
        all_indices: list = METADATA.get("indices", [])
        all_indices: list = [{"index": index_id} for index_id in all_indices] # Return list of dicts for consistency with ESIndex
        return all_indices

    def create_index(self, index_id: str, files: Iterable[tuple[str, dict]]) -> StatusCode:
        index_id = f"{index_id}.{self.name_ext}"
        if self._check_index_exists(index_id):
            return StatusCode.INDEX_ALREADY_EXISTS
        
        index_metadata: dict = {
            "info": self.info,
            "data_store": self.dstore,
            "query_processor": self.qproc,
            "compression": self.compr,
            "optimization": self.optim,
            "documents_indexed": 0
        }

        index_data_path: str = os.path.join(STORAGE_DIR, index_id)
        os.makedirs(index_data_path, exist_ok=True) # Create directory for the index
        self._update_index_metadata(index_id, index_metadata)

        inverted_index, n_docs_indexed = self._build_inverted_index(files, index_data_path)
        
        # Save inverted index to file
        with open(os.path.join(index_data_path, "inverted_index.json"), "w") as f:
            json.dump(inverted_index, f)

        # Update number of documents indexed in metadata
        self._update_index_metadata(index_id, {"documents_indexed": n_docs_indexed})

        # Update global metadata
        self._update_global_metadata("add", index_id)

        return StatusCode.SUCCESS

    def load_index(self, index_id: str) -> StatusCode:
        index_id = self._add_ext_to_index_id(index_id)
        if self._check_index_exists(index_id):
            index_data_path: str = os.path.join(STORAGE_DIR, index_id)
            inverted_index_path: str = os.path.join(index_data_path, "inverted_index.json")
            index_metadata_path: str = os.path.join(index_data_path, "metadata.yaml")

            try:
                with open(index_metadata_path, "r") as f:
                    index_info: dict = yaml.safe_load(f)
                
                self.info = index_info.get("info", "NONE")
                self.dstore = index_info.get("data_store", "NONE")
                self.qproc = index_info.get("query_processor", "NONE")
                self.compr = index_info.get("compression", "NONE")
                self.optim = index_info.get("optimization", "NONE")

                with open(inverted_index_path, "r") as f:
                    self.loaded_index = json.load(f)
                
                return StatusCode.SUCCESS
            except:
                return StatusCode.ERROR_ACCESSING_INDEX
        else:
            return StatusCode.INDEX_NOT_FOUND

    def update_index(self, index_id: str, remove_files: Iterable[str], add_files: Iterable[tuple[str, dict]]) -> StatusCode:
        index_id = self._add_ext_to_index_id(index_id)
        # TODO: Implement index update functionality
        ...

    def query(self, query: str, index_id: str=None) -> str | StatusCode:
        index_id = self._add_ext_to_index_id(index_id)
        # TODO: Implement query functionality
        ...

    def delete_index(self, index_id: str) -> StatusCode:
        index_id = self._add_ext_to_index_id(index_id)
        if not self._check_index_exists(index_id):
            return StatusCode.INDEX_NOT_FOUND

        index_data_path: str = os.path.join(STORAGE_DIR, index_id)
        shutil.rmtree(index_data_path)  # Remove the index directory and all its contents
        self._update_global_metadata("remove", index_id)
        return StatusCode.SUCCESS

    def list_indexed_files(self, index_id: str) -> Iterable[str]:
        index_id = self._add_ext_to_index_id(index_id)
        if not self._check_index_exists(index_id):
            return StatusCode.INDEX_NOT_FOUND

        index_data_path: str = os.path.join(STORAGE_DIR, index_id)

        if self.dstore == DataStore.CUSTOM.name:
            documents_path: str = os.path.join(index_data_path, "documents")
            all_files: list = os.listdir(documents_path)
            all_file_ids: list = [os.path.splitext(filename)[0] for filename in all_files] # Remove .json extension
            return all_file_ids
        
        elif self.dstore == DataStore.MONGODB.name:
            ...
        
        elif self.dstore == DataStore.REDIS.name:
            ...
