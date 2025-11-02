# ======================== IMPORTS ========================
import os
import yaml
import json
import math
import zlib
import redis
import shutil
from pathlib import Path
from typing import Iterable
from .encoder import Encoder
from rocksdict import Rdict
from .index_base import BaseIndex
from .query_processing import QueryProcessingEngine
from constants import IndexInfo, DataStore, Compression, QueryProc, Optimizations, StatusCode, SEARCH_FIELDS, STORAGE_DIR, REDIS_HOST, REDIS_PORT, REDIS_DB


# ======================= GLOBALS ========================
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


# ======================== CLASSES ========================
class CustomIndex(BaseIndex):
    def __init__(self, core: str, info: str="BOOLEAN", dstore: str="CUSTOM", qproc: str="NONE", compr: str="NONE", optim: str="NONE"):
        super().__init__(core, info, dstore, qproc, compr, optim)
        self.core = core
        self.info = info
        self.dstore = dstore
        self.qproc = qproc
        self.compr = compr
        self.encoder = Encoder()
        self.optim = optim
        self.loaded_index = None

        self.name_ext = f"{IndexInfo[info].value}{DataStore[dstore].value}{Compression[compr].value}{Optimizations[optim].value}{QueryProc[qproc].value}"

        # Redis
        self.redis_client = None
        if self.dstore == DataStore.REDIS.name:
            try:
                self.redis_client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    decode_responses=True
                )
                self.redis_client.ping() # Test connection
            except Exception as e:
                print(f"Failed to connect to Redis: {e}")
                self.redis_client = None

        # RocksDB
        self.index_db_handle: Rdict = None
        self.doc_store_handle: Rdict = None

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

    def _build_inverted_index(self, files: Iterable[tuple[str, dict]], index_data_path: str, index_id: str) -> dict:
        # Create place for storing documents if using custom data store
        if self.dstore == DataStore.CUSTOM.name:
            os.mkdir(os.path.join(index_data_path, "documents")) # Folder for storing all the documents
        elif self.dstore == DataStore.ROCKSDB.name:
            doc_store_path: str = os.path.join(index_data_path, "doc_store")
            doc_store = Rdict(doc_store_path)

        inverted_index: dict = {}
        file_count: int = 0

        for uuid, payload in files:
            # Store document in the data store
            if self.dstore == DataStore.CUSTOM.name:
                with open(os.path.join(index_data_path, "documents", f"{uuid}.json"), "w") as f:
                    json.dump(payload, f)
            elif self.dstore == DataStore.ROCKSDB.name:
                doc_store[uuid.encode('utf-8')] = json.dumps(payload).encode('utf-8')
            elif self.dstore == DataStore.REDIS.name:
                if self.redis_client:
                    self.redis_client.hset(f"{index_id}:documents", uuid, json.dumps(payload))

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
        
        if self.dstore == DataStore.ROCKSDB.name and doc_store:
            doc_store.close()

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

        # TODO: Implement optimisations

        return inverted_index, file_count

    def _update_index_metadata(self, index_id: str, items: dict) -> dict | StatusCode:
        index_id = self._add_ext_to_index_id(index_id)
        if index_id == StatusCode.INDEX_NOT_FOUND:
            return StatusCode.INDEX_NOT_FOUND
        
        index_data_path: str = os.path.join(STORAGE_DIR, index_id)

        if self.dstore == DataStore.CUSTOM.name:
            index_metadata_path: str = os.path.join(index_data_path, "metadata.yaml")
            current_info = {}
            if os.path.exists(index_metadata_path):
                with open(index_metadata_path, "r") as f:
                    current_info = yaml.safe_load(f)
            
            current_info.update(items)

            with open(index_metadata_path, "w") as f:
                yaml.dump(current_info, f, default_flow_style=False)
    
        elif self.dstore == DataStore.ROCKSDB.name:
            index_db_path = os.path.join(index_data_path, "index_db")
            try:
                with Rdict(index_db_path) as db:
                    meta_bytes = db.get(b'metadata')
                    current_info = json.loads(meta_bytes.decode('utf-8')) if meta_bytes else {}
                    current_info.update(items)
                    db[b'metadata'] = json.dumps(current_info).encode('utf-8')
            except Exception as e:
                return StatusCode.ERROR_ACCESSING_INDEX
        
        elif self.dstore == DataStore.REDIS.name:
            if not self.redis_client:
                return StatusCode.ERROR_ACCESSING_INDEX
            try:
                key = f"{index_id}:metadata"
                meta_json = self.redis_client.get(key)
                current_info = json.loads(meta_json) if meta_json else {}
                current_info.update(items)
                self.redis_client.set(key, json.dumps(current_info))
            except Exception as e:
                print(f"Redis metadata update error: {e}")
                return StatusCode.ERROR_ACCESSING_INDEX
    
    def _update_global_metadata(self, action: str, index_id: str) -> None:
        index_id = self._add_ext_to_index_id(index_id)
        all_indices: list = METADATA.get("indices", [])
        
        if action == "add" and index_id not in all_indices:
            all_indices.append(index_id)
        elif action == "remove" and index_id in all_indices:
            all_indices.remove(index_id)
        
        METADATA["indices"] = all_indices
        with open(os.path.join(STORAGE_DIR, "metadata.yaml"), "w") as f:
            yaml.dump(METADATA, f, default_flow_style=False)

    def _compress(self, inverted_index: dict) -> bytes:
        # CODE Compression
        if self.compr == Compression.CODE.name:
            for term in inverted_index:
                for doc_id in inverted_index[term]:
                    positions = inverted_index[term][doc_id]["positions"]
                    if positions:
                        # Gap encode
                        gaps = self.encoder.gap_encode(positions)
                        # VarByte encode
                        vb_encoded = self.encoder.varbyte_encode(gaps)
                        inverted_index[term][doc_id]["positions"] = vb_encoded.hex() # Store as hex string for JSON compatibility
        
        # Serialize the index into bytes
        data_bytes = json.dumps(inverted_index).encode('utf-8')

        # CLIB Compression
        if self.compr == Compression.CLIB.name:
            data_bytes = zlib.compress(data_bytes)

        return data_bytes

    def _decompress(self, compr: str, data_bytes: bytes) -> dict:
        if compr == Compression.CLIB.name:
            data_bytes = zlib.decompress(data_bytes)

        # Deserialize from JSON
        # For CODE decompression, we will handle it during query time as needed
        return json.loads(data_bytes.decode('utf-8'))
        
    # Public Methods
    def get_index_info(self, index_id: str) -> dict | StatusCode:
        index_id_full = self._add_ext_to_index_id(index_id)
        if not self._check_index_exists(index_id_full):
            return StatusCode.INDEX_NOT_FOUND

        index_data_path: str = os.path.join(STORAGE_DIR, index_id_full)
        index_info = {}

        try:
            if self.dstore == DataStore.CUSTOM.name:
                index_metadata_path: str = os.path.join(index_data_path, "metadata.yaml")
                with open(index_metadata_path, "r") as f:
                    index_info = yaml.safe_load(f)
            
            elif self.dstore == DataStore.ROCKSDB.name:
                index_db_path = os.path.join(index_data_path, "index_db")

                # Check if the handle is already open from load_index()
                if self.index_db_handle:
                    meta_bytes = self.index_db_handle[b'metadata']
                    index_info = json.loads(meta_bytes.decode('utf-8'))
                else:
                    # If index is not loaded, open it temporarily
                    with Rdict(index_db_path) as db:
                        meta_bytes = db[b'metadata']
                        index_info = json.loads(meta_bytes.decode('utf-8'))
            
            elif self.dstore == DataStore.REDIS.name:
                if not self.redis_client:
                    return StatusCode.ERROR_ACCESSING_INDEX
                key = f"{index_id_full}:metadata"
                meta_json = self.redis_client.get(key)
                index_info = json.loads(meta_json)
            
            return {index_id: index_info}

        except Exception as e:
            print(f"Error getting index info: {e}")
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

        try:
            # Setup storage location
            if self.dstore == DataStore.CUSTOM.name:
                os.makedirs(index_data_path, exist_ok=True) 
            elif self.dstore == DataStore.ROCKSDB.name:
                os.makedirs(index_data_path, exist_ok=True)
            elif self.dstore == DataStore.REDIS.name:
                if not self.redis_client:
                    return StatusCode.ERROR_ACCESSING_INDEX
                # No directory needed, but we create an empty one for consistency
                os.makedirs(index_data_path, exist_ok=True) 
            
            # Create initial metadata
            self._update_index_metadata(index_id, index_metadata)

            # Build index and store documents
            inverted_index, n_docs_indexed = self._build_inverted_index(files, index_data_path, index_id)

            # Compression
            compressed_inv_idx: bytes = self._compress(inverted_index)
            
            # Save inverted index
            if self.dstore == DataStore.CUSTOM.name:
                with open(os.path.join(index_data_path, "inverted_index.bin"), "wb") as f:
                    f.write(compressed_inv_idx)
            elif self.dstore == DataStore.ROCKSDB.name:
                index_db_path = os.path.join(index_data_path, "index_db")
                with Rdict(index_db_path) as db:
                    db[b'inverted_index'] = compressed_inv_idx
            elif self.dstore == DataStore.REDIS.name:
                self.redis_client.set(f"{index_id}:inverted_index", compressed_inv_idx)

            # Update metadata with document count
            self._update_index_metadata(index_id, {"documents_indexed": n_docs_indexed})
            
            # Update global metadata
            self._update_global_metadata("add", index_id)

            return StatusCode.SUCCESS

        except Exception as e:
            print(f"Error creating index: {e}")
            # TODO: Add cleanup logic here if creation fails
            return StatusCode.ERROR_ACCESSING_INDEX

    def load_index(self, index_id: str) -> StatusCode:
        index_id_full = self._add_ext_to_index_id(index_id)
        if not self._check_index_exists(index_id_full):
            return StatusCode.INDEX_NOT_FOUND
        
        index_data_path: str = os.path.join(STORAGE_DIR, index_id_full)

        try:
            index_info = {}
            index_bytes: bytes = None

            # If other RocksDB handles are already open, close them first to prevent a lock conflict on loading.
            if self.index_db_handle:
                self.index_db_handle.close()
                self.index_db_handle = None
            if self.doc_store_handle:
                self.doc_store_handle.close()
                self.doc_store_handle = None

            if self.dstore == DataStore.CUSTOM.name:
                inverted_index_path: str = os.path.join(index_data_path, "inverted_index.bin")
                index_metadata_path: str = os.path.join(index_data_path, "metadata.yaml")

                with open(index_metadata_path, "r") as f:
                    index_info: dict = yaml.safe_load(f)
                with open(inverted_index_path, "rb") as f:
                    index_bytes = f.read()

            elif self.dstore == DataStore.ROCKSDB.name:
                index_db_path = os.path.join(index_data_path, "index_db")
                doc_store_path = os.path.join(index_data_path, "doc_store")
                
                self.index_db_handle = Rdict(index_db_path)
                self.doc_store_handle = Rdict(doc_store_path)
                
                meta_bytes = self.index_db_handle[b'metadata']
                index_info = json.loads(meta_bytes.decode('utf-8'))

                index_bytes = self.index_db_handle[b'inverted_index']
                        
            elif self.dstore == DataStore.REDIS.name:
                if not self.redis_client:
                    return StatusCode.ERROR_ACCESSING_INDEX
                
                meta_json = self.redis_client.get(f"{index_id_full}:metadata")
                index_info = json.loads(meta_json)

                byte_redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
                index_bytes = byte_redis.get(f"{index_id_full}:inverted_index")

            if not index_bytes:
                return StatusCode.ERROR_ACCESSING_INDEX
            
            self.loaded_index = self._decompress(index_info.get("compression"), index_bytes)
            
            # Populate self.info from the loaded metadata
            self.info = index_info.get("info", "NONE")
            self.dstore = index_info.get("data_store", "NONE")
            self.qproc = index_info.get("query_processor", "NONE")
            self.compr = index_info.get("compression", "NONE")
            self.optim = index_info.get("optimization", "NONE")
            
            return StatusCode.SUCCESS
        
        except Exception as e:
            # Clean up handles if loading failed, otherwise they stay locked
            if self.index_db_handle:
                self.index_db_handle.close()
                self.index_db_handle = None
            if self.doc_store_handle:
                self.doc_store_handle.close()
                self.doc_store_handle = None
                
            return StatusCode.ERROR_ACCESSING_INDEX

    def update_index(self, index_id: str, remove_files: Iterable[str], add_files: Iterable[tuple[str, dict]]) -> StatusCode:
        index_id = self._add_ext_to_index_id(index_id)
        # TODO: Implement index update functionality
        ...

    def query(self, query: str, index_id: str=None) -> str | StatusCode:
        index_id = self._add_ext_to_index_id(index_id)
        if index_id == StatusCode.INDEX_NOT_FOUND:
            return StatusCode.INDEX_NOT_FOUND

        if not self.loaded_index:
            return StatusCode.ERROR_ACCESSING_INDEX
        
        engine = QueryProcessingEngine(self.compr, self.qproc)

        # Get all document IDs (for NOT operations)
        all_doc_ids = self.list_indexed_files(index_id)
        if isinstance(all_doc_ids, StatusCode):
            return all_doc_ids

        matching_doc_ids, hits = engine.process_custom_query(query, self.loaded_index, index_id, all_doc_ids, self.dstore, self.doc_store_handle, self.redis_client, self.info)

        if isinstance(matching_doc_ids, StatusCode):
            return matching_doc_ids # Query failed
        
        # Format output to mimic Elasticsearch
        es_like_output = {
            "hits": {
                "total": {"value": len(matching_doc_ids), "relation": "eq"},
                "hits": hits
            }
        }
        
        return json.dumps(es_like_output, indent=2)

    def delete_index(self, index_id: str) -> StatusCode:
        index_id = self._add_ext_to_index_id(index_id)
        if not self._check_index_exists(index_id):
            return StatusCode.INDEX_NOT_FOUND

        index_data_path: str = os.path.join(STORAGE_DIR, index_id)
        
        try:
            if self.dstore == DataStore.CUSTOM.name:
                shutil.rmtree(index_data_path) 
            
            elif self.dstore == DataStore.ROCKSDB.name:
                if self.index_db_handle:
                    self.index_db_handle.close()
                    self.index_db_handle = None
                if self.doc_store_handle:
                    self.doc_store_handle.close()
                    self.doc_store_handle = None
                
                # Rdict databases are directories, so just remove the parent
                shutil.rmtree(index_data_path)
            
            elif self.dstore == DataStore.REDIS.name:
                if not self.redis_client:
                    return StatusCode.ERROR_ACCESSING_INDEX
                
                # Find all keys related to this index
                keys_to_delete = []
                for key in self.redis_client.scan_iter(f"{index_id}:*"):
                    keys_to_delete.append(key)
                
                if keys_to_delete:
                    self.redis_client.delete(*keys_to_delete)
                
                # Remove the empty directory
                if os.path.exists(index_data_path):
                    shutil.rmtree(index_data_path)
            
            self._update_global_metadata("remove", index_id)
            return StatusCode.SUCCESS
        
        except Exception as e:
            return StatusCode.ERROR_ACCESSING_INDEX

    def list_indexed_files(self, index_id: str) -> Iterable[str]:
        index_id = self._add_ext_to_index_id(index_id)
        if not self._check_index_exists(index_id):
            return StatusCode.INDEX_NOT_FOUND

        index_data_path: str = os.path.join(STORAGE_DIR, index_id)

        try:
            if self.dstore == DataStore.CUSTOM.name:
                documents_path: str = os.path.join(index_data_path, "documents")
                all_files: list = os.listdir(documents_path)
                all_file_ids: list = [os.path.splitext(filename)[0] for filename in all_files] # Remove .json extension
                return all_file_ids
            
            elif self.dstore == DataStore.ROCKSDB.name:
                doc_store_path = os.path.join(index_data_path, "doc_store")
                if not os.path.exists(doc_store_path):
                    return []
                
                # Use a handle if it's open, otherwise open a temp one
                if self.doc_store_handle:
                    return [key.decode('utf-8') for key in self.doc_store_handle.keys()]
                else:
                    with Rdict(doc_store_path) as db:
                        return [key.decode('utf-8') for key in db.keys()]
            
            elif self.dstore == DataStore.REDIS.name:
                if not self.redis_client:
                    return []
                
                # Get all fields (UUIDs) from the 'documents' hash
                doc_ids = self.redis_client.hkeys(f"{index_id}:documents")
                return doc_ids
        
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
