# ======================== IMPORTS ========================
import os
import inspect
from typing import Iterable
from dotenv import load_dotenv
from .index_base import BaseIndex
from utils import Style, StatusCode
from elasticsearch import Elasticsearch, helpers


# ======================= GLOBALS ========================
load_dotenv()  # Loads .env file into environment variables
username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")
api_key = os.getenv("API_KEY")
CHUNK_SIZE = 500  # Number of documents to index in one bulk operation


# ======================== CLASSES ========================
class ESIndex(BaseIndex):
    def __init__(self, host: str, port: int, scheme: str, core: str, info: str="NONE", dstore: str="NONE", qproc: str="NONE", compr: str="NONE", optim: str="NONE"):
        super().__init__(core, info, dstore, qproc, compr, optim)
        self.host = host
        self.port = port
        self.scheme = scheme
        status = self.connect_to_cluster()

        if status != StatusCode.SUCCESS:
            raise ConnectionError("Failed to connect to Elasticsearch cluster.")
    
    def connect_to_cluster(self) -> StatusCode:
        self.es_client = Elasticsearch(
            [{'host': self.host, 'port': self.port, 'scheme': self.scheme}],
            basic_auth=(username, password)
        )
        try:
            cluster_info = self.es_client.info()
            print(f"{Style.FG_CYAN}Connected to cluster: {cluster_info['cluster_name']}{Style.RESET}")
            print(f"{Style.FG_CYAN}Elasticsearch version: {cluster_info['version']['number']}{Style.RESET}")
            return StatusCode.SUCCESS
        except Exception as e:
            print(f"{Style.FG_RED}Connection failed: {e}{Style.RESET}")
            return StatusCode.CONNECTION_FAILED

    def get_index_info(self, index_id: str) -> dict | StatusCode:
        try:
            index_info: dict = self.es_client.indices.get(index=index_id)
            res: dict = self.es_client.count(index=index_id)
            index_info[index_id]["docs_count"] = res["count"]
            return index_info
        
        except Exception as e:
            print(f"{Style.FG_RED}Failed to get index info: {e}{Style.RESET}")
            return StatusCode.ERROR_ACCESSING_INDEX

    def list_indices(self) -> Iterable[str]:
        return self.es_client.cat.indices(format="json")

    def create_index(self, index_id: str, files: Iterable[tuple[str, dict]]) -> None:
        # Create new index if it doesn't exist
        if self.es_client.indices.exists(index=index_id):
            print(f"Index '{index_id}' already exists")
        else:
            self.es_client.indices.create(index=index_id)

        # Helper function to resolve if the source is a generator function or an iterable
        def resolve_iterable(source):
            if inspect.isgeneratorfunction(source):
                return source()  # call generator function
            elif hasattr(source, '__iter__'):
                return source
            else:
                raise TypeError("Expected a generator function or iterable.")
        
        # Generator to yield documents in the required format
        def bulk_doc_generator(index_id, files):
            for file in files:
                uuid, doc = file

                yield {
                    "_index": index_id,
                    "_id": uuid,
                    "_source": doc
                }

        failed_flag = False

        # Bulk index documents
        data_iter = resolve_iterable(files)
        for ok, result in helpers.streaming_bulk(client=self.es_client, actions=bulk_doc_generator(index_id, data_iter), chunk_size=CHUNK_SIZE):
            if not ok:
                failed_flag = True
                print("Failed:", result)
        
        if not failed_flag:
            return StatusCode.SUCCESS
        else:
            return StatusCode.INDEXING_FAILED

    def load_index(self, serialized_index_dump: str) -> None:
        ...

    def update_index(self, index_id: str, remove_files: Iterable[tuple[str, str]], add_files: Iterable[tuple[str, str]]) -> None:
        ...

    def query(self, query: str) -> str:
        ...

    def delete_index(self, index_id: str) -> StatusCode:
        res = self.es_client.indices.delete(index=index_id.lower())
        if res.get('acknowledged', False):
            return StatusCode.SUCCESS
        else:
            return StatusCode.ERROR_ACCESSING_INDEX

    def list_indexed_files(self, index_id: str) -> Iterable[str]:
        ...
