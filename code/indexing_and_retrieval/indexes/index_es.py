# ======================== IMPORTS ========================
import os
import inspect
from utils import style
from typing import Iterable
from dotenv import load_dotenv
from .index_base import IndexBase
from elasticsearch import Elasticsearch, helpers


# ======================= GLOBALS ========================
load_dotenv()  # Loads .env file into environment variables
username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")
api_key = os.getenv("API_KEY")
CHUNK_SIZE = 500  # Number of documents to index in one bulk operation


# ======================== CLASSES ========================
class ESIndex(IndexBase):
    def __init__(self, host: str, port: int, scheme: str, core: str, info: str, dstore: str, qproc: str, compr: str, optim: str):
        super().__init__(core, info, dstore, qproc, compr, optim)
        self.host = host
        self.port = port
        self.scheme = scheme
        self.connect_to_cluster()
    
    def connect_to_cluster(self) -> None:
        self.es_client = Elasticsearch(
            [{'host': self.host, 'port': self.port, 'scheme': self.scheme}],
            basic_auth=(username, password)
        )
        try:
            cluster_info = self.es_client.info()
            print(f"{style.FG_CYAN}Connected to cluster: {cluster_info['cluster_name']}{style.RESET}")
            print(f"{style.FG_CYAN}Elasticsearch version: {cluster_info['version']['number']}{style.RESET}")
            print()
        except Exception as e:
            print(f"Connection failed: {e}")

    def list_indices(self) -> Iterable[str]:
        return self.es_client.cat.indices(format="json")

    def create_index(self, index_id: str, files: Iterable[tuple[str, dict]]) -> None:
        # Create new index
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

        # Bulk index documents
        data_iter = resolve_iterable(files)
        for ok, result in helpers.streaming_bulk(client=self.es_client, actions=bulk_doc_generator(index_id, data_iter), chunk_size=CHUNK_SIZE):
            if not ok:
                print("Failed:", result)

    def load_index(self, serialized_index_dump: str) -> None:
        ...

    def update_index(self, index_id: str, remove_files: Iterable[tuple[str, str]], add_files: Iterable[tuple[str, str]]) -> None:
        ...

    def query(self, query: str) -> str:
        ...

    def delete_index(self, index_id: str) -> None:
        self.es_client.indices.delete(index=index_id.lower())

    def list_indexed_files(self, index_id: str) -> Iterable[str]:
        ...
