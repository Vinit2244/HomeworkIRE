# ================== IMPORTS ==================
import re
import os
import json
from typing import List
from utils import Style
from constants import STORAGE_DIR, MAX_RESULTS, DataStore, StatusCode, Compression


# ================== QUERY PARSER ==================
class QueryParser:
    def __init__(self, query):
        self.tokens = self.tokenize(query)
        self.pos = 0

    def tokenize(self, query):
        # Splits by quotes and operators while keeping phrases
        token_pattern = r'"[^"]+"|\(|\)|AND|OR|NOT|PHRASE'
        tokens = re.findall(token_pattern, query)
        return [t.strip() for t in tokens if t.strip()]

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self):
        token = self.peek()
        self.pos += 1
        return token

    def parse(self):
        return self.parse_or()

    # Lowest precedence
    def parse_or(self):
        left = self.parse_and()
        while self.peek() == "OR":
            self.consume()
            right = self.parse_and()
            left = {"OR": [left, right]}
        return left

    # Medium precedence
    def parse_and(self):
        left = self.parse_not()
        while self.peek() == "AND":
            self.consume()
            right = self.parse_not()
            left = {"AND": [left, right]}
        return left

    # Next precedence
    def parse_not(self):
        if self.peek() == "NOT":
            self.consume()
            operand = self.parse_phrase()
            return {"NOT": operand}
        return self.parse_phrase()

    # Highest precedence (PHRASE, parentheses, terms)
    def parse_phrase(self):
        token = self.peek()
        if token == "(":
            self.consume()
            expr = self.parse_or()
            assert self.consume() == ")", "Missing closing parenthesis"
            return expr

        if token == "PHRASE":
            self.consume()
            right = self.parse_phrase()
            return {"PHRASE": right}

        term = self.consume()
        if term.startswith('"') and term.endswith('"'):
            return {"TERM": term.strip('"')}

        raise ValueError(f"Unexpected token: {token}")


# ================== QUERY PROCESSING ENGINE ==================
class QueryProcessingEngine:
    def __init__(self):
        ...

    # Private functions
    def _preprocess_query(self, query: str) -> str:
        return query
    
    def _check_phrase_in_doc(self, doc_id: str, words: list, index: dict) -> bool:
        def get_positions(word: str, doc_id: str) -> list:
            postings = index.get(word, {}).get(doc_id, {})
            if not postings:
                return []
            
            pos_data = postings.get("positions", [])

            if self.compr == Compression.CODE.name:
                if not pos_data:
                    return []
                varbyte_blob = bytes.fromhex(pos_data) # Convert hex string back to bytes
                gaps = self.encoder.varbyte_decode(varbyte_blob)
                return self.encoder.gap_decode(gaps)
            else:
                return pos_data
        
        # Get all positions for the first word in this doc
        first_word_positions = get_positions(words[0], doc_id)
        if not first_word_positions:
            return False

        for pos in first_word_positions:
            # Check if "word2" exists at pos + 1, "word3" at pos + 2, etc.
            match_found = True
            for i, next_word in enumerate(words[1:], start=1):
                next_word_positions = index.get(next_word, {}).get(doc_id, {}).get("positions", [])
                if (pos + i) not in next_word_positions:
                    match_found = False # This sequence failed
                    break
            
            if match_found:
                # Found a full phrase match starting at this position
                return True 
        
        return False # No starting position led to a full match

    def _execute_custom_query(self, node: dict, index: dict, all_docs: set) -> set:
        if "TERM" in node:
            term = node["TERM"].lower() # Lowercase to match index
            term_data = index.get(term)
            if term_data:
                return set(term_data.keys()) # Returns {uuid1, uuid2, ...}
            else:
                return set() # Empty set

        if "AND" in node:
            left_docs = self._execute_custom_query(node["AND"][0], index, all_docs)
            right_docs = self._execute_custom_query(node["AND"][1], index, all_docs)
            return left_docs.intersection(right_docs) # AND = intersection

        if "OR" in node:
            left_docs = self._execute_custom_query(node["OR"][0], index, all_docs)
            right_docs = self._execute_custom_query(node["OR"][1], index, all_docs)
            return left_docs.union(right_docs) # OR = union

        if "NOT" in node:
            operand_docs = self._execute_custom_query(node["NOT"], index, all_docs)
            return all_docs.difference(operand_docs) # NOT = set difference

        if "PHRASE" in node:
            inner = node["PHRASE"]
            phrase_term = inner.get("TERM")
            
            if not phrase_term:
                raise ValueError("PHRASE operator must be followed by a single term (e.g., PHRASE \"hello world\")")
            
            words = phrase_term.lower().split() # Lowercase to match index
            if not words:
                return set()
            
            # Get postings for the first word
            first_word = words[0]
            postings = index.get(first_word)
            if not postings:
                return set() # First word not in index

            candidate_docs = set(postings.keys())
            final_docs = set()

            # For each doc, check if the other words follow in sequence
            for doc_id in candidate_docs:
                if self._check_phrase_in_doc(doc_id, words, index):
                    final_docs.add(doc_id)
            
            return final_docs

        raise ValueError(f"Unknown query node: {node}")

    def _build_es_query(self, node, search_fields: list) -> dict:
            if not node:
                raise ValueError("Query parser returned an empty node.")

            if "TERM" in node:
                # Use "match" for a single term (better relevance than match_phrase)
                return {"multi_match": {
                    "query": node["TERM"],
                    "fields": search_fields
                }}

            if "AND" in node:
                left, right = node["AND"]
                return {
                    "bool": {
                        "must": [self._build_es_query(left, search_fields), self._build_es_query(right, search_fields)]
                    }
                }

            if "OR" in node:
                left, right = node["OR"]
                return {
                    "bool": {
                        "should": [self._build_es_query(left, search_fields), self._build_es_query(right, search_fields)],
                        "minimum_should_match": 1
                    }
                }

            if "NOT" in node:
                return {"bool": {"must_not": [self._build_es_query(node["NOT"], search_fields)]}}

            if "PHRASE" in node:
                inner = node["PHRASE"]
                if "TERM" in inner:
                    return {
                        "multi_match": {
                            "query": inner["TERM"],
                            "fields": search_fields,
                            "type": "phrase"
                        }
                    }
                return self._build_es_query(inner, search_fields)

            raise ValueError(f"Unknown node type: {node}")

    # Public functions
    def parse_query(self, query: str) -> dict | StatusCode:
        try:
            parser = QueryParser(query)
            parsed_tree = parser.parse()
            if parsed_tree is None:
                raise ValueError("Query parser returned None")
        except Exception as e:
            return StatusCode.QUERY_FAILED
        
        return parsed_tree
    
    def process_custom_query(self, query: str, loaded_index, index_id: str, all_doc_ids: List[str], dstore: str) -> dict:
        # Preprocess the query
        preprocessed_query = self._preprocess_query(query)

        # Parse the query
        parsed_tree = self.parse_query(preprocessed_query)
        if isinstance(parsed_tree, StatusCode):
            return parsed_tree
        
        # Convert all_doc_ids to a set for faster operations
        all_doc_ids_set = set(all_doc_ids)
        if not all_doc_ids_set:
            print(f"{Style.FG_ORANGE}Warning: Index contains no documents.{Style.RESET}")
            all_doc_ids_set = set()

        # Execute the query
        try:
            matching_doc_ids = self._execute_custom_query(
                parsed_tree, 
                loaded_index, 
                all_doc_ids_set
            )
        except Exception as e:
            return StatusCode.QUERY_FAILED
        
        # Fetch documents and format results
        index_data_path: str = os.path.join(STORAGE_DIR, index_id)
        hits = []

        for i, doc_id in enumerate(matching_doc_ids):
            # Stop fetching once we reach the limit
            if i >= MAX_RESULTS:
                break
                
            doc_source = None
            try:
                if dstore == DataStore.CUSTOM.name:
                    doc_path = os.path.join(index_data_path, "documents", f"{doc_id}.json")
                    with open(doc_path, "r") as f:
                        doc_source = json.load(f)

                elif dstore == DataStore.ROCKSDB.name:
                    if self.doc_store_handle:
                        doc_bytes = self.doc_store_handle.get(doc_id.encode('utf-8'))
                        if doc_bytes:
                            doc_source = json.loads(doc_bytes.decode('utf-8'))
                
                elif dstore == DataStore.REDIS.name:
                    if self.redis_client:
                        doc_json = self.redis_client.hget(f"{index_id}:documents", doc_id)
                        if doc_json:
                            doc_source = json.loads(doc_json)
                
                if doc_source:
                    hits.append({
                        "_id": doc_id,
                        "_source": doc_source
                        # You could add relevance scores here if using TF-IDF
                    })

            except Exception as e:
                print(f"{Style.FG_ORANGE}Warning: Could not retrieve document {doc_id}. Error: {e}{Style.RESET}")

        return matching_doc_ids, hits

    def process_es_query(self, es_client, index_id: str, query: str, search_fields: list, max_results: int, source: bool=True):
        if not search_fields:
            print(f"{Style.FG_RED}Error: No search_fields provided.{Style.RESET}")
            return StatusCode.QUERY_FAILED
        
        # Preprocess the query
        preprocessed_query = self._preprocess_query(query)

        try:
            parser = QueryParser(preprocessed_query)
            parsed_tree = parser.parse()
            
            if parsed_tree is None:
                raise ValueError(f"QueryParser failed to parse query: '{query}'")

            es_query = self._build_es_query(parsed_tree, search_fields)

            res = es_client.search(
                index=index_id, 
                query=es_query, 
                size=max_results,
                _source=source
            )

            return res

        except Exception as e:
            print(f"{Style.FG_RED}Error during query parsing or execution: {e}{Style.RESET}")
            print(f"Failed query: {query}")
            return StatusCode.QUERY_FAILED
