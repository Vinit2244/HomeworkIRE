# # ======================== IMPORTS ========================
# import os
# import re
# import json
# import math
# import pickle
# import zlib
# from collections import defaultdict
# from typing import Iterable, Dict, List, Set, Tuple
# from pathlib import Path
# import psycopg2
# import redis
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize
# import nltk

# # Download required NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')
# # try:
# #     nltk.data.find('corpora/stopwords')
# # except LookupError:
# #     nltk.download('stopwords')

# from .index_base import BaseIndex, IndexInfo, DataStore, Compression, QueryProc, Optimizations
# from utils import StatusCode, load_config

# # ======================= GLOBALS ========================
# # STOP_WORDS = set(stopwords.words('english'))
# # STEMMER = PorterStemmer()
# # INDEX_BASE_DIR = Path("./custom_indices")
# # INDEX_BASE_DIR.mkdir(exist_ok=True)
# config = load_config()
# STORAGE_DIR: str = config.get("storage_folder_path", "./storage")
# os.makedirs(STORAGE_DIR, exist_ok=True)
# CONTENT_FIELD: str = config.get("content_field", "text")

# # =================== COMPRESSION ===================
# class CompressionHandler:
#     """Handles compression of posting lists"""
    
#     @staticmethod
#     def simple_varbyte_encode(numbers: List[int]) -> bytes:
#         """Simple variable byte encoding"""
#         result = bytearray()
#         for num in numbers:
#             while num >= 128:
#                 result.append((num % 128) | 0x80)
#                 num //= 128
#             result.append(num)
#         return bytes(result)
    
#     @staticmethod
#     def simple_varbyte_decode(data: bytes) -> List[int]:
#         """Simple variable byte decoding"""
#         result = []
#         current = 0
#         shift = 0
#         for byte in data:
#             if byte & 0x80:
#                 current |= (byte & 0x7F) << shift
#                 shift += 7
#             else:
#                 current |= byte << shift
#                 result.append(current)
#                 current = 0
#                 shift = 0
#         return result
    
#     @staticmethod
#     def zlib_compress(data: bytes) -> bytes:
#         """Compress using zlib"""
#         return zlib.compress(data, level=9)
    
#     @staticmethod
#     def zlib_decompress(data: bytes) -> bytes:
#         """Decompress using zlib"""
#         return zlib.decompress(data)

# # =================== TEXT PROCESSING ===================
# class TextProcessor:
#     """Handles tokenization, stop word removal, and stemming"""
    
#     @staticmethod
#     def process_text(text: str) -> List[str]:
#         """Process text: tokenize, lowercase, remove stopwords, stem"""
#         # Tokenize
#         tokens = word_tokenize(text.lower())
        
#         # Remove non-alphanumeric tokens and stopwords
#         tokens = [token for token in tokens if token.isalnum() and token not in STOP_WORDS]
        
#         # Stem
#         tokens = [STEMMER.stem(token) for token in tokens]
        
#         return tokens
    
#     @staticmethod
#     def get_term_positions(text: str) -> Dict[str, List[int]]:
#         """Get term positions in text"""
#         tokens = word_tokenize(text.lower())
#         term_positions = defaultdict(list)
        
#         for pos, token in enumerate(tokens):
#             if token.isalnum() and token not in STOP_WORDS:
#                 stemmed = STEMMER.stem(token)
#                 term_positions[stemmed].append(pos)
        
#         return dict(term_positions)

# # =================== QUERY PARSER ===================
# class QueryParser:
#     """Parses boolean queries with AND, OR, NOT, PHRASE operators"""
    
#     @staticmethod
#     def tokenize_query(query: str) -> List[str]:
#         """Tokenize query into terms and operators"""
#         # Handle quoted phrases
#         pattern = r'"([^"]+)"|(\(|\)|AND|OR|NOT)|(\w+)'
#         tokens = []
#         for match in re.finditer(pattern, query):
#             if match.group(1):  # Quoted phrase
#                 tokens.append(('PHRASE', match.group(1)))
#             elif match.group(2):  # Operator or parenthesis
#                 tokens.append(match.group(2))
#             elif match.group(3):  # Regular term
#                 tokens.append(('TERM', match.group(3)))
#         return tokens
    
#     @staticmethod
#     def parse_query(query: str) -> dict:
#         """Parse query into AST"""
#         tokens = QueryParser.tokenize_query(query)
#         pos = [0]  # Use list to make it mutable in nested functions
        
#         def parse_expr():
#             left = parse_term()
            
#             while pos[0] < len(tokens):
#                 token = tokens[pos[0]]
                
#                 if token == 'OR':
#                     pos[0] += 1
#                     right = parse_expr()
#                     left = {'op': 'OR', 'left': left, 'right': right}
#                 elif token == 'AND':
#                     pos[0] += 1
#                     right = parse_term()
#                     left = {'op': 'AND', 'left': left, 'right': right}
#                 else:
#                     break
            
#             return left
        
#         def parse_term():
#             if pos[0] >= len(tokens):
#                 return None
            
#             token = tokens[pos[0]]
            
#             if token == 'NOT':
#                 pos[0] += 1
#                 expr = parse_term()
#                 return {'op': 'NOT', 'expr': expr}
#             elif token == '(':
#                 pos[0] += 1
#                 expr = parse_expr()
#                 if pos[0] < len(tokens) and tokens[pos[0]] == ')':
#                     pos[0] += 1
#                 return expr
#             elif isinstance(token, tuple):
#                 pos[0] += 1
#                 if token[0] == 'PHRASE':
#                     return {'op': 'PHRASE', 'value': token[1]}
#                 else:  # TERM
#                     return {'op': 'TERM', 'value': token[1]}
            
#             return None
        
#         return parse_expr()

# # =================== CUSTOM INDEX ===================
# class CustomIndex(BaseIndex):
#     """Custom inverted index implementation"""
    
#     def __init__(self, core: str, info: str, dstore: str, qproc: str, compr: str, optim: str):
#         super().__init__(core, info, dstore, qproc, compr, optim)
        
#         self.info_type = IndexInfo[info]
#         self.dstore_type = DataStore[dstore]
#         self.compr_type = Compression[compr]
#         self.qproc_type = QueryProc[qproc]
#         self.optim_type = Optimizations[optim]
        
#         # In-memory index structures
#         self.inverted_index = {}  # term -> postings list
#         self.doc_lengths = {}  # doc_id -> length
#         self.doc_count = 0
#         self.avg_doc_length = 0
#         self.current_index_id = None
        
#         # Database connections
#         self.db_conn = None
#         self.redis_conn = None
        
#         self._init_datastore()
    
#     def _init_datastore(self):
#         """Initialize datastore connections"""
#         if self.dstore_type == DataStore.POSTGRESQL:
#             try:
#                 self.db_conn = psycopg2.connect(
#                     host="localhost",
#                     database="ir_index",
#                     user="postgres",
#                     password="postgres"
#                 )
#                 self._create_postgres_tables()
#             except Exception as e:
#                 print(f"PostgreSQL connection failed: {e}")
#                 print("Falling back to CUSTOM datastore")
#                 self.dstore_type = DataStore.CUSTOM
        
#         elif self.dstore_type == DataStore.REDIS:
#             try:
#                 self.redis_conn = redis.Redis(
#                     host='localhost',
#                     port=6379,
#                     db=0,
#                     decode_responses=False
#                 )
#                 self.redis_conn.ping()
#             except Exception as e:
#                 print(f"Redis connection failed: {e}")
#                 print("Falling back to CUSTOM datastore")
#                 self.dstore_type = DataStore.CUSTOM
    
#     def _create_postgres_tables(self):
#         """Create PostgreSQL tables for index storage"""
#         cursor = self.db_conn.cursor()
        
#         # Postings table
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS postings (
#                 index_id TEXT,
#                 term TEXT,
#                 doc_id TEXT,
#                 positions BYTEA,
#                 word_count INTEGER,
#                 tf_idf REAL,
#                 PRIMARY KEY (index_id, term, doc_id)
#             )
#         """)
        
#         # Index metadata table
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS index_metadata (
#                 index_id TEXT PRIMARY KEY,
#                 doc_count INTEGER,
#                 avg_doc_length REAL,
#                 metadata JSONB
#             )
#         """)
        
#         # Create GIN index for faster searches
#         cursor.execute("""
#             CREATE INDEX IF NOT EXISTS idx_postings_term 
#             ON postings USING GIN (term gin_trgm_ops)
#         """)
        
#         self.db_conn.commit()
#         cursor.close()
    
#     def _get_index_dir(self, index_id: str) -> str:
#         """Get directory path for index"""
#         return os.path.join(STORAGE_DIR, index_id)
    
#     def _compress_postings(self, postings: List[int]) -> bytes:
#         """Compress postings list"""
#         if self.compr_type == Compression.NONE:
#             return pickle.dumps(postings)
#         elif self.compr_type == Compression.CODE:
#             return CompressionHandler.simple_varbyte_encode(postings)
#         else:  # CLIB
#             data = pickle.dumps(postings)
#             return CompressionHandler.zlib_compress(data)
    
#     def _decompress_postings(self, data: bytes) -> List[int]:
#         """Decompress postings list"""
#         if self.compr_type == Compression.NONE:
#             return pickle.loads(data)
#         elif self.compr_type == Compression.CODE:
#             return CompressionHandler.simple_varbyte_decode(data)
#         else:  # CLIB
#             decompressed = CompressionHandler.zlib_decompress(data)
#             return pickle.loads(decompressed)
    
#     def _build_inverted_index(self, files: Iterable[Tuple[str, dict]]) -> dict:
#         """Build inverted index from files"""
#         inverted_index = defaultdict(lambda: defaultdict(dict))
#         doc_lengths = {}
#         doc_count = 0
#         total_length = 0
        
#         for doc_id, doc_data in files:
#             # Get text content (assuming 'text' field)
#             text = doc_data.get(CONTENT_FIELD, '')
#             if not text:
#                 continue
            
#             doc_count += 1
            
#             # Process text based on index info type
#             if self.info_type == IndexInfo.BOOLEAN:
#                 # Boolean index: just doc_ids and positions
#                 term_positions = TextProcessor.get_term_positions(text)
#                 for term, positions in term_positions.items():
#                     inverted_index[term][doc_id] = {
#                         'positions': positions
#                     }
#                 doc_lengths[doc_id] = len(term_positions)
            
#             elif self.dstore_type == DataStore.POSTGRESQL:
#                 cursor = self.db_conn.cursor()
#                 cursor.execute("DELETE FROM postings WHERE index_id = %s", (index_id,))
#                 cursor.execute("DELETE FROM index_metadata WHERE index_id = %s", (index_id,))
#                 self.db_conn.commit()
#                 cursor.close()
            
#             else:  # REDIS
#                 # Delete all keys related to this index
#                 pattern = f"{index_id}:*"
#                 keys = self.redis_conn.keys(pattern)
#                 if keys:
#                     self.redis_conn.delete(*keys)
            
#             return StatusCode.SUCCESS
        
#         except Exception as e:
#             print(f"Error deleting index: {e}")
#             return StatusCode.ERROR_ACCESSING_INDEX
    
#     def list_indices(self) -> Iterable[str]:
#         """List all indices"""
#         try:
#             if self.dstore_type == DataStore.CUSTOM:
#                 indices = []
#                 for path in INDEX_BASE_DIR.iterdir():
#                     if path.is_dir():
#                         indices.append({"index": path.name})
#                 return indices
            
#             elif self.dstore_type == DataStore.POSTGRESQL:
#                 cursor = self.db_conn.cursor()
#                 cursor.execute("SELECT index_id FROM index_metadata")
#                 indices = [{"index": row[0]} for row in cursor.fetchall()]
#                 cursor.close()
#                 return indices
            
#             else:  # REDIS
#                 pattern = "*:metadata"
#                 keys = self.redis_conn.keys(pattern)
#                 indices = [{"index": key.decode().replace(":metadata", "")} for key in keys]
#                 return indices
        
#         except Exception as e:
#             print(f"Error listing indices: {e}")
#             return []
    
#     def list_indexed_files(self, index_id: str) -> Iterable[str]:
#         """List all files in index"""
#         try:
#             status = self.load_index(index_id)
#             if status != StatusCode.SUCCESS:
#                 return []
            
#             return list(self.doc_lengths.keys())
        
#         except Exception as e:
#             print(f"Error listing files: {e}")
#             return []
    
#     def get_index_info(self, index_id: str) -> dict:
#         """Get index information"""
#         try:
#             status = self.load_index(index_id)
#             if status != StatusCode.SUCCESS:
#                 return StatusCode.ERROR_ACCESSING_INDEX
            
#             return {
#                 index_id: {
#                     "docs_count": self.doc_count,
#                     "avg_doc_length": self.avg_doc_length,
#                     "unique_terms": len(self.inverted_index),
#                     "index_type": self.info_type.name,
#                     "datastore": self.dstore_type.name,
#                     "compression": self.compr_type.name,
#                     "query_proc": self.qproc_type.name,
#                     "optimizations": self.optim_type.name
#                 }
#             }
        
#         except Exception as e:
#             print(f"Error getting index info: {e}")
#             return StatusCode.ERROR_ACCESSING_INDEXinfo_type == IndexInfo.WORDCOUNT:
#                 # Word count index: doc_ids, positions, and counts
#                 term_positions = TextProcessor.get_term_positions(text)
#                 for term, positions in term_positions.items():
#                     inverted_index[term][doc_id] = {
#                         'positions': positions,
#                         'count': len(positions)
#                     }
#                 doc_lengths[doc_id] = sum(len(pos) for pos in term_positions.values())
            
#             else:  # TFIDF
#                 # TF-IDF index: doc_ids, positions, counts, and TF-IDF scores
#                 term_positions = TextProcessor.get_term_positions(text)
#                 doc_length = sum(len(pos) for pos in term_positions.values())
                
#                 for term, positions in term_positions.items():
#                     count = len(positions)
#                     tf = count / doc_length if doc_length > 0 else 0
                    
#                     inverted_index[term][doc_id] = {
#                         'positions': positions,
#                         'count': count,
#                         'tf': tf
#                     }
                
#                 doc_lengths[doc_id] = doc_length
            
#             total_length += doc_lengths[doc_id]
        
#         # Calculate IDF and TF-IDF for TFIDF index
#         if self.info_type == IndexInfo.TFIDF:
#             for term, postings in inverted_index.items():
#                 df = len(postings)
#                 idf = math.log(doc_count / df) if df > 0 else 0
                
#                 for doc_id in postings:
#                     tf = postings[doc_id]['tf']
#                     postings[doc_id]['tfidf'] = tf * idf
        
#         avg_doc_length = total_length / doc_count if doc_count > 0 else 0
        
#         return {
#             'index': dict(inverted_index),
#             'doc_lengths': doc_lengths,
#             'doc_count': doc_count,
#             'avg_doc_length': avg_doc_length
#         }
    
#     def _save_to_custom_storage(self, index_id: str, index_data: dict):
#         """Save index to custom disk storage"""
#         index_dir = self._get_index_dir(index_id)
#         index_dir.mkdir(exist_ok=True, parents=True)
        
#         # Save inverted index
#         index_file = index_dir / "inverted_index.pkl"
#         with open(index_file, 'wb') as f:
#             pickle.dump(index_data['index'], f)
        
#         # Save metadata
#         metadata_file = index_dir / "metadata.json"
#         metadata = {
#             'doc_count': index_data['doc_count'],
#             'avg_doc_length': index_data['avg_doc_length'],
#             'doc_lengths': index_data['doc_lengths']
#         }
#         with open(metadata_file, 'w') as f:
#             json.dump(metadata, f)
    
#     def _save_to_postgresql(self, index_id: str, index_data: dict):
#         """Save index to PostgreSQL"""
#         cursor = self.db_conn.cursor()
        
#         # Save metadata
#         cursor.execute("""
#             INSERT INTO index_metadata (index_id, doc_count, avg_doc_length, metadata)
#             VALUES (%s, %s, %s, %s)
#             ON CONFLICT (index_id) DO UPDATE
#             SET doc_count = EXCLUDED.doc_count,
#                 avg_doc_length = EXCLUDED.avg_doc_length,
#                 metadata = EXCLUDED.metadata
#         """, (
#             index_id,
#             index_data['doc_count'],
#             index_data['avg_doc_length'],
#             json.dumps(index_data['doc_lengths'])
#         ))
        
#         # Save postings
#         for term, postings in index_data['index'].items():
#             for doc_id, data in postings.items():
#                 positions = self._compress_postings(data['positions'])
#                 word_count = data.get('count', len(data['positions']))
#                 tf_idf = data.get('tfidf', 0.0)
                
#                 cursor.execute("""
#                     INSERT INTO postings (index_id, term, doc_id, positions, word_count, tf_idf)
#                     VALUES (%s, %s, %s, %s, %s, %s)
#                     ON CONFLICT (index_id, term, doc_id) DO UPDATE
#                     SET positions = EXCLUDED.positions,
#                         word_count = EXCLUDED.word_count,
#                         tf_idf = EXCLUDED.tf_idf
#                 """, (index_id, term, doc_id, positions, word_count, tf_idf))
        
#         self.db_conn.commit()
#         cursor.close()
    
#     def _save_to_redis(self, index_id: str, index_data: dict):
#         """Save index to Redis"""
#         # Save metadata
#         self.redis_conn.hset(
#             f"{index_id}:metadata",
#             mapping={
#                 'doc_count': index_data['doc_count'],
#                 'avg_doc_length': index_data['avg_doc_length'],
#                 'doc_lengths': json.dumps(index_data['doc_lengths'])
#             }
#         )
        
#         # Save postings
#         for term, postings in index_data['index'].items():
#             for doc_id, data in postings.items():
#                 key = f"{index_id}:term:{term}:doc:{doc_id}"
#                 value = pickle.dumps(data)
#                 self.redis_conn.set(key, value)
            
#             # Save document list for each term
#             doc_list = list(postings.keys())
#             self.redis_conn.sadd(f"{index_id}:term:{term}:docs", *doc_list)
    
#     def create_index(self, index_id: str, files: Iterable[Tuple[str, dict]]) -> StatusCode:
#         """Create index from files"""
#         try:
#             print(f"Building inverted index for '{index_id}'...")
#             index_data = self._build_inverted_index(files)
            
#             print(f"Saving index to {self.dstore_type.name} storage...")
#             if self.dstore_type == DataStore.CUSTOM:
#                 self._save_to_custom_storage(index_id, index_data)
#             elif self.dstore_type == DataStore.POSTGRESQL:
#                 self._save_to_postgresql(index_id, index_data)
#             else:  # REDIS
#                 self._save_to_redis(index_id, index_data)
            
#             print(f"Index '{index_id}' created successfully!")
#             return StatusCode.SUCCESS
        
#         except Exception as e:
#             print(f"Error creating index: {e}")
#             return StatusCode.INDEXING_FAILED
    
#     def load_index(self, index_id: str) -> StatusCode:
#         """Load index into memory"""
#         try:
#             self.current_index_id = index_id
            
#             if self.dstore_type == DataStore.CUSTOM:
#                 index_dir = self._get_index_dir(index_id)
                
#                 # Load inverted index
#                 with open(index_dir / "inverted_index.pkl", 'rb') as f:
#                     self.inverted_index = pickle.load(f)
                
#                 # Load metadata
#                 with open(index_dir / "metadata.json", 'r') as f:
#                     metadata = json.load(f)
#                     self.doc_count = metadata['doc_count']
#                     self.avg_doc_length = metadata['avg_doc_length']
#                     self.doc_lengths = metadata['doc_lengths']
            
#             elif self.dstore_type == DataStore.POSTGRESQL:
#                 cursor = self.db_conn.cursor()
                
#                 # Load metadata
#                 cursor.execute("""
#                     SELECT doc_count, avg_doc_length, metadata
#                     FROM index_metadata WHERE index_id = %s
#                 """, (index_id,))
#                 row = cursor.fetchone()
#                 if not row:
#                     return StatusCode.ERROR_ACCESSING_INDEX
                
#                 self.doc_count = row[0]
#                 self.avg_doc_length = row[1]
#                 self.doc_lengths = json.loads(row[2])
                
#                 # Load postings
#                 cursor.execute("""
#                     SELECT term, doc_id, positions, word_count, tf_idf
#                     FROM postings WHERE index_id = %s
#                 """, (index_id,))
                
#                 self.inverted_index = defaultdict(dict)
#                 for term, doc_id, positions, word_count, tf_idf in cursor:
#                     positions_list = self._decompress_postings(bytes(positions))
#                     self.inverted_index[term][doc_id] = {
#                         'positions': positions_list,
#                         'count': word_count,
#                         'tfidf': tf_idf
#                     }
                
#                 self.inverted_index = dict(self.inverted_index)
#                 cursor.close()
            
#             else:  # REDIS
#                 # Load metadata
#                 metadata = self.redis_conn.hgetall(f"{index_id}:metadata")
#                 self.doc_count = int(metadata[b'doc_count'])
#                 self.avg_doc_length = float(metadata[b'avg_doc_length'])
#                 self.doc_lengths = json.loads(metadata[b'doc_lengths'])
                
#                 # Note: For Redis, we'll load postings on-demand during query
#                 self.inverted_index = {}
            
#             return StatusCode.SUCCESS
        
#         except Exception as e:
#             print(f"Error loading index: {e}")
#             return StatusCode.ERROR_ACCESSING_INDEX
    
#     def _get_postings_redis(self, term: str) -> Dict:
#         """Get postings for a term from Redis"""
#         if term in self.inverted_index:
#             return self.inverted_index[term]
        
#         postings = {}
#         doc_ids = self.redis_conn.smembers(f"{self.current_index_id}:term:{term}:docs")
        
#         for doc_id in doc_ids:
#             key = f"{self.current_index_id}:term:{term}:doc:{doc_id.decode()}"
#             data = self.redis_conn.get(key)
#             if data:
#                 postings[doc_id.decode()] = pickle.loads(data)
        
#         self.inverted_index[term] = postings
#         return postings
    
#     def _evaluate_query_tree(self, node: dict) -> Set[str]:
#         """Evaluate query tree and return matching document IDs"""
#         if node is None:
#             return set()
        
#         op = node['op']
        
#         if op == 'TERM':
#             term = STEMMER.stem(node['value'].lower())
            
#             if self.dstore_type == DataStore.REDIS:
#                 postings = self._get_postings_redis(term)
#             else:
#                 postings = self.inverted_index.get(term, {})
            
#             return set(postings.keys())
        
#         elif op == 'PHRASE':
#             phrase = node['value']
#             terms = [STEMMER.stem(t.lower()) for t in word_tokenize(phrase) 
#                      if t.isalnum() and t.lower() not in STOP_WORDS]
            
#             if not terms:
#                 return set()
            
#             # Get postings for all terms
#             all_postings = []
#             for term in terms:
#                 if self.dstore_type == DataStore.REDIS:
#                     postings = self._get_postings_redis(term)
#                 else:
#                     postings = self.inverted_index.get(term, {})
#                 all_postings.append(postings)
            
#             # Find documents containing all terms
#             doc_ids = set(all_postings[0].keys())
#             for postings in all_postings[1:]:
#                 doc_ids &= set(postings.keys())
            
#             # Check for phrase proximity
#             result = set()
#             for doc_id in doc_ids:
#                 # Check if terms appear in sequence
#                 positions_lists = [all_postings[i][doc_id]['positions'] for i in range(len(terms))]
                
#                 for start_pos in positions_lists[0]:
#                     found = True
#                     for i, positions in enumerate(positions_lists[1:], 1):
#                         if start_pos + i not in positions:
#                             found = False
#                             break
#                     if found:
#                         result.add(doc_id)
#                         break
            
#             return result
        
#         elif op == 'AND':
#             left_docs = self._evaluate_query_tree(node['left'])
#             right_docs = self._evaluate_query_tree(node['right'])
#             return left_docs & right_docs
        
#         elif op == 'OR':
#             left_docs = self._evaluate_query_tree(node['left'])
#             right_docs = self._evaluate_query_tree(node['right'])
#             return left_docs | right_docs
        
#         elif op == 'NOT':
#             expr_docs = self._evaluate_query_tree(node['expr'])
#             all_docs = set(self.doc_lengths.keys())
#             return all_docs - expr_docs
        
#         return set()
    
#     def _score_documents(self, doc_ids: Set[str], query: str) -> List[Tuple[str, float]]:
#         """Score documents based on index type"""
#         scores = []
        
#         for doc_id in doc_ids:
#             if self.info_type == IndexInfo.BOOLEAN:
#                 # Boolean: all matching docs have same score
#                 scores.append((doc_id, 1.0))
            
#             elif self.info_type == IndexInfo.WORDCOUNT:
#                 # WordCount: score based on term frequency
#                 query_terms = TextProcessor.process_text(query)
#                 score = 0
                
#                 for term in query_terms:
#                     if self.dstore_type == DataStore.REDIS:
#                         postings = self._get_postings_redis(term)
#                     else:
#                         postings = self.inverted_index.get(term, {})
                    
#                     if doc_id in postings:
#                         score += postings[doc_id].get('count', 0)
                
#                 scores.append((doc_id, score))
            
#             else:  # TFIDF
#                 # TF-IDF: score based on TF-IDF values
#                 query_terms = TextProcessor.process_text(query)
#                 score = 0
                
#                 for term in query_terms:
#                     if self.dstore_type == DataStore.REDIS:
#                         postings = self._get_postings_redis(term)
#                     else:
#                         postings = self.inverted_index.get(term, {})
                    
#                     if doc_id in postings:
#                         score += postings[doc_id].get('tfidf', 0)
                
#                 scores.append((doc_id, score))
        
#         # Sort by score descending
#         scores.sort(key=lambda x: x[1], reverse=True)
#         return scores
    
#     def query(self, query: str, index_id: str = None) -> str:
#         """Execute query and return results"""
#         try:
#             if index_id and index_id != self.current_index_id:
#                 status = self.load_index(index_id)
#                 if status != StatusCode.SUCCESS:
#                     return json.dumps({"error": "Failed to load index"})
            
#             # Parse query
#             query_tree = QueryParser.parse_query(query)
            
#             # Evaluate query based on processing type
#             if self.qproc_type == QueryProc.TERM:
#                 # Term-at-a-time processing
#                 matching_docs = self._evaluate_query_tree(query_tree)
#             else:  # DOC
#                 # Document-at-a-time processing
#                 matching_docs = self._evaluate_query_tree(query_tree)
            
#             # Score documents
#             scored_docs = self._score_documents(matching_docs, query)
            
#             # Format results
#             results = {
#                 "hits": {
#                     "total": {"value": len(scored_docs)},
#                     "hits": [
#                         {
#                             "_index": self.current_index_id,
#                             "_id": doc_id,
#                             "_score": score,
#                             "_source": {"doc_id": doc_id}
#                         }
#                         for doc_id, score in scored_docs[:50]  # Top 50 results
#                     ]
#                 }
#             }
            
#             return json.dumps(results, indent=2)
        
#         except Exception as e:
#             return json.dumps({"error": str(e)})
    
#     def update_index(self, index_id: str, remove_files: Iterable[str], 
#                      add_files: Iterable[Tuple[str, dict]]) -> StatusCode:
#         """Update index by removing and adding documents"""
#         try:
#             # Load current index
#             status = self.load_index(index_id)
#             if status != StatusCode.SUCCESS:
#                 return status
            
#             # Remove files
#             for doc_id in remove_files:
#                 # Remove from inverted index
#                 for term in list(self.inverted_index.keys()):
#                     if doc_id in self.inverted_index[term]:
#                         del self.inverted_index[term][doc_id]
#                     if not self.inverted_index[term]:
#                         del self.inverted_index[term]
                
#                 # Remove from doc_lengths
#                 if doc_id in self.doc_lengths:
#                     del self.doc_lengths[doc_id]
#                     self.doc_count -= 1
            
#             # Add files
#             new_index_data = self._build_inverted_index(add_files)
            
#             # Merge new data
#             for term, postings in new_index_data['index'].items():
#                 if term not in self.inverted_index:
#                     self.inverted_index[term] = {}
#                 self.inverted_index[term].update(postings)
            
#             self.doc_lengths.update(new_index_data['doc_lengths'])
#             self.doc_count += new_index_data['doc_count']
            
#             # Recalculate average doc length
#             total_length = sum(self.doc_lengths.values())
#             self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 0
            
#             # Save updated index
#             updated_data = {
#                 'index': self.inverted_index,
#                 'doc_lengths': self.doc_lengths,
#                 'doc_count': self.doc_count,
#                 'avg_doc_length': self.avg_doc_length
#             }
            
#             if self.dstore_type == DataStore.CUSTOM:
#                 self._save_to_custom_storage(index_id, updated_data)
#             elif self.dstore_type == DataStore.POSTGRESQL:
#                 self._save_to_postgresql(index_id, updated_data)
#             else:  # REDIS
#                 self._save_to_redis(index_id, updated_data)
            
#             return StatusCode.SUCCESS
        
#         except Exception as e:
#             print(f"Error updating index: {e}")
#             return StatusCode.ERROR_ACCESSING_INDEX
    
#     def delete_index(self, index_id: str) -> StatusCode:
#         """Delete index"""
#         try:
#             if self.dstore_type == DataStore.CUSTOM:
#                 index_dir = self._get_index_dir(index_id)
#                 if index_dir.exists():
#                     import shutil
#                     shutil.rmtree(index_dir)
            
#             elif self.