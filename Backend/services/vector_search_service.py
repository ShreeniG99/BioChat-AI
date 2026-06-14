"""
Enhanced Vector Search Service - Phase 1 Implementation
Fixes Elasticsearch connectivity, improves hybrid search, adds citation validation
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import faiss
import json
import os
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ConnectionError, RequestError, NotFoundError
import pickle

logger = logging.getLogger(__name__)


class EnhancedVectorSearchService:
    """Phase 1: Enhanced hybrid search with robust ES handling and citation validation"""
    
    def __init__(self, es_host: str = None, es_user: str = None, es_password: str = None):
        # Elasticsearch configuration with authentication
        self.es_host = es_host or os.getenv('ES_HOST', 'http://localhost:9200')
        self.es_user = es_user or os.getenv('ES_USER', 'elastic')
        self.es_password = es_password or os.getenv('ES_PASSWORD', '')
        
        # Initialize ES client with proper auth
        self.es_client = None
        self.es_available = False
        
        # FAISS setup
        self.index_path = "biomedical_faiss.index"
        self.metadata_path = "biomedical_metadata.pkl"
        self.faiss_index = None
        self.documents = []
        self.embedding_dim = 768  # BioBERT dimension
        
        # Hybrid search parameters
        self.dense_weight = 0.6
        self.sparse_weight = 0.4
        
    async def initialize(self):
        """Initialize both Elasticsearch and FAISS"""
        await self._initialize_elasticsearch()
        await self._initialize_faiss()
        
    async def _initialize_elasticsearch(self):
        """Initialize Elasticsearch with robust error handling"""
        try:
            # Configure ES client with authentication
            es_config = {
                'hosts': [self.es_host],
                'timeout': 30,
                'max_retries': 3,
                'retry_on_timeout': True,
                'verify_certs': False,  # For development
                'ssl_show_warn': False
            }
            
            # Add authentication if provided
            if self.es_password:
                es_config['http_auth'] = (self.es_user, self.es_password)
                logger.info(f"ES auth configured for user: {self.es_user}")
            
            self.es_client = AsyncElasticsearch(**es_config)
            
            # Test connection
            await self.es_client.info()
            
            # Create index if it doesn't exist
            await self._create_index()
            
            self.es_available = True
            logger.info("✅ Elasticsearch connected and initialized successfully")
            
        except ConnectionError as e:
            logger.warning(f"⚠️  Elasticsearch connection failed: {e}")
            logger.warning("🔄 Continuing with FAISS-only mode")
            self.es_available = False
            
        except Exception as e:
            logger.error(f"❌ Elasticsearch setup error: {e}")
            self.es_available = False
            
    async def _create_index(self):
        """Create Elasticsearch index with proper mapping"""
        index_name = "biomedical_docs"
        
        # Check if index exists
        try:
            exists = await self.es_client.indices.exists(index=index_name)
            if exists:
                logger.info(f"✅ Index '{index_name}' already exists")
                return
        except Exception as e:
            logger.warning(f"Could not check index existence: {e}")
            
        # Fixed mapping without unsupported parameters
        mapping = {
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "abstract": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "journal": {
                        "type": "keyword"
                    },
                    "year": {
                        "type": "integer"
                    },
                    "pmid": {
                        "type": "keyword"
                    },
                    "pmcid": {
                        "type": "keyword"
                    },
                    "doi": {
                        "type": "keyword"
                    },
                    "authors": {
                        "type": "text"
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 768
                    },
                    "chunk_id": {
                        "type": "keyword"
                    },
                    "timestamp": {
                        "type": "date"
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "biomedical_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop"]
                        }
                    }
                }
            }
        }
        
        try:
            await self.es_client.indices.create(index=index_name, body=mapping)
            logger.info(f"✅ Created index '{index_name}' with biomedical mapping")
        except RequestError as e:
            if 'already exists' in str(e):
                logger.info(f"✅ Index '{index_name}' already exists")
            else:
                logger.error(f"❌ Failed to create index: {e}")
                raise
                
    async def _initialize_faiss(self):
        """Initialize FAISS index"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                # Load existing index
                self.faiss_index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"✅ Loaded FAISS index: {self.faiss_index.ntotal} vectors, {len(self.documents)} documents")
            else:
                # Create new index
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
                self.documents = []
                logger.info("✅ Created new FAISS index")
                
        except Exception as e:
            logger.error(f"❌ FAISS initialization failed: {e}")
            # Fallback: create new index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.documents = []
            
    async def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to both FAISS and Elasticsearch"""
        if not documents:
            return
            
        valid_docs = []
        embeddings = []
        
        for doc in documents:
            if 'embedding' in doc and doc['embedding'] is not None:
                valid_docs.append(doc)
                embeddings.append(doc['embedding'])
                
        if not valid_docs:
            logger.warning("No valid embeddings found in documents")
            return
            
        # Add to FAISS
        try:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            self.faiss_index.add(embeddings_array)
            self.documents.extend(valid_docs)
            
            # Save index
            faiss.write_index(self.faiss_index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.documents, f)
                
            logger.info(f"✅ Added {len(valid_docs)} documents to FAISS")
            
        except Exception as e:
            logger.error(f"❌ Failed to add documents to FAISS: {e}")
            
        # Add to Elasticsearch if available
        if self.es_available:
            await self._add_to_elasticsearch(valid_docs)
            
    async def _add_to_elasticsearch(self, documents: List[Dict[str, Any]]):
        """Add documents to Elasticsearch"""
        try:
            actions = []
            for i, doc in enumerate(documents):
                action = {
                    "_index": "biomedical_docs",
                    "_id": f"{doc.get('pmid', f'doc_{len(self.documents) + i}')}_{i}",
                    "_source": {
                        "text": doc.get('text', ''),
                        "title": doc.get('metadata', {}).get('title', ''),
                        "abstract": doc.get('metadata', {}).get('abstract', ''),
                        "journal": doc.get('metadata', {}).get('journal', ''),
                        "year": doc.get('metadata', {}).get('year', 0),
                        "pmid": doc.get('metadata', {}).get('pmid', ''),
                        "pmcid": doc.get('metadata', {}).get('pmcid', ''),
                        "doi": doc.get('metadata', {}).get('doi', ''),
                        "authors": doc.get('metadata', {}).get('authors', ''),
                        "embedding": doc.get('embedding', []),
                        "chunk_id": doc.get('chunk_id', f'chunk_{i}'),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                actions.append(action)
                
            # Bulk index
            if actions:
                from elasticsearch.helpers import async_bulk
                success, failed = await async_bulk(self.es_client, actions, refresh=True)
                logger.info(f"✅ Added {success} documents to Elasticsearch")
                if failed:
                    logger.warning(f"⚠️  {len(failed)} documents failed to index")
                    
        except Exception as e:
            logger.error(f"❌ Failed to add documents to Elasticsearch: {e}")
            
    async def search(self, query_embedding: List[float], query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """Enhanced hybrid search with citation validation"""
        dense_results = await self._dense_search(query_embedding, k * 2)
        
        if self.es_available:
            sparse_results = await self._sparse_search(query_text, k * 2)
            # Combine and rerank
            combined_results = self._combine_results(dense_results, sparse_results, k)
        else:
            combined_results = dense_results[:k]
            logger.info("Using FAISS-only search (Elasticsearch unavailable)")
            
        # Add citation validation metadata
        validated_results = self._add_citation_metadata(combined_results)
        
        return validated_results
        
    async def _dense_search(self, query_embedding: List[float], k: int) -> List[Dict[str, Any]]:
        """Dense vector search using FAISS"""
        try:
            if self.faiss_index.ntotal == 0:
                return []
                
            # Normalize query embedding
            query_array = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)
            
            # Search
            scores, indices = self.faiss_index.search(query_array, min(k, self.faiss_index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc['score'] = float(score)
                    doc['search_type'] = 'dense'
                    results.append(doc)
                    
            return results
            
        except Exception as e:
            logger.error(f"❌ Dense search failed: {e}")
            return []
            
    async def _sparse_search(self, query_text: str, k: int) -> List[Dict[str, Any]]:
        """Sparse (BM25) search using Elasticsearch"""
        try:
            if not self.es_available:
                return []
                
            # Multi-field BM25 query
            query = {
                "query": {
                    "multi_match": {
                        "query": query_text,
                        "fields": ["text^2", "title^1.5", "abstract^1.2", "authors^0.8"],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                        "minimum_should_match": "70%"
                    }
                },
                "size": k,
                "_source": ["text", "title", "abstract", "journal", "year", "pmid", "pmcid", "doi", "authors", "chunk_id"]
            }
            
            response = await self.es_client.search(index="biomedical_docs", body=query)
            
            results = []
            for hit in response['hits']['hits']:
                doc = {
                    'text': hit['_source'].get('text', ''),
                    'metadata': {
                        'title': hit['_source'].get('title', ''),
                        'abstract': hit['_source'].get('abstract', ''),
                        'journal': hit['_source'].get('journal', ''),
                        'year': hit['_source'].get('year', ''),
                        'pmid': hit['_source'].get('pmid', ''),
                        'pmcid': hit['_source'].get('pmcid', ''),
                        'doi': hit['_source'].get('doi', ''),
                        'authors': hit['_source'].get('authors', ''),
                    },
                    'score': hit['_score'],
                    'search_type': 'sparse',
                    'chunk_id': hit['_source'].get('chunk_id', '')
                }
                results.append(doc)
                
            return results
            
        except Exception as e:
            logger.error(f"❌ Sparse search failed: {e}")
            return []
            
    def _combine_results(self, dense_results: List[Dict], sparse_results: List[Dict], k: int) -> List[Dict]:
        """Combine and rerank dense and sparse results"""
        # Normalize scores
        if dense_results:
            max_dense = max(r['score'] for r in dense_results)
            for r in dense_results:
                r['normalized_dense_score'] = r['score'] / max_dense if max_dense > 0 else 0
                
        if sparse_results:
            max_sparse = max(r['score'] for r in sparse_results)
            for r in sparse_results:
                r['normalized_sparse_score'] = r['score'] / max_sparse if max_sparse > 0 else 0
                
        # Create combined results with RRF (Reciprocal Rank Fusion)
        all_results = {}
        
        # Add dense results
        for i, result in enumerate(dense_results):
            key = result.get('chunk_id', f"dense_{i}")
            if key not in all_results:
                all_results[key] = result.copy()
                all_results[key]['dense_rank'] = i + 1
                all_results[key]['sparse_rank'] = float('inf')
            
        # Add sparse results
        for i, result in enumerate(sparse_results):
            key = result.get('chunk_id', f"sparse_{i}")
            if key in all_results:
                all_results[key]['sparse_rank'] = i + 1
            else:
                all_results[key] = result.copy()
                all_results[key]['dense_rank'] = float('inf')
                all_results[key]['sparse_rank'] = i + 1
                
        # Calculate combined scores using RRF and weighted combination
        for result in all_results.values():
            # RRF score
            rrf_score = (1.0 / (60 + result['dense_rank'])) + (1.0 / (60 + result['sparse_rank']))
            
            # Weighted score
            dense_score = result.get('normalized_dense_score', 0) * self.dense_weight
            sparse_score = result.get('normalized_sparse_score', 0) * self.sparse_weight
            weighted_score = dense_score + sparse_score
            
            # Combined score
            result['combined_score'] = 0.6 * rrf_score + 0.4 * weighted_score
            
        # Sort and return top k
        combined_list = sorted(all_results.values(), key=lambda x: x['combined_score'], reverse=True)
        return combined_list[:k]
        
    def _add_citation_metadata(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add citation validation metadata to search results"""
        for i, result in enumerate(results):
            result['citation_id'] = i + 1
            result['citation_text'] = result.get('text', '')[:200] + '...' if len(result.get('text', '')) > 200 else result.get('text', '')
            result['citation_valid'] = bool(result.get('text', '').strip())
            result['source_quality'] = self._assess_source_quality(result.get('metadata', {}))
            
        return results
        
    def _assess_source_quality(self, metadata: Dict[str, Any]) -> str:
        """Assess quality of source for citation validation"""
        journal = metadata.get('journal', '').lower()
        year = metadata.get('year', 0)
        
        # High-impact journals
        high_impact = ['nature', 'science', 'cell', 'lancet', 'nejm', 'jama', 'bmj']
        if any(journal_name in journal for journal_name in high_impact):
            return 'high'
            
        # Recent papers
        if isinstance(year, (int, str)) and str(year).isdigit():
            year_int = int(year)
            if year_int >= 2020:
                return 'recent'
            elif year_int >= 2015:
                return 'moderate'
            else:
                return 'older'
                
        return 'unknown'
        
    async def get_document_count(self) -> Dict[str, int]:
        """Get document counts from both indices"""
        counts = {
            'faiss': self.faiss_index.ntotal if self.faiss_index else 0,
            'elasticsearch': 0
        }
        
        if self.es_available:
            try:
                response = await self.es_client.count(index="biomedical_docs")
                counts['elasticsearch'] = response['count']
            except Exception as e:
                logger.error(f"Failed to get ES count: {e}")
                
        return counts
        
    async def health_check(self) -> Dict[str, Any]:
        """Health check for search services"""
        health = {
            'faiss_available': self.faiss_index is not None,
            'faiss_documents': self.faiss_index.ntotal if self.faiss_index else 0,
            'elasticsearch_available': self.es_available,
            'elasticsearch_documents': 0,
            'hybrid_search': self.es_available and self.faiss_index is not None
        }
        
        if self.es_available:
            try:
                es_health = await self.es_client.cluster.health()
                health['elasticsearch_status'] = es_health['status']
                
                count_response = await self.es_client.count(index="biomedical_docs")
                health['elasticsearch_documents'] = count_response['count']
            except Exception as e:
                health['elasticsearch_error'] = str(e)
                
        return health
        
    async def close(self):
        """Close connections"""
        if self.es_client:
            await self.es_client.close()