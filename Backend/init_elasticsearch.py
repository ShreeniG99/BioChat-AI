"""
Initialize Elasticsearch index for BioChatAI
Save this as: init_elasticsearch.py
"""

import os
import sys
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv('enhanced.env')

def wait_for_elasticsearch(es, max_retries=30, delay=2):
    """Wait for Elasticsearch to be ready"""
    for i in range(max_retries):
        try:
            if es.ping():
                print("✅ Elasticsearch is ready!")
                return True
        except Exception as e:
            print(f"⏳ Waiting for Elasticsearch... ({i+1}/{max_retries})")
            time.sleep(delay)
    return False

def create_index():
    """Create the biomedical_docs index with proper mappings"""
    
    # Connect to Elasticsearch
    es = Elasticsearch(
        ['http://localhost:9200'],
        basic_auth=('elastic', os.getenv('ES_PASSWORD')),
        verify_certs=False,
        timeout=30
    )
    
    # Wait for Elasticsearch to be ready
    if not wait_for_elasticsearch(es):
        print("❌ Elasticsearch is not responding. Please check if it's running.")
        sys.exit(1)
    
    index_name = os.getenv('ES_INDEX_NAME', 'biomedical_docs')
    
    # Check if index exists
    if es.indices.exists(index=index_name):
        print(f"⚠️  Index '{index_name}' already exists.")
        response = input("Do you want to delete and recreate it? (y/n): ")
        if response.lower() == 'y':
            es.indices.delete(index=index_name)
            print(f"🗑️  Deleted existing index '{index_name}'")
        else:
            print("Keeping existing index.")
            return
    
    # Define index mappings for hybrid search
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "biomedical_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "stop", "snowball"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                # Document metadata
                "doc_id": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "biomedical_analyzer",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "abstract": {
                    "type": "text",
                    "analyzer": "biomedical_analyzer"
                },
                "content": {
                    "type": "text",
                    "analyzer": "biomedical_analyzer"
                },
                "authors": {"type": "text"},
                "journal": {"type": "text"},
                "publication_date": {"type": "date"},
                "pmid": {"type": "keyword"},
                "doi": {"type": "keyword"},
                
                # Vector embeddings for dense retrieval
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768,  # BioBERT embedding dimension
                    "index": True,
                    "similarity": "cosine"
                },
                
                # Additional fields for RAG
                "chunks": {
                    "type": "nested",
                    "properties": {
                        "text": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 768,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "position": {"type": "integer"}
                    }
                },
                
                # Metadata
                "source": {"type": "keyword"},
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"}
            }
        }
    }
    
    # Create the index
    try:
        es.indices.create(index=index_name, body=index_settings)
        print(f"✅ Successfully created index '{index_name}' with hybrid search mappings!")
        
        # Verify index creation
        info = es.indices.get(index=index_name)
        print(f"📊 Index info: {len(info[index_name]['mappings']['properties'])} fields configured")
        
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_index()
    print("\n🎉 Elasticsearch initialization complete!")
    print("You can now start your FastAPI application.")