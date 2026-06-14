from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

# Load the environment file
load_dotenv("enhanced.env")

# Fetch values
es_host = os.environ.get("ES_HOST")
es_user = os.environ.get("ES_USERNAME")
es_pass = os.environ.get("ES_PASSWORD")

# Quick sanity check
if not all([es_host, es_user, es_pass]):
    raise RuntimeError("Missing one of ES_HOST, ES_USERNAME, or ES_PASSWORD in enhanced.env")

print("Connecting to Elasticsearch at:", es_host)

# Initialize Elasticsearch client
es = Elasticsearch(
    hosts=[es_host],
    basic_auth=(es_user, es_pass),
    request_timeout=30,
    retry_on_timeout=True,
    max_retries=3
)

# Test connection
try:
    info = es.info()
    print("Elasticsearch cluster info:", info)
except Exception as e:
    print("Failed to connect to Elasticsearch:", e)
