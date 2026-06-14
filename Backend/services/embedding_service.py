"""
Embedding Service - BioBERT Integration
Local biomedical text embeddings with caching
"""

import torch
import numpy as np
import hashlib
import json
import os
import logging
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModel
import pickle
from datetime import datetime, timedelta
import asyncio
import threading

logger = logging.getLogger(__name__)

class EmbeddingService:
    """BioBERT embedding service with local caching"""

    def __init__(self, model_name: str = "dmis-lab/biobert-v1.1", 
                 cache_dir: str = "embeddings_cache",
                 max_cache_size: int = 10000):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.model_loaded = False

        # Caching
        self.memory_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Thread safety
        self.lock = threading.Lock()

        # Initialize
        self._setup_cache_dir()
        self._load_model()

    def _setup_cache_dir(self):
        """Create cache directory structure"""
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "chunks"), exist_ok=True)

    def _load_model(self):
        """Load BioBERT model and tokenizer"""
        try:
            logger.info(f"Loading BioBERT model: {self.model_name}")
            logger.info(f"Using device: {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            self.model_loaded = True
            logger.info("✅ BioBERT model loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load BioBERT model: {e}")
            logger.info("Note: BioBERT download requires internet connection")
            self.model_loaded = False

    def is_ready(self) -> bool:
        """Check if embedding service is ready"""
        return self.model_loaded

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """Get file path for cached embedding"""
        return os.path.join(self.cache_dir, "chunks", f"{cache_key}.pkl")

    def _load_from_disk_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from disk cache"""
        cache_path = self._get_cache_path(cache_key)

        if os.path.exists(cache_path):
            try:
                # Check if cache is still valid (30 days)
                mod_time = os.path.getmtime(cache_path)
                if datetime.now().timestamp() - mod_time > 30 * 24 * 3600:
                    os.remove(cache_path)
                    return None

                with open(cache_path, 'rb') as f:
                    embedding = pickle.load(f)
                    return embedding

            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
                return None

        return None

    def _save_to_disk_cache(self, cache_key: str, embedding: np.ndarray):
        """Save embedding to disk cache"""
        try:
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding to cache: {e}")

    async def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for single text"""
        if not self.model_loaded:
            logger.error("BioBERT model not loaded")
            return None

        if not text or not text.strip():
            return None

        # Clean and preprocess text
        cleaned_text = self._preprocess_biomedical_text(text)
        cache_key = self._get_cache_key(cleaned_text)

        # Check memory cache first
        with self.lock:
            if cache_key in self.memory_cache:
                self.cache_hits += 1
                return self.memory_cache[cache_key]

        # Check disk cache
        embedding = self._load_from_disk_cache(cache_key)
        if embedding is not None:
            with self.lock:
                self.memory_cache[cache_key] = embedding
                self.cache_hits += 1
                # Manage memory cache size
                if len(self.memory_cache) > self.max_cache_size:
                    self._clean_memory_cache()
            return embedding

        # Generate new embedding
        try:
            embedding = await self._generate_embedding(cleaned_text)

            if embedding is not None:
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)

                # Cache the embedding
                with self.lock:
                    self.memory_cache[cache_key] = embedding
                    self.cache_misses += 1
                    if len(self.memory_cache) > self.max_cache_size:
                        self._clean_memory_cache()

                self._save_to_disk_cache(cache_key, embedding)

            return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    def _preprocess_biomedical_text(self, text: str) -> str:
        """Preprocess text for biomedical embeddings"""
        # Expand common biomedical abbreviations
        abbreviations = {
            'HTN': 'hypertension',
            'MI': 'myocardial infarction',
            'COPD': 'chronic obstructive pulmonary disease',
            'CHF': 'congestive heart failure',
            'CAD': 'coronary artery disease',
            'DM': 'diabetes mellitus',
            'CKD': 'chronic kidney disease',
            'mRNA': 'messenger RNA',
            'DNA': 'deoxyribonucleic acid',
            'RNA': 'ribonucleic acid',
            'HIV': 'human immunodeficiency virus',
            'COVID-19': 'coronavirus disease 2019',
            'SARS-CoV-2': 'severe acute respiratory syndrome coronavirus 2'
        }

        # Replace abbreviations
        words = text.split()
        for i, word in enumerate(words):
            clean_word = word.strip('.,;:()[]{}')
            if clean_word in abbreviations:
                words[i] = word.replace(clean_word, abbreviations[clean_word])

        return ' '.join(words)

    async def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding using BioBERT"""
        try:
            # Tokenize with truncation for long texts
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                return embedding.squeeze()

        except Exception as e:
            logger.error(f"BioBERT inference error: {e}")
            return None

    async def embed_batch(self, texts: List[str], batch_size: int = 16) -> List[Optional[np.ndarray]]:
        """Generate embeddings for batch of texts"""
        if not self.model_loaded:
            logger.error("BioBERT model not loaded")
            return [None] * len(texts)

        embeddings = []

        # Process in batches to manage memory
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []

            for text in batch_texts:
                embedding = await self.embed_text(text)
                batch_embeddings.append(embedding)

            embeddings.extend(batch_embeddings)

            # Allow other tasks to run
            await asyncio.sleep(0.01)

        logger.info(f"Generated embeddings for {len(texts)} texts")
        return embeddings

    def _clean_memory_cache(self):
        """Clean memory cache when it gets too large"""
        if len(self.memory_cache) <= self.max_cache_size * 0.8:
            return

        # Remove oldest entries (simple FIFO)
        items_to_remove = len(self.memory_cache) - int(self.max_cache_size * 0.7)
        keys_to_remove = list(self.memory_cache.keys())[:items_to_remove]

        for key in keys_to_remove:
            del self.memory_cache[key]

        logger.info(f"Cleaned memory cache, removed {items_to_remove} entries")

    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate statistics"""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return self.cache_hits / total_requests

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            "memory_cache_size": len(self.memory_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.get_cache_hit_rate(),
            "disk_cache_files": len([f for f in os.listdir(os.path.join(self.cache_dir, "chunks")) if f.endswith('.pkl')])
        }

    def chunk_document(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Split document into overlapping chunks for better embedding"""
        if not text or len(text.split()) <= chunk_size:
            return [text]

        words = text.split()
        chunks = []

        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))

            if end >= len(words):
                break

            start += chunk_size - overlap

        return chunks

    async def embed_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Embed document with chunking strategy"""
        chunks_data = []

        # Combine title and abstract for better context
        title = document.get("title", "")
        abstract = document.get("abstract", "")
        full_text = document.get("full_text", "")

        # Create text content with sections
        sections = []
        if title:
            sections.append(f"Title: {title}")
        if abstract:
            sections.append(f"Abstract: {abstract}")
        if full_text:
            sections.append(f"Full Text: {full_text}")

        full_content = " ".join(sections)

        # Chunk the content
        chunks = self.chunk_document(full_content, chunk_size=400, overlap=80)

        # Generate embeddings for each chunk
        embeddings = await self.embed_batch(chunks, batch_size=8)

        # Create chunk documents
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding is not None:
                chunk_doc = {
                    "chunk_id": f"{document.get('pmid', 'unknown')}_{i}",
                    "text": chunk_text,
                    "embedding": embedding.tolist(),  # Convert to list for JSON serialization
                    "metadata": {
                        "pmid": document.get("pmid"),
                        "title": document.get("title"),
                        "journal": document.get("journal"),
                        "year": document.get("year"),
                        "doi": document.get("doi"),
                        "pmcid": document.get("pmcid"),
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                }
                chunks_data.append(chunk_doc)

        logger.info(f"Created {len(chunks_data)} embeddings for document {document.get('pmid')}")
        return chunks_data
