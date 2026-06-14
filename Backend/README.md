# BioRAG - Advanced Biomedical Literature Mining Backend

A production-ready FastAPI backend for biomedical literature mining using state-of-the-art RAG (Retrieval-Augmented Generation) technology.

## 🚀 Features

- **Real PubMed Integration**: Live document ingestion from PubMed/PMC using NCBI E-utilities
- **Hybrid Search**: Combines FAISS dense retrieval with Elasticsearch BM25 sparse retrieval
- **BioBERT Embeddings**: Specialized biomedical embeddings with comprehensive caching
- **Local Generation**: Microsoft BioGPT for answer generation (no external API costs)
- **BioASQ Evaluation**: Production-grade evaluation following biomedical QA standards
- **Production Ready**: Comprehensive error handling, logging, authentication, and monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │────│   FastAPI        │────│   PubMed        │
│   (React)       │    │   Backend        │    │   E-utilities   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
            ┌───────▼────┐ ┌────▼────┐ ┌────▼──────┐
            │   FAISS    │ │  Elastic│ │  BioBERT  │
            │   Index    │ │ Search  │ │  Encoder  │
            └────────────┘ └─────────┘ └───────────┘
```

## ⚡ Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone and setup
cd backend
cp .env.example .env

# Start services
docker-compose up --build

# Check health
curl http://localhost:8000/health
```

### Option 2: Local Development

```bash
# Install Python 3.12.8 dependencies
pip install -r requirements.txt

# Start Elasticsearch (required)
docker run -d --name elasticsearch \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0

# Run backend
python main.py
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_health.py -v
pytest tests/test_auth.py -v
pytest tests/test_query.py -v
pytest tests/test_evaluation.py -v

# Run with coverage
pytest --cov=. --cov-report=html
```

## 📡 API Endpoints

### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/signup` - User registration

### Core Functionality
- `POST /api/query` - Main biomedical query endpoint
- `GET /api/history` - Chat history
- `GET /api/metrics` - System performance metrics

### Evaluation
- `POST /api/evaluate/compare` - BioASQ evaluation against baselines

### System
- `GET /health` - Health check

## 🔧 Configuration

Key environment variables:

```bash
# JWT Authentication
JWT_SECRET=your_secret_key
JWT_EXPIRATION_HOURS=24

# Elasticsearch
ES_HOST=localhost:9200
ES_INDEX=biomedical_docs

# Models
BIOBERT_MODEL=dmis-lab/biobert-v1.1
GENERATION_MODEL=microsoft/BioGPT

# Caching
EMBEDDING_CACHE_DIR=embeddings_cache
FAISS_INDEX_DIR=faiss_index
MAX_MEMORY_CACHE_SIZE=10000
```

## 🏥 Demo Credentials

For testing and demonstration:

- **Email**: user@example.com
- **Password**: password123

## 📊 Performance Metrics

The system tracks comprehensive metrics:

- **Response Time**: Average query processing time
- **Cache Hit Rate**: Embedding cache efficiency  
- **Document Coverage**: Total indexed documents
- **Citation Accuracy**: Grounding quality
- **BioASQ Scores**: Evaluation against biomedical QA standards

## 🔬 Biomedical Features

### Medical Abbreviation Expansion
- HTN → hypertension
- MI → myocardial infarction  
- CAR-T → chimeric antigen receptor T-cell
- COVID-19 → coronavirus disease 2019

### Intelligent Chunking
- Section-aware processing (abstract, methods, results)
- 400-token chunks with 80-token overlap
- Citation provenance tracking

### Quality Filtering
- Recent publications preferred (2015+)
- High-impact journal boosting
- Comprehensive error handling

## 🌐 Frontend Integration

CORS configured for `localhost:3000`. The backend provides exactly the response format expected by the frontend:

```json
{
  "answer": "Biomedical answer with citations [1][2]...",
  "confidence_score": 0.92,
  "citations": [
    {
      "id": 1,
      "title": "Research Title",
      "pmcid": "PMC1234567",
      "doi": "10.1038/example",
      "journal": "Nature Medicine",
      "year": 2024
    }
  ],
  "follow_up_questions": ["Question 1", "Question 2", "Question 3"],
  "processing_time": 3.24,
  "query_analysis": {
    "intent": "mechanism",
    "entities": ["CAR-T", "therapy"],
    "complexity_score": 0.8
  }
}
```

## 🚨 Error Handling

Comprehensive error handling with structured responses:

- **Authentication errors**: 401 with clear messages
- **Validation errors**: 422 with field-specific details  
- **Rate limiting**: 429 with retry information
- **Server errors**: 500 with sanitized error messages
- **Graceful degradation**: System continues with partial failures

## 📈 Scalability

Production considerations:

- **Horizontal scaling**: Stateless design enables load balancing
- **Database**: SQLite for demo, easily upgraded to PostgreSQL
- **Caching**: Multi-level caching (memory + disk + distributed)
- **Monitoring**: Built-in metrics and health checks
- **Resource management**: Configurable batch sizes and timeouts

## 🔒 Security

- JWT authentication with configurable expiration
- Input validation and sanitization
- SQL injection prevention
- Rate limiting per user
- CORS configuration
- Secure headers

## 🐛 Troubleshooting

### Common Issues

**BioBERT model not loading:**
```bash
# Ensure internet connection for initial download
# Check available disk space (~3GB required)
```

**Elasticsearch connection failed:**
```bash
# Check Elasticsearch is running
curl http://localhost:9200

# Check Docker container
docker ps | grep elasticsearch
```

**Slow response times:**
```bash
# Check system resources
# Reduce batch sizes in config
# Enable GPU if available
```

**CORS errors:**
```bash
# Verify frontend URL in CORS settings
# Check browser network tab for preflight requests
```

## 📝 Development

### Adding New Features

1. **New endpoints**: Add to `main.py` with proper authentication
2. **New services**: Create in `services/` directory  
3. **New evaluations**: Extend `evaluation_service.py`
4. **New text processing**: Add to `utils/helpers.py`

### Code Quality

- Type hints throughout
- Comprehensive logging
- Error handling patterns
- Test coverage >80%
- Documentation strings

## 📚 Documentation

- **API Documentation**: Available at `http://localhost:8000/docs` (FastAPI auto-generated)
- **Code Documentation**: Inline docstrings and comments
- **Architecture**: See `docs/` directory (if available)

## 🤝 Contributing

1. Follow existing code patterns
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass
5. Check code quality with linters

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- **BioBERT**: dmis-lab/biobert-v1.1
- **BioGPT**: Microsoft BioGPT
- **BioASQ**: Biomedical Question Answering standards
- **PubMed**: NCBI literature database
- **FastAPI**: Modern Python web framework

---

**Built with ❤️ for the biomedical research community**
