---
base_model: microsoft/biogpt
library_name: peft
pipeline_tag: text-generation
license: mit
language:
- en
tags:
- biomedical
- pubmed
- question-answering
- lora
- transformers
- peft
datasets:
- pubmed_qa
---

# BioChatAI – Fine-tuned BioGPT with LoRA for Biomedical Question Answering

## Overview

**BioChatAI** is a full-stack AI research assistant that answers biomedical questions using real, 
peer-reviewed scientific literature from **PubMed**. It does not hallucinate — every answer is 
grounded in fetched articles with numbered citations [1], [2], [3]...

<img width="1600" height="756" alt="image" src="https://github.com/user-attachments/assets/c0846a25-8ed1-4c4d-9d54-69b49672a11e" />
<img width="1600" height="756" alt="image" src="https://github.com/user-attachments/assets/ccd9d33f-1bf1-46dd-b2fb-fb69051e4328" />

---

## Details

- **Developed by:**  
  Shreenidhi G, Tharika N S, Thanushika Sri R A, Harithra S  
- **Supervised by:** Dr. Mithun Kumar  
- **Institution:** School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore, India  
- **Model type:** Autoregressive Language Model (GPT-based)  
- **Base model:** microsoft/biogpt (234M parameters)  
- **Fine-tuning method:** LoRA (Low-Rank Adaptation)  
- **Language:** English (Biomedical domain)  
- **License:** MIT  

---

## Workflow

![BioChatAI Workflow](./workflow(1).png)
<img width="1920" height="1080" alt="Workflow (1)" src="https://github.com/user-attachments/assets/4bbbdaca-36d6-47d9-96dd-2287a1d5e57e" />

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla HTML, CSS, JavaScript |
| Backend | FastAPI (Python) + Uvicorn |
| Authentication | JWT (HS256) + bcrypt (12 rounds) |
| Database | SQLite (`users.db`) |
| Embeddings | BioBERT (`dmis-lab/biobert-v1.1`) — 768-dim vectors |
| Dense Search | FAISS (Facebook AI Similarity Search) |
| Sparse Search | Elasticsearch (BM25) |
| Hybrid Fusion | Reciprocal Rank Fusion (RRF) |
| Answer Generation | Microsoft BioGPT (fine-tuned with LoRA) |
| Data Source | NCBI PubMed E-utilities API |
| ML Framework | HuggingFace Transformers + PyTorch |

---


## Key Mathematics

### BioBERT Embeddings
v ∈ ℝ⁷⁶⁸, L2-normalized: v̂ = v / ||v||₂
Query: q̂ ∈ ℝ⁷⁶⁸ Articles: D = { d̂₁ ... d̂ₙ } ∈ ℝⁿˣ⁷⁶⁸

### Cosine Similarity (FAISS)
cosine_similarity(q̂, d̂ᵢ) = q̂ · d̂ᵢ (dot product of normalized vectors)

### Hybrid Search — Reciprocal Rank Fusion
RRF(d) = 1/(60 + rank_dense) + 1/(60 + rank_sparse)
final_score = 0.6 × RRF + 0.4 × (0.6×dense + 0.4×sparse)

### Confidence Score
confidence = 0.25×search_quality + 0.15×count_factor + 0.20×length_factor
+ 0.15×sentence_factor + 0.20×citation_factor + 0.05×bioterm_density
Final: min(0.98, max(0.1, confidence))

---

## Quick Start
### Option 1: Docker (Recommended)
```bash
cd Backend
cp enhanced.env .env
docker-compose up --build
```
### Option 2: Local Development
```bash
# 1. Start Elasticsearch
docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0

# 2. Install Python dependencies
pip install -r Backend/requirements.txt

# 3. Run backend
python Backend/main.py

# 4. Open frontend/index.html in browser

```
---
## Evaluation and Performance
<img width="1600" height="949" alt="image" src="https://github.com/user-attachments/assets/ef39181d-4ab0-48cb-a302-a7898f961da3" />
---

## Compute Infrastructure
#### Hardware

CPU-based training

Minimum 16 GB RAM recommended

#### Software

PyTorch 2.0+

Transformers 4.30+

PEFT 0.17.1

Python 3.8+

##### Environmental Impact

Hardware type: Consumer-grade CPU

Training duration: ~1 hour

Estimated energy usage: ~0.5 kWh

Estimated carbon footprint: < 0.1 kg CO₂eq

LoRA fine-tuning reduced trainable parameters by 99.47%, significantly lowering computational and environmental costs.

## Citation

If you use this model in your research, please cite:
```bibtex
@misc{biochatai2024,
  title={BioChatAI: Parameter-Efficient Fine-tuning of BioGPT for Biomedical Question Answering},
  author={Shreenidhi G and Tharika N S and Thanushika Sri R A and Harithra S},
  year={2024},
  institution={School of Artificial Intelligence, Amrita Vishwa Vidyapeetham},
  supervisor={Dr. Mithun Kumar}
}
```
## Model Card Authors

Shreenidhi G

Tharika N S

Thanushika Sri R A

Harithra S

Supervisor: Dr. Mithun Kumar

## Framework Versions

PEFT 0.17.1

Transformers 4.30+

PyTorch 2.0+

Contact

For questions or issues, please open an issue on the repository
