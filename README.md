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

## Model Overview

**BioChatAI** is a parameter-efficient, domain-adapted biomedical language model fine-tuned from **Microsoft’s BioGPT** using **LoRA (Low-Rank Adaptation)**.  
It is designed for **biomedical question answering, literature synthesis, and research assistance**, while retaining the core biomedical knowledge of the base BioGPT model.

The model was trained on curated biomedical question–answer pairs derived from **PubMedQA**, enabling accurate, citation-aware responses with minimal computational cost.

---

## Model Details

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

## Intended Uses

### Direct Use

BioChatAI is suitable for:

- Biomedical question answering  
- Literature synthesis and summarization  
- Clinical and biomedical research assistance  
- Medical education support  
- Automated literature review  

### Downstream Use

The model can be integrated into:

- Research assistant applications  
- Medical chatbots (non-clinical)  
- Healthcare information systems  
- Academic research platforms  
- Clinical decision-support tools (with expert oversight)  

### Out-of-Scope Use

This model **must not** be used for:

- Direct clinical diagnosis or treatment decisions  
- Patient-facing medical advice without professional supervision  
- Emergency or life-critical medical situations  
- Replacing licensed healthcare professionals  

---

## Training Details

### Training Data

- **Dataset:** PubMedQA (curated subset)  
- **Size:** 50 high-quality biomedical QA pairs  
- **Source:** Peer-reviewed PubMed abstracts  
- **Domains:** Oncology, cardiology, neurology, immunology  
- **Data characteristics:** Context-rich abstracts with verified citations  

### Training Procedure

- **Fine-tuning approach:** LoRA (parameter-efficient fine-tuning)  
- **Target modules:** Query (Q) and Value (V) projection layers  

#### LoRA Configuration

- Rank (r): 8  
- Alpha: 16  
- Dropout: 0.1  
- Trainable parameters: ~1.2M  
- Percentage of total parameters trained: **0.53%**  

#### Training Hyperparameters

- Training regime: CPU-based  
- Learning rate: 2e-4  
- Batch size: 2  
- Gradient accumulation steps: 4  
- Epochs: 3  
- Optimizer: AdamW  
- Weight decay: 0.01  
- Warmup steps: 100  
- Max sequence length: 512  

#### Training Performance

- Training time: ~61 minutes (CPU)  
- Final training loss: 2.89  
- Loss reduction: 11% vs. base model  
- Checkpoint size: ~4.8 MB (LoRA adapters only)  
- Memory usage: ~8 GB RAM  

---

## Evaluation

### Evaluation Data

- **Dataset:** PubMedQA evaluation subset  
- **Size:** 10 held-out biomedical questions  
- **Coverage:** Multi-domain biomedical topics  

### Metrics

#### Language Generation Quality

- BLEU-1: 0.116  
- BLEU-2: 0.052  
- Perplexity: 11% improvement over base BioGPT  

#### Biomedical Performance (BioASQ-inspired)

- F1 Score: 0.82  
- Citation Accuracy: 94%  
- Semantic Similarity: 0.87 (BioBERT-based)  

#### System Performance

- Average response time: ~30 seconds  
- Context relevance score: 0.85  

### Results Summary

- 11% reduction in training loss  
- High factual consistency with citations  
- Strong domain-specific language generation  
- Efficient adaptation with minimal trainable parameters  

---

## Bias, Risks, and Limitations

### Known Limitations

- Fine-tuned on a small dataset (50 QA pairs)  
- English-only biomedical content  
- Reflects publication bias present in biomedical literature  
- Training data current only up to 2024  
- Not validated for real-world clinical deployment  

### Bias Considerations

- Potential demographic and geographic bias toward Western biomedical research  
- Limited representation of rare diseases  

### Recommendations

Users should:

- Verify outputs against original research papers  
- Use the model strictly as a research assistant  
- Cross-check with updated clinical guidelines  
- Consult healthcare professionals for medical decisions  

---

## How to Use the Model

```python
from transformers import BioGptTokenizer, BioGptForCausalLM
from peft import PeftModel

# Load base model and tokenizer
base_model_name = "microsoft/biogpt"
tokenizer = BioGptTokenizer.from_pretrained(base_model_name)
base_model = BioGptForCausalLM.from_pretrained(base_model_name)

# Load LoRA adapter
peft_model_id = "your-username/biochatai-biogpt-lora"
model = PeftModel.from_pretrained(base_model, peft_model_id)

# Generate a response
query = "What is the mechanism of action of mRNA vaccines?"
inputs = tokenizer(query, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)

```

## Technical Specifications
### Architecture

Base architecture: BioGPT (GPT-2 variant)

Total parameters: 234M

Transformer layers: 16

Attention heads: 16

Hidden size: 1024

Vocabulary size: 42,384 (biomedical-specific)

### Compute Infrastructure
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
