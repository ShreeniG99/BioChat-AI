"""
Quick evaluation to generate real metrics
"""
import asyncio
from Backend.services.lightweight_rag_service import LightweightRAGService

async def evaluate_queries():
    service = LightweightRAGService()
    
    test_queries = [
        "What is the mechanism of mRNA vaccines?",
        "How does CAR-T therapy work?",
        "What causes Alzheimer's disease?",
        "Explain CRISPR gene editing",
        "How do ACE inhibitors work?",
        "What are the side effects of chemotherapy?",
        "Describe the process of DNA replication",
        "What is the role of mitochondria in cells?",
        "How does insulin regulate blood sugar?",
        "What is the function of the liver?"
        
    ]
    
    results = []
    for query in test_queries:
        result = await service.process_query(query)
        
        # Manual quality assessment (you judge each)
        quality = input(f"Rate answer quality (H/M/L): {result['answer'][:100]}... ")
        
        results.append({
            'query': query,
            'quality': quality,
            'has_citations': len(result['citations']) > 0,
            'confidence': result['confidence_score']
        })
    
    return results

# Run evaluation
results = asyncio.run(evaluate_queries())
# Save results and generate real confusion matrix