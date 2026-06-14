"""
BioChatAI System Evaluation
Evaluates system performance on 50 biomedical queries and generates metrics
Location: Backend/evaluation/evaluate_system.py
"""
import asyncio
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.lightweight_rag_service import LightweightRAGService

# Test queries covering diverse biomedical topics
TEST_QUERIES = [
    # Vaccines & Immunology
    "What is the mechanism of mRNA vaccines?",
    "How does CAR-T cell therapy work?",
    "What is herd immunity?",
    "How do monoclonal antibodies treat cancer?",
    "What is the difference between innate and adaptive immunity?",
    
    # Neuroscience & Brain Disorders
    "What causes Alzheimer's disease?",
    "How does Parkinson's disease affect dopamine neurons?",
    "What is the blood-brain barrier?",
    "How do SSRIs work for depression?",
    "What causes epileptic seizures?",
    
    # Genetics & Gene Editing
    "Explain CRISPR gene editing",
    "What is gene therapy?",
    "How does RNA interference work?",
    "What are the ethical concerns of genetic engineering?",
    "What is epigenetics?",
    
    # Cardiovascular System
    "How do ACE inhibitors work?",
    "What causes atherosclerosis?",
    "How does aspirin prevent heart attacks?",
    "What is the role of statins in cardiovascular health?",
    "What causes hypertension?",
    
    # Cancer Biology
    "What are the side effects of chemotherapy?",
    "How does radiation therapy kill cancer cells?",
    "What is tumor angiogenesis?",
    "How do checkpoint inhibitors work in immunotherapy?",
    "What is the difference between benign and malignant tumors?",
    
    # Molecular Biology
    "Describe the process of DNA replication",
    "What is the role of mitochondria in cells?",
    "How does protein synthesis occur?",
    "What is apoptosis?",
    "How do enzymes catalyze reactions?",
    
    # Endocrinology & Metabolism
    "How does insulin regulate blood sugar?",
    "What causes Type 2 diabetes?",
    "What is the function of the thyroid gland?",
    "How does cortisol affect the body?",
    "What is metabolic syndrome?",
    
    # Infectious Diseases
    "How do antibiotics work?",
    "What is antibiotic resistance?",
    "How does HIV affect the immune system?",
    "What is the difference between bacteria and viruses?",
    "How do vaccines provide immunity?",
    
    # Organ Systems
    "What is the function of the liver?",
    "How do kidneys filter blood?",
    "What is the role of the pancreas?",
    "How does the respiratory system exchange gases?",
    "What is the function of the spleen?",
    
    # Pharmacology
    "How do beta blockers work?",
    "What is the mechanism of NSAIDs?",
    "How does warfarin prevent blood clots?",
    "What are the side effects of corticosteroids?",
    "How do proton pump inhibitors reduce stomach acid?",
]

assert len(TEST_QUERIES) == 50, f"Expected 50 queries, got {len(TEST_QUERIES)}"


async def evaluate_queries(auto_mode=False):
    """
    Evaluate system on test queries
    
    Args:
        auto_mode: If True, automatically assigns quality based on confidence
                   If False, prompts user for manual rating
    """
    print("\n" + "="*70)
    print("BioChatAI System Evaluation")
    print("="*70 + "\n")
    
    service = LightweightRAGService()
    results = []
    
    print(f"📊 Evaluating {len(TEST_QUERIES)} biomedical queries...")
    print(f"⚙️  Mode: {'Automatic' if auto_mode else 'Manual'} quality assessment\n")
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] Processing: {query[:60]}...")
        
        try:
            result = await service.process_query(query, max_results=5)
            
            # Extract key metrics
            has_citations = len(result.get('citations', [])) > 0
            num_citations = len(result.get('citations', []))
            confidence = result.get('confidence_score', 0)
            answer = result.get('answer', '')
            processing_time = result.get('processing_time', 0)
            
            # Quality assessment
            if auto_mode:
                # Automatic quality based on confidence and citations
                if confidence >= 0.85 and num_citations >= 3:
                    quality = 'H'  # High
                elif confidence >= 0.70 and num_citations >= 2:
                    quality = 'M'  # Medium
                else:
                    quality = 'L'  # Low
                
                print(f"   ✓ Answer: {answer[:80]}...")
                print(f"   📚 Citations: {num_citations}")
                print(f"   🎯 Confidence: {confidence:.2f}")
                print(f"   ⭐ Quality: {quality}")
            else:
                # Manual quality assessment
                print(f"\n   📝 Answer: {answer[:200]}...")
                print(f"   📚 Citations: {num_citations}")
                print(f"   🎯 Confidence: {confidence:.2f}")
                
                quality = input("\n   Rate quality (H=High, M=Medium, L=Low): ").upper()
                while quality not in ['H', 'M', 'L']:
                    quality = input("   Please enter H, M, or L: ").upper()
            
            # Store result
            results.append({
                'query_id': i,
                'query': query,
                'answer': answer,
                'quality': quality,
                'has_citations': has_citations,
                'num_citations': num_citations,
                'confidence': confidence,
                'processing_time': processing_time,
                'search_results_count': result.get('search_results_count', 0)
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'query_id': i,
                'query': query,
                'quality': 'L',
                'has_citations': False,
                'num_citations': 0,
                'confidence': 0,
                'error': str(e)
            })
    
    # Close service session
    await service.close()
    
    return results


def save_results(results):
    """Save evaluation results to JSON file"""
    
    # Create output directory
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"evaluation_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    # Calculate and save summary statistics
    total = len(results)
    high_quality = sum(1 for r in results if r['quality'] == 'H')
    medium_quality = sum(1 for r in results if r['quality'] == 'M')
    low_quality = sum(1 for r in results if r['quality'] == 'L')
    
    with_citations = sum(1 for r in results if r.get('has_citations', False))
    avg_confidence = sum(r.get('confidence', 0) for r in results) / total
    avg_processing_time = sum(r.get('processing_time', 0) for r in results) / total
    avg_citations = sum(r.get('num_citations', 0) for r in results) / total
    
    summary = {
        'evaluation_date': datetime.now().isoformat(),
        'total_queries': total,
        'quality_distribution': {
            'high': high_quality,
            'medium': medium_quality,
            'low': low_quality
        },
        'quality_percentages': {
            'high': f"{(high_quality/total)*100:.1f}%",
            'medium': f"{(medium_quality/total)*100:.1f}%",
            'low': f"{(low_quality/total)*100:.1f}%"
        },
        'metrics': {
            'citation_rate': f"{(with_citations/total)*100:.1f}%",
            'avg_confidence': f"{avg_confidence:.3f}",
            'avg_processing_time': f"{avg_processing_time:.2f}s",
            'avg_citations_per_query': f"{avg_citations:.1f}"
        }
    }
    
    summary_file = output_dir / f"evaluation_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary saved to: {summary_file}")
    
    return summary, results_file


def print_summary(results):
    """Print evaluation summary"""
    
    total = len(results)
    high_quality = sum(1 for r in results if r['quality'] == 'H')
    medium_quality = sum(1 for r in results if r['quality'] == 'M')
    low_quality = sum(1 for r in results if r['quality'] == 'L')
    
    with_citations = sum(1 for r in results if r.get('has_citations', False))
    avg_confidence = sum(r.get('confidence', 0) for r in results) / total
    avg_processing_time = sum(r.get('processing_time', 0) for r in results) / total
    avg_citations = sum(r.get('num_citations', 0) for r in results) / total
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70 + "\n")
    
    print(f"📊 Total Queries: {total}")
    print(f"\n🎯 Quality Distribution:")
    print(f"   High Quality:   {high_quality:2d} ({(high_quality/total)*100:5.1f}%)")
    print(f"   Medium Quality: {medium_quality:2d} ({(medium_quality/total)*100:5.1f}%)")
    print(f"   Low Quality:    {low_quality:2d} ({(low_quality/total)*100:5.1f}%)")
    
    print(f"\n📚 Citation Metrics:")
    print(f"   Queries with citations: {with_citations}/{total} ({(with_citations/total)*100:.1f}%)")
    print(f"   Avg citations per query: {avg_citations:.1f}")
    
    print(f"\n⚡ Performance Metrics:")
    print(f"   Avg confidence score: {avg_confidence:.3f}")
    print(f"   Avg processing time: {avg_processing_time:.2f}s")
    
    accuracy = (high_quality + medium_quality) / total
    print(f"\n🎓 Overall Accuracy: {accuracy*100:.1f}% (High + Medium quality)")
    
    print("\n" + "="*70 + "\n")


async def main():
    """Main execution function"""
    
    print("\n🔬 BioChatAI Evaluation Script")
    print("📍 Location: Backend/evaluation/evaluate_system.py\n")
    
    # Ask user for mode
    mode = input("Choose evaluation mode:\n  [1] Automatic (fast, based on confidence)\n  [2] Manual (you rate each answer)\n\nEnter 1 or 2: ")
    
    auto_mode = (mode == '1')
    
    if not auto_mode:
        print("\n⚠️  Manual mode: You'll rate each of 50 answers.")
        print("    This will take ~15-20 minutes.")
        confirm = input("\nContinue? (y/n): ")
        if confirm.lower() != 'y':
            print("Evaluation cancelled.")
            return
    
    # Run evaluation
    results = await evaluate_queries(auto_mode=auto_mode)
    
    # Print summary
    print_summary(results)
    
    # Save results
    summary, results_file = save_results(results)
    
    print(f"✨ Evaluation complete! Results saved to 'figures/' directory.")
    print(f"\n📊 Next step: Run visualization script to generate plots:")
    print(f"   python Backend/visualization/create_plots.py --results {results_file}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()