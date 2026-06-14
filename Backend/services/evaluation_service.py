"""
Evaluation Service - CORRECTED
Fixed evaluation metrics, scoring, and comprehensive testing
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import random

logger = logging.getLogger(__name__)

class EvaluationService:
    """CORRECTED: Comprehensive RAG evaluation service"""

    def __init__(self, rag_service):
        self.rag_service = rag_service
        
        # Evaluation test queries for different biomedical domains
        self.biomedical_test_queries = [
            # Mechanism queries
            "What is the mechanism of action of metformin in diabetes?",
            "How does CRISPR-Cas9 gene editing work?",
            "What is the pathophysiology of Alzheimer's disease?",
            
            # Treatment queries
            "What are the current treatment options for lung cancer?",
            "How effective is immunotherapy for melanoma?",
            "What are the side effects of ACE inhibitors?",
            
            # Diagnostic queries
            "What are the diagnostic criteria for rheumatoid arthritis?",
            "How is COVID-19 diagnosed in asymptomatic patients?",
            "What biomarkers are used for early Alzheimer's detection?",
            
            # Research queries
            "What are the latest developments in CAR-T cell therapy?",
            "What does recent research say about the gut microbiome and mental health?",
            "What are the current clinical trials for Parkinson's disease?",
            
            # Comparative queries
            "What is the difference between Type 1 and Type 2 diabetes?",
            "How do mRNA vaccines compare to traditional vaccines?",
            "What are the pros and cons of robotic vs. laparoscopic surgery?"
        ]

    async def run_comprehensive_evaluation(self, test_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive evaluation with multiple metrics"""
        start_time = datetime.utcnow()
        
        # Use provided queries or default biomedical queries
        queries = test_queries or self.biomedical_test_queries
        
        logger.info(f"🔍 Starting comprehensive evaluation with {len(queries)} queries")
        
        # Run evaluation for each query
        evaluation_results = []
        for i, query in enumerate(queries):
            logger.info(f"📋 Evaluating query {i+1}/{len(queries)}: {query[:50]}...")
            
            try:
                result = await self.rag_service.process_query(query, max_results=10)
                
                # Calculate metrics for this query
                query_metrics = self._evaluate_single_response(query, result)
                query_metrics["query"] = query
                query_metrics["query_index"] = i + 1
                
                evaluation_results.append(query_metrics)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Query {i+1} failed: {e}")
                evaluation_results.append({
                    "query": query,
                    "query_index": i + 1,
                    "error": str(e),
                    "success": False
                })

        # Calculate aggregate metrics
        evaluation_summary = self._calculate_aggregate_metrics(evaluation_results)
        
        # Add metadata
        evaluation_summary["evaluation_metadata"] = {
            "start_time": start_time.isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "total_evaluation_time": (datetime.utcnow() - start_time).total_seconds(),
            "num_queries_evaluated": len(queries),
            "num_successful": len([r for r in evaluation_results if r.get("success", True)]),
            "num_failed": len([r for r in evaluation_results if not r.get("success", True)]),
            "evaluation_version": "2.0"
        }
        
        # Add individual query results
        evaluation_summary["individual_results"] = evaluation_results
        
        # Save evaluation results
        await self._save_evaluation_results(evaluation_summary)
        
        total_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"✅ Evaluation completed in {total_time:.2f}s")
        
        return evaluation_summary

    def _evaluate_single_response(self, query: str, response: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single query-response pair"""
        metrics = {"success": True}
        
        # Extract response components
        answer = response.get("answer", "")
        citations = response.get("citations", [])
        confidence_score = response.get("confidence_score", 0.0)
        processing_time = response.get("processing_time", 0.0)
        
        # 1. Answer Quality Metrics
        metrics["answer_length"] = len(answer.split())
        metrics["answer_char_length"] = len(answer)
        metrics["answer_sentence_count"] = len([s for s in answer.split('.') if s.strip()])
        
        # 2. Citation Metrics
        metrics["citation_count"] = len(citations)
        metrics["citations_per_100_words"] = (len(citations) / max(len(answer.split()), 1)) * 100
        
        # Validate citations
        valid_citations = 0
        high_quality_citations = 0
        recent_citations = 0
        
        for citation in citations:
            if citation.get("pmid") or citation.get("doi"):
                valid_citations += 1
            
            if citation.get("source_quality") == "high":
                high_quality_citations += 1
                
            year = citation.get("year")
            if year and str(year).isdigit() and int(year) >= 2020:
                recent_citations += 1
        
        metrics["valid_citations"] = valid_citations
        metrics["citation_validity_rate"] = valid_citations / max(len(citations), 1)
        metrics["high_quality_citation_rate"] = high_quality_citations / max(len(citations), 1)
        metrics["recent_citation_rate"] = recent_citations / max(len(citations), 1)
        
        # 3. Confidence and Performance Metrics
        metrics["confidence_score"] = confidence_score
        metrics["processing_time"] = processing_time
        metrics["search_results_count"] = response.get("search_results_count", 0)
        
        # 4. Content Analysis
        metrics["biomedical_term_coverage"] = self._calculate_biomedical_coverage(answer)
        metrics["answer_completeness"] = self._assess_answer_completeness(query, answer)
        metrics["answer_relevance"] = self._assess_answer_relevance(query, answer)
        
        # 5. Query Analysis Integration
        query_analysis = response.get("query_analysis", {})
        metrics["query_complexity"] = query_analysis.get("complexity_score", 0.0)
        metrics["query_intent"] = query_analysis.get("intent", "unknown")
        
        # 6. Overall Score
        metrics["overall_score"] = self._calculate_overall_score(metrics)
        
        return metrics

    def _calculate_biomedical_coverage(self, answer: str) -> float:
        """Calculate biomedical term coverage in answer"""
        biomedical_terms = [
            'protein', 'gene', 'dna', 'rna', 'enzyme', 'hormone', 'pathway',
            'mechanism', 'therapy', 'treatment', 'diagnosis', 'clinical',
            'patient', 'study', 'trial', 'research', 'evidence', 'efficacy',
            'biomarker', 'prognosis', 'etiology', 'pathogenesis', 'therapeutic'
        ]
        
        answer_lower = answer.lower()
        found_terms = sum(1 for term in biomedical_terms if term in answer_lower)
        
        return min(1.0, found_terms / 10)  # Normalize to 0-1 scale

    def _assess_answer_completeness(self, query: str, answer: str) -> float:
        """Assess how complete the answer is relative to the query"""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common stop words
        stop_words = {'what', 'how', 'why', 'when', 'where', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but'}
        query_content_words = query_words - stop_words
        
        if not query_content_words:
            return 0.5
        
        # Calculate overlap
        overlap = len(query_content_words.intersection(answer_words))
        coverage = overlap / len(query_content_words)
        
        # Bonus for longer, more detailed answers
        length_bonus = min(0.3, len(answer.split()) / 200)
        
        return min(1.0, coverage + length_bonus)

    def _assess_answer_relevance(self, query: str, answer: str) -> float:
        """Assess answer relevance to the query"""
        # Simple keyword-based relevance
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # Extract key medical/scientific terms from query
        key_terms = []
        if 'mechanism' in query_lower:
            key_terms.extend(['mechanism', 'pathway', 'process', 'function'])
        if 'treatment' in query_lower or 'therapy' in query_lower:
            key_terms.extend(['treatment', 'therapy', 'drug', 'medication'])
        if 'diagnosis' in query_lower:
            key_terms.extend(['diagnosis', 'diagnostic', 'symptom', 'sign'])
        
        # Count relevant terms in answer
        relevance_score = 0.5  # Base relevance
        for term in key_terms:
            if term in answer_lower:
                relevance_score += 0.1
        
        return min(1.0, relevance_score)

    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score for the response"""
        components = []
        
        # Answer quality (30%)
        length_score = min(1.0, metrics.get("answer_length", 0) / 150)  # Target ~150 words
        components.append(("length", length_score, 0.15))
        
        completeness_score = metrics.get("answer_completeness", 0)
        components.append(("completeness", completeness_score, 0.15))
        
        # Citation quality (25%)
        citation_score = min(1.0, metrics.get("citation_count", 0) / 5)  # Target ~5 citations
        components.append(("citations", citation_score, 0.15))
        
        citation_validity = metrics.get("citation_validity_rate", 0)
        components.append(("citation_validity", citation_validity, 0.10))
        
        # Relevance and accuracy (25%)
        relevance_score = metrics.get("answer_relevance", 0)
        components.append(("relevance", relevance_score, 0.15))
        
        biomedical_coverage = metrics.get("biomedical_term_coverage", 0)
        components.append(("biomedical_coverage", biomedical_coverage, 0.10))
        
        # System performance (20%)
        confidence_score = metrics.get("confidence_score", 0)
        components.append(("confidence", confidence_score, 0.10))
        
        # Processing efficiency (lower is better, so invert)
        processing_time = metrics.get("processing_time", 10)
        efficiency_score = max(0, 1 - (processing_time / 30))  # 30s is poor performance
        components.append(("efficiency", efficiency_score, 0.10))
        
        # Calculate weighted score
        total_score = sum(score * weight for _, score, weight in components)
        
        return min(1.0, total_score)

    def _calculate_aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all queries"""
        successful_results = [r for r in results if r.get("success", True)]
        
        if not successful_results:
            return {
                "error": "No successful evaluations",
                "citation_accuracy": {"citation_accuracy": 0.0},
                "answer_relevance": {"composite_relevance": 0.0},
                "biomedical_f1": {"composite_biomedical_score": 0.0},
                "comprehensive_score": {"comprehensive_score": 0.0}
            }
        
        # Citation Accuracy Metrics
        citation_metrics = {
            "total_citations": sum(r.get("citation_count", 0) for r in successful_results),
            "average_citations_per_answer": sum(r.get("citation_count", 0) for r in successful_results) / len(successful_results),
            "citation_validity_rate": sum(r.get("citation_validity_rate", 0) for r in successful_results) / len(successful_results),
            "high_quality_citation_rate": sum(r.get("high_quality_citation_rate", 0) for r in successful_results) / len(successful_results),
            "recent_citation_rate": sum(r.get("recent_citation_rate", 0) for r in successful_results) / len(successful_results)
        }
        
        citation_accuracy = (
            citation_metrics["citation_validity_rate"] * 0.4 +
            citation_metrics["high_quality_citation_rate"] * 0.3 +
            citation_metrics["recent_citation_rate"] * 0.3
        )
        citation_metrics["citation_accuracy"] = citation_accuracy
        
        # Answer Relevance Metrics
        relevance_metrics = {
            "average_answer_length": sum(r.get("answer_length", 0) for r in successful_results) / len(successful_results),
            "average_completeness": sum(r.get("answer_completeness", 0) for r in successful_results) / len(successful_results),
            "average_relevance": sum(r.get("answer_relevance", 0) for r in successful_results) / len(successful_results),
            "biomedical_coverage": sum(r.get("biomedical_term_coverage", 0) for r in successful_results) / len(successful_results)
        }
        
        composite_relevance = (
            relevance_metrics["average_completeness"] * 0.4 +
            relevance_metrics["average_relevance"] * 0.4 +
            relevance_metrics["biomedical_coverage"] * 0.2
        )
        relevance_metrics["composite_relevance"] = composite_relevance
        
        # Biomedical F1-like Score
        precision = sum(r.get("biomedical_term_coverage", 0) for r in successful_results) / len(successful_results)
        recall = sum(r.get("answer_completeness", 0) for r in successful_results) / len(successful_results)
        
        biomedical_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        biomedical_metrics = {
            "biomedical_precision": precision,
            "biomedical_recall": recall,
            "biomedical_f1": biomedical_f1,
            "composite_biomedical_score": biomedical_f1
        }
        
        # Comprehensive Score
        comprehensive_score = sum(r.get("overall_score", 0) for r in successful_results) / len(successful_results)
        
        # Performance Metrics
        performance_metrics = {
            "average_processing_time": sum(r.get("processing_time", 0) for r in successful_results) / len(successful_results),
            "average_confidence": sum(r.get("confidence_score", 0) for r in successful_results) / len(successful_results),
            "success_rate": len(successful_results) / len(results)
        }
        
        return {
            "citation_accuracy": citation_metrics,
            "answer_relevance": relevance_metrics,
            "biomedical_f1": biomedical_metrics,
            "comprehensive_score": {"comprehensive_score": comprehensive_score},
            "performance_metrics": performance_metrics,
            "benchmark_comparison": self._generate_benchmark_comparison(comprehensive_score, citation_accuracy, composite_relevance)
        }

    def _generate_benchmark_comparison(self, comprehensive_score: float, citation_accuracy: float, relevance_score: float) -> Dict[str, Any]:
        """Generate benchmark comparison data"""
        # Simulated benchmark data (in real implementation, these would be actual benchmarks)
        benchmarks = {
            "baseline_rag": {"comprehensive": 0.65, "citation": 0.70, "relevance": 0.68},
            "pubmed_baseline": {"comprehensive": 0.58, "citation": 0.85, "relevance": 0.60},
            "biomedical_llm": {"comprehensive": 0.72, "citation": 0.75, "relevance": 0.74}
        }
        
        current_scores = {
            "comprehensive": comprehensive_score,
            "citation": citation_accuracy, 
            "relevance": relevance_score
        }
        
        comparisons = {}
        for benchmark_name, benchmark_scores in benchmarks.items():
            improvements = {}
            for metric, current_score in current_scores.items():
                benchmark_score = benchmark_scores.get(metric, 0.5)
                improvement = ((current_score - benchmark_score) / benchmark_score) * 100 if benchmark_score > 0 else 0
                improvements[f"{metric}_improvement"] = improvement
            
            comparisons[benchmark_name] = {
                "scores": benchmark_scores,
                "improvements": improvements,
                "overall_improvement": sum(improvements.values()) / len(improvements)
            }
        
        return comparisons

    async def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to database"""
        try:
            from Backend.services.auth_service import get_auth_service
            
            auth_service = get_auth_service()
            await auth_service.save_evaluation_results(
                evaluation_type="comprehensive_rag_evaluation",
                results=results
            )
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")

    async def quick_evaluation(self, num_queries: int = 5) -> Dict[str, Any]:
        """Run a quick evaluation with a subset of queries"""
        sample_queries = random.sample(self.biomedical_test_queries, min(num_queries, len(self.biomedical_test_queries)))
        
        return await self.run_comprehensive_evaluation(sample_queries)

    def get_evaluation_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable evaluation summary"""
        try:
            comprehensive_score = results.get("comprehensive_score", {}).get("comprehensive_score", 0)
            citation_accuracy = results.get("citation_accuracy", {}).get("citation_accuracy", 0)
            relevance_score = results.get("answer_relevance", {}).get("composite_relevance", 0)
            
            metadata = results.get("evaluation_metadata", {})
            total_queries = metadata.get("num_queries_evaluated", 0)
            success_rate = metadata.get("num_successful", 0) / max(total_queries, 1)
            
            summary = f"""
📊 BioRAG Evaluation Summary
============================

🎯 Overall Performance:
• Comprehensive Score: {comprehensive_score:.1%} 
• Citation Accuracy: {citation_accuracy:.1%}
• Answer Relevance: {relevance_score:.1%}

📈 System Metrics:
• Queries Evaluated: {total_queries}
• Success Rate: {success_rate:.1%}
• Avg Processing Time: {results.get('performance_metrics', {}).get('average_processing_time', 0):.2f}s

🏆 Performance Grade: {self._get_performance_grade(comprehensive_score)}
"""
            
            return summary
            
        except Exception as e:
            return f"Error generating summary: {e}"

    def _get_performance_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 0.9:
            return "A+ (Excellent)"
        elif score >= 0.8:
            return "A (Very Good)"
        elif score >= 0.7:
            return "B (Good)"
        elif score >= 0.6:
            return "C (Satisfactory)"
        elif score >= 0.5:
            return "D (Needs Improvement)"
        else:
            return "F (Poor)"