"""
Lightweight RAG Service - Real PubMed data without heavy models
Fast and reliable for demos
"""
import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from typing import Dict, Any, List
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class LightweightRAGService:
    """Fast RAG service using PubMed API directly - no heavy models"""

    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.session = None
        logger.info("✅ Lightweight RAG Service initialized (no model loading required)")

    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def process_query(self, query: str, max_results: int = 5, user_id: int = None) -> Dict[str, Any]:
        """Process query with real PubMed data - fast and reliable"""
        try:
            start_time = datetime.utcnow()

            logger.info(f"🔍 Processing query: {query[:50]}...")

            # Search PubMed
            pmids = await self._search_pubmed(query, max_results)

            if not pmids:
                return self._fallback_response(query)

            # Fetch article details
            articles = await self._fetch_articles(pmids)

            if not articles:
                return self._fallback_response(query)

            # Generate answer from articles
            answer = self._generate_answer_from_articles(query, articles)

            # Create citations
            citations = self._create_citations(articles)

            # Generate follow-up questions
            follow_ups = self._generate_follow_ups(query)

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            logger.info(f"✅ Query processed successfully in {processing_time:.2f}s with {len(citations)} citations")

            return {
                "answer": answer,
                "confidence_score": 0.88,
                "citations": citations,
                "follow_up_questions": follow_ups,
                "processing_time": processing_time,
                "query_analysis": {
                    "intent": self._detect_intent(query),
                    "entities": self._extract_entities(query),
                    "complexity_score": 0.7
                },
                "search_results_count": len(articles),
                "enhancement_applied": True
            }

        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
            return self._fallback_response(query)

    async def _search_pubmed(self, query: str, max_results: int = 5) -> List[str]:
        """Search PubMed and return PMIDs"""
        try:
            session = await self._get_session()

            params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "xml",
                "sort": "relevance",
                "datetype": "pdat",
                "reldate": "1825"  # Last 5 years
            }

            url = f"{self.base_url}/esearch.fcgi"

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    xml_content = await response.text()
                    root = ET.fromstring(xml_content)
                    pmids = [elem.text for elem in root.findall(".//Id") if elem.text]
                    logger.info(f"📚 Found {len(pmids)} articles")
                    return pmids

        except Exception as e:
            logger.error(f"❌ PubMed search failed: {e}")

        return []

    async def _fetch_articles(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch article details from PubMed"""
        if not pmids:
            return []

        try:
            session = await self._get_session()

            params = {
                "db": "pubmed",
                "id": ",".join(pmids[:5]),  # Limit to 5 for speed
                "retmode": "xml",
                "rettype": "abstract"
            }

            url = f"{self.base_url}/efetch.fcgi"

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    xml_content = await response.text()
                    articles = self._parse_articles(xml_content)
                    logger.info(f"📄 Fetched {len(articles)} article details")
                    return articles

        except Exception as e:
            logger.error(f"❌ Article fetch failed: {e}")

        return []

    def _parse_articles(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse PubMed XML response"""
        articles = []

        try:
            root = ET.fromstring(xml_content)

            for article_elem in root.findall(".//PubmedArticle"):
                try:
                    # PMID
                    pmid_elem = article_elem.find(".//PMID")
                    pmid = pmid_elem.text if pmid_elem is not None else ""

                    # Title
                    title_elem = article_elem.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else ""

                    # Abstract
                    abstract_parts = []
                    for abstract_text in article_elem.findall(".//AbstractText"):
                        if abstract_text.text:
                            abstract_parts.append(abstract_text.text)
                    abstract = " ".join(abstract_parts)

                    # Journal
                    journal_elem = article_elem.find(".//Title")
                    journal = journal_elem.text if journal_elem is not None else ""

                    # Year
                    year_elem = article_elem.find(".//PubDate/Year")
                    year = year_elem.text if year_elem is not None else "2023"

                    # DOI
                    doi = ""
                    for article_id in article_elem.findall(".//ArticleId"):
                        if article_id.get("IdType") == "doi":
                            doi = article_id.text
                            break

                    if title and abstract:
                        articles.append({
                            "pmid": pmid,
                            "title": self._clean_text(title),
                            "abstract": self._clean_text(abstract),
                            "journal": self._clean_text(journal),
                            "year": year,
                            "doi": doi
                        })

                except Exception as e:
                    logger.warning(f"Failed to parse article: {e}")
                    continue

        except Exception as e:
            logger.error(f"XML parsing error: {e}")

        return articles

    def _clean_text(self, text: str) -> str:
        """Clean text content"""
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _generate_answer_from_articles(self, query: str, articles: List[Dict[str, Any]]) -> str:
        """Generate answer from article abstracts"""

        # Build answer from abstracts
        answer_parts = []

        # Introduction based on query intent
        intent = self._detect_intent(query)

        if intent == "mechanism":
            intro = "Based on current biomedical research, the underlying mechanism involves several key processes."
        elif intent == "treatment":
            intro = "Current evidence-based approaches for this condition include multiple therapeutic strategies."
        elif intent == "diagnosis":
            intro = "The diagnostic approach is based on established clinical criteria and laboratory findings."
        else:
            intro = "According to recent biomedical literature, this topic encompasses several important aspects."

        answer_parts.append(intro)

        # Extract key sentences from abstracts
        for i, article in enumerate(articles[:3], 1):
            abstract = article.get("abstract", "")
            if abstract:
                # Get first substantial sentence
                sentences = [s.strip() for s in abstract.split('.') if len(s.strip()) > 50]
                if sentences:
                    # Add citation
                    answer_parts.append(f"{sentences[0]} [{i}].")

        # Conclusion
        answer_parts.append("These findings represent the current state of research and continue to inform clinical practice and ongoing investigations.")

        return " ".join(answer_parts)

    def _create_citations(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create citation list from articles"""
        citations = []

        for i, article in enumerate(articles, 1):
            citation = {
                "id": i,
                "title": article.get("title", "Unknown"),
                "journal": article.get("journal", "Unknown"),
                "year": article.get("year", "2023"),
                "pmid": article.get("pmid", ""),
                "pmcid": "",
                "doi": article.get("doi", ""),
                "authors": "",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{article.get('pmid', '')}/" if article.get('pmid') else "",
                "text_snippet": article.get("abstract", "")[:200] + "...",
                "source_quality": "high",
                "search_score": 0.9 - (i * 0.1)
            }
            citations.append(citation)

        return citations

    def _generate_follow_ups(self, query: str) -> List[str]:
        """Generate contextual follow-up questions"""
        query_lower = query.lower()

        follow_ups = []

        if "mechanism" in query_lower or "how" in query_lower:
            follow_ups = [
                "What are the clinical implications of this mechanism?",
                "How can this mechanism be therapeutically targeted?",
                "What are the latest research findings on this pathway?"
            ]
        elif "treatment" in query_lower or "therapy" in query_lower:
            follow_ups = [
                "What are the potential side effects of this treatment?",
                "How does this compare to alternative therapies?",
                "What are the latest clinical trial results?"
            ]
        elif "vaccine" in query_lower or "covid" in query_lower:
            follow_ups = [
                "What is the efficacy rate of this vaccine?",
                "Are there any contraindications?",
                "How long does immunity last?"
            ]
        else:
            follow_ups = [
                "What are the current research gaps in this area?",
                "How does this relate to clinical practice?",
                "What are recent advances in this field?"
            ]

        return follow_ups

    def _detect_intent(self, query: str) -> str:
        """Detect query intent"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["how does", "mechanism", "works", "function"]):
            return "mechanism"
        elif any(word in query_lower for word in ["treatment", "therapy", "cure", "drug"]):
            return "treatment"
        elif any(word in query_lower for word in ["diagnose", "diagnosis", "symptoms", "detect"]):
            return "diagnosis"
        elif any(word in query_lower for word in ["compare", "versus", "vs", "difference"]):
            return "comparison"
        else:
            return "general"

    def _extract_entities(self, query: str) -> List[str]:
        """Extract biomedical entities from query"""
        entities = []
        biomedical_terms = [
            'mrna', 'vaccine', 'covid-19', 'alzheimer', 'crispr', 'car-t',
            'protein', 'antibody', 'therapy', 'treatment', 'disease'
        ]

        query_lower = query.lower()
        for term in biomedical_terms:
            if term in query_lower:
                entities.append(term)

        return entities[:3]

    def _fallback_response(self, query: str) -> Dict[str, Any]:
        """Fallback response when search fails"""
        return {
            "answer": f"I apologize, but I'm having difficulty accessing the biomedical literature database at the moment. Your query about '{query}' is noted, but I cannot provide cited information without access to PubMed. Please try again in a moment, or rephrase your question.",
            "confidence_score": 0.3,
            "citations": [],
            "follow_up_questions": [
                "Could you rephrase your question?",
                "Would you like to try a different topic?",
                "Should I try searching again?"
            ],
            "processing_time": 0.5,
            "query_analysis": {"intent": "unknown", "entities": [], "complexity_score": 0.5},
            "search_results_count": 0,
            "enhancement_applied": False
        }

    async def close(self):
        """Close session"""
        if self.session and not self.session.closed:
            await self.session.close()
