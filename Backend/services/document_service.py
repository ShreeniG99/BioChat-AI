"""
Document Service - PubMed/PMC Integration (CORRECTED)
Fixed async/await patterns, error handling, and rate limiting
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
import re
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import time
import random
from urllib.parse import quote

logger = logging.getLogger(__name__)

class DocumentService:
    """CORRECTED: PubMed/PMC document ingestion service using NCBI E-utilities"""

    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.rate_limit_delay = 0.34  # ~3 requests per second (NCBI guideline)
        self.last_request_time = 0
        self.session = None
        self.retry_attempts = 3
        self.timeout = 30

    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def _rate_limit(self):
        """Enforce NCBI rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    async def _make_request(self, url: str, params: Dict[str, Any]) -> str:
        """Make rate-limited request with retries"""
        await self._rate_limit()
        session = await self._get_session()
        
        for attempt in range(self.retry_attempts):
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 429:  # Rate limited
                        wait_time = 2 ** attempt + random.uniform(0, 1)
                        logger.warning(f"Rate limited, waiting {wait_time:.1f}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"HTTP {response.status} for {url}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Request failed: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                    
        raise Exception(f"Failed to fetch after {self.retry_attempts} attempts")

    async def search_pubmed(self, query: str, max_results: int = 50, 
                           date_filter: Optional[str] = None) -> List[str]:
        """Search PubMed for PMIDs using ESearch"""
        try:
            # Enhanced query with biomedical filters
            enhanced_query = self._enhance_biomedical_query(query)
            
            params = {
                "db": "pubmed",
                "term": enhanced_query,
                "retmax": max_results,
                "retmode": "xml",
                "sort": "relevance",
                "tool": "biorag",
                "email": "biorag@example.com"
            }
            
            # Add date filter if specified
            if date_filter:
                params["datetype"] = "pdat"
                params["reldate"] = date_filter
            else:
                # Default to last 5 years for fresher results
                params["datetype"] = "pdat"
                params["reldate"] = "1825"  # ~5 years in days
                
            url = f"{self.base_url}/esearch.fcgi"
            xml_content = await self._make_request(url, params)
            
            # Parse XML response
            root = ET.fromstring(xml_content)
            pmids = []
            for id_elem in root.findall(".//Id"):
                if id_elem.text:
                    pmids.append(id_elem.text)
                    
            logger.info(f"✅ Found {len(pmids)} PMIDs for query: {query}")
            return pmids
            
        except Exception as e:
            logger.error(f"❌ PubMed search error: {e}")
            return []

    def _enhance_biomedical_query(self, query: str) -> str:
        """Enhanced query with biomedical search strategies"""
        # Medical abbreviation expansion
        abbreviations = {
            'HTN': 'hypertension OR HTN',
            'MI': 'myocardial infarction OR MI OR heart attack',
            'CAR-T': 'chimeric antigen receptor T-cell OR CAR-T OR CAR T',
            'COPD': 'chronic obstructive pulmonary disease OR COPD',
            'CHF': 'congestive heart failure OR CHF OR heart failure',
            'CAD': 'coronary artery disease OR CAD',
            'DM': 'diabetes mellitus OR diabetes OR DM',
            'CKD': 'chronic kidney disease OR CKD',
            'CRISPR': 'CRISPR-Cas OR CRISPR OR gene editing',
            'mRNA': 'messenger RNA OR mRNA OR RNA vaccine'
        }
        
        query_upper = query.upper()
        for abbrev, expansion in abbreviations.items():
            if abbrev in query_upper:
                query = query.replace(abbrev, f"({expansion})")
        
        # Add publication type filters for high-quality sources
        quality_filters = [
            'AND (Clinical Trial[ptyp] OR Randomized Controlled Trial[ptyp] OR Review[ptyp] OR Meta-Analysis[ptyp] OR Systematic Review[ptyp])',
            'AND (humans[MeSH Terms] OR clinical[All Fields])'
        ]
        
        # Don't add filters to every query to maintain diversity
        if len(query.split()) > 2 and random.random() < 0.7:
            query += random.choice(quality_filters)
            
        return query

    async def fetch_abstracts(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch abstracts for given PMIDs using EFetch"""
        if not pmids:
            return []
            
        try:
            # Process in batches of 50 (NCBI recommendation)
            batch_size = 50
            all_documents = []
            
            for i in range(0, len(pmids), batch_size):
                batch_pmids = pmids[i:i + batch_size]
                batch_docs = await self._fetch_abstract_batch(batch_pmids)
                all_documents.extend(batch_docs)
                
            logger.info(f"✅ Fetched {len(all_documents)} abstracts")
            return all_documents
            
        except Exception as e:
            logger.error(f"❌ Abstract fetch error: {e}")
            return []

    async def _fetch_abstract_batch(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch a batch of abstracts"""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
            "tool": "biorag",
            "email": "biorag@example.com"
        }
        
        url = f"{self.base_url}/efetch.fcgi"
        xml_content = await self._make_request(url, params)
        return self._parse_pubmed_xml(xml_content)

    def _parse_pubmed_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse PubMed XML response"""
        documents = []
        try:
            root = ET.fromstring(xml_content)
            for article in root.findall(".//PubmedArticle"):
                try:
                    doc = self._extract_article_data(article)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to parse article: {e}")
                    continue
                    
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            
        return documents

    def _extract_article_data(self, article_elem) -> Optional[Dict[str, Any]]:
        """Extract structured data from PubMed article XML"""
        try:
            # PMID
            pmid_elem = article_elem.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None
            if not pmid:
                return None

            # Article details
            article_detail = article_elem.find(".//Article")
            if article_detail is None:
                return None

            # Title
            title_elem = article_detail.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""

            # Abstract
            abstract_parts = []
            for abstract_text in article_detail.findall(".//AbstractText"):
                text = abstract_text.text or ""
                label = abstract_text.get("Label", "")
                if label and text:
                    abstract_parts.append(f"{label}: {text}")
                elif text:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)

            # Authors
            authors = []
            for author in article_detail.findall(".//Author"):
                lastname = author.find(".//LastName")
                firstname = author.find(".//ForeName")
                if lastname is not None and firstname is not None:
                    authors.append(f"{firstname.text} {lastname.text}")

            # Journal
            journal_elem = article_detail.find(".//Title")
            journal = journal_elem.text if journal_elem is not None else ""

            # Publication date
            pub_date_elem = article_elem.find(".//PubDate")
            year = ""
            if pub_date_elem is not None:
                year_elem = pub_date_elem.find(".//Year")
                year = year_elem.text if year_elem is not None else ""

            # DOI
            doi = ""
            for article_id in article_elem.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = article_id.text
                    break

            # PMC ID
            pmcid = ""
            for article_id in article_elem.findall(".//ArticleId"):
                if article_id.get("IdType") == "pmc":
                    pmcid = article_id.text
                    break

            # Keywords/MeSH terms
            keywords = []
            for mesh in article_elem.findall(".//MeshHeading"):
                descriptor = mesh.find(".//DescriptorName")
                if descriptor is not None:
                    keywords.append(descriptor.text)

            # Create document
            document = {
                "pmid": pmid,
                "title": self._clean_text(title),
                "abstract": self._clean_text(abstract),
                "full_text": "",  # Will be populated from PMC if available
                "authors": authors[:10],  # Limit to first 10 authors
                "journal": self._clean_text(journal),
                "year": year,
                "doi": doi,
                "pmcid": pmcid,
                "keywords": keywords[:20],  # Limit keywords
                "source": "pubmed",
                "ingestion_date": datetime.utcnow().isoformat(),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }

            return document

        except Exception as e:
            logger.error(f"Error extracting article data: {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove XML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove extra punctuation
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()

    async def search_and_fetch(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Complete search and fetch pipeline"""
        try:
            # Search for PMIDs
            pmids = await self.search_pubmed(query, max_results)
            if not pmids:
                logger.warning(f"No PMIDs found for query: {query}")
                return []

            # Fetch abstracts
            documents = await self.fetch_abstracts(pmids)

            # Filter out low-quality documents
            quality_docs = []
            for doc in documents:
                if self._is_quality_document(doc):
                    quality_docs.append(doc)

            logger.info(f"✅ Retrieved {len(quality_docs)} quality documents for: {query}")
            return quality_docs

        except Exception as e:
            logger.error(f"❌ Search and fetch error: {e}")
            return []

    def _is_quality_document(self, doc: Dict[str, Any]) -> bool:
        """Filter for document quality"""
        # Must have title and abstract
        if not doc.get("title") or not doc.get("abstract"):
            return False

        # Abstract should be substantial
        if len(doc["abstract"]) < 100:
            return False

        # Should have recent publication date
        year = doc.get("year", "")
        if year and year.isdigit():
            if int(year) < 2015:  # Focus on recent research
                return False

        return True

    async def close(self):
        """Close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()