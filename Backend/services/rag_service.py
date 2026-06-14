"""
Enhanced RAG Service - Phase 1 Implementation
Improved answer generation, paragraph-length responses, citation validation
"""

import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime
import re
import random
from transformers import pipeline
import torch

logger = logging.getLogger(__name__)


class EnhancedRAGService:
    """Phase 1: Enhanced RAG with paragraph generation and citation validation"""

    def __init__(self, document_service, embedding_service, vector_search_service):
        self.document_service = document_service
        self.embedding_service = embedding_service
        self.vector_search_service = vector_search_service

        # Initialize generation model with enhanced settings
        self.generator = None
        self.generator_loaded = False
        self._load_generation_model()

        # Enhanced query analysis patterns
        self.intent_patterns = {
            'mechanism': [r'how does', r'mechanism', r'pathway', r'process', r'function', r'works'],
            'treatment': [r'treatment', r'therapy', r'drug', r'medication', r'cure', r'therapeutic'],
            'diagnosis': [r'diagnose', r'symptoms', r'signs', r'detect', r'screening'],
            'research': [r'study', r'research', r'trial', r'evidence', r'findings', r'clinical trial'],
            'comparison': [r'versus', r'vs', r'compare', r'difference', r'better', r'alternative'],
            'definition': [r'what is', r'define', r'meaning', r'explain', r'description'],
            'epidemiology': [r'prevalence', r'incidence', r'epidemiology', r'statistics', r'mortality'],
            'adverse_effects': [r'side effects', r'adverse', r'toxicity', r'contraindication', r'safety']
        }

        # Enhanced biomedical entity patterns
        self.entity_patterns = {
            'disease': [r'cancer', r'diabetes', r'hypertension', r'alzheimer', r'covid', r'disease', r'syndrome', r'disorder', r'carcinoma', r'sarcoma'],
            'drug': [r'drug', r'medication', r'therapy', r'treatment', r'compound', r'molecule', r'inhibitor', r'agonist', r'antagonist'],
            'gene': [r'gene', r'protein', r'enzyme', r'receptor', r'antibody', r'biomarker', r'mutation'],
            'procedure': [r'surgery', r'procedure', r'imaging', r'test', r'assay', r'biopsy', r'endoscopy'],
            'anatomy': [r'heart', r'brain', r'liver', r'kidney', r'lung', r'blood', r'cell', r'tissue']
        }

        # Citation validation patterns
        self.citation_patterns = [
            r'\[(\d+)\]',  # [1], [2], etc.
            r'\[(\d+)[,\s]*(\d+)\]',  # [1,2] or [1 2]
            r'\[(\d+)-(\d+)\]'  # [1-3]
        ]

    def _load_generation_model(self):
        """Load generation model with enhanced configuration"""
        try:
            logger.info("Loading BioGPT generation model with enhanced settings...")
            model_name = "microsoft/BioGPT"
            device = 0 if torch.cuda.is_available() else -1

            self.generator = pipeline(
                "text-generation",
                model=model_name,
                device=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                max_new_tokens=400,  # Increased for paragraph generation
                do_sample=True,
                temperature=0.5,  # Balanced creativity/factuality
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
                pad_token_id=50256
            )
            self.generator_loaded = True
            logger.info("✅ BioGPT model loaded successfully with enhanced settings")

        except Exception as e:
            logger.error(f"❌ Failed to load BioGPT, using fallback: {e}")
            try:
                self.generator = pipeline(
                    "text-generation",
                    model="distilgpt2",
                    device=-1,
                    max_new_tokens=350,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.08
                )
                self.generator_loaded = True
                logger.info("✅ Fallback model (DistilGPT2) loaded")
            except Exception as fallback_error:
                logger.error(f"❌ Fallback model also failed: {fallback_error}")
                self.generator_loaded = False

    async def process_query(self, query: str, max_results: int = 12,
                            include_full_text: bool = False, user_id: int = None) -> Dict[str, Any]:
        """Enhanced RAG pipeline with citation validation"""
        try:
            start_time = datetime.utcnow()

            # 1. Enhanced query analysis
            query_analysis = self._analyze_query(query)
            logger.info(f"Query analysis: {query_analysis}")

            # 2. Generate query embedding
            query_embedding = await self.embedding_service.embed_text(query)
            if query_embedding is None:
                return self._fallback_response(query, "Failed to generate query embedding")

            # 3. Enhanced hybrid retrieval
            search_results = await self.vector_search_service.search(
                query_embedding=query_embedding,
                query_text=query,
                k=max_results * 2
            )

            # 4. Smart document fetching based on result quality
            if len(search_results) < 8 or self._needs_more_evidence(search_results, query):
                logger.info("Fetching additional documents due to low result count or quality")
                fresh_chunks = await self._fetch_fresh_documents(query, max_results=25)
                if fresh_chunks:
                    await self.vector_search_service.add_documents(fresh_chunks)
                    # Re-search with new documents
                    search_results = await self.vector_search_service.search(
                        query_embedding=query_embedding,
                        query_text=query,
                        k=max_results * 2
                    )

            # 5. Enhanced context assembly with citation preparation
            context, citations = self._assemble_context(search_results, query, max_results)

            # 6. Generate comprehensive answer
            answer = await self._generate_comprehensive_answer(query, context, query_analysis, citations)

            # 7. Validate and enhance citations
            answer, validated_citations = self._validate_and_enhance_citations(answer, citations, search_results)

            # 8. Calculate enhanced confidence score
            confidence_score = self._calculate_enhanced_confidence(search_results, answer, validated_citations)

            # 9. Quality check and potential retry
            if confidence_score < 0.7 or len(answer.split()) < 100:
                logger.info("Low confidence or short answer detected, attempting enhancement")
                enhanced_answer = await self._enhance_answer(query, context, answer, query_analysis)
                if enhanced_answer and len(enhanced_answer) > len(answer):
                    answer = enhanced_answer
                    confidence_score = self._calculate_enhanced_confidence(search_results, answer, validated_citations)

            # 10. Generate contextual follow-up questions
            follow_up_questions = self._generate_contextual_follow_ups(query, answer, query_analysis, validated_citations)

            # 11. Save enhanced history
            if user_id:
                await self._save_query_history(user_id, query, answer, confidence_score)

            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"✅ Enhanced query processed successfully in {processing_time:.2f}s")

            return {
                "answer": answer,
                "confidence_score": confidence_score,
                "citations": validated_citations,
                "follow_up_questions": follow_up_questions,
                "processing_time": processing_time,
                "query_analysis": query_analysis,
                "search_results_count": len(search_results),
                "enhancement_applied": confidence_score >= 0.7 and len(answer.split()) >= 100
            }

        except Exception as e:
            logger.error(f"❌ Enhanced RAG processing failed: {e}")
            return self._fallback_response(query, str(e))

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Enhanced query analysis with biomedical focus"""
        query_lower = query.lower()
        
        # Detect intent
        intent = "general"
        intent_confidence = 0.0
        for intent_type, patterns in self.intent_patterns.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, query_lower))
            confidence = matches / len(patterns)
            if confidence > intent_confidence:
                intent = intent_type
                intent_confidence = confidence

        # Detect entities
        entities = []
        entity_scores = {}
        for entity_type, patterns in self.entity_patterns.items():
            matches = [pattern for pattern in patterns if re.search(pattern, query_lower)]
            if matches:
                entities.append(entity_type)
                entity_scores[entity_type] = len(matches) / len(patterns)

        # Calculate complexity
        complexity_factors = [
            len(query.split()) / 25.0,  # Length factor
            len(entities) / 10.0,  # Entity density
            intent_confidence,  # Intent clarity
            1.0 if any(word in query_lower for word in ['clinical', 'trial', 'study', 'research']) else 0.0
        ]
        complexity_score = min(1.0, sum(complexity_factors) / len(complexity_factors))

        # Detect question type
        question_type = "factual"
        if any(word in query_lower for word in ['how', 'why', 'mechanism']):
            question_type = "explanatory"
        elif any(word in query_lower for word in ['compare', 'versus', 'difference']):
            question_type = "comparative"
        elif any(word in query_lower for word in ['latest', 'recent', 'new', 'current']):
            question_type = "current_research"

        return {
            "intent": intent,
            "intent_confidence": intent_confidence,
            "entities": entities,
            "entity_scores": entity_scores,
            "complexity_score": complexity_score,
            "question_type": question_type,
            "query_length": len(query),
            "word_count": len(query.split()),
            "biomedical_terms": self._count_biomedical_terms(query_lower)
        }

    def _count_biomedical_terms(self, query_lower: str) -> int:
        """Count biomedical terms in query"""
        biomedical_terms = [
            'protein', 'gene', 'dna', 'rna', 'enzyme', 'hormone', 'neurotransmitter',
            'pathogenesis', 'etiology', 'prognosis', 'biomarker', 'therapeutic',
            'clinical', 'diagnosis', 'treatment', 'therapy', 'drug', 'medication'
        ]
        return sum(1 for term in biomedical_terms if term in query_lower)

    def _needs_more_evidence(self, results: List[Dict[str, Any]], query: str) -> bool:
        """Determine if additional evidence is needed"""
        if not results:
            return True
            
        # Check result quality
        avg_score = sum(r.get('combined_score', r.get('score', 0)) for r in results) / len(results)
        if avg_score < 0.3:
            return True
            
        # Check for recent query terms in results
        query_terms = set(query.lower().split())
        result_texts = ' '.join(r.get('text', '') for r in results[:5]).lower()
        term_coverage = sum(1 for term in query_terms if term in result_texts) / len(query_terms)
        
        return term_coverage < 0.5

    async def _fetch_fresh_documents(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Enhanced fresh document fetching"""
        try:
            logger.info(f"Fetching fresh documents for query: {query}")
            
            # Enhanced query for PubMed
            enhanced_query = self._enhance_pubmed_query(query)
            docs = await self.document_service.search_and_fetch(enhanced_query, max_results)
            
            if not docs:
                # Try simplified query
                simplified_query = self._simplify_query(query)
                docs = await self.document_service.search_and_fetch(simplified_query, max_results)
            
            if not docs:
                return []

            fresh_chunks = []
            for doc in docs:
                chunks = await self.embedding_service.embed_document(doc)
                fresh_chunks.extend(chunks)

            logger.info(f"Generated {len(fresh_chunks)} fresh document chunks")
            return fresh_chunks
        except Exception as e:
            logger.error(f"Failed to fetch fresh documents: {e}")
            return []

    def _enhance_pubmed_query(self, query: str) -> str:
        """Enhance query for PubMed search"""
        # Add MeSH terms and field qualifiers
        enhanced = query
        
        # Add recent publication filter
        enhanced += " AND (\"2020\"[Date - Publication] : \"2024\"[Date - Publication])"
        
        # Add study type preferences
        if any(term in query.lower() for term in ['treatment', 'therapy', 'drug']):
            enhanced += " AND (clinical trial[pt] OR randomized controlled trial[pt])"
        elif any(term in query.lower() for term in ['mechanism', 'pathway']):
            enhanced += " AND (review[pt] OR research support[pt])"
            
        return enhanced

    def _simplify_query(self, query: str) -> str:
        """Simplify query for broader search"""
        # Extract key biomedical terms
        words = query.split()
        important_words = []
        
        for word in words:
            word_lower = word.lower()
            if (len(word) > 3 and 
                word_lower not in ['what', 'how', 'when', 'where', 'why', 'does', 'are', 'the', 'and', 'for']):
                important_words.append(word)
                
        return ' '.join(important_words[:5])  # Use top 5 important words

    def _assemble_context(self, search_results: List[Dict[str, Any]], query: str, max_results: int):
        """Enhanced context assembly with citation metadata"""
        reranked = self._rerank_results(search_results, query)
        top_results = reranked[:max_results]
        
        context_parts, citations = [], []
        total_length = 0
        max_context_length = 4000  # Limit context length
        
        for i, res in enumerate(top_results):
            if total_length >= max_context_length:
                break
                
            cid = i + 1
            text = res.get('text', '')
            meta = res.get('metadata', {})
            
            if text:
                # Smart text truncation
                available_length = max_context_length - total_length
                if len(text) > available_length:
                    # Truncate at sentence boundary
                    sentences = text.split('. ')
                    truncated_text = ""
                    for sentence in sentences:
                        if len(truncated_text + sentence) < available_length - 50:
                            truncated_text += sentence + ". "
                        else:
                            break
                    text = truncated_text.strip()
                    if not text.endswith('.'):
                        text += "..."
                
                # Format with citation marker
                context_parts.append(f"[{cid}] {text}")
                total_length += len(text)
                
                # Enhanced citation metadata
                citations.append({
                    "id": cid,
                    "title": meta.get('title', 'Unknown'),
                    "journal": meta.get('journal', 'Unknown'),
                    "year": meta.get('year', 'Unknown'),
                    "pmid": meta.get('pmid', ''),
                    "pmcid": meta.get('pmcid', ''),
                    "doi": meta.get('doi', ''),
                    "authors": meta.get('authors', ''),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{meta.get('pmid', '')}/" if meta.get('pmid') else "",
                    "text_snippet": text[:200] + "..." if len(text) > 200 else text,
                    "source_quality": res.get('source_quality', 'unknown'),
                    "search_score": res.get('combined_score', res.get('score', 0))
                })
        
        context = "\n\n".join(context_parts)
        return context, citations

    def _rerank_results(self, results: List[Dict[str, Any]], query: str):
        """Enhanced result reranking"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        def enhanced_score(res):
            base_score = res.get('combined_score', res.get('score', 0))
            
            # Text relevance boost
            text = res.get('text', '').lower()
            word_overlap = len(query_words.intersection(set(text.split()))) / len(query_words) if query_words else 0
            relevance_boost = word_overlap * 0.3
            
            # Journal quality boost
            journal = res.get('metadata', {}).get('journal', '').lower()
            quality_boost = 0
            high_impact_journals = [
                'new england journal of medicine', 'nature', 'science', 'cell',
                'lancet', 'jama', 'bmj', 'nejm', 'circulation', 'jacc'
            ]
            for journal_name in high_impact_journals:
                if journal_name in journal:
                    quality_boost = 0.25
                    break
                    
            # Recency boost
            year = res.get('metadata', {}).get('year', '0')
            recency_boost = 0
            if str(year).isdigit():
                year_int = int(year)
                if year_int >= 2022:
                    recency_boost = 0.2
                elif year_int >= 2020:
                    recency_boost = 0.15
                elif year_int >= 2018:
                    recency_boost = 0.1
                    
            # Source type boost
            source_boost = 0
            if res.get('search_type') == 'dense':
                source_boost = 0.05  # Slight preference for semantic matches
                
            return base_score + relevance_boost + quality_boost + recency_boost + source_boost
            
        return sorted(results, key=enhanced_score, reverse=True)

    async def _generate_comprehensive_answer(self, query: str, context: str,
                                           query_analysis: Dict[str, Any],
                                           citations: List[Dict[str, Any]]) -> str:
        """Generate comprehensive, paragraph-length answers"""
        
        if not self.generator_loaded:
            return self._template_comprehensive_answer(query, context, query_analysis)

        # Build enhanced prompt based on query analysis
        intent = query_analysis.get('intent', 'general')
        question_type = query_analysis.get('question_type', 'factual')
        
        # Intent-specific prompt templates
        prompt_templates = {
            'mechanism': self._build_mechanism_prompt,
            'treatment': self._build_treatment_prompt,
            'diagnosis': self._build_diagnosis_prompt,
            'comparison': self._build_comparison_prompt,
            'research': self._build_research_prompt,
            'definition': self._build_definition_prompt
        }
        
        prompt_builder = prompt_templates.get(intent, self._build_general_prompt)
        prompt = prompt_builder(query, context, citations)
        
        try:
            # Generate with enhanced parameters
            output = self.generator(
                prompt,
                max_new_tokens=380,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
                return_full_text=False
            )
            
            answer = output[0]['generated_text'].strip()
            answer = self._post_process_answer(answer, citations)
            
            # Quality check and potential retry
            if self._is_answer_inadequate(answer):
                logger.info("Initial answer inadequate, retrying with modified prompt")
                enhanced_prompt = self._enhance_prompt_for_retry(prompt, answer)
                output = self.generator(
                    enhanced_prompt,
                    max_new_tokens=400,
                    temperature=0.6,
                    top_p=0.92,
                    return_full_text=False
                )
                answer = output[0]['generated_text'].strip()
                answer = self._post_process_answer(answer, citations)
            
            return answer if answer else self._template_comprehensive_answer(query, context, query_analysis)
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return self._template_comprehensive_answer(query, context, query_analysis)

    def _build_mechanism_prompt(self, query: str, context: str, citations: List[Dict]) -> str:
        """Build mechanism-focused prompt"""
        return f"""You are a biomedical expert. Provide a comprehensive explanation of the biological mechanism in response to this question using ONLY the provided scientific literature context.

Question: {query}

Scientific Context:
{context}

Instructions:
- Write 6-8 complete sentences explaining the mechanism step-by-step
- Include specific molecular processes, pathways, and interactions
- Use numbered citations [1], [2], etc. that correspond to the context sources
- Focus on cause-and-effect relationships
- Explain both upstream triggers and downstream effects
- Be scientifically precise but accessible

Answer:"""

    def _build_treatment_prompt(self, query: str, context: str, citations: List[Dict]) -> str:
        """Build treatment-focused prompt"""
        return f"""You are a clinical expert. Provide comprehensive treatment information based on current evidence using ONLY the provided scientific literature.

Question: {query}

Scientific Evidence:
{context}

Instructions:
- Write 6-8 complete sentences covering treatment approaches
- Include first-line therapies, alternatives, and recent advances
- Mention dosing, efficacy, and safety considerations when available
- Use numbered citations [1], [2], etc. from the provided sources
- Prioritize evidence-based treatments over experimental ones
- Include any relevant clinical guidelines or recommendations

Answer:"""

    def _build_diagnosis_prompt(self, query: str, context: str, citations: List[Dict]) -> str:
        """Build diagnosis-focused prompt"""
        return f"""You are a diagnostic expert. Provide comprehensive diagnostic information using ONLY the provided medical literature.

Question: {query}

Medical Literature:
{context}

Instructions:
- Write 6-8 complete sentences covering diagnostic approaches
- Include clinical criteria, laboratory tests, and imaging when relevant
- Mention diagnostic accuracy, sensitivity, and specificity if available
- Use numbered citations [1], [2], etc. from the context
- Address differential diagnosis considerations
- Include any recent advances in diagnostic techniques

Answer:"""

    def _build_comparison_prompt(self, query: str, context: str, citations: List[Dict]) -> str:
        """Build comparison-focused prompt"""
        return f"""You are a medical expert. Provide a comprehensive comparison using ONLY the provided scientific literature.

Question: {query}

Scientific Literature:
{context}

Instructions:
- Write 6-8 complete sentences comparing the options systematically
- Include efficacy, safety, cost-effectiveness, and patient factors
- Use evidence from clinical trials when available
- Use numbered citations [1], [2], etc. from the provided sources
- Present balanced analysis of advantages and disadvantages
- Include recommendations based on patient populations or clinical scenarios

Answer:"""

    def _build_research_prompt(self, query: str, context: str, citations: List[Dict]) -> str:
        """Build research-focused prompt"""
        return f"""You are a research expert. Summarize the current research evidence using ONLY the provided scientific literature.

Question: {query}

Research Literature:
{context}

Instructions:
- Write 6-8 complete sentences summarizing key research findings
- Include study designs, sample sizes, and primary outcomes when available
- Mention limitations and future research directions
- Use numbered citations [1], [2], etc. from the context
- Prioritize systematic reviews and meta-analyses
- Address clinical implications of the research

Answer:"""

    def _build_definition_prompt(self, query: str, context: str, citations: List[Dict]) -> str:
        """Build definition-focused prompt"""
        return f"""You are a medical educator. Provide a comprehensive definition and explanation using ONLY the provided scientific literature.

Question: {query}

Scientific Literature:
{context}

Instructions:
- Write 6-8 complete sentences defining and explaining the concept
- Include etymology, classification, and key characteristics
- Provide clinical context and significance
- Use numbered citations [1], [2], etc. from the sources
- Include relevant subtypes or related concepts
- Explain practical implications or applications

Answer:"""

    def _build_general_prompt(self, query: str, context: str, citations: List[Dict]) -> str:
        """Build general comprehensive prompt"""
        return f"""You are a biomedical expert. Provide a comprehensive, evidence-based answer using ONLY the provided scientific literature.

Question: {query}

Scientific Literature:
{context}

Instructions:
- Write 6-8 complete sentences providing a thorough answer
- Cover multiple aspects relevant to the question
- Use numbered citations [1], [2], etc. from the provided sources
- Be scientifically accurate and cite evidence appropriately
- Include current understanding and any recent developments
- Maintain focus on biomedical and clinical relevance

Answer:"""

    def _post_process_answer(self, answer: str, citations: List[Dict]) -> str:
        """Post-process generated answer"""
        # Remove prompt artifacts
        answer = re.sub(r'^(Answer\s*:?\s*)', '', answer, flags=re.IGNORECASE).strip()
        answer = re.sub(r'^(Based on.*?:)', '', answer, flags=re.IGNORECASE).strip()
        
        # Ensure proper sentence structure
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        # Ensure adequate length (6-8 sentences)
        if len(sentences) < 6:
            # Try to expand or use template
            return self._expand_short_answer(answer, citations)
        elif len(sentences) > 8:
            sentences = sentences[:8]
            
        answer = ' '.join(sentences)
        
        # Ensure final punctuation
        if not answer.endswith(('.', '!', '?')):
            answer += '.'
            
        # Validate citations exist
        if not re.search(r'\[\d+\]', answer) and citations:
            answer = answer.rstrip('.') + ' [1].'
            
        return answer

    def _expand_short_answer(self, answer: str, citations: List[Dict]) -> str:
        """Expand short answers using citation context"""
        if not citations:
            return answer
            
        # Add context from first few citations
        additional_context = []
        for i, citation in enumerate(citations[:3]):
            snippet = citation.get('text_snippet', '')
            if snippet and len(snippet) > 50:
                additional_context.append(f"{snippet} [{i+1}]")
                
        if additional_context:
            answer += " " + " ".join(additional_context)
            
        return answer

    def _is_answer_inadequate(self, answer: str) -> bool:
        """Check if answer needs improvement"""
        word_count = len(answer.split())
        sentence_count = len(re.split(r'[.!?]+', answer))
        has_citations = bool(re.search(r'\[\d+\]', answer))
        
        return (word_count < 80 or 
                sentence_count < 4 or 
                not has_citations)

    def _enhance_prompt_for_retry(self, original_prompt: str, previous_answer: str) -> str:
        """Enhance prompt for retry attempt"""
        return f"""{original_prompt}

Note: Provide a more detailed and comprehensive answer. The previous response was too brief. Focus on:
- Providing specific details and mechanisms
- Including more context from the scientific literature
- Ensuring proper citation of sources
- Writing at least 6 complete sentences

Answer:"""

    def _template_comprehensive_answer(self, query: str, context: str, query_analysis: Dict[str, Any]) -> str:
        """Template-based comprehensive answer generation"""
        intent = query_analysis.get('intent', 'general')
        
        # Extract key information from context
        context_sentences = [s.strip() for s in context.split('\n\n') if s.strip()]
        
        # Build comprehensive answer
        intro_templates = {
            'mechanism': f"The mechanism underlying {query.lower().replace('how does', '').replace('mechanism of', '').strip()} involves multiple interconnected biological processes.",
            'treatment': f"Treatment approaches for {query.lower().replace('treatment for', '').replace('how to treat', '').strip()} are based on current clinical evidence and guidelines.",
            'diagnosis': f"The diagnosis of {query.lower().replace('how to diagnose', '').replace('diagnosis of', '').strip()} relies on established clinical criteria and diagnostic tools.",
            'definition': f"Understanding {query.lower().replace('what is', '').replace('define', '').strip()} requires examining its biological and clinical characteristics.",
        }
        
        intro = intro_templates.get(intent, f"Regarding {query.lower()}, current biomedical literature provides comprehensive insights.")
        
        # Use context sentences with citations
        body_sentences = []
        for i, sentence in enumerate(context_sentences[:5]):
            if '[' in sentence:  # Already has citation
                body_sentences.append(sentence)
            else:
                body_sentences.append(f"{sentence} [{i+1}]")
                
        # Add conclusion
        conclusion = "These findings contribute to our current understanding and inform clinical practice and future research directions."
        
        comprehensive_answer = f"{intro} {' '.join(body_sentences)} {conclusion}"
        
        return comprehensive_answer

    def _validate_and_enhance_citations(self, answer: str, citations: List[Dict], search_results: List[Dict]) -> tuple:
        """Validate citations and enhance citation metadata"""
        # Find all citation markers in answer
        citation_matches = re.findall(r'\[(\d+)\]', answer)
        used_citations = set(int(match) for match in citation_matches)
        
        # Validate citations exist and are accessible
        validated_citations = []
        citation_mapping = {}
        
        for i, citation in enumerate(citations):
            citation_id = citation['id']
            if citation_id in used_citations:
                # Enhance citation with additional metadata
                enhanced_citation = citation.copy()
                enhanced_citation['used_in_answer'] = True
                enhanced_citation['validation_status'] = 'valid'
                
                # Add quality assessment
                enhanced_citation['citation_quality'] = self._assess_citation_quality(citation, search_results)
                
                validated_citations.append(enhanced_citation)
                citation_mapping[citation_id] = len(validated_citations)
            else:
                # Mark as unused but keep for reference
                citation['used_in_answer'] = False
                citation['validation_status'] = 'unused'
                
        # Check for orphaned citations (citations in answer but not in list)
        max_citation_id = len(citations)
        orphaned_citations = [cid for cid in used_citations if cid > max_citation_id]
        
        if orphaned_citations:
            logger.warning(f"Found orphaned citations: {orphaned_citations}")
            # Remove orphaned citations from answer
            for orphaned_id in orphaned_citations:
                answer = answer.replace(f'[{orphaned_id}]', '')
                
        # Add missing citations for unsupported statements
        enhanced_answer = self._add_missing_citations(answer, validated_citations)
        
        return enhanced_answer, validated_citations

    def _assess_citation_quality(self, citation: Dict, search_results: List[Dict]) -> str:
        """Assess the quality of a citation"""
        # Find corresponding search result
        search_result = None
        for result in search_results:
            if result.get('metadata', {}).get('pmid') == citation.get('pmid'):
                search_result = result
                break
                
        if not search_result:
            return 'unknown'
            
        score = search_result.get('combined_score', 0)
        journal_quality = citation.get('source_quality', 'unknown')
        year = citation.get('year', '0')
        
        # Calculate quality score
        quality_score = 0
        
        # Search relevance
        if score > 0.7:
            quality_score += 3
        elif score > 0.5:
            quality_score += 2
        elif score > 0.3:
            quality_score += 1
            
        # Journal impact
        if journal_quality == 'high':
            quality_score += 3
        elif journal_quality == 'recent':
            quality_score += 2
        elif journal_quality == 'moderate':
            quality_score += 1
            
        # Recency
        if str(year).isdigit() and int(year) >= 2020:
            quality_score += 2
        elif str(year).isdigit() and int(year) >= 2015:
            quality_score += 1
            
        # Classify quality
        if quality_score >= 6:
            return 'high'
        elif quality_score >= 4:
            return 'good'
        elif quality_score >= 2:
            return 'moderate'
        else:
            return 'low'

    def _add_missing_citations(self, answer: str, citations: List[Dict]) -> str:
        """Add citations to unsupported statements"""
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        enhanced_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                # Check if sentence has citation
                if not re.search(r'\[\d+\]', sentence):
                    # Find best matching citation
                    best_citation = self._find_best_citation_for_sentence(sentence, citations)
                    if best_citation:
                        sentence = sentence.rstrip('.!?') + f" [{best_citation['id']}]."
                        
                enhanced_sentences.append(sentence)
                
        return ' '.join(enhanced_sentences)

    def _find_best_citation_for_sentence(self, sentence: str, citations: List[Dict]) -> Dict:
        """Find best citation for a given sentence"""
        if not citations:
            return None
            
        sentence_lower = sentence.lower()
        best_citation = None
        best_score = 0
        
        for citation in citations:
            snippet = citation.get('text_snippet', '').lower()
            if snippet:
                # Simple word overlap scoring
                sentence_words = set(sentence_lower.split())
                snippet_words = set(snippet.split())
                overlap = len(sentence_words.intersection(snippet_words))
                score = overlap / max(len(sentence_words), 1)
                
                if score > best_score:
                    best_score = score
                    best_citation = citation
                    
        return best_citation if best_score > 0.2 else None

    def _calculate_enhanced_confidence(self, search_results: List[Dict], answer: str, citations: List[Dict]) -> float:
        """Calculate enhanced confidence score"""
        if not search_results:
            return 0.1
            
        # Base search quality
        search_scores = [r.get('combined_score', r.get('score', 0)) for r in search_results]
        avg_search_score = sum(search_scores) / len(search_scores) if search_scores else 0
        
        # Result count factor
        count_factor = min(1.0, len(search_results) / 10.0)
        
        # Answer quality factors
        word_count = len(answer.split())
        answer_length_factor = min(1.0, word_count / 120.0)  # Target ~120 words
        
        sentence_count = len(re.split(r'[.!?]+', answer))
        sentence_factor = min(1.0, sentence_count / 6.0)  # Target 6+ sentences
        
        # Citation quality
        citation_factor = 0
        if citations:
            valid_citations = [c for c in citations if c.get('used_in_answer', False)]
            citation_coverage = len(valid_citations) / max(len(citations), 1)
            citation_quality_scores = [self._citation_quality_score(c.get('citation_quality', 'unknown')) 
                                     for c in valid_citations]
            avg_citation_quality = sum(citation_quality_scores) / len(citation_quality_scores) if citation_quality_scores else 0
            citation_factor = 0.7 * citation_coverage + 0.3 * avg_citation_quality
        
        # Biomedical term density
        biomedical_density = self._calculate_biomedical_density(answer)
        
        # Combined confidence
        confidence = (
            avg_search_score * 0.25 +
            count_factor * 0.15 +
            answer_length_factor * 0.2 +
            sentence_factor * 0.15 +
            citation_factor * 0.2 +
            biomedical_density * 0.05
        )
        
        return min(0.98, max(0.1, confidence))

    def _citation_quality_score(self, quality: str) -> float:
        """Convert citation quality to numeric score"""
        quality_mapping = {
            'high': 1.0,
            'good': 0.8,
            'moderate': 0.6,
            'low': 0.4,
            'unknown': 0.2
        }
        return quality_mapping.get(quality, 0.2)

    def _calculate_biomedical_density(self, text: str) -> float:
        """Calculate density of biomedical terms in text"""
        biomedical_terms = [
            'clinical', 'therapeutic', 'diagnosis', 'treatment', 'therapy', 'medical',
            'disease', 'syndrome', 'disorder', 'condition', 'pathology', 'symptom',
            'drug', 'medication', 'pharmaceutical', 'compound', 'molecule',
            'protein', 'gene', 'enzyme', 'receptor', 'biomarker', 'pathway',
            'study', 'trial', 'research', 'evidence', 'analysis', 'investigation'
        ]
        
        text_lower = text.lower()
        text_words = set(text_lower.split())
        biomedical_word_count = sum(1 for term in biomedical_terms if term in text_words)
        
        return min(1.0, biomedical_word_count / 20.0)  # Normalize to 0-1

    async def _enhance_answer(self, query: str, context: str, current_answer: str, query_analysis: Dict) -> str:
        """Enhance short or low-quality answers"""
        if not self.generator_loaded:
            return current_answer
            
        enhancement_prompt = f"""The following answer needs enhancement to be more comprehensive and informative.

Original Question: {query}

Current Answer: {current_answer}

Additional Context: {context[:1000]}

Please provide an enhanced version that:
- Expands on the current answer with more details
- Maintains accuracy and proper citations
- Includes 6-8 complete sentences
- Provides more comprehensive coverage of the topic

Enhanced Answer:"""

        try:
            output = self.generator(
                enhancement_prompt,
                max_new_tokens=400,
                temperature=0.6,
                top_p=0.9,
                return_full_text=False
            )
            
            enhanced_answer = output[0]['generated_text'].strip()
            enhanced_answer = self._post_process_answer(enhanced_answer, [])
            
            # Only return if significantly better
            if len(enhanced_answer.split()) > len(current_answer.split()) * 1.3:
                return enhanced_answer
                
        except Exception as e:
            logger.error(f"Answer enhancement failed: {e}")
            
        return current_answer

    def _generate_contextual_follow_ups(self, query: str, answer: str, query_analysis: Dict, citations: List[Dict]) -> List[str]:
        """Generate contextual follow-up questions"""
        intent = query_analysis.get('intent', 'general')
        entities = query_analysis.get('entities', [])
        
        follow_ups = []
        
        # Intent-based follow-ups
        if intent == 'mechanism':
            follow_ups.extend([
                "What are the clinical implications of this mechanism?",
                "How can this mechanism be therapeutically targeted?",
                "What factors can disrupt this biological process?"
            ])
        elif intent == 'treatment':
            follow_ups.extend([
                "What are the potential side effects and contraindications?",
                "How does this treatment compare to alternative approaches?",
                "What are the latest clinical trial results?"
            ])
        elif intent == 'diagnosis':
            follow_ups.extend([
                "What is the diagnostic accuracy of these tests?",
                "How early can this condition be detected?",
                "What are emerging diagnostic technologies for this condition?"
            ])
        elif intent == 'comparison':
            follow_ups.extend([
                "What factors determine the choice between these options?",
                "Are there any combination approaches?",
                "What do recent meta-analyses conclude?"
            ])
            
        # Entity-based follow-ups
        if 'disease' in entities:
            follow_ups.extend([
                "What are the risk factors and prevention strategies?",
                "How has the understanding of this condition evolved recently?"
            ])
        if 'drug' in entities:
            follow_ups.extend([
                "What is the mechanism of action?",
                "Are there any drug interactions to consider?"
            ])
        if 'gene' in entities:
            follow_ups.extend([
                "What role does this gene play in disease pathogenesis?",
                "Are there therapeutic approaches targeting this gene?"
            ])
            
        # Citation-based follow-ups
        if citations:
            high_quality_citations = [c for c in citations if c.get('citation_quality') in ['high', 'good']]
            if len(high_quality_citations) >= 3:
                follow_ups.append("What do the most recent high-quality studies show?")
                
        # General follow-ups
        follow_ups.extend([
            "What are the current research gaps and future directions?",
            "How does this relate to clinical practice guidelines?",
            "What are the implications for patient care?"
        ])
        
        # Remove duplicates and select best
        unique_follow_ups = list(dict.fromkeys(follow_ups))  # Preserve order while removing duplicates
        
        # Score and select top 3
        scored_follow_ups = []
        for question in unique_follow_ups:
            score = self._score_follow_up_question(question, query, answer, query_analysis)
            scored_follow_ups.append((question, score))
            
        scored_follow_ups.sort(key=lambda x: x[1], reverse=True)
        return [q[0] for q in scored_follow_ups[:3]]

    def _score_follow_up_question(self, question: str, original_query: str, answer: str, query_analysis: Dict) -> float:
        """Score follow-up question relevance"""
        score = 0.5  # Base score
        
        # Prefer questions that build on the original query
        original_words = set(original_query.lower().split())
        question_words = set(question.lower().split())
        word_overlap = len(original_words.intersection(question_words))
        score += word_overlap * 0.1
        
        # Prefer questions related to the identified intent
        intent = query_analysis.get('intent', 'general')
        intent_keywords = {
            'mechanism': ['clinical', 'therapeutic', 'target', 'process'],
            'treatment': ['side effects', 'compare', 'alternative', 'trial'],
            'diagnosis': ['accuracy', 'early', 'detect', 'test'],
            'research': ['study', 'research', 'evidence', 'trial']
        }
        
        if intent in intent_keywords:
            for keyword in intent_keywords[intent]:
                if keyword in question.lower():
                    score += 0.2
                    break
                    
        # Prefer practical, actionable questions
        practical_keywords = ['clinical', 'patient', 'practice', 'care', 'implication']
        if any(keyword in question.lower() for keyword in practical_keywords):
            score += 0.15
            
        return score

    async def _save_query_history(self, user_id: int, query: str, answer: str, confidence: float):
        """Save enhanced query history"""
        try:
            from services.auth_service import get_auth_service
            auth = get_auth_service()
            
            title = query[:50] + ('...' if len(query) > 50 else '')
            preview = answer[:100] + ('...' if len(answer) > 100 else '')
            
            await auth.save_chat_history(
                user_id=user_id,
                title=title,
                preview=preview,
                query_text=query,
                response_text=answer,
                metadata={
                    'confidence_score': confidence,
                    'response_length': len(answer.split()),
                    'processing_version': 'enhanced_v1'
                }
            )
        except Exception as e:
            logger.error(f"Failed to save enhanced query history: {e}")

    def _fallback_response(self, query: str, error_message: str) -> Dict[str, Any]:
        """Enhanced fallback response"""
        return {
            "answer": f"I apologize, but I encountered an issue processing your query about '{query}'. "
                     f"This might be due to connectivity issues or insufficient relevant literature in the current index. "
                     f"Please try rephrasing your question or check back later as we continuously update our biomedical knowledge base.",
            "confidence_score": 0.1,
            "citations": [],
            "follow_up_questions": [
                "Could you rephrase your question with different terms?",
                "Are you looking for specific recent research or general information?",
                "Would you like to focus on a particular aspect of this topic?"
            ],
            "processing_time": 0.0,
            "query_analysis": {"intent": "error", "entities": [], "complexity_score": 0.0},
            "enhancement_applied": False,
            "error": error_message,
            "suggestions": [
                "Try using more specific biomedical terminology",
                "Focus on a single aspect of your question",
                "Check if the topic has recent research literature"
            ]
        }