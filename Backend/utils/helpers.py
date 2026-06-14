"""
Biomedical Text Processing Utilities
Specialized utilities for biomedical literature processing
"""

import re
import string
from typing import List, Dict, Any, Optional, Tuple
import unicodedata
import logging

logger = logging.getLogger(__name__)

class BiomedicalTextProcessor:
    """Specialized text processing for biomedical content"""

    def __init__(self):
        # Medical abbreviation dictionary
        self.medical_abbreviations = {
            # Cardiovascular
            'HTN': 'hypertension',
            'MI': 'myocardial infarction',
            'CHF': 'congestive heart failure',
            'CAD': 'coronary artery disease',
            'AF': 'atrial fibrillation',
            'PCI': 'percutaneous coronary intervention',
            'CABG': 'coronary artery bypass grafting',
            'ECG': 'electrocardiogram',
            'EKG': 'electrocardiogram',

            # Respiratory
            'COPD': 'chronic obstructive pulmonary disease',
            'ARDS': 'acute respiratory distress syndrome',
            'PE': 'pulmonary embolism',
            'DVT': 'deep vein thrombosis',

            # Endocrine
            'DM': 'diabetes mellitus',
            'T1DM': 'type 1 diabetes mellitus',
            'T2DM': 'type 2 diabetes mellitus',
            'HbA1c': 'glycated hemoglobin',
            'TSH': 'thyroid stimulating hormone',

            # Nephrology
            'CKD': 'chronic kidney disease',
            'ESRD': 'end stage renal disease',
            'GFR': 'glomerular filtration rate',
            'AKI': 'acute kidney injury',

            # Infectious Disease
            'HIV': 'human immunodeficiency virus',
            'AIDS': 'acquired immunodeficiency syndrome',
            'HBV': 'hepatitis B virus',
            'HCV': 'hepatitis C virus',
            'COVID-19': 'coronavirus disease 2019',
            'SARS-CoV-2': 'severe acute respiratory syndrome coronavirus 2',
            'MERS-CoV': 'middle east respiratory syndrome coronavirus',

            # Molecular Biology
            'DNA': 'deoxyribonucleic acid',
            'RNA': 'ribonucleic acid',
            'mRNA': 'messenger ribonucleic acid',
            'tRNA': 'transfer ribonucleic acid',
            'rRNA': 'ribosomal ribonucleic acid',
            'PCR': 'polymerase chain reaction',
            'RT-PCR': 'reverse transcription polymerase chain reaction',
            'ELISA': 'enzyme-linked immunosorbent assay',
            'WB': 'western blot',
            'FISH': 'fluorescence in situ hybridization',

            # Immunology
            'NK': 'natural killer',
            'CTL': 'cytotoxic T lymphocyte',
            'Th1': 'T helper 1',
            'Th2': 'T helper 2',
            'Treg': 'regulatory T cell',
            'IL': 'interleukin',
            'TNF': 'tumor necrosis factor',
            'IFN': 'interferon',

            # Oncology
            'NSCLC': 'non-small cell lung cancer',
            'SCLC': 'small cell lung cancer',
            'ALL': 'acute lymphoblastic leukemia',
            'AML': 'acute myeloid leukemia',
            'CLL': 'chronic lymphocytic leukemia',
            'CML': 'chronic myeloid leukemia',
            'NHL': 'non-Hodgkin lymphoma',
            'CAR-T': 'chimeric antigen receptor T cell',

            # Gene Editing
            'CRISPR': 'clustered regularly interspaced short palindromic repeats',
            'Cas9': 'CRISPR associated protein 9',
            'gRNA': 'guide ribonucleic acid',
            'PAM': 'protospacer adjacent motif',

            # Clinical
            'ICU': 'intensive care unit',
            'OR': 'operating room',
            'ER': 'emergency room',
            'ED': 'emergency department',
            'IV': 'intravenous',
            'PO': 'per os by mouth',
            'IM': 'intramuscular',
            'SC': 'subcutaneous',
            'BID': 'twice daily',
            'TID': 'three times daily',
            'QID': 'four times daily',
            'PRN': 'as needed'
        }

        # Drug classes and mechanisms
        self.drug_classes = {
            'ACE inhibitor': 'angiotensin converting enzyme inhibitor',
            'ARB': 'angiotensin receptor blocker',
            'CCB': 'calcium channel blocker',
            'NSAID': 'nonsteroidal anti-inflammatory drug',
            'SSRI': 'selective serotonin reuptake inhibitor',
            'SNRI': 'serotonin norepinephrine reuptake inhibitor',
            'PPI': 'proton pump inhibitor',
            'H2RA': 'histamine 2 receptor antagonist'
        }

        # Biomedical entity patterns
        self.entity_patterns = {
            'gene': r'\b[A-Z][A-Z0-9]{1,10}\b',  # Gene symbols (simplified)
            'protein': r'\b[A-Z][a-z]{2,}(?:-\d+)?\b(?:\s+protein)?',  # Protein names
            'drug': r'\b[a-z]{3,}(?:mab|ib|nib|pril|sartan|olol|pine)\b',  # Drug suffixes
            'disease': r'\b(?:syndrome|disease|disorder|carcinoma|adenoma|sarcoma)\b',
            'procedure': r'\b(?:surgery|biopsy|resection|transplantation|therapy)\b'
        }

    def expand_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations in text"""
        expanded_text = text

        # Expand medical abbreviations
        for abbrev, expansion in self.medical_abbreviations.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            expanded_text = re.sub(pattern, expansion, expanded_text, flags=re.IGNORECASE)

        # Expand drug class abbreviations
        for abbrev, expansion in self.drug_classes.items():
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            expanded_text = re.sub(pattern, expansion, expanded_text, flags=re.IGNORECASE)

        return expanded_text

    def normalize_biomedical_text(self, text: str) -> str:
        """Comprehensive biomedical text normalization"""
        if not text:
            return ""

        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)

        # Fix common OCR/encoding errors
        text = self._fix_encoding_errors(text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Normalize hyphens and dashes
        text = re.sub(r'[—–−]', '-', text)

        # Fix spacing around punctuation
        text = re.sub(r'\s*([.,:;!?])\s*', r'\1 ', text)
        text = re.sub(r'\s+([.,:;!?])', r'\1', text)

        # Normalize parentheses spacing
        text = re.sub(r'\s*\(\s*', ' (', text)
        text = re.sub(r'\s*\)\s*', ') ', text)

        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _fix_encoding_errors(self, text: str) -> str:
        """Fix common encoding errors in biomedical text"""
        # Common encoding fixes
        fixes = {
            'Â±': '±',
            'Î±': 'α',
            'Î²': 'β',
            'Î³': 'γ',
            'Î´': 'δ',
            'Îµ': 'ε',
            'âˆ'': '−',
            'â‰¤': '≤',
            'â‰¥': '≥',
            'Â°': '°',
            'Âµ': 'μ'
        }

        for wrong, correct in fixes.items():
            text = text.replace(wrong, correct)

        return text

    def extract_biomedical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract biomedical entities from text"""
        entities = {entity_type: [] for entity_type in self.entity_patterns}

        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = list(set(matches))  # Remove duplicates

        return entities

    def segment_biomedical_text(self, text: str, max_length: int = 500) -> List[str]:
        """Segment biomedical text at appropriate boundaries"""
        if len(text.split()) <= max_length:
            return [text]

        segments = []
        current_segment = ""

        # Split by sentences first
        sentences = self._split_sentences(text)

        for sentence in sentences:
            sentence_words = sentence.split()

            # If adding this sentence would exceed max_length, start new segment
            if len(current_segment.split()) + len(sentence_words) > max_length and current_segment:
                segments.append(current_segment.strip())
                current_segment = sentence
            else:
                current_segment += " " + sentence if current_segment else sentence

        if current_segment:
            segments.append(current_segment.strip())

        return segments

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling biomedical abbreviations"""
        # Common abbreviations that shouldn't trigger sentence breaks
        abbreviations = {'Dr.', 'Mr.', 'Ms.', 'Prof.', 'et al.', 'i.e.', 'e.g.', 'vs.', 'Fig.', 'Table'}

        # Split by periods, but be careful with abbreviations
        sentences = []
        current_sentence = ""

        words = text.split()
        for i, word in enumerate(words):
            current_sentence += word + " "

            if word.endswith('.') and word not in abbreviations:
                # Check if next word starts with capital (likely new sentence)
                if i + 1 < len(words) and words[i + 1][0].isupper():
                    sentences.append(current_sentence.strip())
                    current_sentence = ""

        if current_sentence:
            sentences.append(current_sentence.strip())

        return sentences

    def clean_citation_text(self, text: str) -> str:
        """Clean text by removing in-text citations"""
        # Remove common citation patterns
        patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2020)
            r'\[[^\]]*\d+[^\]]*\]',  # [1], [1-5], [Smith et al.]
            r'\b(?:ref|reference)s?\.?\s*\d+',  # ref. 1, references 1-5
        ]

        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # Clean up extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)

        return cleaned.strip()

    def extract_numerical_values(self, text: str) -> List[Dict[str, Any]]:
        """Extract numerical values with their units and context"""
        # Pattern for numbers with units
        number_pattern = r'(\d+(?:\.\d+)?)\s*(mg|g|kg|ml|l|μg|ng|pg|μM|nM|mM|M|%|mmHg|bpm|°C|°F)'

        matches = re.findall(number_pattern, text, re.IGNORECASE)

        numerical_values = []
        for value, unit in matches:
            numerical_values.append({
                'value': float(value),
                'unit': unit,
                'context': self._extract_context_around_number(text, f"{value} {unit}")
            })

        return numerical_values

    def _extract_context_around_number(self, text: str, number_str: str, window: int = 20) -> str:
        """Extract context words around a numerical value"""
        words = text.split()

        # Find the position of the number
        for i, word in enumerate(words):
            if number_str in word:
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                return ' '.join(words[start:end])

        return ""

    def identify_section_headers(self, text: str) -> List[Dict[str, Any]]:
        """Identify section headers in biomedical text"""
        common_headers = [
            'abstract', 'introduction', 'background', 'methods', 'methodology',
            'materials and methods', 'results', 'findings', 'discussion',
            'conclusion', 'conclusions', 'acknowledgments', 'references',
            'supplementary', 'appendix'
        ]

        sections = []
        lines = text.split('\n')

        for i, line in enumerate(lines):
            line_clean = line.strip().lower()

            # Check if line matches section header patterns
            for header in common_headers:
                if (line_clean == header or 
                    line_clean.startswith(header + ':') or
                    (len(line.split()) <= 3 and header in line_clean)):

                    sections.append({
                        'header': line.strip(),
                        'type': header,
                        'line_number': i,
                        'position': text.find(line)
                    })
                    break

        return sections

    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key biomedical phrases from text"""
        # Clean the text
        cleaned_text = self.normalize_biomedical_text(text)

        # Common biomedical phrase patterns
        phrase_patterns = [
            # Mechanism descriptions
            r'\b(?:inhibit|activate|induce|suppress|enhance|regulate)s?\s+[a-z\s]{5,30}',
            # Treatment descriptions
            r'\b(?:treat|therapy|treatment)\s+(?:of|for|with)\s+[a-z\s]{5,30}',
            # Association descriptions
            r'\bassociated\s+with\s+[a-z\s]{5,30}',
            # Function descriptions
            r'\bfunction\s+(?:of|in)\s+[a-z\s]{5,30}',
            # Effect descriptions
            r'\beffect\s+(?:of|on)\s+[a-z\s]{5,30}'
        ]

        key_phrases = []
        for pattern in phrase_patterns:
            matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
            for match in matches:
                if 10 <= len(match) <= 100:  # Reasonable phrase length
                    key_phrases.append(match.strip())

        # Remove duplicates and sort by length (prefer longer phrases)
        unique_phrases = list(set(key_phrases))
        unique_phrases.sort(key=len, reverse=True)

        return unique_phrases[:max_phrases]

    def validate_biomedical_content(self, text: str) -> Dict[str, Any]:
        """Validate biomedical content quality"""
        validation_results = {
            'length_adequate': len(text.split()) >= 20,
            'has_biomedical_terms': False,
            'has_numerical_data': False,
            'has_proper_structure': False,
            'readability_score': 0.0
        }

        # Check for biomedical terms
        biomedical_indicators = [
            r'\b(?:study|research|analysis|trial)\b',
            r'\b(?:patient|subject|participant)s?\b',
            r'\b(?:treatment|therapy|drug|medication)s?\b',
            r'\b(?:gene|protein|enzyme|receptor)s?\b',
            r'\b(?:cell|tissue|organ)s?\b',
            r'\b(?:disease|syndrome|disorder)s?\b'
        ]

        biomedical_matches = 0
        for pattern in biomedical_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                biomedical_matches += 1

        validation_results['has_biomedical_terms'] = biomedical_matches >= 2

        # Check for numerical data
        numerical_patterns = [
            r'\d+(?:\.\d+)?\s*%',  # Percentages
            r'\d+(?:\.\d+)?\s*(?:mg|g|ml|μg|ng)',  # Measurements
            r'p\s*[<>=]\s*0\.\d+',  # P-values
            r'\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?'  # Error margins
        ]

        for pattern in numerical_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                validation_results['has_numerical_data'] = True
                break

        # Check structure (sentences, paragraphs)
        sentences = text.split('.')
        paragraphs = text.split('\n\n')
        validation_results['has_proper_structure'] = len(sentences) >= 3 and len(paragraphs) >= 1

        # Simple readability score
        words = text.split()
        sentences_count = len([s for s in sentences if s.strip()])
        if sentences_count > 0:
            avg_words_per_sentence = len(words) / sentences_count
            # Biomedical text typically has longer sentences
            if 15 <= avg_words_per_sentence <= 25:
                validation_results['readability_score'] = 1.0
            elif 10 <= avg_words_per_sentence <= 30:
                validation_results['readability_score'] = 0.7
            else:
                validation_results['readability_score'] = 0.4

        return validation_results

# Global processor instance
_processor = None

def get_biomedical_processor():
    """Get global biomedical text processor instance"""
    global _processor
    if _processor is None:
        _processor = BiomedicalTextProcessor()
    return _processor

# Convenience functions
def expand_medical_abbreviations(text: str) -> str:
    """Expand medical abbreviations in text"""
    return get_biomedical_processor().expand_abbreviations(text)

def normalize_text(text: str) -> str:
    """Normalize biomedical text"""
    return get_biomedical_processor().normalize_biomedical_text(text)

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract biomedical entities"""
    return get_biomedical_processor().extract_biomedical_entities(text)

def segment_text(text: str, max_length: int = 500) -> List[str]:
    """Segment biomedical text"""
    return get_biomedical_processor().segment_biomedical_text(text, max_length)

def validate_content(text: str) -> Dict[str, Any]:
    """Validate biomedical content quality"""
    return get_biomedical_processor().validate_biomedical_content(text)
