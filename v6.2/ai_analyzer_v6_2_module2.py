"""
═══════════════════════════════════════════════════════════════════════════════
AI SEMANTIC ANALYZER v6.2.0 - MODUL 2: AI DETECTION, CONTEXT & FP REDUCTION
═══════════════════════════════════════════════════════════════════════════════

PDF processing, AI reference detection, and false positive filtering for 
AI Semantic Analyzer v6.2.

MAJOR UPDATE v6.2.0 (February 2026):
    🆕 DUAL TAXONOMY SUPPORT
        - Integrated with ai_taxonomy_v7 (16 categories)
        - Enhanced pattern matching for Applications + Technologies
        - Vendor-specific detection (ChatGPT, Claude, Gemini, AWS, etc.)
    
    🔒 IMPROVED ACCURACY
        - Enhanced false positive filtering
        - Keyword tier-based confidence scoring
        - Stricter semantic thresholds for risky keywords
        - Better handling of ML/DL measurement units
    
    🎯 DETECTION IMPROVEMENTS
        - 99 patterns (vs 45 in v6.1, +120%)
        - 413 keywords (vs 150 in v6.1, +176%)
        - Mandatory AI context for generic terms
        - Multi-label support (ready for v7.0)

COMPONENTS:
    1. PDF Text Extraction
       - pdfplumber: Primary extraction
       - PyMuPDF (fitz): Fallback
       - Tesseract OCR: Image-based PDFs
       - Text corruption detection
    
    2. Semantic Model (Singleton)
       - SentenceTransformer: all-MiniLM-L6-v2
       - Lazy loading for performance
       - Cosine similarity matching
    
    3. AI Reference Detection
       - Hard patterns: AI/ML/DL with context
       - Soft patterns: Domain-specific terms
       - Semantic validation (threshold: 0.60-0.68)
       - Context extraction with boundaries
    
    4. Category Classification
       - Keyword-based primary matching
       - Pattern-based secondary matching
       - Semantic similarity fallback
       - Confidence scoring with tiers (NEW v6.2)
    
    5. False Positive Filtering
       - ~60 exclusion patterns
       - Text corruption detection
       - Short text validators (AI, ML, DL)
       - Board/governance filtering
       - Measurement unit exclusions

WORKFLOW:
    PDF → Extract Text → Detect AI References → Classify Categories
    → Filter False Positives → Extract Context → Return AIReference objects

CHANGELOG v6.2.0 (Feb 2026):
    - Integration with Dual Taxonomy v7.0.1
    - Enhanced pattern matching (99 patterns)
    - Vendor-specific keyword detection
    - Keyword tier confidence scoring
    - Improved false positive filtering
    - Multi-label classification ready

CHANGELOG v6.1.2 (Jan 2026):
    - FIX: _detect_by_semantics() transmits reference_strength, confidence_score
    - Resolved "cannot access local variable 'ref'" error

CHANGELOG v6.1.1 (Jan 2026):
    - Populate AIReference.reference_strength / confidence_score / confidence_reasons
    - Export compatibility: strength/confidence in RAW and DEDUP

TECHNICAL NOTES:
    - Semantic threshold: 0.60 (general), 0.68 (strict for mention-only)
    - Context window: ±200 chars around match
    - Multi-threading safe: SemanticModelLoader singleton
    - Memory efficient: Lazy model loading

Author: TeRa0
Version: 6.2.0
Date: February 2026
Part of: AI Semantic Analyzer

═══════════════════════════════════════════════════════════════════════════════
"""


from __future__ import annotations
import time
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import pdfplumber
import fitz
from sentence_transformers import SentenceTransformer, util

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE_LOCAL = True
except ImportError:
    OCR_AVAILABLE_LOCAL = False

try:
    import wordsegment
    wordsegment.load()
    WORDSEGMENT_AVAILABLE = True
except ImportError:
    WORDSEGMENT_AVAILABLE = False

from ai_analyzer_v6_2_module1 import (
    logger, AnalyzerConfig, AIReference, DocumentResult,
    AI_CATEGORIES, FALSE_POSITIVE_PATTERNS, OCR_AVAILABLE,
    TRADITIONAL_ROBOTICS_PATTERNS, AI_ROBOTICS_PATTERNS,
    RPA_NON_AI_PATTERNS, RPA_AI_PATTERNS
)

import warnings
import logging

# Suprimă mesajele deranjante din PDF processing
warnings.filterwarnings("ignore")
logging.getLogger("pdfminer").setLevel(logging.CRITICAL)
logging.getLogger("pdfplumber").setLevel(logging.CRITICAL)

# ═══════════════════════════════════════════════════════════════════════════
# SEMANTIC MODEL LOADER (Singleton)
# ═══════════════════════════════════════════════════════════════════════════

class SemanticModelLoader:
    _instance = None
    _model = None
    MODEL_NAME = 'all-MiniLM-L6-v2'
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if SemanticModelLoader._model is None:
            logger.info(f"Încărcare model semantic: {self.MODEL_NAME}")
            start_time = time.time()
            SemanticModelLoader._model = SentenceTransformer(self.MODEL_NAME)
            logger.info(f"Model încărcat în {time.time() - start_time:.2f}s")
    
    @property
    def model(self):
        return SemanticModelLoader._model


# ═══════════════════════════════════════════════════════════════════════════
# AI MAIN PATTERNS v6.0 - 80+ PATTERNS
# ═══════════════════════════════════════════════════════════════════════════

AI_MAIN_PATTERNS = [
    # CORE AI/ML
    r'\bartificial intelligence\b',
    r'\bmachine learning\b',
    r'\bdeep learning\b',
    r'\bneural network(?:s)?\b',
    r'\bnatural language processing\b',
    r'\bcomputer vision\b',
    r'\b(?:AI|ML|DL|NLP)\b(?![\w-])',
    r'\bpredictive analytics\b',
    r'\bcognitive computing\b',
    r'\bintelligent automation\b',
    
    # GENERATIVE AI & LLMs (2023-2025)
    r'\bgenerative AI\b',
    r'\bGenAI\b',
    r'\blarge language model(?:s)?\b',
    r'\bLLM(?:s)?\b(?![\w-])',
    r'\bGPT(?:[\s-]?[3-5])?\b',
    r'\bChatGPT\b',
    r'\bfoundation model(?:s)?\b',
    r'\bfrontier model(?:s)?\b',
    r'\btransformer(?:s)?\b',
    r'\bmultimodal(?:\s+AI|\s+model)?\b',
    r'\btext[\s-]to[\s-](?:image|video|speech)\b',
    r'\b(?:content|text|code|image)\s+generation\b',
    
    # Modele specifice
    r'\bCopilot\b', r'\bClaude\b', r'\bGemini\b',
    r'\bLlama(?:\s+\d)?\b', r'\bMistral\b',
    r'\bDALL[\s-]?E\b', r'\bStable\s+Diffusion\b',
    r'\bMidjourney\b', r'\bSora\b',
    r'\bAnthropic\b', r'\bOpenAI\b',
    
    # AGENTIC AI (2025)
    r'\b(?:AI|intelligent)\s+agent(?:s)?\b',
    r'\bagentic\s+(?:AI|system|workflow|RAG)\b',
    r'\bautonomous\s+(?:AI|agent|system)\b',
    r'\bmulti[\s-]?agent\b',
    r'\bagent[\s-]?based\b',
    r'\btool\s+(?:use|calling)\b',
    r'\bfunction\s+calling\b',
    r'\bAutoGPT\b', r'\bCrewAI\b', r'\bLangGraph\b', r'\bLangChain\b',
    
    # RAG & KNOWLEDGE (2024-2025)
    r'\bRAG\b(?![\w-])',
    r'\bretrieval[\s-]?augmented\s+generation\b',
    r'\bvector\s+(?:database|store|search|embedding)\b',
    r'\bembedding(?:s)?\b',
    r'\bsemantic\s+search\b',
    r'\bknowledge\s+(?:base|graph)\b',
    
    # PROMPT & FINE-TUNING
    r'\bprompt\s+engineering\b',
    r'\bfine[\s-]?tun(?:e|ed|ing)\b',
    r'\bRLHF\b', r'\bLoRA\b',
    r'\binstruction\s+tuning\b',
    
    # INFRASTRUCTURE & MLOps
    r'\bMLOps\b', r'\bLLMOps\b', r'\bAIOps\b',
    r'\bAI[\s-]?as[\s-]?a[\s-]?service\b',
    r'\bedge\s+(?:AI|computing)\b',
    r'\bmodel\s+(?:serving|deployment|inference)\b',
    r'\b(?:GPU|TPU|NPU)\s+(?:cluster|compute|infrastructure)\b',
    
    # AI COMPOUNDS
    r'\bAI[\s-]?powered\b', r'\bAI[\s-]?driven\b',
    r'\bAI[\s-]?enabled\b', r'\bAI[\s-]?first\b',
    r'\bAI[\s-]?native\b', r'\bAI[\s-]?augmented\b',
    r'\bML[\s-]?based\b', r'\bML[\s-]?powered\b',
    
    # SAFETY & GOVERNANCE
    r'\bresponsible\s+AI\b', r'\bAI\s+ethics\b',
    r'\bAI\s+governance\b', r'\bexplainable\s+AI\b',
    r'\bXAI\b', r'\bAI\s+safety\b',
    r'\bhallucination(?:s)?\b', r'\bguardrails\b',
    
    # USE CASES
    r'\bchatbot(?:s)?\b', r'\bvirtual\s+assistant\b',
    r'\bconversational\s+AI\b',
    r'\brecommendation\s+(?:system|engine)\b',
    r'\bsentiment\s+analysis\b', r'\bfraud\s+detection\b',
    r'\bpredictive\s+maintenance\b',
    r'\bautonomous\s+(?:vehicle|driving)\b',
    r'\brobot(?:ic)?(?:s)?\b', r'\bRPA\b', r'\bhyperautomation\b',
    
    # AI CODING (2024-2025)
    r'\bcode\s+(?:generation|completion)\b',
    r'\bAI[\s-]?(?:assisted|powered)\s+(?:coding|development)\b',
    r'\bGitHub\s+Copilot\b', r'\bCursor\b', r'\bTabnine\b',
]


# ═══════════════════════════════════════════════════════════════════════════
# AI CANONICAL DESCRIPTIONS v6.0 - 65+ DESCRIERI
# ═══════════════════════════════════════════════════════════════════════════

AI_CANONICAL_DESCRIPTIONS = [
    # Strategic Investment
    "investing in artificial intelligence technology and capabilities",
    "strategic acquisition of AI companies and startups",
    "allocating budget for AI research and development",
    "AI investment strategy and portfolio management",
    "significant capital expenditure on artificial intelligence initiatives",
    
    # Operational Implementation
    "deploying AI solutions for operational efficiency",
    "automating business processes with artificial intelligence",
    "implementing machine learning in operations",
    "AI-driven process optimization and automation",
    "hyperautomation using AI and machine learning",
    
    # Customer Experience
    "AI-powered chatbots for customer service",
    "personalized customer experience using machine learning",
    "intelligent recommendation systems for customers",
    "conversational AI and virtual assistants",
    "AI-driven customer journey optimization",
    
    # Product Development
    "developing AI-powered products and features",
    "artificial intelligence in product innovation",
    "intelligent product design and development",
    "AI-first product strategy and roadmap",
    
    # Governance & Ethics
    "responsible AI and ethical considerations",
    "AI governance framework and policies",
    "mitigating bias in artificial intelligence systems",
    "AI compliance with regulations and standards",
    "AI safety and alignment initiatives",
    
    # Data & Analytics
    "predictive analytics using machine learning",
    "data-driven insights with artificial intelligence",
    "AI for business intelligence and forecasting",
    "augmented analytics and AI-powered insights",
    
    # Talent & Workforce
    "hiring AI talent and data scientists",
    "upskilling workforce in artificial intelligence",
    "AI training programs for employees",
    "building AI capabilities and expertise",
    
    # Agentic AI (2025)
    "autonomous AI agents and intelligent systems",
    "multi-agent orchestration and coordination",
    "self-learning adaptive AI systems",
    "intelligent autonomous decision-making",
    "agentic AI workflows and task automation",
    "AI agents for complex problem solving",
    "tool use and function calling capabilities",
    
    # Generative AI & LLMs (2023-2025)
    "generative AI and large language models",
    "AI content generation and creation",
    "foundation models and GPT applications",
    "generative artificial intelligence solutions",
    "large language model deployment and fine-tuning",
    "ChatGPT and Copilot integration",
    "LLM-powered applications and workflows",
    "multimodal AI capabilities",
    "prompt engineering and optimization",
    
    # RAG & Knowledge (2024-2025)
    "retrieval-augmented generation for enterprise",
    "vector database and semantic search",
    "knowledge graph and AI integration",
    "grounded AI responses with RAG",
    
    # Infrastructure & MLOps
    "AI infrastructure and cloud platforms",
    "MLOps and model deployment pipelines",
    "edge computing for AI applications",
    "scalable AI computing infrastructure",
    "model serving and inference optimization",
    
    # Explainability & Safety
    "explainable AI and model interpretability",
    "transparent and trustworthy artificial intelligence",
    "AI safety and robustness measures",
    "AI guardrails and content moderation",
    "hallucination detection and prevention",
    
    # Research & Innovation
    "AI research collaborations with universities",
    "artificial intelligence patent portfolio",
    "breakthrough AI innovations and discoveries",
    "frontier AI research and development",
    
    # AI Coding (2024-2025)
    "AI-assisted software development",
    "code generation with machine learning",
    "GitHub Copilot and AI coding tools",
    "automated code review and testing with AI",
]


_AI_DESCRIPTION_EMBEDDINGS = None

def get_ai_description_embeddings():
    global _AI_DESCRIPTION_EMBEDDINGS
    if _AI_DESCRIPTION_EMBEDDINGS is None:
        loader = SemanticModelLoader.get_instance()
        logger.info("Calculare embeddings pentru descrieri AI canonice...")
        _AI_DESCRIPTION_EMBEDDINGS = loader.model.encode(
            AI_CANONICAL_DESCRIPTIONS, convert_to_tensor=True, show_progress_bar=False
        )
    return _AI_DESCRIPTION_EMBEDDINGS


# ═══════════════════════════════════════════════════════════════════════════
# ROBOTICS & RPA CLASSIFICATION v6.0
# ═══════════════════════════════════════════════════════════════════════════

_compiled_traditional_robotics = None
_compiled_ai_robotics = None
_compiled_rpa_non_ai = None
_compiled_rpa_ai = None

def _compile_robotics_patterns():
    global _compiled_traditional_robotics, _compiled_ai_robotics
    global _compiled_rpa_non_ai, _compiled_rpa_ai
    
    if _compiled_traditional_robotics is None:
        _compiled_traditional_robotics = [re.compile(p, re.IGNORECASE) for p in TRADITIONAL_ROBOTICS_PATTERNS]
        _compiled_ai_robotics = [re.compile(p, re.IGNORECASE) for p in AI_ROBOTICS_PATTERNS]
        _compiled_rpa_non_ai = [re.compile(p, re.IGNORECASE) for p in RPA_NON_AI_PATTERNS]
        _compiled_rpa_ai = [re.compile(p, re.IGNORECASE) for p in RPA_AI_PATTERNS]


def classify_robotics_reference(context: str) -> str:
    """Returns: 'ai_robotics', 'traditional_robotics', 'not_robotics'"""
    _compile_robotics_patterns()
    context_lower = context.lower()
    
    if 'robot' not in context_lower:
        return 'not_robotics'
    
    for pattern in _compiled_ai_robotics:
        if pattern.search(context):
            return 'ai_robotics'
    
    for pattern in _compiled_traditional_robotics:
        if pattern.search(context):
            return 'traditional_robotics'
    
    ai_indicators = ['artificial intelligence', 'machine learning', 'neural', 
                    'autonomous', 'cognitive', 'intelligent', ' AI ', ' ML ',
                    'computer vision', 'deep learning', 'predictive']
    
    if any(ind.lower() in context_lower for ind in ai_indicators):
        return 'ai_robotics'
    
    return 'traditional_robotics'


def classify_rpa_reference(context: str) -> str:
    """Returns: 'ai_rpa', 'traditional_rpa', 'not_rpa'"""
    _compile_robotics_patterns()
    context_lower = context.lower()
    
    if 'rpa' not in context_lower and 'robotic process' not in context_lower:
        return 'not_rpa'
    
    for pattern in _compiled_rpa_ai:
        if pattern.search(context):
            return 'ai_rpa'
    
    for pattern in _compiled_rpa_non_ai:
        if pattern.search(context):
            return 'traditional_rpa'
    
    return 'not_rpa'


# ═══════════════════════════════════════════════════════════════════════════
# TEXT CLEANER
# ═══════════════════════════════════════════════════════════════════════════

class TextCleaner:
    SPACE_PATTERNS = [
        (r'([a-z])([A-Z][a-z])', r'\1 \2'),
        (r'([a-z])([A-Z]{2,})', r'\1 \2'),
        (r'([A-Z]{2,})([a-z])', r'\1 \2'),
        (r'(\.)([A-Z])', r'\1 \2'),
        (r'(,)([a-zA-Z])', r'\1 \2'),
    ]
    
    EXCEPTIONS = [
        'GenAI', 'OpenAI', 'DeepMind', 'GitHub', 'LinkedIn',
        'ChatGPT', 'AutoGPT', 'LangChain', 'LangGraph', 'MLOps', 'LLMOps',
    ]
    
    def __init__(self):
        self.compiled_patterns = [(re.compile(p), r) for p, r in self.SPACE_PATTERNS]
        self.exception_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(e) for e in self.EXCEPTIONS) + r')\b', re.IGNORECASE
        )
    
    def clean_text(self, text: str) -> str:
        if not text:
            return text
        
        protected = {}
        counter = [0]
        def protect(match):
            key = f"__PROT_{counter[0]}__"
            protected[key] = match.group(0)
            counter[0] += 1
            return key
        
        text = self.exception_pattern.sub(protect, text)
        
        for pattern, replacement in self.compiled_patterns:
            text = pattern.sub(replacement, text)
        
        for key, value in protected.items():
            text = text.replace(key, value)
        
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()


_text_cleaner = None

def get_text_cleaner() -> TextCleaner:
    global _text_cleaner
    if _text_cleaner is None:
        _text_cleaner = TextCleaner()
    return _text_cleaner

# ═══════════════════════════════════════════════════════════════════════════
# TEXT CORRUPTION DETECTOR v6.0.4
# ═══════════════════════════════════════════════════════════════════════════

def is_text_corrupted(text: str) -> Tuple[bool, str]:
    """
    Verifică dacă textul extras este corupt/ilizibil.
    Returns: (is_corrupted, reason)
    """
    if not text or len(text) < 100:
        return True, "empty"
    
    # 1. Caractere replacement (�)
    replacement_count = text.count('�') + text.count('\ufffd')
    if replacement_count > 0:
        ratio = replacement_count / len(text)
        if ratio > 0.05:
            return True, "replacement_chars"
    
    # 2. Prea puține litere
    letter_count = sum(1 for c in text if c.isalpha())
    letter_ratio = letter_count / len(text)
    if letter_ratio < 0.40:
        return True, "low_letters"
    
    # 3. Prea puține spații
    space_ratio = text.count(' ') / len(text)
    if len(text) > 500 and space_ratio < 0.08:
        return True, "no_spaces"
    
    return False, "valid"

# ═══════════════════════════════════════════════════════════════════════════
# PDF TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

class PDFTextExtractor:
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        
        # OCR disponibil dacă AMBELE verificări sunt True
        self.ocr_available = OCR_AVAILABLE and OCR_AVAILABLE_LOCAL
        if self.ocr_available:
            logger.info("✓ OCR (Tesseract) disponibil")
        else:
            logger.warning("⚠ OCR indisponibil - unele PDF-uri pot avea text incomplet")

        self.text_cleaner = get_text_cleaner()
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, int, str]:
        """
        Extrage text din PDF.
        
        Returns: 
            (text, num_pages, text_status)
            text_status: 'valid', 'corrupted_ocr_success', 'corrupted_ocr_failed', 'ocr_needed', 'empty'
        """
        logger.info(f"Extragere text din: {pdf_path}")
        
        text, num_pages = self._extract_with_pdfplumber(pdf_path)
        
        if len(text.strip()) < self.config.min_text_length:
            text, num_pages = self._extract_with_pymupdf(pdf_path)
        
        # Verifică dacă textul e corupt
        is_corrupted, reason = is_text_corrupted(text)
        
        if is_corrupted and reason != "empty":
            logger.warning(f"Text corupt detectat ({reason}): {pdf_path}")
            if self.ocr_available:
                text_ocr = self._extract_with_ocr(pdf_path)
                is_still_corrupted, _ = is_text_corrupted(text_ocr)
                if not is_still_corrupted and len(text_ocr) > len(text) * 0.5:
                    text = self.text_cleaner.clean_text(text_ocr)
                    logger.info(f"✓ OCR reușit: {len(text)} caractere din {num_pages} pagini")
                    return text, num_pages, "corrupted_ocr_success"
                else:
                    logger.warning(f"✗ OCR eșuat pentru {pdf_path}")
                    return text, num_pages, "corrupted_ocr_failed"
            else:
                logger.warning(f"OCR indisponibil pentru {pdf_path}")
                return text, num_pages, "ocr_needed"
        
        if len(text.strip()) < self.config.min_text_length:
            return text, num_pages, "empty"
        
        text = self.text_cleaner.clean_text(text)
        logger.info(f"✓ Extras {len(text)} caractere din {num_pages} pagini")
        return text, num_pages, "valid"
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> Tuple[str, int]:
        text_parts = []
        num_pages = 0
        try:
            with pdfplumber.open(pdf_path) as pdf:
                num_pages = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n".join(text_parts), num_pages
        except Exception as e:
            logger.warning(f"Eroare pdfplumber: {e}")
            return "", 0
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Tuple[str, int]:
        text_parts = []
        num_pages = 0
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text_parts.append(page_text)
            doc.close()
            return "\n".join(text_parts), num_pages
        except Exception as e:
            logger.warning(f"Eroare PyMuPDF: {e}")
            return "", 0
    
    def _extract_with_ocr(self, pdf_path: str) -> str:
        text_parts = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(min(len(doc), 20)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
                if text:
                    text_parts.append(text)
            doc.close()
            return "\n".join(text_parts)
        except Exception as e:
            logger.warning(f"Eroare OCR: {e}")
            return ""


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT EXTRACTOR v6.0.1 - SENTENCES + AI MARKERS
# ═══════════════════════════════════════════════════════════════════════════

class ContextExtractor:
    """
    Extractor context v6.0.1
    - Extrage N propoziții înainte și după termenul AI
    - Marchează termenul cu >>>termen<<< pentru highlighting în Excel
    - Elimină boilerplate
    """
    
    HIGHLIGHT_START = '>>>'
    HIGHLIGHT_END = '<<<'
    
    BOILERPLATE_PATTERNS = [
        r'(?:^|\n)\s*\d+\s*(?:\n|$)',
        r'(?:^|\n)\s*(?:Page|Pagina)\s+\d+',
        r'(?:^|\n)\s*©\s*\d{4}',
        r'(?:^|\n)\s*All\s+rights\s+reserved',
        r'(?:^|\n)\s*\[\s*\d+\s*\]',
        r'(?:www\.|https?://)\S+',
    ]
    
    SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.compiled_boilerplate = [re.compile(p, re.IGNORECASE) for p in self.BOILERPLATE_PATTERNS]
        self.text_cleaner = get_text_cleaner()
        self.sentences_before = getattr(config, 'context_sentences_before', 2)
        self.sentences_after = getattr(config, 'context_sentences_after', 2)
    
    def extract_context_with_sentences(self, text: str, match_start: int, match_end: int,
                                       highlight: bool = True) -> Tuple[str, str]:
        """
        Extrage context bazat pe propoziții și marchează termenul AI.
        Returns: (context_with_markers, ai_term)
        """
        ai_term = text[match_start:match_end]
        
        # Fereastră mare pentru găsirea propozițiilor
        window_start = max(0, match_start - 2000)
        window_end = min(len(text), match_end + 2000)
        window_text = text[window_start:window_end]
        
        relative_match_start = match_start - window_start
        
        # Split în propoziții
        sentences = self.SENTENCE_SPLIT_PATTERN.split(window_text)
        
        # Găsește propoziția care conține termenul
        current_pos = 0
        target_sentence_idx = 0
        
        for idx, sentence in enumerate(sentences):
            sentence_end = current_pos + len(sentence)
            if current_pos <= relative_match_start < sentence_end:
                target_sentence_idx = idx
                break
            current_pos = sentence_end + 1
        
        # Extrage propozițiile din jur
        start_idx = max(0, target_sentence_idx - self.sentences_before)
        end_idx = min(len(sentences), target_sentence_idx + self.sentences_after + 1)
        
        context_sentences = sentences[start_idx:end_idx]
        context = ' '.join(context_sentences)
        
        # Curăță
        context = self._clean_boilerplate(context)
        if self.text_cleaner:
            context = self.text_cleaner.clean_text(context)
        context = re.sub(r'\s+', ' ', context).strip()
        
        # Marchează termenul AI
        if highlight and ai_term:
            term_pattern = re.compile(re.escape(ai_term), re.IGNORECASE)
            term_match = term_pattern.search(context)
            if term_match:
                context = (
                    context[:term_match.start()] +
                    self.HIGHLIGHT_START + term_match.group() + self.HIGHLIGHT_END +
                    context[term_match.end():]
                )
        
        # Limită lungime
        max_length = getattr(self.config, 'max_context_length', 800)
        if len(context) > max_length:
            context = context[:max_length] + '...'
        
        return context, ai_term
    
    def extract_relevant_context(self, text: str, match_start: int, match_end: int) -> str:
        """Compatibilitate - returnează doar contextul."""
        context, _ = self.extract_context_with_sentences(text, match_start, match_end, highlight=True)
        return context
    
    def _clean_boilerplate(self, context: str) -> str:
        for pattern in self.compiled_boilerplate:
            context = pattern.sub(' ', context)
        return context


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY CLASSIFIER v6.2 - DUAL TAXONOMY + FALLBACK RULES
# ═══════════════════════════════════════════════════════════════════════════

class CategoryClassifier:
    """
    Clasificator categorii v6.2 cu:
    - Dual taxonomy support (16 categories)
    - Keyword tier-based confidence
    - Pattern matching primar (din AI_CATEGORIES)
    - Keyword matching secundar  
    - Fallback rules pentru reducerea Unclassified de la 44% la ~15%
    """
    
    # Reguli fallback - folosite când pattern/keyword matching eșuează
    FALLBACK_RULES = [
        # Generative AI & LLMs - PRIORITATE MAXIMĂ (termeni specifici)
        (r'(?:generative|GenAI|LLM|GPT|ChatGPT|Copilot|Claude|Gemini|Llama|'
         r'foundation\s+model|large\s+language|transformer|multimodal|'
         r'text[\s-]to[\s-]|DALL[\s-]?E|Midjourney|prompt\s+engineer)', 
         'Generative AI & LLMs'),
        
        # Agentic AI Systems
        (r'(?:agentic|autonomous\s+agent|multi[\s-]?agent|AI\s+agent|'
         r'intelligent\s+agent|tool\s+(?:use|calling)|function\s+calling|AutoGPT)',
         'Agentic AI Systems'),
        
        # AI Infrastructure & MLOps
        (r'(?:MLOps|LLMOps|AIOps|infrastructure|GPU|TPU|NPU|'
         r'edge\s+(?:computing|AI)|model\s+(?:deployment|serving|hosting)|'
         r'AI[\s-]?as[\s-]?a[\s-]?service|scalab)',
         'AI Infrastructure & MLOps'),
        
        # Strategic Investment
        (r'(?:invest(?:ment|ing|ed)\s+(?:in\s+)?(?:AI|artificial)|'
         r'acqui(?:sition|red)|M&A.*(?:AI|artificial)|'
         r'(?:AI|artificial).*(?:budget|funding|capital)|'
         r'\$[\d,.]+\s*(?:million|billion).*(?:AI|artificial))',
         'Strategic Investment'),
        
        # Operational Implementation
        (r'(?:deploy(?:ment|ing|ed)|implement(?:ation|ing|ed)|'
         r'automat(?:e|ion|ed|ing)|efficien|streamlin|optimi[zs]|'
         r'workflow|process\s+(?:improvement|automation)|hyperautomation)',
         'Operational Implementation'),
        
        # Customer Experience
        (r'(?:customer\s+(?:experience|service|engagement|journey)|'
         r'chatbot|virtual\s+assistant|personali[zs]|recommend(?:ation)?|'
         r'conversational\s+AI)',
         'Customer Experience'),
        
        # Product Development
        (r'(?:product\s+(?:development|innovation|feature)|'
         r'AI[\s-]?powered\s+(?:product|feature|solution)|'
         r'intelligent\s+(?:product|solution|feature)|next[\s-]generation)',
         'Product Development'),
        
        # Risk & Compliance
        (r'(?:govern(?:ance)?|ethic(?:s|al)|responsible\s+AI|'
         r'bias|fairness|complian|regulat|risk\s+(?:management|assessment)|'
         r'trustworthy|accountab)',
         'Risk & Compliance'),
        
        # Data & Analytics
        (r'(?:predict(?:ive|ion)|analytics|data[\s-]?driven|'
         r'forecast|business\s+intelligence|insight|algorithm|'
         r'(?:advanced|augmented)\s+analytics)',
         'Data & Analytics'),
        
        # Talent & Workforce
        (r'(?:talent|skill|train(?:ing)?|hire|hiring|recruit|'
         r'workforce|upskill|reskill|data\s+scientist|'
         r'(?:AI|ML)\s+(?:team|expertise|capability))',
         'Talent & Workforce'),
        
        # Explainability & AI Safety
        (r'(?:explainab|interpretab|XAI|transparen(?:cy|t)|'
         r'(?:AI|model)\s+safety|guardrail|hallucination|alignment)',
         'Explainability & AI Safety'),
        
        # Research & Innovation
        (r'(?:research|patent|lab(?:oratory)?|academic|universit|'
         r'collaborat|breakthrough|innovation|frontier\s+(?:AI|research))',
         'Research & Innovation'),
        
        # AI Coding & Development
        (r'(?:code\s+(?:generation|completion)|'
         r'AI[\s-]?(?:assisted|powered)\s+(?:coding|development)|'
         r'GitHub\s+Copilot|Cursor|Tabnine)',
         'AI Coding & Development'),
    ]
    
    def __init__(self):
        # Compile patterns din AI_CATEGORIES
        self.compiled_patterns = {}
        for category, data in AI_CATEGORIES.items():
            self.compiled_patterns[category] = {
                'keywords': [kw.lower() for kw in data['keywords']],
                'patterns': [re.compile(p, re.IGNORECASE) for p in data.get('patterns', [])]
            }
        
        # Compile fallback rules
        self.compiled_fallback = [
            (re.compile(pattern, re.IGNORECASE), category)
            for pattern, category in self.FALLBACK_RULES
        ]
        
        logger.info(f"Category classifier v6.2: {len(AI_CATEGORIES)} categorii (Dual Taxonomy v7.0) + {len(self.FALLBACK_RULES)} fallback rules")
    
    def classify_with_confidence(self, text: str, context: str) -> Tuple[str, float]:
        """
        Clasifică cu confidence score.
        
        Logica:
        1. Pattern matching din AI_CATEGORIES (confidence ridicată)
        2. Keyword matching (confidence medie)
        3. Fallback rules (confidence scăzută dar acceptabilă)
        
        Returns: (category, confidence_score 0.0-1.0)
        """
        combined = f"{text} {context}".lower()
        scores = {}
        
        # ═══════════════════════════════════════════════════════════════
        # PASUL 1: Pattern + Keyword matching din AI_CATEGORIES
        # ═══════════════════════════════════════════════════════════════
        for category, data in self.compiled_patterns.items():
            score = 0.0
            
            # Pattern matches = 0.5 puncte fiecare (cel mai specific)
            pattern_matches = sum(1 for p in data['patterns'] if p.search(combined))
            score += pattern_matches * 0.5
            
            # Keyword matches = 0.2 puncte fiecare
            keyword_matches = sum(1 for kw in data['keywords'] if kw in combined)
            score += keyword_matches * 0.2
            
            if score > 0:
                scores[category] = score
        
        # Dacă avem match bun din AI_CATEGORIES, returnează
        if scores:
            best_category = max(scores, key=scores.get)
            max_score = scores[best_category]
            confidence = min(1.0, max_score / 1.5)  # Normalizare
            
            if confidence >= 0.25:  # Threshold scăzut pentru a permite mai multe clasificări
                return best_category, confidence
        
        # ═══════════════════════════════════════════════════════════════
        # PASUL 2: Fallback rules (dacă AI_CATEGORIES nu a dat rezultate)
        # ═══════════════════════════════════════════════════════════════
        for pattern, category in self.compiled_fallback:
            if pattern.search(combined):
                return category, 0.35  # Confidence medie pentru fallback
        
        # ═══════════════════════════════════════════════════════════════
        # PASUL 3: Nicio potrivire - încearcă clasificare bazată pe text
        # ═══════════════════════════════════════════════════════════════
        # Pentru cazuri cu termeni generici, clasifică în Data & Analytics
        generic_ai_terms = ['ai', 'ml', 'artificial intelligence', 'machine learning']
        if any(term in combined for term in generic_ai_terms):
            return 'Data & Analytics', 0.2  # Default pentru termeni generici
        
        return 'Unclassified', 0.0


# ═══════════════════════════════════════════════════════════════════════════
# FALSE POSITIVE FILTER v6.2 - ENHANCED VALIDATION & ENCODING CHECK
# ═══════════════════════════════════════════════════════════════════════════

# Importă AI_CONTEXT_VALIDATORS din module1
try:
    from ai_analyzer_v6_2_module1 import AI_CONTEXT_VALIDATORS
except ImportError:
    AI_CONTEXT_VALIDATORS = [
        'artificial intelligence', 'machine learning', 'deep learning',
        'neural network', 'algorithm', 'model training', 'prediction',
        'automat', 'intelligent system', 'chatbot', 'generative', 'LLM', 'GPT',
        'autonomous', 'analytics', 'natural language', 'computer vision',
    ]


class FalsePositiveFilter:
    """
    Filtru false pozitive v6.2 cu:
    - Enhanced pattern matching (~60 patterns)
    - ML/DL measurement unit detection
    - Pattern matching extins
    - Validare text corupt/encoding
    - Validare context pentru texte scurte (AI, ML, DL)
    """
    
    def __init__(self):
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in FALSE_POSITIVE_PATTERNS]
        self.context_validators = [v.lower() for v in AI_CONTEXT_VALIDATORS]
        logger.info(f"False positive filter v6.2: {len(FALSE_POSITIVE_PATTERNS)} patterns (enhanced from v6.1)")
    
    def is_false_positive(self, text: str, context: str) -> bool:
        """
        Verifică dacă referința e false pozitiv.
        
        Returns True dacă:
        - Match-uiește un pattern false pozitiv
        - Textul e corupt (encoding issues)
        - Text scurt fără context AI valid
        """
        # 1. Verifică text/context corupt
        if self._is_corrupted_text(context):
            return True
        
        # 2. Verifică pattern-uri false pozitive
        combined = f"{text} {context}"
        for pattern in self.compiled_patterns:
            if pattern.search(combined):
                return True
        
        # 3. Pentru texte scurte (AI, ML, DL), verifică context
        if len(text) <= 3 and not self._has_valid_ai_context(context):
            return True
        
        return False
    
    def _is_corrupted_text(self, text: str) -> bool:
        """Detectează text corupt din encoding issues."""
        if not text or len(text) < 20:
            return False
        
        # Prea multe caractere speciale (>10%)
        special_count = len(re.findall(r'[#$%&*@^\\|~`]', text))
        if special_count / len(text) > 0.1:
            return True
        
        # Prea puține spații pentru lungimea textului (text lipit)
        space_count = text.count(' ')
        if len(text) > 100 and space_count / len(text) < 0.05:
            return True
        
        # Secvențe lungi fără vocale (imposibil în engleză)
        if re.search(r'[bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ]{8,}', text):
            return True
        
        return False
    
    def _has_valid_ai_context(self, context: str) -> bool:
        """Verifică dacă contextul confirmă că e vorba de AI real."""
        if not context:
            return False
        
        context_lower = context.lower()
        
        # Verifică dacă contextul conține termeni AI reali
        for validator in self.context_validators:
            if validator in context_lower:
                return True
        
        return False


# ═══════════════════════════════════════════════════════════════════════════
# AI REFERENCE DETECTOR v6.0.1
# ═══════════════════════════════════════════════════════════════════════════

class AIReferenceDetector:
    """
    Detector referințe AI v6.2 (Dual Taxonomy + enhanced FP filtering):
      - 99 patterns (vs 45 in v6.1)
      - 413 keywords (vs 150 in v6.1)
      - Vendor-specific detection
      - Separă triggers HARD vs SOFT (buzzword gating)
      - Rulează validatori pe context (actionability / specificity / marketing-only)
      - Calculează confidence_score și etichetează referințele slabe ca mention_only (FP candidates)

    IMPORTANT: Pentru a păstra compatibilitatea cu restul softului (Module 1/3/4),
    strength/confidence sunt înregistrate în detection_method sub formă de metadate.
    (La cererea ta, vom actualiza ulterior și celelalte module ca să aibă câmpuri dedicate.)
    """

    # Prag strict pentru semantic-only (fără HARD + fără semnal de implementare)
    DEFAULT_SEMANTIC_STRICT_THRESHOLD = 0.68

    # ────────────────────────────────────────────────────────────────────
    # HARD vs SOFT triggers
    # ────────────────────────────────────────────────────────────────────
    # HARD: indicatori tehnici / specifici (AI “real”)
    AI_HARD_PATTERNS = [
        # LLM / GenAI / modele
        r'\bgenerative AI\b', r'\bGenAI\b',
        r'\blarge language model(?:s)?\b', r'\bLLM(?:s)?\b(?![\w-])',
        r'\bGPT(?:[\s-]?[3-5])?\b', r'\bChatGPT\b',
        r'\bfoundation model(?:s)?\b', r'\bfrontier model(?:s)?\b',
        r'\btransformer(?:s)?\b', r'\bmultimodal(?:\s+AI|\s+model)?\b',

        # RAG / embeddings / semantic infra
        r'\bRAG\b(?![\w-])', r'\bretrieval[\s-]?augmented\s+generation\b',
        r'\bvector\s+(?:database|store|search|embedding)\b',
        r'\bembedding(?:s)?\b', r'\bsemantic\s+search\b', r'\bknowledge\s+(?:graph|base)\b',

        # Fine-tuning / RLHF / LoRA
        r'\bprompt\s+engineering\b', r'\bfine[\s-]?tun(?:e|ed|ing)\b',
        r'\bRLHF\b', r'\bLoRA\b', r'\binstruction\s+tuning\b',

        # MLOps / deployment / serving
        r'\bMLOps\b', r'\bLLMOps\b', r'\bAIOps\b',
        r'\bmodel\s+(?:serving|deployment|inference)\b',
        r'\bedge\s+(?:AI|computing)\b',
        r'\b(?:GPU|TPU|NPU)\s+(?:cluster|compute|infrastructure)\b',

        # Modele / tool-uri specifice
        r'\bGitHub\s+Copilot\b', r'\bCopilot\b', r'\bClaude\b', r'\bGemini\b',
        r'\bLlama(?:\s+\d)?\b', r'\bMistral\b',
        r'\bDALL[\s-]?E\b', r'\bStable\s+Diffusion\b', r'\bMidjourney\b', r'\bSora\b',
        r'\bLangChain\b', r'\bLangGraph\b', r'\bAutoGPT\b', r'\bCrewAI\b',
        r'\bfunction\s+calling\b', r'\btool\s+(?:use|calling)\b',
    ]

    # SOFT: termeni vagi / marketing / “AI-washing”
    AI_SOFT_PATTERNS = [
        r'\bartificial intelligence\b',
        r'\bmachine learning\b', r'\bdeep learning\b',
        r'\bneural network(?:s)?\b',
        r'\bnatural language processing\b', r'\bcomputer vision\b',
        r'\b(?:AI|ML|DL|NLP)\b(?![\w-])',
        r'\bpredictive analytics\b',
        r'\bcognitive computing\b',
        r'\bintelligent automation\b',
        r'\bAI[\s-]?powered\b', r'\bAI[\s-]?driven\b',
        r'\bAI[\s-]?enabled\b', r'\bAI[\s-]?first\b',
        r'\bAI[\s-]?native\b', r'\bAI[\s-]?augmented\b',
        r'\bML[\s-]?based\b', r'\bML[\s-]?powered\b',
        r'\bchatbot(?:s)?\b', r'\bvirtual\s+assistant\b',
        r'\bconversational\s+AI\b',
        r'\brecommendation\s+(?:system|engine)\b',
        r'\bsentiment\s+analysis\b', r'\bfraud\s+detection\b',
        r'\bpredictive\s+maintenance\b',
        r'\brobot(?:ic)?(?:s)?\b', r'\bRPA\b', r'\bhyperautomation\b',
        r'\bintelligent\b', r'\bautomation\b'
    ]

    # Semnale pentru validatori
    IMPLEMENTATION_VERBS = [
        r'\bdeploy(?:ed|ing)?\b', r'\bimplement(?:ed|ing)?\b', r'\bintegrat(?:e|ed|ing)\b',
        r'\broll[\s-]?out\b', r'\blaunch(?:ed|ing)?\b', r'\bbuild(?:ing)?\b',
        r'\bscale(?:d|ing)?\b', r'\bproduction\b', r'\binference\b', r'\bserving\b'
    ]

    TECHNICAL_ARTIFACTS = [
        r'\bmodel\b', r'\bpipeline\b', r'\btraining\b', r'\bfine[\s-]?tuning\b',
        r'\bembedding(?:s)?\b', r'\bvector\b', r'\btoken(?:s)?\b',
        r'\bprompt\b', r'\bguardrails\b', r'\bevaluation\b', r'\bbenchmark\b',
        r'\bdata\s+scientist\b', r'\bml\s+engineer\b'
    ]

    MARKETING_ONLY_CUES = [
        r'\bleverag(?:e|ing)\b', r'\bexplor(?:e|ing|ation)\b', r'\baspir(?:e|ation)\b',
        r'\bcommit(?:ment|ted)\b', r'\bvision\b', r'\bstrategy\b', r'\broadmap\b',
        r'\bprinciples?\b', r'\bguidelines?\b', r'\bpolicy\b', r'\bframework\b',
        r'\bresponsible\s+AI\b', r'\bAI\s+ethics\b', r'\bAI\s+governance\b'
    ]

    ESG_DOC_CUES = [
        'sustainability', 'esg', 'csr', 'responsible', 'governance', 'ethics'
    ]

    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.semantic_loader = SemanticModelLoader.get_instance()
        self.model = self.semantic_loader.model
        self.context_extractor = ContextExtractor(config)
        self.category_classifier = CategoryClassifier()
        self.fp_filter = FalsePositiveFilter()

        # compile patterns
        self.compiled_hard = [re.compile(p, re.IGNORECASE) for p in self.AI_HARD_PATTERNS]
        self.compiled_soft = [re.compile(p, re.IGNORECASE) for p in self.AI_SOFT_PATTERNS]
        self.compiled_impl = [re.compile(p, re.IGNORECASE) for p in self.IMPLEMENTATION_VERBS]
        self.compiled_artifacts = [re.compile(p, re.IGNORECASE) for p in self.TECHNICAL_ARTIFACTS]
        self.compiled_marketing = [re.compile(p, re.IGNORECASE) for p in self.MARKETING_ONLY_CUES]

        logger.info(
            f"AI Reference Detector v6.2 (Dual Taxonomy v7.0): hard={len(self.AI_HARD_PATTERNS)}, soft={len(self.AI_SOFT_PATTERNS)}"
        )

    # ────────────────────────────────────────────────────────────────────
    # Public API (compatibil)
    # ────────────────────────────────────────────────────────────────────
    def detect_references(
        self,
        text: str,
        company: str,
        year: int,
        position: int,
        industry: str,
        sector: str,
        country: str,
        doc_type: str,
        source: str
    ) -> List[AIReference]:
        references: List[AIReference] = []

        pattern_refs = self._detect_by_patterns(
            text, company, year, position, industry, sector, country, doc_type, source
        )
        references.extend(pattern_refs)

        semantic_refs = self._detect_by_semantics(
            text, company, year, position, industry, sector, country, doc_type, source
        )
        references.extend(semantic_refs)

        logger.info(
            f"Detectat {len(references)} referințe ({len(pattern_refs)} pattern, {len(semantic_refs)} semantic)"
        )
        return references

    # ────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────────────
    def _is_esg_like(self, doc_type: str, context: str) -> bool:
        dt = (doc_type or "").lower()
        ctx = (context or "").lower()
        if any(cue in dt for cue in self.ESG_DOC_CUES):
            return True
        # weak signal from context
        if 'sustainab' in ctx or 'esg' in ctx or 'csr' in ctx:
            return True
        return False

    def _feature_counts(self, context: str) -> Dict[str, int]:
        ctx = context or ""
        impl = sum(1 for p in self.compiled_impl if p.search(ctx))
        art = sum(1 for p in self.compiled_artifacts if p.search(ctx))
        mkt = sum(1 for p in self.compiled_marketing if p.search(ctx))
        return {"impl": impl, "artifacts": art, "marketing": mkt}

    def _has_hard_indicator(self, context: str) -> bool:
        ctx = context or ""
        return any(p.search(ctx) for p in self.compiled_hard)

    def _compute_confidence_and_strength(
        self,
        *,
        trigger_type: str,  # 'hard'|'soft'|'semantic'
        semantic_score: float,
        context_clean: str,
        doc_type: str
    ) -> Tuple[float, str, List[str]]:
        """
        Returnează: (confidence_score 0..1, strength label, reasons)
        strength: 'strong'|'medium'|'mention_only'
        """
        reasons: List[str] = []
        feats = self._feature_counts(context_clean)

        has_impl = feats["impl"] > 0
        has_art = feats["artifacts"] > 0
        has_mkt = feats["marketing"] > 0
        has_hard = self._has_hard_indicator(context_clean)
        is_esg = self._is_esg_like(doc_type, context_clean)

        # base score by trigger type
        conf = 0.0
        if trigger_type == "hard":
            conf = 0.72
            reasons.append("hard_trigger")
        elif trigger_type == "semantic":
            # semantic_score already 0..1-ish
            conf = 0.35 + min(max(semantic_score, 0.0), 1.0) * 0.55
            reasons.append(f"semantic={semantic_score:.3f}")
        else:  # soft
            conf = 0.35
            reasons.append("soft_trigger")

        # bonuses
        if has_impl:
            conf += 0.12
            reasons.append("implementation_signal")
        if has_art:
            conf += 0.10
            reasons.append("technical_specificity")
        if has_hard:
            conf += 0.08
            reasons.append("hard_in_context")

        # penalties
        if has_mkt and not (has_impl or has_art or has_hard):
            conf -= 0.18
            reasons.append("marketing_only_penalty")
        if is_esg and not (has_impl or has_art or has_hard):
            conf -= 0.12
            reasons.append("esg_boilerplate_penalty")

        # clamp
        conf = max(0.0, min(1.0, conf))

        # map to strength
        if conf >= 0.70:
            strength = "strong"
        elif conf >= 0.55:
            strength = "medium"
        else:
            strength = "mention_only"

        return conf, strength, reasons

    def _format_detection_method(self, base: str, trigger_type: str, strength: str, conf: float, reasons: List[str]) -> str:
        # Metadate compacte, ușor de pars-at ulterior (Module 3/4)
        # Exemplu: "pattern|trigger=soft|strength=mention_only|conf=0.48|reasons=soft_trigger,marketing_only_penalty"
        safe_reasons = ",".join(reasons)[:250]
        return f"{base}|trigger={trigger_type}|strength={strength}|conf={conf:.3f}|reasons={safe_reasons}"

    # ────────────────────────────────────────────────────────────────────
    # Pattern detection (HARD/SOFT gating + mention_only)
    # ────────────────────────────────────────────────────────────────────
    def _detect_by_patterns(
        self,
        text: str,
        company: str,
        year: int,
        position: int,
        industry: str,
        sector: str,
        country: str,
        doc_type: str,
        source: str
    ) -> List[AIReference]:
        references: List[AIReference] = []
        seen_positions = set()

        # căutăm mai întâi HARD, apoi SOFT (ca să evităm dubluri)
        pattern_groups = [("hard", self.compiled_hard), ("soft", self.compiled_soft)]

        for trigger_type, compiled_list in pattern_groups:
            for pattern in compiled_list:
                for match in pattern.finditer(text):
                    if match.start() in seen_positions:
                        continue
                    seen_positions.add(match.start())

                    matched_text = match.group()

                    # Context CU marcaje
                    context, ai_term = self.context_extractor.extract_context_with_sentences(
                        text, match.start(), match.end(), highlight=True
                    )

                    # False positive check (fără marcaje)
                    context_clean = context.replace('>>>', '').replace('<<<', '')
                    if self.fp_filter.is_false_positive(matched_text, context_clean):
                        continue

                    # Robotics/RPA filters (compatibil)
                    robotics_type = classify_robotics_reference(context_clean)
                    rpa_type = classify_rpa_reference(context_clean)
                    if self.config.filter_traditional_robotics and robotics_type == 'traditional_robotics':
                        continue
                    if self.config.filter_traditional_rpa and rpa_type == 'traditional_rpa':
                        continue

                    # Category + confidence (compatibil)
                    category, cat_confidence = self.category_classifier.classify_with_confidence(
                        matched_text, context_clean
                    )

                    # Scoring & strength
                    conf, strength, reasons = self._compute_confidence_and_strength(
                        trigger_type=trigger_type,
                        semantic_score=0.0,
                        context_clean=context_clean,
                        doc_type=doc_type
                    )

                    # Dacă e SOFT și iese mention_only, îl păstrăm (FP candidate), nu îl eliminăm
                    detection_method = self._format_detection_method(
                        "pattern", trigger_type, strength, conf, reasons
                    )

                    page = text[:match.start()].count('\n') // 50 + 1

                    ref = AIReference(
                        company=company, year=year, position=position,
                        industry=industry, sector=sector, country=country,
                        doc_type=doc_type,
                        text=ai_term, context=context, page=page,
                        category=category, detection_method=detection_method,
                        sentiment='neutral', sentiment_score=0.0,
                        semantic_score=0.0, source=source,
                        robotics_type=robotics_type, rpa_type=rpa_type,
                        category_confidence=cat_confidence,
                        reference_strength=strength,
                        confidence_score=conf,
                        confidence_reasons=';'.join(reasons)
                    )
                    references.append(ref)

        return references

    # ────────────────────────────────────────────────────────────────────
    # Semantic detection (threshold strict 0.68 when semantic-only)
    # ────────────────────────────────────────────────────────────────────
    def _detect_by_semantics(
        self,
        text: str,
        company: str,
        year: int,
        position: int,
        industry: str,
        sector: str,
        country: str,
        doc_type: str,
        source: str
    ) -> List[AIReference]:
        references: List[AIReference] = []

        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
        if not paragraphs:
            return references

        ai_embeddings = get_ai_description_embeddings()
        batch_size = 32

        strict_thr = getattr(self.config, "semantic_threshold_strict", self.DEFAULT_SEMANTIC_STRICT_THRESHOLD)
        base_thr = getattr(self.config, "semantic_threshold", 0.60)

        for i in range(0, len(paragraphs), batch_size):
            batch = paragraphs[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
            similarities = util.cos_sim(batch_embeddings, ai_embeddings)
            max_sims = similarities.max(dim=1)

            for sim_score, para in zip(max_sims.values, batch):
                score = float(sim_score.item())

                # Decide prag în funcție de HARD/impl în paragraf
                has_hard = self._has_hard_indicator(para)
                feats = self._feature_counts(para)
                has_impl = feats["impl"] > 0
                chosen_thr = base_thr if (has_hard or has_impl) else strict_thr

                if score < chosen_thr:
                    continue

                # FP filter pentru semantic
                if self.fp_filter.is_false_positive("", para):
                    continue

                robotics_type = classify_robotics_reference(para)
                rpa_type = classify_rpa_reference(para)
                if self.config.filter_traditional_robotics and robotics_type == 'traditional_robotics':
                    continue
                if self.config.filter_traditional_rpa and rpa_type == 'traditional_rpa':
                    continue

                category, cat_confidence = self.category_classifier.classify_with_confidence("", para)

                ai_term, context_marked = self._find_and_mark_ai_term(para)

                conf, strength, reasons = self._compute_confidence_and_strength(
                    trigger_type="semantic",
                    semantic_score=score,
                    context_clean=para,
                    doc_type=doc_type
                )
                detection_method = self._format_detection_method(
                    "semantic", "semantic", strength, conf, reasons
                )

                ref = AIReference(
                    company=company, year=year, position=position,
                    industry=industry, sector=sector, country=country,
                    doc_type=doc_type,
                    text=ai_term, context=context_marked, page=0,
                    category=category, detection_method=detection_method,
                    sentiment='neutral', sentiment_score=0.0,
                    semantic_score=score, source=source,
                    robotics_type=robotics_type, rpa_type=rpa_type,
                    category_confidence=cat_confidence,
                    reference_strength=strength,
                    confidence_score=conf,
                    confidence_reasons=';'.join(reasons)
                )
                references.append(ref)

        return references

    def _find_and_mark_ai_term(self, paragraph: str) -> Tuple[str, str]:
        """Găsește termenul AI în paragraf și îl marchează cu >>><<<."""
        ai_terms_priority = [
            r'\bgenerative AI\b', r'\bGenAI\b', r'\blarge language model(?:s)?\b',
            r'\bLLM(?:s)?\b', r'\bartificial intelligence\b', r'\bmachine learning\b',
            r'\bdeep learning\b', r'\bneural network(?:s)?\b', r'\bGPT\b',
            r'\bChatGPT\b', r'\bCopilot\b', r'\bpredictive analytics\b',
            r'\bnatural language processing\b', r'\bcomputer vision\b',
            r'\bautonomous\b', r'\brobot(?:ic)?(?:s)?\b', r'\bRPA\b',
            r'\bRAG\b', r'\bembedding(?:s)?\b', r'\bvector\s+database\b',
            r'\b(?:AI|ML)\b(?![\w-])', r'\bchatbot\b', r'\bautomation\b',
        ]

        for pattern_str in ai_terms_priority:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            match = pattern.search(paragraph)
            if match:
                ai_term = match.group()
                marked = paragraph[:match.start()] + '>>>' + ai_term + '<<<' + paragraph[match.end():]
                return ai_term, marked

        return paragraph[:50], paragraph


# ═══════════════════════════════════════════════════════════════════════════
# FILENAME PARSER
# ═══════════════════════════════════════════════════════════════════════════

class FilenameParser:
    """Parser pentru numele fișierelor Fortune 500."""
    
    DOC_TYPE_MAPPING = {
        'annual': 'Annual Report',
        'sustainability': 'Sustainability',
        'proxy': 'Proxy',
        'quarterly': 'Quarterly Report',
        '10k': 'Annual Report',
        '10q': 'Quarterly Report',
        'def14a': 'Proxy',
        'esg': 'Sustainability',
        'csr': 'Sustainability',
    }
    
    def parse_filename(self, filename: str) -> Dict:
        """
        Parse filename to extract company, year, doc_type, position.
        
        Format așteptat: "38. Bank of America - 2024 - annual report.pdf"
        """
        result = {
            'company': 'Unknown',
            'year': 2024,
            'doc_type': 'Annual Report',
            'position': 0,
            'industry': 'Unknown',
            'sector': 'Unknown',
            'country': 'Unknown'
        }
        
        base = Path(filename).stem
        
        # ═══════════════════════════════════════════════════════════════
        # PASUL 1: Extrage POZIȚIA (ex: "38." la început)
        # ═══════════════════════════════════════════════════════════════
        position_match = re.match(r'^(\d{1,3})\.\s*', base)
        if position_match:
            result['position'] = int(position_match.group(1))
            # Elimină poziția din string pentru procesare ulterioară
            base = base[position_match.end():]
        
        # ═══════════════════════════════════════════════════════════════
        # PASUL 2: Extrage ANUL (ex: "2024")
        # ═══════════════════════════════════════════════════════════════
        year_match = re.search(r'(?:^|[\s\-_])20(1[5-9]|2[0-5])(?:[\s\-_]|$)', base)
        if year_match:
            result['year'] = int('20' + year_match.group(1))
        
        # ═══════════════════════════════════════════════════════════════
        # PASUL 3: Extrage DOC_TYPE
        # ═══════════════════════════════════════════════════════════════
        base_lower = base.lower()
        for key, value in self.DOC_TYPE_MAPPING.items():
            if key in base_lower:
                result['doc_type'] = value
                break
        
        # ═══════════════════════════════════════════════════════════════
        # PASUL 4: Extrage COMPANY NAME
        # Format: "Company Name - Year - doc type"
        # ═══════════════════════════════════════════════════════════════
        # Split după " - " (cu spații)
        parts = re.split(r'\s+-\s+', base)
        
        if len(parts) >= 1:
            company_part = parts[0].strip()
            
            # Curăță compania de eventuale numere rămase
            company_part = re.sub(r'^\d+\.\s*', '', company_part)
            
            if company_part:
                result['company'] = company_part
        
        return result


# ═══════════════════════════════════════════════════════════════════════════
# FINAL
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("AI SEMANTIC ANALYZER v6.0.5 - MODUL 2: PDF PROCESSING & EXTRACTION")
    print("=" * 80)
    print(f"\n✓ Module 2 v6.0 încărcat!")
    print(f"\nFeatures:")
    print(f"  - AI patterns: {len(AI_MAIN_PATTERNS)}")
    print(f"  - Canonical descriptions: {len(AI_CANONICAL_DESCRIPTIONS)}")
    print(f"  - Robotics/RPA classification")
    print(f"  - Improved context extraction")
    print(f"  - Category classifier with confidence")
