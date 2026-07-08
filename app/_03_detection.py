"""
═══════════════════════════════════════════════════════════════════════════════
AI SEMANTIC ANALYZER - AI DETECTION, CONTEXT & FP REDUCTION
═══════════════════════════════════════════════════════════════════════════════

PDF processing, AI reference detection, dual-taxonomy classification, and
false-positive filtering.

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
       - Negation filtering (drop hard / down-weight soft negations)
       - Confidence scoring with keyword tiers

    4. Category Classification
       - Classic taxonomy (CategoryClassifier): keyword + pattern + fallback
       - EU_Semantics taxonomy (EUCategoryClassifier), in parallel

    5. False Positive Filtering
       - ~60 exclusion patterns
       - Text corruption detection
       - Short text validators (AI, ML, DL)
       - Measurement unit exclusions

WORKFLOW:
    PDF -> Extract Text -> Detect AI References -> Classify Categories
    -> Filter False Positives -> Extract Context -> Return AIReference objects

TECHNICAL NOTES:
    - Semantic threshold: 0.60 (general), 0.68 (strict for mention-only)
    - Multi-threading safe: SemanticModelLoader singleton
    - Memory efficient: Lazy model loading

Author: TeRa0
Part of: AI Semantic Analyzer
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import time
import re
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

# Verify tesseract binary is on PATH; pytesseract being importable is not enough.
# Without this check, pytesseract.image_to_string() fails per-PDF instead of cleanly skipping.
if OCR_AVAILABLE_LOCAL:
    import shutil as _shutil
    if _shutil.which("tesseract") is None:
        OCR_AVAILABLE_LOCAL = False

try:
    import wordsegment
    wordsegment.load()
    WORDSEGMENT_AVAILABLE = True
except ImportError:
    WORDSEGMENT_AVAILABLE = False

from _02_core import (
    logger, AnalyzerConfig, AIReference, DocumentResult,
    AI_CATEGORIES, AI_APPLICATIONS, AI_TECHNOLOGIES,
    EU_CATEGORIES, FALLBACK_NAME_TO_CODE,
    FALSE_POSITIVE_PATTERNS, OCR_AVAILABLE,
    TRADITIONAL_ROBOTICS_PATTERNS, AI_ROBOTICS_PATTERNS,
    RPA_NON_AI_PATTERNS, RPA_AI_PATTERNS
)

import warnings
import logging

# Suppress noisy messages from PDF processing
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
            logger.info(f"Loading semantic model: {self.MODEL_NAME}")
            start_time = time.time()
            SemanticModelLoader._model = SentenceTransformer(self.MODEL_NAME)
            logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
    
    @property
    def model(self):
        return SemanticModelLoader._model


# ═══════════════════════════════════════════════════════════════════════════
# AI MAIN PATTERNS - 80+ PATTERNS
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
    
    # Specific models
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
# AI CANONICAL DESCRIPTIONS - 65+ DESCRIPTIONS
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
        logger.info("Computing embeddings for canonical AI descriptions...")
        _AI_DESCRIPTION_EMBEDDINGS = loader.model.encode(
            AI_CANONICAL_DESCRIPTIONS, convert_to_tensor=True, show_progress_bar=False
        )
    return _AI_DESCRIPTION_EMBEDDINGS


# ═══════════════════════════════════════════════════════════════════════════
# ROBOTICS & RPA CLASSIFICATION
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
# TEXT CORRUPTION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

def is_text_corrupted(text: str) -> tuple[bool, str]:
    """
    Check whether the extracted text is corrupted/unreadable.
    Returns: (is_corrupted, reason)
    """
    if not text or len(text) < 100:
        return True, "empty"
    
    # 1. Replacement characters
    replacement_count = text.count('�') + text.count('\ufffd')
    if replacement_count > 0:
        ratio = replacement_count / len(text)
        if ratio > 0.05:
            return True, "replacement_chars"
    
    # 2. Too few letters
    letter_count = sum(1 for c in text if c.isalpha())
    letter_ratio = letter_count / len(text)
    if letter_ratio < 0.40:
        return True, "low_letters"
    
    # 3. Too few spaces
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

        # OCR available only if BOTH checks are True
        self.ocr_available = OCR_AVAILABLE and OCR_AVAILABLE_LOCAL
        if self.ocr_available:
            logger.info("OCR (Tesseract) available")
        else:
            logger.warning("OCR unavailable - some PDFs may have incomplete text")

        self.text_cleaner = get_text_cleaner()

    def extract_text_from_pdf(self, pdf_path: str) -> tuple[str, int, str]:
        """
        Extract text from PDF.

        Returns:
            (text, num_pages, text_status)
            text_status: 'valid', 'corrupted_ocr_success', 'corrupted_ocr_failed', 'ocr_needed', 'empty'
        """
        logger.info(f"Extracting text from: {pdf_path}")

        # PyMuPDF (fitz) is the PRIMARY extractor: far lighter on memory and
        # faster than pdfplumber for plain-text extraction. pdfplumber is kept
        # as a fallback for the rare PDFs where fitz yields too little text.
        text, num_pages = self._extract_with_pymupdf(pdf_path)

        if len(text.strip()) < self.config.min_text_length:
            text, num_pages = self._extract_with_pdfplumber(pdf_path)

        # Check whether the text is corrupted
        is_corrupted, reason = is_text_corrupted(text)

        # Try OCR for corrupted-non-empty AND fully-empty extractions
        # (scanned PDFs / xref-damaged PDFs land in the "empty" branch).
        needs_ocr = is_corrupted or len(text.strip()) < self.config.min_text_length

        if needs_ocr:
            if reason and reason != "empty":
                logger.warning(f"Corrupted text detected ({reason}): {pdf_path}")
            else:
                logger.warning(f"No extractable text in {pdf_path} - attempting OCR")
            if self.ocr_available:
                text_ocr = self._extract_with_ocr(pdf_path)
                is_still_corrupted, _ = is_text_corrupted(text_ocr)
                if not is_still_corrupted and len(text_ocr) >= max(self.config.min_text_length, len(text) * 0.5):
                    text = self.text_cleaner.clean_text(text_ocr)
                    logger.info(f"OCR succeeded: {len(text)} characters from {num_pages} pages")
                    return text, num_pages, "corrupted_ocr_success"
                else:
                    logger.warning(f"OCR failed for {pdf_path}")
                    return text, num_pages, "corrupted_ocr_failed"
            else:
                logger.warning(f"OCR unavailable for {pdf_path}")
                return text, num_pages, "ocr_needed"

        if len(text.strip()) < self.config.min_text_length:
            return text, num_pages, "empty"

        text = self.text_cleaner.clean_text(text)
        logger.info(f"Extracted {len(text)} characters from {num_pages} pages")
        return text, num_pages, "valid"

    def _extract_with_pdfplumber(self, pdf_path: str) -> tuple[str, int]:
        text_parts: list[str] = []
        num_pages = 0
        try:
            with pdfplumber.open(pdf_path) as pdf:
                num_pages = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                    # Release per-page caches immediately so a long PDF doesn't
                    # retain every page's parsed objects until the doc closes.
                    page.flush_cache()
                    try:
                        page.get_textmap.cache_clear()
                    except Exception:
                        pass
            text = "\n".join(text_parts)
            text_parts.clear()
            return text, num_pages
        except Exception as e:
            logger.warning(f"pdfplumber error: {e}")
            return "", 0

    def _extract_with_pymupdf(self, pdf_path: str) -> tuple[str, int]:
        text_parts: list[str] = []
        num_pages = 0
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text_parts.append(page_text)
            doc.close()
            text = "\n".join(text_parts)
            text_parts.clear()
            return text, num_pages
        except Exception as e:
            logger.warning(f"PyMuPDF error: {e}")
            return "", 0

    def _extract_with_ocr(self, pdf_path: str) -> str:
        text_parts: list[str] = []
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
            text = "\n".join(text_parts)
            text_parts.clear()
            return text
        except Exception as e:
            logger.warning(f"OCR via PyMuPDF failed ({e}); trying pdfplumber rendering")
            text_parts.clear()
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages[:20]:
                        img = page.to_image(resolution=200).original
                        page_text = pytesseract.image_to_string(img)
                        if page_text:
                            text_parts.append(page_text)
                        page.flush_cache()
                text = "\n".join(text_parts)
                text_parts.clear()
                return text
            except Exception as e2:
                logger.warning(f"OCR error (both backends failed): {e2}")
                return ""


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT EXTRACTOR - SENTENCES + AI MARKERS
# ═══════════════════════════════════════════════════════════════════════════

class ContextExtractor:
    """
    Context extractor
    - Extracts N sentences before and after the AI term
    - Marks the term with >>>term<<< for highlighting in Excel
    - Strips boilerplate
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
                                       highlight: bool = True) -> tuple[str, str]:
        """
        Extract sentence-based context and mark the AI term.
        Returns: (context_with_markers, ai_term)
        """
        ai_term = text[match_start:match_end]

        # Large window for locating sentences
        window_start = max(0, match_start - 2000)
        window_end = min(len(text), match_end + 2000)
        window_text = text[window_start:window_end]

        relative_match_start = match_start - window_start

        # Split into sentences
        sentences = self.SENTENCE_SPLIT_PATTERN.split(window_text)

        # Find the sentence that contains the term
        current_pos = 0
        target_sentence_idx = 0

        for idx, sentence in enumerate(sentences):
            sentence_end = current_pos + len(sentence)
            if current_pos <= relative_match_start < sentence_end:
                target_sentence_idx = idx
                break
            current_pos = sentence_end + 1

        # Extract surrounding sentences
        start_idx = max(0, target_sentence_idx - self.sentences_before)
        end_idx = min(len(sentences), target_sentence_idx + self.sentences_after + 1)

        context_sentences = sentences[start_idx:end_idx]
        context = ' '.join(context_sentences)

        # Clean
        context = self._clean_boilerplate(context)
        if self.text_cleaner:
            context = self.text_cleaner.clean_text(context)
        context = re.sub(r'\s+', ' ', context).strip()

        # Mark the AI term
        if highlight and ai_term:
            # Boundary-aware: do NOT match the term as a substring inside a
            # larger word (e.g. "AI" inside "tailored"/"available"/"contain").
            # Alnum lookarounds instead of \b so multi-word ("machine
            # learning") and hyphenated ("AI-powered") terms still match.
            term_pattern = re.compile(
                r'(?<![A-Za-z0-9])' + re.escape(ai_term) + r'(?![A-Za-z0-9])',
                re.IGNORECASE,
            )
            term_match = term_pattern.search(context)
            if term_match:
                context = (
                    context[:term_match.start()] +
                    self.HIGHLIGHT_START + term_match.group() + self.HIGHLIGHT_END +
                    context[term_match.end():]
                )

        # Length limit
        max_length = getattr(self.config, 'max_context_length', 800)
        if len(context) > max_length:
            context = context[:max_length] + '...'

        return context, ai_term

    def extract_relevant_context(self, text: str, match_start: int, match_end: int) -> str:
        """Compatibility helper - returns only the context."""
        context, _ = self.extract_context_with_sentences(text, match_start, match_end, highlight=True)
        return context
    
    def _clean_boilerplate(self, context: str) -> str:
        for pattern in self.compiled_boilerplate:
            context = pattern.sub(' ', context)
        return context


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY CLASSIFIER - CLASSIC TAXONOMY + FALLBACK RULES
# ═══════════════════════════════════════════════════════════════════════════

class CategoryClassifier:
    """
    Classic-taxonomy classifier with:
    - Primary pattern matching (from AI_CATEGORIES)
    - Secondary keyword matching (keyword tier-based confidence)
    - Fallback rules to keep the Unclassified rate low
    """

    # Fallback rules - used when pattern/keyword matching fails
    FALLBACK_RULES = [
        # Generative AI & LLMs - HIGHEST PRIORITY (specific terms)
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
    
    # Accept an axis match only when normalized confidence clears this floor.
    AXIS_THRESHOLD = 0.25

    def __init__(self):
        # Compile each axis SEPARATELY so a reference can be classified
        # independently on Applications (A) and Technologies (B). Keywords are
        # wrapped in \b...\b so short tokens like "ai", "BI", "IPA" don't
        # substring-match inside unrelated words ("sustainable", "abilities").
        self.compiled_apps = self._compile_axis(AI_APPLICATIONS)
        self.compiled_techs = self._compile_axis(AI_TECHNOLOGIES)

        logger.info(
            f"Category classifier (dual-axis): {len(AI_APPLICATIONS)} application "
            f"+ {len(AI_TECHNOLOGIES)} technology categories"
        )

    @staticmethod
    def _compile_axis(taxonomy: dict[str, dict]) -> dict[str, dict]:
        compiled = {}
        for category, data in taxonomy.items():
            compiled[category] = {
                'keyword_patterns': [
                    re.compile(r'\b' + re.escape(kw.lower()) + r'\b')
                    for kw in data['keywords']
                ],
                'patterns': [re.compile(p, re.IGNORECASE) for p in data.get('patterns', [])]
            }
        return compiled

    def _score_axis(self, combined: str, compiled: dict[str, dict]) -> tuple[str, float]:
        """Score one taxonomy axis and return its best (category, confidence).

        Pattern matches score 0.5 each (most specific), keyword matches 0.2
        each; the winning raw score is normalized by /1.5 and capped at 1.0.
        Returns ('none', 0.0) when no category clears AXIS_THRESHOLD — there is
        NO fallback, so an axis may legitimately be 'none'.
        """
        scores = {}
        for category, data in compiled.items():
            score = 0.0
            score += sum(1 for p in data['patterns'] if p.search(combined)) * 0.5
            score += sum(1 for p in data['keyword_patterns'] if p.search(combined)) * 0.2
            if score > 0:
                scores[category] = score

        if scores:
            best_category = max(scores, key=scores.get)
            confidence = min(1.0, scores[best_category] / 1.5)
            if confidence >= self.AXIS_THRESHOLD:
                return best_category, confidence

        return 'none', 0.0

    def classify_dual(self, text: str, context: str) -> tuple[str, float, str, float]:
        """Classify a reference independently on both classic axes.

        Returns: (category_a, conf_a, category_b, conf_b). Each axis is the
        best-scoring category for that taxonomy, or 'none' if nothing matched.
        A reference can therefore land in an A-category, a B-category, both, or
        neither — the EU_Semantics axis (classified in parallel) captures
        references that match neither A nor B.
        """
        combined = f"{text} {context}".lower()
        category_a, conf_a = self._score_axis(combined, self.compiled_apps)
        category_b, conf_b = self._score_axis(combined, self.compiled_techs)
        return category_a, conf_a, category_b, conf_b


# ═══════════════════════════════════════════════════════════════════════════
# EU_SEMANTICS CLASSIFIER (JRC AI Watch) — parallel to CategoryClassifier
# ═══════════════════════════════════════════════════════════════════════════

class EUCategoryClassifier:
    """Classify a detected AI reference under the EU_SEMANTICS taxonomy
    (European Commission JRC "AI Watch").

    Runs in PARALLEL with the classic CategoryClassifier on the SAME detected
    references (no extra PDF processing). Returns (domain, subdomain, confidence).
    References with no EU keyword/pattern match → ('Unclassified', 'Unclassified', 0.0).
    """

    def __init__(self):
        # Compile keyword + pattern matchers per EU leaf. Keywords use \b…\b so
        # short tokens don't substring-match inside unrelated words.
        self.compiled: dict[str, dict] = {}
        for leaf_code, data in EU_CATEGORIES.items():
            self.compiled[leaf_code] = {
                'domain': data['domain'],
                'subdomain': data['subdomain'],
                'keyword_patterns': [
                    re.compile(r'\b' + re.escape(kw.lower()) + r'\b')
                    for kw in data['keywords']
                ],
                'patterns': [
                    re.compile(p, re.IGNORECASE) for p in data.get('patterns', [])
                ],
            }
        logger.info(f"EU classifier: {len(EU_CATEGORIES)} EU_Semantics leaves (JRC AI Watch)")

    def classify_with_confidence(self, text: str, context: str) -> tuple[str, str, float]:
        """Return (eu_domain, eu_subdomain, confidence 0.0-1.0) for a reference."""
        combined = f"{text} {context}".lower()
        scores: dict[str, float] = {}
        for leaf_code, data in self.compiled.items():
            score = 0.0
            score += sum(1 for p in data['patterns'] if p.search(combined)) * 0.5
            score += sum(1 for p in data['keyword_patterns'] if p.search(combined)) * 0.2
            if score > 0:
                scores[leaf_code] = score

        if not scores:
            return 'Unclassified', 'Unclassified', 0.0

        best = max(scores, key=scores.get)
        confidence = min(1.0, scores[best] / 1.5)
        info = self.compiled[best]
        return info['domain'], info['subdomain'], confidence


# ═══════════════════════════════════════════════════════════════════════════
# FALSE POSITIVE FILTER - VALIDATION & ENCODING CHECK
# ═══════════════════════════════════════════════════════════════════════════

# Import AI_CONTEXT_VALIDATORS from core
try:
    from _02_core import AI_CONTEXT_VALIDATORS
except ImportError:
    AI_CONTEXT_VALIDATORS = [
        'artificial intelligence', 'machine learning', 'deep learning',
        'neural network', 'algorithm', 'model training', 'prediction',
        'automat', 'intelligent system', 'chatbot', 'generative', 'LLM', 'GPT',
        'autonomous', 'analytics', 'natural language', 'computer vision',
    ]


class FalsePositiveFilter:
    """
    False positive filter with:
    - Pattern matching (~60 patterns)
    - ML/DL measurement unit detection
    - Corrupted text / encoding validation
    - Context validation for short texts (AI, ML, DL)
    """

    def __init__(self):
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in FALSE_POSITIVE_PATTERNS]
        self.context_validators = [v.lower() for v in AI_CONTEXT_VALIDATORS]
        logger.info(f"False positive filter: {len(FALSE_POSITIVE_PATTERNS)} patterns")

    def is_false_positive(self, text: str, context: str) -> bool:
        """
        Check whether the reference is a false positive.

        Returns True if:
        - It matches a false positive pattern
        - The text is corrupted (encoding issues)
        - Short text without a valid AI context
        """
        # 1. Check for corrupted text/context
        if self._is_corrupted_text(context):
            return True

        # 2. Check false positive patterns
        combined = f"{text} {context}"
        for pattern in self.compiled_patterns:
            if pattern.search(combined):
                return True

        # 3. For short texts (AI, ML, DL), validate context
        if len(text) <= 3 and not self._has_valid_ai_context(context):
            return True

        return False

    def _is_corrupted_text(self, text: str) -> bool:
        """Detect corrupted text from encoding issues."""
        if not text or len(text) < 20:
            return False

        # Too many special characters (>10%)
        special_count = len(re.findall(r'[#$%&*@^\\|~`]', text))
        if special_count / len(text) > 0.1:
            return True

        # Too few spaces for the text length (run-on text)
        space_count = text.count(' ')
        if len(text) > 100 and space_count / len(text) < 0.05:
            return True

        # Long sequences without vowels (impossible in English)
        if re.search(r'[bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ]{8,}', text):
            return True

        return False

    def _has_valid_ai_context(self, context: str) -> bool:
        """Check whether the context confirms it is a genuine AI reference."""
        if not context:
            return False

        context_lower = context.lower()

        # Check whether the context contains real AI terms
        for validator in self.context_validators:
            if validator in context_lower:
                return True

        return False


# ═══════════════════════════════════════════════════════════════════════════
# AI REFERENCE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# NEGATION FILTER
# ═══════════════════════════════════════════════════════════════════════════
# Two-tier filter applied AFTER pattern + semantic detection, BEFORE references
# are returned to the pipeline.
#
#   HARD_NEGATION  → drop the reference entirely (unambiguous "we do not use AI")
#   SOFT_NEGATION  → keep, but mark is_negated_context=True, force strength to
#                    'mention_only', multiply confidence by 0.4. Captures
#                    risk-factor / hypothetical / disclaimer language.

class NegationFilter:
    """Context-window negation detector. See module-level note above."""

    # Hard negation: company DENIES adoption / DISCLAIMS use.
    # Drop the reference — it is not evidence of AI adoption.
    HARD_NEGATION_PATTERNS = [
        r'\b(?:do(?:es)?|did)\s+not\s+(?:currently\s+)?(?:use|deploy|leverage|employ|utilize|rely\s+on|operate)\b.{0,40}\b(?:AI|ML|GenAI|artificial\s+intelligence|machine\s+learning)\b',
        r'\b(?:AI|ML|GenAI|artificial\s+intelligence|machine\s+learning)\b.{0,40}\b(?:is|are)\s+not\s+(?:currently\s+)?(?:used|deployed|employed|leveraged|integrated)\b',
        r'\b(?:no|not)\s+(?:current\s+)?(?:plans?|intent(?:ion)?|strateg(?:y|ic)\s+(?:plan)?)\s+to\s+(?:deploy|implement|use|adopt|integrate)\b.{0,40}\b(?:AI|ML|GenAI|artificial\s+intelligence)\b',
        r'\bnone\s+of\s+(?:our|the)\s+(?:products|services|operations|systems|business(?:es)?)\b.{0,40}\b(?:use|leverage|rely|employ)\b.{0,40}\b(?:AI|ML|GenAI)\b',
        r'\b(?:absent|without|lacking)\s+(?:any\s+)?(?:AI|ML|machine\s+learning)\b',
        r'\b(?:AI|ML|GenAI)\b.{0,30}\bnot\s+(?:material|significant|core|substantial|a\s+significant)\b',
    ]

    # Soft negation: hypothetical / risk-factor / disclaimer language.
    # Keep the reference (it IS a mention) but down-weight it heavily — these
    # are obligatory regulatory disclosures, not evidence of operational use.
    SOFT_NEGATION_PATTERNS = [
        r'\b(?:AI|ML|GenAI|artificial\s+intelligence|machine\s+learning)\b.{0,60}\b(?:may|could|might|can|would)\s+(?:pose|create|introduce|present|cause|result\s+in|lead\s+to)\b.{0,40}\b(?:risks?|harms?|threats?|concerns?|liabilit(?:y|ies)|exposures?|losses|damages?|adverse|negative)\b',
        r'\b(?:risks?|threats?|harms?|concerns?|liabilit(?:y|ies))\s+(?:associated|related|linked)\s+(?:with|to)\b.{0,40}\b(?:AI|ML|GenAI|artificial\s+intelligence|machine\s+learning)\b',
        r'\bcannot\s+(?:guarantee|ensure|assure|predict)\b.{0,60}\b(?:AI|ML|GenAI|artificial\s+intelligence|machine\s+learning)\b',
        r'\b(?:if|should|in\s+the\s+event\s+that)\s+(?:AI|ML|GenAI|artificial\s+intelligence|our\s+AI)\b.{0,40}\b(?:fail|malfunction|err|breach|violate|expose)',
        r'\b(?:hypothetical(?:ly)?|potential(?:ly)?)\b.{0,30}\b(?:AI|ML|GenAI|artificial\s+intelligence)\b',
        r'\b(?:AI|ML|GenAI)\b.{0,30}\b(?:will|may)\s+not\s+(?:be|become|prove)\b',
        r'\b(?:regulator(?:y|s)|legal|compliance)\s+(?:uncertaint(?:y|ies)|risk(?:s)?)\b.{0,40}\b(?:AI|ML|GenAI|artificial\s+intelligence)\b',
        r'\b(?:AI|ML|GenAI|artificial\s+intelligence)\b.{0,40}\b(?:regulator(?:y|s)|legal|compliance)\s+(?:uncertaint(?:y|ies)|risk(?:s)?|exposure)\b',
        r'\bbias(?:ed|es)?\s+(?:in|of|within)\s+(?:AI|ML|GenAI|our\s+models?|algorithms?)\b',
    ]

    SOFT_NEGATION_CONFIDENCE_FACTOR = 0.4

    def __init__(self) -> None:
        self.compiled_hard = [re.compile(p, re.IGNORECASE) for p in self.HARD_NEGATION_PATTERNS]
        self.compiled_soft = [re.compile(p, re.IGNORECASE) for p in self.SOFT_NEGATION_PATTERNS]
        logger.info(
            f"Negation filter: hard={len(self.HARD_NEGATION_PATTERNS)}, soft={len(self.SOFT_NEGATION_PATTERNS)}"
        )

    def classify(self, context: str) -> str:
        """Return 'hard', 'soft', or 'none'."""
        if not context:
            return "none"
        for p in self.compiled_hard:
            if p.search(context):
                return "hard"
        for p in self.compiled_soft:
            if p.search(context):
                return "soft"
        return "none"


class AIReferenceDetector:
    """
    AI reference detector (dual taxonomy + FP filtering):
      - Vendor-specific detection
      - Separates HARD vs SOFT triggers (buzzword gating)
      - Runs context validators (actionability / specificity / marketing-only)
      - Computes confidence_score and flags weak references as mention_only (FP candidates)
      - Applies NegationFilter post-detection (drop hard / down-weight soft)

    NOTE: To keep compatibility with the rest of the codebase (core / analysis / export),
    strength/confidence are also encoded inside detection_method as metadata.
    """

    # Strict threshold for semantic-only hits (no HARD trigger, no implementation signal)
    DEFAULT_SEMANTIC_STRICT_THRESHOLD = 0.68

    # ────────────────────────────────────────────────────────────────────
    # HARD vs SOFT triggers
    # ────────────────────────────────────────────────────────────────────
    # HARD: technical / specific indicators ("real" AI)
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

    # SOFT: vague / marketing / "AI-washing" terms
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

    # Validator signals
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
        self.eu_classifier = EUCategoryClassifier()
        self.fp_filter = FalsePositiveFilter()
        self.negation_filter = NegationFilter()

        # Integrity counter: references unclassified on ALL THREE axes
        # (A, B, EU). Expected ~0 — a detected AI reference should be captured
        # by at least one taxonomy axis. Surfaced in run stats.
        self.tri_axis_unclassified = 0

        # compile patterns
        self.compiled_hard = [re.compile(p, re.IGNORECASE) for p in self.AI_HARD_PATTERNS]
        self.compiled_soft = [re.compile(p, re.IGNORECASE) for p in self.AI_SOFT_PATTERNS]
        self.compiled_impl = [re.compile(p, re.IGNORECASE) for p in self.IMPLEMENTATION_VERBS]
        self.compiled_artifacts = [re.compile(p, re.IGNORECASE) for p in self.TECHNICAL_ARTIFACTS]
        self.compiled_marketing = [re.compile(p, re.IGNORECASE) for p in self.MARKETING_ONLY_CUES]

        logger.info(
            f"AI Reference Detector (dual taxonomy): hard={len(self.AI_HARD_PATTERNS)}, soft={len(self.AI_SOFT_PATTERNS)}"
        )

    @staticmethod
    def _is_tri_axis_unclassified(ref: AIReference) -> bool:
        """True when a reference is unclassified on all three axes (A, B, EU).

        Per the taxonomy design a detected AI reference must be captured by at
        least one axis; 'none' on A and B is only valid when EU classified it.
        """
        return (
            ref.category_a == 'none' and ref.category_b == 'none'
            and (not ref.eu_subdomain or ref.eu_subdomain == 'Unclassified')
        )

    # ────────────────────────────────────────────────────────────────────
    # Public API
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
    ) -> list[AIReference]:
        references: list[AIReference] = []

        pattern_refs = self._detect_by_patterns(
            text, company, year, position, industry, sector, country, doc_type, source
        )
        references.extend(pattern_refs)

        semantic_refs = self._detect_by_semantics(
            text, company, year, position, industry, sector, country, doc_type, source
        )
        references.extend(semantic_refs)

        # Post-detection negation filter. Hard negations are dropped;
        # soft (risk/hypothetical) negations are kept but down-weighted so
        # they no longer inflate the adoption index.
        filtered: list[AIReference] = []
        n_hard_drop = 0
        n_soft_downgrade = 0
        for ref in references:
            verdict = self.negation_filter.classify(ref.context)
            if verdict == "hard":
                n_hard_drop += 1
                continue
            if verdict == "soft":
                ref.is_negated_context = True
                ref.confidence_score = round(
                    ref.confidence_score * NegationFilter.SOFT_NEGATION_CONFIDENCE_FACTOR, 3
                )
                ref.reference_strength = "mention_only"
                ref.confidence_reasons = (
                    (ref.confidence_reasons + ";" if ref.confidence_reasons else "")
                    + "negation_soft"
                )
                n_soft_downgrade += 1
            filtered.append(ref)

        if n_hard_drop or n_soft_downgrade:
            logger.info(
                f"Negation filter: dropped {n_hard_drop} hard, downgraded {n_soft_downgrade} soft"
            )

        # Integrity check: a detected reference should be captured by at least
        # one taxonomy axis. Count/log any that are 'none' on A, B AND EU.
        n_tri = sum(1 for ref in filtered if self._is_tri_axis_unclassified(ref))
        if n_tri:
            self.tri_axis_unclassified += n_tri
            logger.warning(
                f"Integrity: {n_tri} reference(s) unclassified on all three axes "
                f"(A/B/EU) in this document; running total={self.tri_axis_unclassified}"
            )

        logger.info(
            f"Detected {len(filtered)} references ({len(pattern_refs)} pattern, {len(semantic_refs)} semantic; negation-filtered)"
        )
        return filtered

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

    def _feature_counts(self, context: str) -> dict[str, int]:
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
    ) -> tuple[float, str, list[str]]:
        """
        Returns: (confidence_score in 0..1, strength label, reasons)
        strength: 'strong'|'medium'|'mention_only'
        """
        reasons: list[str] = []
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

    def _format_detection_method(self, base: str, trigger_type: str, strength: str, conf: float, reasons: list[str]) -> str:
        # Compact metadata, easy to parse downstream (analysis / export)
        # Example: "pattern|trigger=soft|strength=mention_only|conf=0.48|reasons=soft_trigger,marketing_only_penalty"
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
    ) -> list[AIReference]:
        references: list[AIReference] = []
        seen_positions = set()

        # Look for HARD triggers first, then SOFT ones (avoids duplicate hits)
        pattern_groups = [("hard", self.compiled_hard), ("soft", self.compiled_soft)]

        for trigger_type, compiled_list in pattern_groups:
            for pattern in compiled_list:
                for match in pattern.finditer(text):
                    if match.start() in seen_positions:
                        continue
                    seen_positions.add(match.start())

                    matched_text = match.group()

                    # Context with markers
                    context, ai_term = self.context_extractor.extract_context_with_sentences(
                        text, match.start(), match.end(), highlight=True
                    )

                    # False positive check (without markers)
                    context_clean = context.replace('>>>', '').replace('<<<', '')
                    if self.fp_filter.is_false_positive(matched_text, context_clean):
                        continue

                    # Robotics / RPA filters
                    robotics_type = classify_robotics_reference(context_clean)
                    rpa_type = classify_rpa_reference(context_clean)
                    if self.config.filter_traditional_robotics and robotics_type == 'traditional_robotics':
                        continue
                    if self.config.filter_traditional_rpa and rpa_type == 'traditional_rpa':
                        continue

                    # Dual-axis classification (classic taxonomy): A and B independent
                    category_a, conf_a, category_b, conf_b = self.category_classifier.classify_dual(
                        matched_text, context_clean
                    )
                    # EU_Semantics parallel classification (same reference)
                    eu_domain, eu_subdomain, eu_conf = self.eu_classifier.classify_with_confidence(
                        matched_text, context_clean
                    )

                    # Scoring & strength
                    conf, strength, reasons = self._compute_confidence_and_strength(
                        trigger_type=trigger_type,
                        semantic_score=0.0,
                        context_clean=context_clean,
                        doc_type=doc_type
                    )

                    # If SOFT and ends up as mention_only we keep it as a FP candidate
                    detection_method = self._format_detection_method(
                        "pattern", trigger_type, strength, conf, reasons
                    )

                    page = text[:match.start()].count('\n') // 50 + 1

                    ref = AIReference(
                        company=company, year=year, position=position,
                        industry=industry, sector=sector, country=country,
                        doc_type=doc_type,
                        text=ai_term, context=context, page=page,
                        category_a=category_a, category_b=category_b,
                        detection_method=detection_method,
                        sentiment='neutral', sentiment_score=0.0,
                        semantic_score=0.0, source=source,
                        robotics_type=robotics_type, rpa_type=rpa_type,
                        category_a_confidence=conf_a, category_b_confidence=conf_b,
                        eu_domain=eu_domain, eu_subdomain=eu_subdomain,
                        eu_confidence=eu_conf,
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
    ) -> list[AIReference]:
        references: list[AIReference] = []

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

                # Pick threshold based on HARD/impl signals in the paragraph
                has_hard = self._has_hard_indicator(para)
                feats = self._feature_counts(para)
                has_impl = feats["impl"] > 0
                chosen_thr = base_thr if (has_hard or has_impl) else strict_thr

                if score < chosen_thr:
                    continue

                # FP filter for semantic hits
                if self.fp_filter.is_false_positive("", para):
                    continue

                robotics_type = classify_robotics_reference(para)
                rpa_type = classify_rpa_reference(para)
                if self.config.filter_traditional_robotics and robotics_type == 'traditional_robotics':
                    continue
                if self.config.filter_traditional_rpa and rpa_type == 'traditional_rpa':
                    continue

                category_a, conf_a, category_b, conf_b = self.category_classifier.classify_dual("", para)
                eu_domain, eu_subdomain, eu_conf = self.eu_classifier.classify_with_confidence("", para)

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
                    category_a=category_a, category_b=category_b,
                    detection_method=detection_method,
                    sentiment='neutral', sentiment_score=0.0,
                    semantic_score=score, source=source,
                    robotics_type=robotics_type, rpa_type=rpa_type,
                    category_a_confidence=conf_a, category_b_confidence=conf_b,
                    eu_domain=eu_domain, eu_subdomain=eu_subdomain,
                    eu_confidence=eu_conf,
                    reference_strength=strength,
                    confidence_score=conf,
                    confidence_reasons=';'.join(reasons)
                )
                references.append(ref)

        return references

    def _find_and_mark_ai_term(self, paragraph: str) -> tuple[str, str]:
        """Find the AI term in the paragraph and wrap it with >>>...<<<."""
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
    """Parser for company report PDF filenames (company, year, doc type)."""
    
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
    
    def parse_filename(self, filename: str) -> dict:
        """
        Parse filename to extract company, year, doc_type, position.

        Expected format: "38. Bank of America - 2024 - annual report.pdf"
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
        # STEP 1: extract the POSITION (e.g. "38." at the start)
        # ═══════════════════════════════════════════════════════════════
        position_match = re.match(r'^(\d{1,3})\.\s*', base)
        if position_match:
            result['position'] = int(position_match.group(1))
            # Strip the position from the string for further processing
            base = base[position_match.end():]

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: extract the YEAR (e.g. "2024")
        # ═══════════════════════════════════════════════════════════════
        year_match = re.search(r'(?:^|[\s\-_])20(1[5-9]|2[0-5])(?:[\s\-_]|$)', base)
        if year_match:
            result['year'] = int('20' + year_match.group(1))

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: extract the DOC_TYPE
        # ═══════════════════════════════════════════════════════════════
        base_lower = base.lower()
        for key, value in self.DOC_TYPE_MAPPING.items():
            if key in base_lower:
                result['doc_type'] = value
                break

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: extract the COMPANY NAME
        # Format: "Company Name - Year - doc type"
        # ═══════════════════════════════════════════════════════════════
        # Split on " - " (with spaces)
        parts = re.split(r'\s+-\s+', base)

        if len(parts) >= 1:
            company_part = parts[0].strip()

            # Strip any residual digits from the company name
            company_part = re.sub(r'^\d+\.\s*', '', company_part)

            if company_part:
                result['company'] = company_part

        return result


# ═══════════════════════════════════════════════════════════════════════════
# FINAL
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("AI SEMANTIC ANALYZER - DETECTION MODULE")
    print("=" * 80)
    print("\n✓ detection module loaded.")
    print("\nFeatures:")
    print(f"  - AI patterns: {len(AI_MAIN_PATTERNS)}")
    print(f"  - Canonical descriptions: {len(AI_CANONICAL_DESCRIPTIONS)}")
    print("  - Robotics / RPA classification")
    print("  - Sentence-aware context extraction")
    print("  - Category classifier with confidence scoring")
