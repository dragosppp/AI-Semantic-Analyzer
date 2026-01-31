"""
═══════════════════════════════════════════════════════════════════════════════
AI SEMANTIC ANALYZER v6.1.1 - MODUL 1: CORE & CONFIGURARE
═══════════════════════════════════════════════════════════════════════════════

CHANGELOG v6.1.1 (Ian 2026):
    - DB: persistă reference_strength / confidence_score / confidence_reasons în ai_references_raw
    - AIReference: sector/country default 'Unknown' pentru compatibilitate
    - Version bump 6.1.1

CHANGELOG v6.1.0 (Ian 2026)
        - Reducere false positives: scor de încredere + strength (mention_only)
        - Prag semantic strict pentru semantic-only: 0.68

    CHANGELOG v6.0.6 (Ian 2026)
        - introducere industrie si tara
    
    CHANGELOG v6.0.5 (Ianuarie 2026):
        - 13 categorii AI specializate
        - 80+ termeni GenAI 2025 noi
        - Filtrare Robotics tradițional vs AI-robotics
        - Filtrare RPA tradițional vs Intelligent Automation
        - Threshold semantic 0.60
        - False positives extinse (~60 patterns)
        - Suport 2020-2025
        - Context validators pentru texte scurte (AI, ML, DL)
        - Text corruption detection


CHANGELOG v6.0 (Ianuarie 2025):
    - 17 categorii AI (de la 12) - adăugat Agentic, GenAI, MLOps, Safety, Coding
    - 80+ termeni GenAI 2025 noi
    - Filtrare Robotics tradițional vs AI-robotics
    - Filtrare RPA tradițional vs Intelligent Automation
    - Threshold semantic 0.55 → 0.60
    - False positives extinse (~40 patterns)
    - Suport 2020-2025

Autor: TeRa0
Versiune: 6.0.5
Data: Ianuarie 2025

═══════════════════════════════════════════════════════════════════════════════
"""


# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

import os
import re
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

import pandas as pd
import numpy as np

import pdfplumber
import fitz  # PyMuPDF

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    warnings.warn("pytesseract not available. OCR functionality disabled.")

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    warnings.warn("transformers/torch not available. Using TextBlob fallback.")

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm import tqdm

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════
# VERSION CONSTANT
# ═══════════════════════════════════════════════════════════════════════════

ANALYZER_VERSION = "6.1.0"
ANALYZER_DATE = "Ianuarie 2026"




# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(log_file: str = "ai_analyzer_v6.log", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("AIAnalyzerV6")
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURARE INTERACTIVĂ
# ═══════════════════════════════════════════════════════════════════════════

CONFIG_FILE = "analyzer_config_v6.json"

def load_saved_config() -> Optional[Dict]:
    if Path(CONFIG_FILE).exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Nu am putut încărca configurația salvată: {e}")
    return None

def save_config(config_dict: Dict):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Configurație salvată în {CONFIG_FILE}")
    except Exception as e:
        logger.error(f"Eroare la salvarea configurației: {e}")

def get_user_input(prompt: str, default: str = "") -> str:
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()

def validate_path(path: str, path_type: str = "folder") -> bool:
    p = Path(path)
    if path_type == "folder":
        return p.exists() and p.is_dir()
    elif path_type == "file":
        return p.exists() and p.is_file()
    return False

def interactive_config() -> Dict:
    print("\n" + "="*70)
    print("AI SEMANTIC ANALYZER v6.1.1 - CONFIGURARE")
    print("="*70)
    
    saved_config = load_saved_config()
    
    if saved_config:
        print("\n✓ Configurație salvată găsită:")
        print(f"  - Folder PDF-uri: {saved_config.get('input_folder', 'N/A')}")
        print(f"  - Fortune 500 CSV: {saved_config.get('fortune500_csv', 'N/A')}")
        print(f"  - Folder output: {saved_config.get('output_folder', 'N/A')}")
        
        keep = input("\nPăstrezi configurația? (da/nu) [da]: ").strip().lower()
        if keep in ['', 'da', 'd', 'yes', 'y']:
            return saved_config
    
    config_dict = {}
    
    print("\n--- LOCAȚIE PDF-URI ---")
    while True:
        default_pdf = saved_config.get('input_folder', '') if saved_config else ''
        pdf_folder = get_user_input("Calea către folderul cu PDF-uri", default_pdf)
        if validate_path(pdf_folder, "folder"):
            config_dict['input_folder'] = pdf_folder
            pdf_count = len(list(Path(pdf_folder).glob("*.pdf")))
            print(f"  ✓ Găsite {pdf_count} fișiere PDF")
            break
        print(f"  ✗ Folderul '{pdf_folder}' nu există.")
    
    print("\n--- FIȘIER FORTUNE 500 CSV ---")
    while True:
        default_csv = saved_config.get('fortune500_csv', '') if saved_config else ''
        csv_file = get_user_input("Calea către CSV Fortune 500 (sau 'skip')", default_csv)
        if csv_file.lower() == "skip" or csv_file == "":
            config_dict['fortune500_csv'] = None
            break
        elif validate_path(csv_file, "file"):
            config_dict['fortune500_csv'] = csv_file
            break
        print(f"  ✗ Fișierul '{csv_file}' nu există.")
    
    print("\n--- FOLDER OUTPUT ---")
    default_output = saved_config.get('output_folder', 'Results_v6') if saved_config else 'Results_v6'
    output_folder = get_user_input("Folder output", default_output)
    config_dict['output_folder'] = output_folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    print("\n--- SETĂRI ---")
    top_n_input = get_user_input("Procesează doar primele N companii? (gol=toate)", "")
    config_dict['top_n'] = int(top_n_input) if top_n_input.isdigit() else None
    
    save_choice = input("\nSalvezi configurația? (da/nu) [da]: ").strip().lower()
    if save_choice in ['', 'da', 'd', 'yes', 'y']:
        save_config(config_dict)
    
    return config_dict


# ═══════════════════════════════════════════════════════════════════════════
# ANALYZER CONFIG
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AnalyzerConfig:
    """Configurație centralizată pentru AI Semantic Analyzer v6.0.6"""
    
    # Paths
    input_folder: str = "Fortune500_PDFs"
    output_folder: str = "Results_v6"
    database_name: str = "fortune500_ai_analysis_v6.db"
    fortune500_csv: Optional[str] = None
    
    # Thresholds - ACTUALIZAT v6.0
    semantic_threshold: float = 0.60  # Crescut de la 0.55
    # NOU v6.1: prag strict pentru semantic-only (reduce FP)
    semantic_threshold_strict: float = 0.68
    # NOU v6.1: prag relaxat folosit când există HARD triggers / implementare
    semantic_threshold_relaxed: float = 0.60
    deduplication_threshold: float = 0.85
    
    # Context
    context_chars: int = 150
    min_text_length: int = 100
    
    # Processing
    max_workers: int = 4
    batch_size: int = 10
    top_n: Optional[int] = None
    
    # NOU v6.0: Filtrare
    filter_traditional_robotics: bool = True
    filter_traditional_rpa: bool = True
    
    # V 6.0.1 Context pentru referințe
    context_sentences_before: int = 2
    context_sentences_after: int = 2
    max_context_length: int = 2000

    # AI Adoption Index weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        'intensity': 0.15,
        'semantic': 0.20,
        'diversity': 0.15,
        'sentiment': 0.15,
        'maturity': 0.15,
        'future': 0.10,
        'commitment': 0.10
    })
    
    # Ani
    start_year: int = 2020
    end_year: int = 2025
    
    @classmethod
    def from_interactive(cls) -> 'AnalyzerConfig':
        config_dict = interactive_config()
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'AnalyzerConfig':
        return cls(
            input_folder=config_dict.get('input_folder', 'Fortune500_PDFs'),
            output_folder=config_dict.get('output_folder', 'Results_v6'),
            fortune500_csv=config_dict.get('fortune500_csv'),
            top_n=config_dict.get('top_n')
        )
    
    def __post_init__(self):
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Suma ponderilor trebuie să fie 1.0, este {total_weight}")
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        logger.info(f"Config v6.1: semantic_threshold={self.semantic_threshold}, strict={self.semantic_threshold_strict}, robotics_filter={self.filter_traditional_robotics}")


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AIReference:
    """Referință AI detectată."""
    # Câmpuri obligatorii (fără default) - trebuie să fie primele
    company: str
    year: int
    position: int
    industry: str
    sector: str
    country: str
    doc_type: str
    text: str
    context: str
    page: int
    category: str
    detection_method: str
    sentiment: str
    sentiment_score: float
    semantic_score: float
    source: str
    # Câmpuri opționale (cu default) - trebuie să fie la final
    robotics_type: str = "not_robotics"  # 'ai_robotics', 'traditional_robotics', 'not_robotics'
    rpa_type: str = "not_rpa"  # 'ai_rpa', 'traditional_rpa', 'not_rpa'
    sentiment_confidence: str = "standard"  # 'standard', 'governance_adjusted', 'confirmed_negative'
    category_confidence: float = 1.0
    # NOU v6.1: reducere false positives
    reference_strength: str = "unknown"  # "strong" | "medium" | "mention_only" | "unknown"
    confidence_score: float = 0.0  # 0..1
    confidence_reasons: str = ""  # string / JSON pentru audit
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DocumentResult:
    """Rezultatul procesării unui document."""
    company: str
    year: int
    position: int
    industry: str
    sector: str
    country: str
    doc_type: str
    source: str
    total_pages: int
    text_length: int
    references: List[AIReference] = field(default_factory=list)
    total_refs: int = 0
    pattern_refs: int = 0
    semantic_refs: int = 0
    categories_count: Dict[str, int] = field(default_factory=dict)
    sentiment_distribution: Dict[str, int] = field(default_factory=dict)
    processing_time: float = 0.0
    
    def add_reference(self, ref: AIReference):
        self.references.append(ref)
        self.total_refs += 1
        if ref.detection_method == 'pattern':
            self.pattern_refs += 1
        elif ref.detection_method == 'semantic':
            self.semantic_refs += 1
        self.categories_count[ref.category] = self.categories_count.get(ref.category, 0) + 1
        self.sentiment_distribution[ref.sentiment] = self.sentiment_distribution.get(ref.sentiment, 0) + 1


@dataclass
class AIAdoptionIndex:
    """AI Adoption Index per companie-an."""
    company: str
    year: int
    position: int
    industry: str
    sector: str
    country: str
    intensity_index: float = 0.0
    semantic_index: float = 0.0
    diversity_index: float = 0.0
    sentiment_index: float = 0.0
    maturity_index: float = 0.0
    future_index: float = 0.0
    commitment_index: float = 0.0
    ai_adoption_index: float = 0.0
    total_refs: int = 0
    total_pages: int = 0
    categories_used: int = 0
    rank_in_year: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# AI CATEGORIES v6.1 - 13 CATEGORII
# ═══════════════════════════════════════════════════════════════════════════

AI_CATEGORIES = {
    'Strategic Investment': {
        'description': 'Investiții strategice în tehnologii AI',
        'keywords': [
            'invest', 'acquisition', 'M&A', 'budget', 'funding', 'capital',
            'partnership', 'venture', 'strategic initiative', 'AI spending',
            'AI investment', 'AI budget', 'allocated', 'committed funds'
        ],
        'patterns': [
            r'invest(?:ed|ing|ment).*(?:AI|artificial intelligence)',
            r'(?:AI|artificial intelligence).*(?:budget|funding|capital)',
            r'acquir(?:ed|ing|ition).*(?:AI|machine learning)',
            r'\$[\d,.]+\s*(?:million|billion|M|B).*(?:AI|artificial intelligence)',
        ]
    },
    
    'Operational Implementation': {
        'description': 'Implementări operaționale: automatizare, eficiență',
        'keywords': [
            'deploy', 'automate', 'RPA', 'efficiency', 'streamline', 
            'optimize', 'workflow', 'process improvement', 'implementation',
            'rollout', 'integration', 'operational excellence', 'productivity',
            'intelligent automation', 'hyperautomation', 'IPA'
        ],
        'patterns': [
            r'deploy(?:ed|ing|ment).*(?:AI|ML|machine learning)',
            r'automat(?:e|ed|ing|ion).*(?:process|workflow|task)',
            r'(?:AI|ML).*(?:efficiency|optimization|productivity)',
            r'(?:intelligent|hyper)[\s-]?automation',
        ]
    },
    
    'Customer Experience': {
        'description': 'AI pentru experiența clienților',
        'keywords': [
            'chatbot', 'personalization', 'customer experience', 'CX',
            'customer service', 'support', 'recommendation', 'engagement',
            'virtual assistant', 'conversational AI', 'customer journey'
        ],
        'patterns': [
            r'(?:AI|ML).*(?:chatbot|virtual assistant|conversational)',
            r'personali[zs](?:e|ed|ation).*(?:customer|experience)',
            r'customer.*(?:experience|service).*(?:AI|ML)',
            r'(?:recommendation|recommender)\s+(?:system|engine)',
        ]
    },
    
    'Product Development': {
        'description': 'Dezvoltare produse cu AI',
        'keywords': [
            'R&D', 'innovation', 'AI-powered product', 'feature', 
            'development', 'design', 'prototype', 'intelligent',
            'AI-enabled', 'smart product', 'next-generation', 'AI-first'
        ],
        'patterns': [
            r'AI[\s-]?powered.*(?:product|feature|solution)',
            r'intelligent.*(?:system|solution|product)',
            r'R&D.*(?:AI|artificial intelligence)',
            r'(?:develop|build|create).*AI.*(?:product|solution)',
        ]
    },
    
    'Risk & Compliance': {
        'description': 'Guvernanță AI, etică, compliance',
        'keywords': [
            'governance', 'ethics', 'responsible AI', 'bias', 'fairness',
            'transparency', 'audit', 'compliance', 'regulation', 'GDPR',
            'AI risk', 'model risk', 'AI policy', 'ethical AI', 'trustworthy AI'
        ],
        'patterns': [
            r'responsible\s+(?:AI|artificial intelligence)',
            r'(?:AI|ML).*(?:ethics|governance|compliance|regulation)',
            r'(?:bias|fairness).*(?:detection|mitigation|audit)',
            r'AI\s+(?:risk|policy|governance)\s+framework',
        ]
    },
    
    'Data & Analytics': {
        'description': 'Analytics predictive, ML pentru insights',
        'keywords': [
            'predictive analytics', 'machine learning', 'data science',
            'insights', 'forecasting', 'modeling', 'algorithm',
            'data-driven', 'business intelligence', 'advanced analytics'
        ],
        'patterns': [
            r'predictive\s+(?:analytics|model|insight)',
            r'machine learning.*(?:model|algorithm|prediction)',
            r'data[\s-]?driven.*(?:insight|decision|strategy)',
            r'(?:advanced|augmented)\s+analytics',
        ]
    },
    
    'Talent & Workforce': {
        'description': 'Talent AI, training, upskilling',
        'keywords': [
            'upskill', 'training', 'AI talent', 'reskill', 'hiring',
            'recruitment', 'workforce', 'education', 'capability',
            'data scientist', 'ML engineer', 'AI team', 'AI expertise'
        ],
        'patterns': [
            r'(?:AI|ML).*(?:talent|skill|training|team)',
            r'(?:upskill|reskill)(?:ing)?.*(?:AI|data|digital)',
            r'hir(?:e|ed|ing).*(?:AI|ML|data).*(?:engineer|scientist)',
        ]
    },
    
    # ═══════════════════════════════════════════════════════════════════
    # CATEGORII NOI v6.0
    # ═══════════════════════════════════════════════════════════════════
    
    'Agentic AI Systems': {
        'description': 'Sisteme autonome, AI agents (TREND 2025)',
        'keywords': [
            'autonomous', 'agent', 'agentic', 'orchestration', 'multi-agent',
            'self-learning', 'adaptive system', 'AI agent', 'autonomous AI',
            'tool use', 'function calling', 'agent framework', 'AutoGPT',
            'computer use', 'browser agent', 'task automation', 'workflow agent'
        ],
        'patterns': [
            r'(?:autonomous|agentic)\s+(?:system|AI|agent|workflow)',
            r'(?:AI|intelligent)\s+agent(?:s)?',
            r'multi[\s-]?agent\s+(?:system|orchestration)',
            r'agent(?:ic)?\s+(?:RAG|retrieval|workflow)',
            r'(?:tool|function)\s+(?:use|calling)',
        ]
    },
    
    'Generative AI & LLMs': {
        'description': 'GenAI, LLM, content generation (TREND 2023-2025)',
        'keywords': [
            'generative AI', 'GenAI', 'large language model', 'LLM',
            'GPT', 'ChatGPT', 'content generation', 'text generation',
            'foundation model', 'transformer', 'Copilot', 'Claude',
            'Gemini', 'Llama', 'multimodal', 'text-to-image', 'DALL-E',
            'prompt engineering', 'fine-tuning', 'RAG', 'vector database'
        ],
        'patterns': [
            r'generativ(?:e)?\s*(?:AI|artificial intelligence)',
            r'(?:large\s+)?language\s+model(?:s)?',
            r'\bLLM(?:s)?\b',
            r'\bGPT[\s-]?[3-5]?\b',
            r'\bChatGPT\b',
            r'foundation\s+model(?:s)?',
            r'\bGenAI\b',
            r'(?:retrieval[\s-]?augmented|RAG)',
            r'(?:vector|embedding)\s+(?:database|store)',
        ]
    },
    
    'AI Infrastructure & MLOps': {
        'description': 'Infrastructură AI: cloud, MLOps, edge',
        'keywords': [
            'MLOps', 'cloud AI', 'edge computing', 'GPU', 'infrastructure',
            'platform', 'compute', 'model deployment', 'scalability',
            'AI cloud', 'model serving', 'inference', 'AI-as-a-service',
            'TPU', 'NPU', 'AI accelerator', 'NVIDIA', 'model hosting'
        ],
        'patterns': [
            r'\bMLOps\b',
            r'(?:AI|ML)\s+(?:platform|infrastructure|stack)',
            r'(?:edge|cloud)\s+(?:AI|computing|deployment)',
            r'(?:model|AI)\s+(?:deployment|serving|hosting)',
            r'(?:GPU|TPU|NPU)\s+(?:compute|cluster)',
            r'AI[\s-]?as[\s-]?a[\s-]?service',
        ]
    },
    
    'Explainability & AI Safety': {
        'description': 'XAI, responsible AI, safety',
        'keywords': [
            'explainability', 'XAI', 'interpretability', 'transparency',
            'safety', 'trustworthy', 'accountability', 'fairness',
            'AI safety', 'guardrails', 'hallucination', 'alignment',
            'red teaming', 'model evaluation', 'bias detection'
        ],
        'patterns': [
            r'(?:explainab|interpretab)(?:le|ility)',
            r'\bXAI\b',
            r'(?:trustworthy|transparent|accountable)\s+(?:AI|ML)',
            r'(?:AI|model)\s+(?:safety|alignment|guardrails)',
            r'(?:hallucination|confabulation)\s+(?:detection|prevention)',
        ]
    },
    
    'Research & Innovation': {
        'description': 'Cercetare AI, brevete, colaborări academice',
        'keywords': [
            'patent', 'research', 'lab', 'collaboration', 'academic',
            'publication', 'breakthrough', 'discovery', 'innovation',
            'R&D center', 'AI research', 'frontier research'
        ],
        'patterns': [
            r'(?:AI|ML)\s+(?:patent|research|lab|center)',
            r'research.*(?:AI|artificial intelligence)',
            r'collaborat(?:e|ion).*(?:university|academic)',
            r'(?:AI|ML)\s+(?:breakthrough|innovation)',
        ]
    },
    
    'AI Coding & Development': {
        'description': 'AI pentru dezvoltare software (NOU 2024-2025)',
        'keywords': [
            'AI coding', 'code generation', 'code completion', 'Copilot',
            'AI-assisted development', 'vibe coding', 'AI IDE',
            'code review AI', 'automated testing', 'AI debugging'
        ],
        'patterns': [
            r'(?:AI|ML)[\s-]?(?:assisted|powered)\s+(?:coding|development)',
            r'code\s+(?:generation|completion|suggestion)',
            r'(?:GitHub|Microsoft)\s+Copilot',
            r'(?:AI|automated)\s+(?:code review|testing|debugging)',
        ]
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# FALSE POSITIVES v6.0 - EXTINS
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# FALSE POSITIVES v6.1 - EXTINS (~60 patterns)
# ═══════════════════════════════════════════════════════════════════════════

FALSE_POSITIVE_PATTERNS = [
    # ═══════════════════════════════════════════════════════════════════
    # FORMAT GREȘIT / CODURI / IDs
    # ═══════════════════════════════════════════════════════════════════
    r'\bAI[\s-]?\d+',           # AI-123, AI 456
    r'\d+[\s-]?AI\b',           # 123-AI
    r'AI[\s-]?\d{4,}',          # AI cu numere lungi
    r'#AI\b', r'@AI\b',         # Hashtags
    r'\bAI\.(com|org|net|io)\b', # Domenii
    r'\.ai\b',                   # .ai TLD
    r'www\..*\.ai',              # URLs
    
    # ═══════════════════════════════════════════════════════════════════
    # ACRONIME NON-AI (extinse)
    # ═══════════════════════════════════════════════════════════════════
    r'\bSEAI\b', r'\bKAI\b', r'\bRAI(?:L|N)\b',
    r'\bAI(?:DS|MS)\b',          # AIDS, AIMS
    r'\bPAID\b', r'\bMAIN\b', r'\bGAIN\b', r'\bPAIN\b',
    r'\bSAID\b', r'\bWAIT\b', r'\bFAIL\b',
    r'\bFAIR\b(?!\s+(?:AI|algorithm|model|machine|lending))',
    r'\bAIR\b(?!.*(?:artificial|intelligence|AI[\s-]?powered))',
    
    # ═══════════════════════════════════════════════════════════════════
    # LOCAȚII (extinse)
    # ═══════════════════════════════════════════════════════════════════
    r'\bDubai\b', r'\bBangkok\b', r'\bThai(?:land)?\b',
    r'\bHawaii\b', r'\bSamurai\b', r'\bBonsai\b',
    r'\bShangh?ai\b', r'\bMumbai\b', r'\bChennai\b',
    
    # ═══════════════════════════════════════════════════════════════════
    # UNITĂȚI DE MĂSURĂ (NOU v6.1 - pentru DL, ML false pozitive)
    # ═══════════════════════════════════════════════════════════════════
    r'\bDL\b(?=\s*(?:of\s+)?(?:water|waste|emissions?))',
    r'\bML\b(?=\s*(?:of\s+)?(?:water|metric|tons?|gallons?|liters?))',
    r'\bML\b(?=\s*\d)',                    # ML urmat de număr (ML 9.5)
    r'\d+\.?\d*\s*(?:ML|DL)\b',            # 9.5 ML
    r'\bDL\b.*(?:metric|intensity|measurement)',
    r'\bML\b.*(?:metric|intensity|measurement)',
    
    # ═══════════════════════════════════════════════════════════════════
    # TEXT CORUPT / ENCODING ISSUES (NOU v6.1)
    # ═══════════════════════════════════════════════════════════════════
    r'[#$%&*@^]{3,}',                      # 3+ caractere speciale consecutive
    r'[A-Za-z0-9]{40,}',                   # Strings >40 chars fără spații
    r'\b[bcdfghjklmnpqrstvwxz]{5,}\b',     # 5+ consoane consecutive (imposibil)
    
    # ═══════════════════════════════════════════════════════════════════
    # BOARD OF DIRECTORS / GOVERNANCE FALSE POZITIVE (NOU v6.1)
    # ═══════════════════════════════════════════════════════════════════
    r'(?:BOD|board\s+of\s+directors?).*?(?:composed|experience|knowledge).*?AI',
    r'(?:communication|media|security),?\s*AI,?\s*(?:and\s+)?(?:cloud|digital)',
    r'(?:spouses?|lineal|descendants?|ascendants?).*?(?:AI|artificial)',
    r'directors?.*?(?:expertise|experience|knowledge).*?(?:AI|artificial)',
    r'Form\s+the\s+BOD.*?AI',
    r'BOD\s+regulations?.*?AI',
    r'skills?\s+gap.*?directors?.*?AI',
    
    # ═══════════════════════════════════════════════════════════════════
    # BIOGRAFIE / CV / CAREER FALSE POZITIVE (NOU v6.1)
    # ═══════════════════════════════════════════════════════════════════
    r'(?:CEO|CFO|CTO|COO|founder|president)\s+(?:of|at)\s+\w+\s*AI\b',
    r'career\s+highlights.*?(?:AI|artificial)',
    r'joined\s+the\s+board.*?(?:AI|artificial)',
    r'director\s+since\s+\d{4}.*?AI',
    r'board\s+committees?.*?(?:AI|artificial)',
    
    # ═══════════════════════════════════════════════════════════════════
    # EMBEDDING NON-AI (NOU v6.1)
    # ═══════════════════════════════════════════════════════════════════
    r'\b[Ee]mbedding\s+(?:policy|policies|values?|principles?|practices?)',
    r'\b[Ee]mbedding\s+(?:ESG|sustainability|diversity|inclusion|culture)',
    r'\b[Ee]mbedded\s+in\s+(?:our|the)\s+(?:culture|organization|strategy)',
    
    # ═══════════════════════════════════════════════════════════════════
    # HEADERS / STRUCTURĂ DOCUMENT FALSE POZITIVE (NOU v6.1)
    # ═══════════════════════════════════════════════════════════════════
    r'AI\s+Company\s+Business\s+Overview',     # Header repetitiv SK
    r'Special\s+Report.*?AI\s+Company',
    
    # ═══════════════════════════════════════════════════════════════════
    # CONTEXTE NON-AI SPECIFICE
    # ═══════════════════════════════════════════════════════════════════
    r'\bair\s+(?:quality|pollution|travel|cargo|freight|conditioning|transport)\b',
    r'\bfair\s+(?:value|market|trade|housing|lending|price)\b',
]


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT VALIDATORS v6.1 - pentru texte scurte (AI, ML, DL singure)
# ═══════════════════════════════════════════════════════════════════════════

# Dacă textul găsit e foarte scurt (<=3 chars), contextul TREBUIE să conțină
# cel puțin unul din acești termeni pentru a fi considerat valid
AI_CONTEXT_VALIDATORS = [
    'artificial intelligence', 'machine learning', 'deep learning',
    'neural network', 'algorithm', 'model training', 'prediction',
    'automat', 'intelligent system', 'chatbot', 'generative', 'LLM', 'GPT',
    'autonomous', 'analytics', 'natural language', 'computer vision',
    'data science', 'cognitive', 'transformer', 'inference',
]


# ═══════════════════════════════════════════════════════════════════════════
# ROBOTICS & RPA CLASSIFICATION v6.0
# ═══════════════════════════════════════════════════════════════════════════

TRADITIONAL_ROBOTICS_PATTERNS = [
    r'\bindustrial\s+robot(?:s|ics)?\b',
    r'\brobotic\s+arm(?:s)?\b',
    r'\bmanufacturing\s+robot(?:s)?\b',
    r'\bassembly\s+robot(?:s)?\b',
    r'\bwarehouse\s+robot(?:s)?\b',
    r'\bpick(?:ing)?[\s-]?(?:and[\s-]?)?pack(?:ing)?\s+robot(?:s)?\b',
]

AI_ROBOTICS_PATTERNS = [
    r'\b(?:AI|ML|intelligent)[\s-]?(?:powered|driven|enabled)\s+robot(?:s|ics)?\b',
    r'\bautonomous\s+robot(?:s|ics)?\b',
    r'\bcognitive\s+robot(?:s|ics)?\b',
    r'\brobot(?:s|ics)?.*(?:machine learning|neural network|computer vision|AI)\b',
    r'\b(?:machine learning|AI|deep learning).*robot(?:s|ics)?\b',
]

RPA_NON_AI_PATTERNS = [
    r'\brobotic\s+process\s+automation\b(?!.*(?:AI|ML|intelligent|cognitive))',
    r'\bRPA\b(?!.*(?:AI|ML|intelligent|cognitive|machine learning))',
    r'\brule[\s-]?based\s+automation\b',
]

RPA_AI_PATTERNS = [
    r'\b(?:intelligent|cognitive|AI[\s-]?powered)\s+(?:RPA|automation)\b',
    r'\bRPA.*(?:with|using|combined|enhanced).*(?:AI|ML|machine learning)\b',
    r'\bhyperautomation\b',
    r'\bintelligent\s+(?:process\s+)?automation\b',
]

# Document types
DOCUMENT_TYPES = ['Annual Report', 'Sustainability', 'Proxy', 'Quarterly Report']


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class DatabaseManager:
    """Manager pentru baza de date SQLite v6.0."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._connect()
    
    def _connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Conectat la DB: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Eroare conectare DB: {e}")
            raise
    
    def create_tables(self):
        cursor = self.conn.cursor()
        
        # Tabel 1: Referințe brute - ACTUALIZAT v6.0
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_references_raw (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                year INTEGER NOT NULL,
                position INTEGER,
                industry TEXT,
                sector TEXT,
                country TEXT,
                doc_type TEXT,
                page INTEGER,
                text TEXT,
                context TEXT,
                category TEXT,
                sentiment TEXT,
                sentiment_score REAL,
                semantic_score REAL,
                detection_method TEXT,
                source TEXT,
                robotics_type TEXT DEFAULT 'not_robotics',
                rpa_type TEXT DEFAULT 'not_rpa',
                sentiment_confidence TEXT DEFAULT 'standard',
                category_confidence REAL DEFAULT 1.0,
                reference_strength TEXT DEFAULT 'unknown',
                confidence_score REAL DEFAULT 0.0,
                confidence_reasons TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company, year, doc_type, page, text)
            )
        ''')

        # v6.1.1: asigură coloanele noi pentru FP scoring (compatibil DB vechi)
        for coldef in [
            ("reference_strength", "TEXT DEFAULT 'unknown'"),
            ("confidence_score", "REAL DEFAULT 0.0"),
            ("confidence_reasons", "TEXT DEFAULT ''"),
        ]:
            try:
                cursor.execute(f"ALTER TABLE ai_references_raw ADD COLUMN {coldef[0]} {coldef[1]}")
            except sqlite3.Error:
                pass

        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_company_year ON ai_references_raw(company, year)')
        
        # Tabel 2: Referințe deduplicate
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_references_deduplicated (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                year INTEGER NOT NULL,
                position INTEGER,
                industry TEXT,
                sector TEXT,
                country TEXT,
                text TEXT,
                context TEXT,
                category TEXT,
                sources TEXT,
                doc_count INTEGER,
                total_occurrences INTEGER,
                avg_sentiment_score REAL,
                avg_semantic_score REAL,
                original_refs TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company, year, context)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dedup_company_year ON ai_references_deduplicated(company, year)')
        
        # Tabel 3: AI Adoption Index
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS adoption_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                year INTEGER NOT NULL,
                position INTEGER,
                industry TEXT,
                sector TEXT,
                country TEXT,
                intensity_index REAL,
                semantic_index REAL,
                diversity_index REAL,
                sentiment_index REAL,
                maturity_index REAL,
                future_index REAL,
                commitment_index REAL,
                ai_adoption_index REAL,
                rank_in_year INTEGER,
                total_refs INTEGER,
                total_pages INTEGER,
                categories_used INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company, year)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_index_year ON adoption_index(year, ai_adoption_index DESC)')
        
        # Tabel 4: Index per industrie
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS adoption_index_industry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                industry TEXT NOT NULL,
                year INTEGER NOT NULL,
                avg_intensity_index REAL,
                avg_semantic_index REAL,
                avg_diversity_index REAL,
                avg_sentiment_index REAL,
                avg_maturity_index REAL,
                avg_future_index REAL,
                avg_commitment_index REAL,
                ai_adoption_index_industry REAL,
                rank_among_industries INTEGER,
                num_companies INTEGER,
                total_refs INTEGER,
                min_index REAL,
                max_index REAL,
                std_deviation REAL,
                companies_list TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(industry, year)
            )
        ''')
        
        self.conn.commit()
        logger.info("Tabele v6.0 create cu succes")
    
    def insert_raw_reference(self, ref: AIReference):
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO ai_references_raw
                (company, year, position, industry, sector, country, doc_type, page,
                 text, context, category, sentiment, sentiment_score, 
                 semantic_score, detection_method, source,
                 robotics_type, rpa_type, sentiment_confidence, category_confidence,
                 reference_strength, confidence_score, confidence_reasons)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ref.company, ref.year, ref.position, ref.industry,
                ref.sector, ref.country,
                ref.doc_type, ref.page, ref.text, ref.context,
                ref.category, ref.sentiment, ref.sentiment_score,
                ref.semantic_score, ref.detection_method, ref.source,
                ref.robotics_type, ref.rpa_type, ref.sentiment_confidence, ref.category_confidence,
                getattr(ref, 'reference_strength', 'unknown'), getattr(ref, 'confidence_score', 0.0), getattr(ref, 'confidence_reasons', '')
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Eroare inserare referință: {e}")
    
    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("Conexiune DB închisă")


# ═══════════════════════════════════════════════════════════════════════════
# FINAL MODUL 1
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("AI SEMANTIC ANALYZER v6.1.1 - MODUL 1: CORE & CONFIGURARE")
    print("=" * 80)
    print("\n✓ Module 1 v6.0.6 încărcat!")
    print(f"\nCategorii AI: {len(AI_CATEGORIES)}")
    print(f"False positive patterns: {len(FALSE_POSITIVE_PATTERNS)}")
    print(f"Robotics patterns: {len(TRADITIONAL_ROBOTICS_PATTERNS) + len(AI_ROBOTICS_PATTERNS)}")
