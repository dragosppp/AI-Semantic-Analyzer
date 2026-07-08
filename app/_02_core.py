"""
═══════════════════════════════════════════════════════════════════════════════
AI SEMANTIC ANALYZER - CORE & CONFIGURATION
═══════════════════════════════════════════════════════════════════════════════

Core module providing the foundational components: configuration, data models,
the SQLite database manager, and the shared false-positive pattern /
context-validator constants.

COMPONENTS:
    1. Configuration management
       - AnalyzerConfig dataclass
       - Interactive configuration wizard
       - JSON config save/load
       - Path validation

    2. Data models
       - AIReference: individual AI reference with metadata
       - DocumentResult: per-PDF processing result
       - AIAdoptionIndex: company-year adoption index

    3. Taxonomy wiring
       - Re-exports AI_APPLICATIONS / AI_TECHNOLOGIES from taxonomy.py
       - Combined AI_CATEGORIES

    4. Database operations
       - SQLite schema creation / migration
       - Raw + deduplicated reference storage
       - Adoption-index persistence

    5. False-positive patterns (~60 exclusions)
    6. AI context validators for short tokens (AI, ML, DL)

DEPENDENCIES:
    - taxonomy (local): dual-taxonomy module
    - pandas, numpy: data processing
    - sqlite3: persistence
    - sentence_transformers: semantic analysis (used downstream)
    - transformers (optional): FinBERT sentiment (used downstream)

Author: TeRa0
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

import json
import logging
import sqlite3
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pytesseract  # noqa: F401
    from PIL import Image  # noqa: F401
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    warnings.warn("pytesseract not available. OCR functionality disabled.")

try:
    from transformers import (  # noqa: F401
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )
    import torch  # noqa: F401
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    warnings.warn("transformers/torch not available. Using TextBlob fallback.")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════════
# RUNTIME CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

PARALLEL_THREADS = 5          # Default number of PDFs processed concurrently


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(
    log_file: str = "ai_analyzer.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return the project-wide logger."""
    log = logging.getLogger("AIAnalyzer")
    log.setLevel(level)
    log.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    log.addHandler(ch)

    return log


logger = setup_logging()


# ═══════════════════════════════════════════════════════════════════════════
# INTERACTIVE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CONFIG_FILE = "analyzer_config.json"


def load_saved_config() -> dict[str, Any] | None:
    """Load a previously saved configuration file, if present."""
    if Path(CONFIG_FILE).exists():
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"Could not load saved configuration: {exc}")
    return None


def save_config(config_dict: dict[str, Any]) -> None:
    """Persist the given configuration dict to disk."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Configuration saved to {CONFIG_FILE}")
    except Exception as exc:
        logger.error(f"Failed to save configuration: {exc}")


def get_user_input(prompt: str, default: str = "") -> str:
    """Prompt the user for a value, returning ``default`` if empty."""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input or default
    return input(f"{prompt}: ").strip()


def validate_path(path: str, path_type: str = "folder") -> bool:
    """Return True if the path exists and matches the requested kind."""
    p = Path(path)
    if path_type == "folder":
        return p.exists() and p.is_dir()
    if path_type == "file":
        return p.exists() and p.is_file()
    return False


def interactive_config() -> dict[str, Any]:
    """Run the interactive configuration wizard."""
    print("\n" + "=" * 70)
    print("AI SEMANTIC ANALYZER - CONFIGURATION")
    print("=" * 70)

    saved_config = load_saved_config()

    if saved_config:
        meta_csv = saved_config.get('company_metadata_csv',
                                    saved_config.get('fortune500_csv', 'N/A'))
        print("\n✓ Saved configuration found:")
        print(f"  - PDF folder:           {saved_config.get('input_folder', 'N/A')}")
        print(f"  - Company metadata CSV: {meta_csv}")
        print(f"  - Output folder:        {saved_config.get('output_folder', 'N/A')}")

        keep = input("\nKeep this configuration? (yes/no) [yes]: ").strip().lower()
        if keep in ("", "yes", "y", "da", "d"):
            return saved_config

    config_dict: dict[str, Any] = {}

    print("\n--- PDF LOCATION ---")
    while True:
        default_pdf = saved_config.get("input_folder", "") if saved_config else ""
        pdf_folder = get_user_input("Path to the PDF folder", default_pdf)
        if validate_path(pdf_folder, "folder"):
            config_dict["input_folder"] = pdf_folder
            pdf_count = len(list(Path(pdf_folder).glob("*.pdf")))
            print(f"  ✓ Found {pdf_count} PDF files")
            break
        print(f"  ✗ Folder '{pdf_folder}' does not exist.")

    print("\n--- COMPANY METADATA CSV (optional) ---")
    while True:
        default_csv = (saved_config.get("company_metadata_csv",
                                        saved_config.get("fortune500_csv", "")) if saved_config else "")
        csv_file = get_user_input(
            "Path to a company metadata CSV (position/sector/industry/country) or 'skip'",
            default_csv,
        )
        if csv_file.lower() == "skip" or csv_file == "":
            config_dict["company_metadata_csv"] = None
            break
        if validate_path(csv_file, "file"):
            config_dict["company_metadata_csv"] = csv_file
            break
        print(f"  ✗ File '{csv_file}' does not exist.")

    print("\n--- OUTPUT FOLDER ---")
    default_output = saved_config.get("output_folder", "RESULT") if saved_config else "RESULT"
    output_folder = get_user_input("Output folder", default_output)
    config_dict["output_folder"] = output_folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    print("\n--- SETTINGS ---")
    top_n_input = get_user_input("Process only the first N companies? (empty = all)", "")
    config_dict["top_n"] = int(top_n_input) if top_n_input.isdigit() else None

    save_choice = input("\nSave configuration? (yes/no) [yes]: ").strip().lower()
    if save_choice in ("", "yes", "y", "da", "d"):
        save_config(config_dict)

    return config_dict


# ═══════════════════════════════════════════════════════════════════════════
# ANALYZER CONFIG
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AnalyzerConfig:
    """Centralized configuration for the AI Semantic Analyzer."""

    # Paths
    input_folder: str = "input_pdfs"
    output_folder: str = "RESULT"
    database_name: str = "ai_adoption_analysis.db"
    company_metadata_csv: str | None = None

    # Semantic thresholds
    semantic_threshold: float = 0.60
    # Strict threshold for semantic-only hits (reduces false positives)
    semantic_threshold_strict: float = 0.68
    # Relaxed threshold used when HARD triggers / implementation signals exist
    semantic_threshold_relaxed: float = 0.60
    deduplication_threshold: float = 0.85

    # Context extraction
    context_chars: int = 150
    min_text_length: int = 100

    # Processing
    max_workers: int = 4
    batch_size: int = 10
    top_n: int | None = None
    parallel_threads: int = PARALLEL_THREADS

    # Filtering toggles
    filter_traditional_robotics: bool = True
    filter_traditional_rpa: bool = True

    # Reference context window
    context_sentences_before: int = 2
    context_sentences_after: int = 2
    max_context_length: int = 2000

    # AI Adoption Index weights (must sum to 1.0)
    weights: dict[str, float] = field(default_factory=lambda: {
        "intensity": 0.15,
        "semantic": 0.20,
        "diversity": 0.15,
        "sentiment": 0.15,
        "maturity": 0.15,
        "future": 0.10,
        "commitment": 0.10,
    })

    # Year range
    start_year: int = 2020
    end_year: int = 2025

    @classmethod
    def from_interactive(cls) -> AnalyzerConfig:
        """Build a config from the interactive wizard."""
        config_dict = interactive_config()
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> AnalyzerConfig:
        """Build a config from a plain dict (e.g. saved JSON)."""
        return cls(
            input_folder=config_dict.get("input_folder", "input_pdfs"),
            output_folder=config_dict.get("output_folder", "RESULT"),
            company_metadata_csv=config_dict.get(
                "company_metadata_csv", config_dict.get("fortune500_csv")
            ),
            top_n=config_dict.get("top_n"),
            parallel_threads=config_dict.get("parallel_threads", PARALLEL_THREADS),
        )

    def __post_init__(self) -> None:
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1.0 (currently {total_weight})")
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Config: semantic_threshold={self.semantic_threshold}, "
            f"strict={self.semantic_threshold_strict}, "
            f"robotics_filter={self.filter_traditional_robotics}, "
            f"parallel_threads={self.parallel_threads}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AIReference:
    """A single detected AI reference with its metadata."""

    # Required fields (must come first — no defaults)
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
    category_a: str          # Applications axis (A1–A7) or 'none'
    category_b: str          # Technologies axis (B1–B8) or 'none'
    detection_method: str
    sentiment: str
    sentiment_score: float
    semantic_score: float
    source: str

    # Optional fields (with defaults)
    robotics_type: str = "not_robotics"           # 'ai_robotics' | 'traditional_robotics' | 'not_robotics'
    rpa_type: str = "not_rpa"                     # 'ai_rpa' | 'traditional_rpa' | 'not_rpa'
    sentiment_confidence: str = "standard"        # 'standard' | 'governance_adjusted' | 'confirmed_negative'
    category_a_confidence: float = 0.0            # confidence of the Applications-axis match (0.0 if 'none')
    category_b_confidence: float = 0.0            # confidence of the Technologies-axis match (0.0 if 'none')

    # False-positive scoring
    reference_strength: str = "unknown"           # 'strong' | 'medium' | 'mention_only' | 'unknown'
    confidence_score: float = 0.0                 # 0..1
    confidence_reasons: str = ""                  # string / JSON for audit
    is_negated_context: bool = False              # negation filter flagged this ref
    # Parallel EU_Semantics (JRC AI Watch) classification
    eu_domain: str = "Unclassified"
    eu_subdomain: str = "Unclassified"
    eu_confidence: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentResult:
    """Outcome of processing a single PDF document."""

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
    references: list[AIReference] = field(default_factory=list)
    total_refs: int = 0
    pattern_refs: int = 0
    semantic_refs: int = 0
    categories_count: dict[str, int] = field(default_factory=dict)
    sentiment_distribution: dict[str, int] = field(default_factory=dict)
    processing_time: float = 0.0

    def add_reference(self, ref: AIReference) -> None:
        self.references.append(ref)
        self.total_refs += 1
        if ref.detection_method == "pattern":
            self.pattern_refs += 1
        elif ref.detection_method == "semantic":
            self.semantic_refs += 1
        for cat in (ref.category_a, ref.category_b):
            if cat and cat != 'none':
                self.categories_count[cat] = self.categories_count.get(cat, 0) + 1
        self.sentiment_distribution[ref.sentiment] = (
            self.sentiment_distribution.get(ref.sentiment, 0) + 1
        )


@dataclass
class AIAdoptionIndex:
    """AI Adoption Index for a single company-year."""

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
    # EU_Semantics parallel scoring (only diversity differs from classic)
    diversity_index_eu: float = 0.0
    ai_adoption_index_eu: float = 0.0
    total_refs: int = 0
    total_pages: int = 0
    categories_used: int = 0
    categories_used_eu: int = 0
    rank_in_year: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# AI CATEGORIES — DUAL TAXONOMY (imported from external module)
# ═══════════════════════════════════════════════════════════════════════════

from _01_taxonomy import (  # noqa: E402
    AI_APPLICATIONS,
    AI_TECHNOLOGIES,
    EU_SEMANTICS,
    EU_DOMAINS,
    EU_LEAF_UNITS,
    FALLBACK_NAME_TO_CODE,
)

# Combined dict exposed to downstream consumers.
AI_CATEGORIES: dict[str, dict] = {}
AI_CATEGORIES.update(AI_APPLICATIONS)
AI_CATEGORIES.update(AI_TECHNOLOGIES)

# EU_Semantics taxonomy (parallel classification) — exposed for detection/export.
EU_CATEGORIES: dict[str, dict] = EU_SEMANTICS


# ═══════════════════════════════════════════════════════════════════════════
# FALSE-POSITIVE PATTERNS (~60 exclusions)
# ═══════════════════════════════════════════════════════════════════════════

FALSE_POSITIVE_PATTERNS: list[str] = [
    # ═══════════════════════════════════════════════════════════════════
    # WRONG FORMAT / CODES / IDs
    # ═══════════════════════════════════════════════════════════════════
    r'\bAI[\s-]?\d+',            # AI-123, AI 456
    r'\d+[\s-]?AI\b',            # 123-AI
    r'AI[\s-]?\d{4,}',           # AI with long digit sequences
    r'#AI\b', r'@AI\b',          # hashtags / mentions
    r'\bAI\.(com|org|net|io)\b', # domains
    r'\.ai\b',                   # .ai TLD
    r'www\..*\.ai',              # URLs

    # ═══════════════════════════════════════════════════════════════════
    # NON-AI ACRONYMS (extended)
    # ═══════════════════════════════════════════════════════════════════
    r'\bSEAI\b', r'\bKAI\b', r'\bRAI(?:L|N)\b',
    r'\bAI(?:DS|MS)\b',          # AIDS, AIMS
    r'\bPAID\b', r'\bMAIN\b', r'\bGAIN\b', r'\bPAIN\b',
    r'\bSAID\b', r'\bWAIT\b', r'\bFAIL\b',
    r'\bFAIR\b(?!\s+(?:AI|algorithm|model|machine|lending))',
    r'\bAIR\b(?!.*(?:artificial|intelligence|AI[\s-]?powered))',

    # ═══════════════════════════════════════════════════════════════════
    # LOCATIONS (extended)
    # ═══════════════════════════════════════════════════════════════════
    r'\bDubai\b', r'\bBangkok\b', r'\bThai(?:land)?\b',
    r'\bHawaii\b', r'\bSamurai\b', r'\bBonsai\b',
    r'\bShangh?ai\b', r'\bMumbai\b', r'\bChennai\b',

    # ═══════════════════════════════════════════════════════════════════
    # MEASUREMENT UNITS (DL / ML false positives)
    # ═══════════════════════════════════════════════════════════════════
    r'\bDL\b(?=\s*(?:of\s+)?(?:water|waste|emissions?))',
    r'\bML\b(?=\s*(?:of\s+)?(?:water|metric|tons?|gallons?|liters?))',
    r'\bML\b(?=\s*\d)',                    # ML followed by a number (ML 9.5)
    r'\d+\.?\d*\s*(?:ML|DL)\b',            # 9.5 ML
    r'\bDL\b.*(?:metric|intensity|measurement)',
    r'\bML\b.*(?:metric|intensity|measurement)',

    # ═══════════════════════════════════════════════════════════════════
    # CORRUPTED TEXT / ENCODING ISSUES
    # ═══════════════════════════════════════════════════════════════════
    r'[#$%&*@^]{3,}',                      # 3+ consecutive special chars
    r'[A-Za-z0-9]{40,}',                   # strings >40 chars without spaces
    r'\b[bcdfghjklmnpqrstvwxz]{5,}\b',     # 5+ consecutive consonants (impossible)

    # ═══════════════════════════════════════════════════════════════════
    # BOARD OF DIRECTORS / GOVERNANCE FALSE POSITIVES
    # ═══════════════════════════════════════════════════════════════════
    r'(?:BOD|board\s+of\s+directors?).*?(?:composed|experience|knowledge).*?AI',
    r'(?:communication|media|security),?\s*AI,?\s*(?:and\s+)?(?:cloud|digital)',
    r'(?:spouses?|lineal|descendants?|ascendants?).*?(?:AI|artificial)',
    r'directors?.*?(?:expertise|experience|knowledge).*?(?:AI|artificial)',
    r'Form\s+the\s+BOD.*?AI',
    r'BOD\s+regulations?.*?AI',
    r'skills?\s+gap.*?directors?.*?AI',

    # ═══════════════════════════════════════════════════════════════════
    # BIOGRAPHY / CV / CAREER FALSE POSITIVES
    # ═══════════════════════════════════════════════════════════════════
    r'(?:CEO|CFO|CTO|COO|founder|president)\s+(?:of|at)\s+\w+\s*AI\b',
    r'career\s+highlights.*?(?:AI|artificial)',
    r'joined\s+the\s+board.*?(?:AI|artificial)',
    r'director\s+since\s+\d{4}.*?AI',
    r'board\s+committees?.*?(?:AI|artificial)',

    # ═══════════════════════════════════════════════════════════════════
    # NON-AI "EMBEDDING" SENSES
    # ═══════════════════════════════════════════════════════════════════
    r'\b[Ee]mbedding\s+(?:policy|policies|values?|principles?|practices?)',
    r'\b[Ee]mbedding\s+(?:ESG|sustainability|diversity|inclusion|culture)',
    r'\b[Ee]mbedded\s+in\s+(?:our|the)\s+(?:culture|organization|strategy)',

    # ═══════════════════════════════════════════════════════════════════
    # DOCUMENT HEADERS / STRUCTURE FALSE POSITIVES
    # ═══════════════════════════════════════════════════════════════════
    r'AI\s+Company\s+Business\s+Overview',     # repetitive SK header
    r'Special\s+Report.*?AI\s+Company',

    # ═══════════════════════════════════════════════════════════════════
    # SPECIFIC NON-AI CONTEXTS
    # ═══════════════════════════════════════════════════════════════════
    r'\bair\s+(?:quality|pollution|travel|cargo|freight|conditioning|transport)\b',
    r'\bfair\s+(?:value|market|trade|housing|lending|price)\b',
]


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT VALIDATORS — required when the matched token is very short (AI/ML/DL)
# ═══════════════════════════════════════════════════════════════════════════
#
# When the raw match is ≤3 chars, the surrounding context MUST contain at
# least one of these terms for the hit to be considered valid.

AI_CONTEXT_VALIDATORS: list[str] = [
    "artificial intelligence", "machine learning", "deep learning",
    "neural network", "algorithm", "model training", "prediction",
    "automat", "intelligent system", "chatbot", "generative", "LLM", "GPT",
    "autonomous", "analytics", "natural language", "computer vision",
    "data science", "cognitive", "transformer", "inference",
]


# ═══════════════════════════════════════════════════════════════════════════
# ROBOTICS & RPA CLASSIFICATION PATTERNS
# ═══════════════════════════════════════════════════════════════════════════

TRADITIONAL_ROBOTICS_PATTERNS: list[str] = [
    r'\bindustrial\s+robot(?:s|ics)?\b',
    r'\brobotic\s+arm(?:s)?\b',
    r'\bmanufacturing\s+robot(?:s)?\b',
    r'\bassembly\s+robot(?:s)?\b',
    r'\bwarehouse\s+robot(?:s)?\b',
    r'\bpick(?:ing)?[\s-]?(?:and[\s-]?)?pack(?:ing)?\s+robot(?:s)?\b',
]

AI_ROBOTICS_PATTERNS: list[str] = [
    r'\b(?:AI|ML|intelligent)[\s-]?(?:powered|driven|enabled)\s+robot(?:s|ics)?\b',
    r'\bautonomous\s+robot(?:s|ics)?\b',
    r'\bcognitive\s+robot(?:s|ics)?\b',
    r'\brobot(?:s|ics)?.*(?:machine learning|neural network|computer vision|AI)\b',
    r'\b(?:machine learning|AI|deep learning).*robot(?:s|ics)?\b',
]

RPA_NON_AI_PATTERNS: list[str] = [
    r'\brobotic\s+process\s+automation\b(?!.*(?:AI|ML|intelligent|cognitive))',
    r'\bRPA\b(?!.*(?:AI|ML|intelligent|cognitive|machine learning))',
    r'\brule[\s-]?based\s+automation\b',
]

RPA_AI_PATTERNS: list[str] = [
    r'\b(?:intelligent|cognitive|AI[\s-]?powered)\s+(?:RPA|automation)\b',
    r'\bRPA.*(?:with|using|combined|enhanced).*(?:AI|ML|machine learning)\b',
    r'\bhyperautomation\b',
    r'\bintelligent\s+(?:process\s+)?automation\b',
]

# Known document types (used by filename parser and export layer)
DOCUMENT_TYPES: list[str] = ["Annual Report", "Sustainability", "Proxy", "Quarterly Report"]


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class DatabaseManager:
    """SQLite database manager for the analyzer."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None
        self._connect()

    def _connect(self) -> None:
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            # WAL allows concurrent reads during writes and is more crash-safe.
            # synchronous=NORMAL is durable across app crashes (only at risk on power loss).
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            logger.info(f"Connected to DB: {self.db_path}")
        except sqlite3.Error as exc:
            logger.error(f"DB connection error: {exc}")
            raise

    def create_tables(self) -> None:
        """Create the full schema (idempotent)."""
        assert self.conn is not None
        cursor = self.conn.cursor()

        # Table 1: raw AI references
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
                category_a TEXT DEFAULT 'none',
                category_b TEXT DEFAULT 'none',
                sentiment TEXT,
                sentiment_score REAL,
                semantic_score REAL,
                detection_method TEXT,
                source TEXT,
                robotics_type TEXT DEFAULT 'not_robotics',
                rpa_type TEXT DEFAULT 'not_rpa',
                sentiment_confidence TEXT DEFAULT 'standard',
                category_a_confidence REAL DEFAULT 0.0,
                category_b_confidence REAL DEFAULT 0.0,
                reference_strength TEXT DEFAULT 'unknown',
                confidence_score REAL DEFAULT 0.0,
                confidence_reasons TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                eu_domain TEXT DEFAULT 'Unclassified',
                eu_subdomain TEXT DEFAULT 'Unclassified',
                eu_confidence REAL DEFAULT 0.0,
                UNIQUE(company, year, doc_type, page, text)
            )
        ''')

        # Backward-compatible migrations: ensure FP-scoring, EU and dual-axis columns exist.
        for col_name, col_def in (
            ("reference_strength", "TEXT DEFAULT 'unknown'"),
            ("confidence_score", "REAL DEFAULT 0.0"),
            ("confidence_reasons", "TEXT DEFAULT ''"),
            ("eu_domain", "TEXT DEFAULT 'Unclassified'"),
            ("eu_subdomain", "TEXT DEFAULT 'Unclassified'"),
            ("eu_confidence", "REAL DEFAULT 0.0"),
            ("category_a", "TEXT DEFAULT 'none'"),
            ("category_b", "TEXT DEFAULT 'none'"),
            ("category_a_confidence", "REAL DEFAULT 0.0"),
            ("category_b_confidence", "REAL DEFAULT 0.0"),
        ):
            try:
                cursor.execute(f"ALTER TABLE ai_references_raw ADD COLUMN {col_name} {col_def}")
            except sqlite3.Error:
                pass

        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_raw_company_year '
            'ON ai_references_raw(company, year)'
        )

        # Table 2: deduplicated references
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
                category_a TEXT DEFAULT 'none',
                category_b TEXT DEFAULT 'none',
                sources TEXT,
                doc_count INTEGER,
                total_occurrences INTEGER,
                avg_sentiment_score REAL,
                avg_semantic_score REAL,
                original_refs TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                eu_domain TEXT DEFAULT 'Unclassified',
                eu_subdomain TEXT DEFAULT 'Unclassified',
                UNIQUE(company, year, context)
            )
        ''')

        # EU and dual-axis columns (backward-compatible).
        for col_name, col_def in (
            ("eu_domain", "TEXT DEFAULT 'Unclassified'"),
            ("eu_subdomain", "TEXT DEFAULT 'Unclassified'"),
            ("category_a", "TEXT DEFAULT 'none'"),
            ("category_b", "TEXT DEFAULT 'none'"),
        ):
            try:
                cursor.execute(
                    f"ALTER TABLE ai_references_deduplicated ADD COLUMN {col_name} {col_def}"
                )
            except sqlite3.Error:
                pass

        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_dedup_company_year '
            'ON ai_references_deduplicated(company, year)'
        )

        # Table 3: AI Adoption Index
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
                diversity_index_eu REAL,
                ai_adoption_index_eu REAL,
                categories_used_eu INTEGER,
                UNIQUE(company, year)
            )
        ''')

        # EU columns (backward-compatible).
        for col_name, col_def in (
            ("diversity_index_eu", "REAL DEFAULT 0.0"),
            ("ai_adoption_index_eu", "REAL DEFAULT 0.0"),
            ("categories_used_eu", "INTEGER DEFAULT 0"),
        ):
            try:
                cursor.execute(f"ALTER TABLE adoption_index ADD COLUMN {col_name} {col_def}")
            except sqlite3.Error:
                pass

        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_index_year '
            'ON adoption_index(year, ai_adoption_index DESC)'
        )

        # Table 3b: per-doc-type adoption index — robustness check
        # for the pooled (company, year) headline number. Lets the paper
        # report the index separately for Annual Report vs ESG vs 10-K.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS adoption_index_by_doctype (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                year INTEGER NOT NULL,
                doc_type TEXT NOT NULL,
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
                total_refs INTEGER,
                total_pages INTEGER,
                categories_used INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                diversity_index_eu REAL,
                ai_adoption_index_eu REAL,
                categories_used_eu INTEGER,
                UNIQUE(company, year, doc_type)
            )
        ''')

        # EU columns (backward-compatible).
        for col_name, col_def in (
            ("diversity_index_eu", "REAL DEFAULT 0.0"),
            ("ai_adoption_index_eu", "REAL DEFAULT 0.0"),
            ("categories_used_eu", "INTEGER DEFAULT 0"),
        ):
            try:
                cursor.execute(
                    f"ALTER TABLE adoption_index_by_doctype ADD COLUMN {col_name} {col_def}"
                )
            except sqlite3.Error:
                pass

        cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_index_by_doctype_year '
            'ON adoption_index_by_doctype(year, doc_type, ai_adoption_index DESC)'
        )

        # Table 4: industry-level aggregated index
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
        logger.info("Database tables created successfully")

    def insert_raw_reference(self, ref: AIReference) -> None:
        """Insert a single raw AI reference (ignoring duplicates)."""
        assert self.conn is not None
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO ai_references_raw
                (company, year, position, industry, sector, country, doc_type, page,
                 text, context, category_a, category_b, sentiment, sentiment_score,
                 semantic_score, detection_method, source,
                 robotics_type, rpa_type, sentiment_confidence,
                 category_a_confidence, category_b_confidence,
                 reference_strength, confidence_score, confidence_reasons,
                 eu_domain, eu_subdomain, eu_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ref.company, ref.year, ref.position, ref.industry,
                ref.sector, ref.country,
                ref.doc_type, ref.page, ref.text, ref.context,
                ref.category_a, ref.category_b, ref.sentiment, ref.sentiment_score,
                ref.semantic_score, ref.detection_method, ref.source,
                ref.robotics_type, ref.rpa_type, ref.sentiment_confidence,
                ref.category_a_confidence, ref.category_b_confidence,
                getattr(ref, "reference_strength", "unknown"),
                getattr(ref, "confidence_score", 0.0),
                getattr(ref, "confidence_reasons", ""),
                getattr(ref, "eu_domain", "Unclassified"),
                getattr(ref, "eu_subdomain", "Unclassified"),
                getattr(ref, "eu_confidence", 0.0),
            ))
            self.conn.commit()
        except sqlite3.Error as exc:
            logger.error(f"Failed to insert reference: {exc}")

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            logger.info("DB connection closed")


# ═══════════════════════════════════════════════════════════════════════════
# MODULE SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("AI SEMANTIC ANALYZER - CORE MODULE")
    print("=" * 80)
    print("\n✓ Core module loaded successfully.")
    print(f"\nAI categories:                 {len(AI_CATEGORIES)}")
    print(f"False-positive patterns:       {len(FALSE_POSITIVE_PATTERNS)}")
    print(
        f"Robotics patterns:             "
        f"{len(TRADITIONAL_ROBOTICS_PATTERNS) + len(AI_ROBOTICS_PATTERNS)}"
    )
