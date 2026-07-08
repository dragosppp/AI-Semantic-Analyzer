"""
═══════════════════════════════════════════════════════════════════════════════
AI SEMANTIC ANALYZER - INTERACTIVE MENU INTERFACE
═══════════════════════════════════════════════════════════════════════════════

Interactive command-line interface that orchestrates the full pipeline.

FEATURES:
    1️⃣  Process PDFs & Extract AI References
        - Multi-threaded PDF processing
        - Pattern matching + semantic analysis
        - Dual taxonomy classification (classic + EU_Semantics)
        - False positive filtering
        - Context extraction

    2️⃣  Analyze Existing Dataset
        - Statistical analysis across companies/years/sectors
        - Temporal trend analysis
        - Category distribution analysis

    3️⃣  Deduplicate References
        - Semantic deduplication (cosine similarity)
        - Configurable similarity threshold
        - Preserves highest confidence instances

    4️⃣  Sentiment Analysis
        - FinBERT sentiment scoring
        - Context-aware sentiment classification
        - Batch processing with progress tracking

    5️⃣  Generate Visualizations
        - Interactive Plotly charts (HTML)
        - Temporal trends, category distributions
        - Company comparisons, sector analysis

    6️⃣  Export Data
        - Excel: Raw + deduplicated datasets
        - JSON: Structured export with metadata
        - SQLite: Full database export

    7️⃣  Database Management
        - View statistics
        - Query interface
        - Backup/restore

    8️⃣  Configuration
        - Save/load analysis settings
        - Update company metadata list
        - Configure semantic thresholds

MENU WORKFLOW:
    Main Menu → Select Option → Execute → Return to Menu

    Each function includes progress indicators (tqdm), error handling,
    result summaries, and export confirmations.

INTEGRATION:
    Orchestrates all modules:
    - core: Config, models, DB
    - detection: PDF processing & reference detection
    - analysis: Statistics, deduplication, sentiment
    - export: Excel, JSON, Plotly exports

Author: TeRa0
Part of: AI Semantic Analyzer
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import re
import sys
import json
import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

# Import all modules
try:
    from _02_core import (
        logger, AnalyzerConfig, AIReference, DocumentResult,
        AIAdoptionIndex, DatabaseManager, AI_CATEGORIES,
        load_saved_config, save_config, CONFIG_FILE,
        PARALLEL_THREADS
    )
    from _03_detection import (
        PDFTextExtractor, AIReferenceDetector, FilenameParser,
        SemanticModelLoader, get_ai_description_embeddings
    )
    from _04_analysis import (
        FinBERTSentimentAnalyzer, ImprovedSentimentAnalyzer,
        SemanticDeduplicator, AIAdoptionIndexCalculator
    )
    from _05_export import (
        ExcelExporter, VisualizationGenerator, GroupAggregator,
        AnalysisPipeline
    )
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Ensure all module files (_02_core.py, _03_detection.py, _04_analysis.py, _05_export.py) are in the same folder.")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# EXTENDED DATABASE MANAGER - with processed document tracking and text_status
# ═══════════════════════════════════════════════════════════════════════════════

class ExtendedDatabaseManager(DatabaseManager):
    """Extended Database Manager with processed document tracking and text_status."""

    def create_tables(self):
        """Create all tables, including processed_documents with text_status."""
        # Call the parent class method
        super().create_tables()

        cursor = self.conn.cursor()

        # Table for tracking processed documents (includes text_status)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT UNIQUE NOT NULL,
                company TEXT,
                year INTEGER,
                position INTEGER,
                industry TEXT,
                doc_type TEXT,
                total_pages INTEGER,
                text_length INTEGER,
                refs_found INTEGER DEFAULT 0,
                first_processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                process_count INTEGER DEFAULT 1,
                file_hash TEXT,
                text_status TEXT DEFAULT 'valid'
            )
        """)
        
        # Add occurrence_count column to ai_references_raw if it doesn't exist
        try:
            cursor.execute("ALTER TABLE ai_references_raw ADD COLUMN occurrence_count INTEGER DEFAULT 1")
            logger.info("Added column occurrence_count to ai_references_raw")
        except:
            pass  # Column already exists

        # Add ref_hash column for unique identification
        try:
            cursor.execute("ALTER TABLE ai_references_raw ADD COLUMN ref_hash TEXT")
            logger.info("Added column ref_hash to ai_references_raw")
        except:
            pass

        # Dual taxonomy support + FP meta columns
        try:
            cursor.execute("ALTER TABLE ai_references_raw ADD COLUMN reference_strength TEXT DEFAULT 'unknown'")
            logger.info("Added column reference_strength to ai_references_raw")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE ai_references_raw ADD COLUMN confidence_score REAL DEFAULT 0.0")
            logger.info("Added column confidence_score to ai_references_raw")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE ai_references_raw ADD COLUMN confidence_reasons TEXT DEFAULT ''")
            logger.info("Added column confidence_reasons to ai_references_raw")
        except:
            pass

        # Add text_status column if it doesn't exist (for existing DBs)
        try:
            cursor.execute("ALTER TABLE processed_documents ADD COLUMN text_status TEXT DEFAULT 'valid'")
            logger.info("Added column text_status to processed_documents")
        except:
            pass  # Column already exists

        # Index for fast lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_source ON processed_documents(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ref_hash ON ai_references_raw(ref_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_text_status ON processed_documents(text_status)")

        self.conn.commit()
        logger.info("✓ Extended tables created/verified")

    def is_document_processed(self, source: str) -> bool:
        """Check whether a document has already been processed."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM processed_documents WHERE source = ?", (source,))
        return cursor.fetchone() is not None

    def get_processed_documents(self) -> set:
        """Return the set of processed document sources."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT source FROM processed_documents")
        return set(row[0] for row in cursor.fetchall())
    
    def mark_document_processed(self, doc_result: 'DocumentResult', refs_found: int,
                               file_hash: str = None, text_status: str = 'valid'):
        """
        Mark a document as processed, or update it if it already exists.

        Args:
            doc_result: The document processing result
            refs_found: Number of AI references found
            file_hash: File hash (optional)
            text_status: Status of the extracted text - 'valid', 'corrupted_ocr_success',
                        'corrupted_ocr_failed', 'ocr_needed', 'empty', 'error'
        """
        cursor = self.conn.cursor()

        # Check whether the document already exists
        cursor.execute("SELECT id, process_count FROM processed_documents WHERE source = ?",
                       (doc_result.source,))
        existing = cursor.fetchone()

        if existing:
            # Update — increment process_count
            cursor.execute("""
                UPDATE processed_documents
                SET last_processed_at = CURRENT_TIMESTAMP,
                    process_count = process_count + 1,
                    refs_found = ?,
                    file_hash = COALESCE(?, file_hash),
                    text_status = ?
                WHERE source = ?
            """, (refs_found, file_hash, text_status, doc_result.source))
            logger.debug(f"Updated document: {doc_result.source} (run #{existing[1]+1}, status={text_status})")
        else:
            # Insert new record
            cursor.execute("""
                INSERT INTO processed_documents
                (source, company, year, position, industry, doc_type,
                 total_pages, text_length, refs_found, file_hash, text_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (doc_result.source, doc_result.company, doc_result.year,
                  doc_result.position, doc_result.industry, doc_result.doc_type,
                  doc_result.total_pages, doc_result.text_length, refs_found,
                  file_hash, text_status))
            logger.debug(f"New document processed: {doc_result.source} (status={text_status})")
        
        self.conn.commit()
    
    def insert_or_update_reference(self, ref: 'AIReference') -> tuple[bool, bool]:
        """
        Insert or update a reference.
        Returns: (is_new, was_updated)
        """
        cursor = self.conn.cursor()

        # Generate unique hash for the reference
        ref_hash = self._generate_ref_hash(ref)

        # Check whether the reference already exists (via UNIQUE constraint fields)
        cursor.execute("""
            SELECT id, occurrence_count FROM ai_references_raw 
            WHERE company = ? AND year = ? AND doc_type = ? AND page = ? AND text = ?
        """, (ref.company, ref.year, ref.doc_type, ref.page, ref.text))
        
        existing = cursor.fetchone()
        
        if existing:
            # Reference exists — increment occurrence_count
            ref_id, current_count = existing
            new_count = (current_count or 1) + 1
            
            cursor.execute("""
                UPDATE ai_references_raw 
                SET occurrence_count = ?,
                    ref_hash = COALESCE(ref_hash, ?),
                    sentiment_score = ?,
                    semantic_score = ?,
                    reference_strength = COALESCE(?, reference_strength),
                    confidence_score = CASE WHEN ? > COALESCE(confidence_score, 0) THEN ? ELSE COALESCE(confidence_score, 0) END,
                    confidence_reasons = CASE WHEN ? != '' THEN ? ELSE COALESCE(confidence_reasons, '') END
                WHERE id = ?
            """, (new_count, ref_hash, ref.sentiment_score, ref.semantic_score,
                  getattr(ref, "reference_strength", None),
                  float(getattr(ref, "confidence_score", 0.0) or 0.0),
                  float(getattr(ref, "confidence_score", 0.0) or 0.0),
                  getattr(ref, "confidence_reasons", "") or "",
                  getattr(ref, "confidence_reasons", "") or "",
                  ref_id))
            
            self.conn.commit()
            logger.debug(f"Reference updated (occurrence={new_count}): {ref.text[:50]}...")
            return (False, True)  # Not new, but was updated
        else:
            # Reference does not exist — insert new
            try:
                cursor.execute("""
                    INSERT INTO ai_references_raw
                    (company, year, position, industry, sector, country, doc_type, page, text, context,
                     category_a, category_b, sentiment, sentiment_score, semantic_score, detection_method,
                     source, occurrence_count, ref_hash, reference_strength, confidence_score, confidence_reasons,
                     eu_domain, eu_subdomain, eu_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
                """, (ref.company, ref.year, ref.position, ref.industry,
                      getattr(ref, "sector", "Unknown"), getattr(ref, "country", "Unknown"),
                      ref.doc_type, ref.page, ref.text, ref.context,
                      ref.category_a, ref.category_b, ref.sentiment, ref.sentiment_score, ref.semantic_score,
                      ref.detection_method, ref.source, ref_hash,
                      getattr(ref, "reference_strength", "unknown"),
                      float(getattr(ref, "confidence_score", 0.0) or 0.0),
                      getattr(ref, "confidence_reasons", "") or "",
                      getattr(ref, "eu_domain", "Unclassified"),
                      getattr(ref, "eu_subdomain", "Unclassified"),
                      float(getattr(ref, "eu_confidence", 0.0) or 0.0)))
                self.conn.commit()
                logger.debug(f"New reference inserted: {ref.text[:50]}...")
                return (True, False)  # Is new

            except Exception as e:
                # If UNIQUE constraint still fires (race condition), update instead
                if "UNIQUE constraint failed" in str(e):
                    logger.warning(f"UNIQUE constraint — updating existing reference")
                    cursor.execute("""
                        UPDATE ai_references_raw 
                        SET occurrence_count = COALESCE(occurrence_count, 1) + 1,
                            ref_hash = COALESCE(ref_hash, ?)
                        WHERE company = ? AND year = ? AND doc_type = ? AND page = ? AND text = ?
                    """, (ref_hash, ref.company, ref.year, ref.doc_type, ref.page, ref.text))
                    self.conn.commit()
                    return (False, True)
                else:
                    raise e
    
    def _generate_ref_hash(self, ref: 'AIReference') -> str:
        """
        Generate a unique SHA-256 hash for a reference.

        Hash includes:
        - Company
        - Year
        - Doc type
        - Page
        - Text (first 200 characters for maximum uniqueness)

        Returns:
            SHA-256 hash (32 hex characters)
        """
        # Normalize text for consistency
        normalized_text = ' '.join(ref.text.split())[:200]
        # Build the unique string
        unique_str = f"{ref.company}|{ref.year}|{ref.doc_type}|{ref.page}|{normalized_text}"
        # Generate SHA-256 and return the first 32 characters
        return hashlib.sha256(unique_str.encode('utf-8')).hexdigest()[:32]

    def get_processing_stats(self) -> dict:
        """Get detailed processing statistics, including text_status breakdown."""
        cursor = self.conn.cursor()

        stats = {}

        # Processed documents
        cursor.execute("SELECT COUNT(*) FROM processed_documents")
        stats['total_documents_processed'] = cursor.fetchone()[0]

        # Documents processed more than once
        cursor.execute("SELECT COUNT(*) FROM processed_documents WHERE process_count > 1")
        stats['documents_reprocessed'] = cursor.fetchone()[0]

        # Total processing runs (sum of process_count)
        cursor.execute("SELECT SUM(process_count) FROM processed_documents")
        stats['total_processing_runs'] = cursor.fetchone()[0] or 0
        
        # References
        try:
            cursor.execute("SELECT COUNT(*), SUM(COALESCE(occurrence_count, 1)) FROM ai_references_raw")
            row = cursor.fetchone()
            stats['unique_references'] = row[0] or 0
            stats['total_occurrences'] = row[1] or 0
        except:
            cursor.execute("SELECT COUNT(*) FROM ai_references_raw")
            stats['unique_references'] = cursor.fetchone()[0] or 0
            stats['total_occurrences'] = stats['unique_references']
        
        # Documents with no references
        cursor.execute("SELECT COUNT(*) FROM processed_documents WHERE refs_found = 0")
        stats['documents_without_refs'] = cursor.fetchone()[0]

        # text_status statistics
        try:
            cursor.execute("""
                SELECT text_status, COUNT(*) FROM processed_documents 
                GROUP BY text_status
            """)
            stats['by_text_status'] = {row[0] or 'unknown': row[1] for row in cursor.fetchall()}
        except:
            stats['by_text_status'] = {}
        
        # Documents with text issues
        try:
            cursor.execute("""
                SELECT COUNT(*) FROM processed_documents 
                WHERE text_status IN ('corrupted_ocr_failed', 'ocr_needed', 'empty', 'error')
            """)
            stats['documents_with_text_issues'] = cursor.fetchone()[0] or 0
        except:
            stats['documents_with_text_issues'] = 0
        
        return stats
    
    def insert_raw_reference(self, ref: 'AIReference'):
        """Compatibility method for existing code. Delegates to insert_or_update_reference."""
        return self.insert_or_update_reference(ref)


# ═══════════════════════════════════════════════════════════════════════════════
# METADATA EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class MetadataExtractor:
    """Metadata extractor from filename."""

    @staticmethod
    def is_chinese_document(filename: str) -> bool:
        """Check whether the document is a Chinese-language document."""
        filename_clean = filename.strip()
        return (filename_clean.endswith(' CN') or 
                filename_clean.endswith('_CN') or 
                filename_clean.endswith('-CN') or
                filename_clean.upper().endswith(' CN'))


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT PROCESSOR - with text_status return
# ═══════════════════════════════════════════════════════════════════════════════

class DocumentProcessor:
    """
    PDF document processor.

    process_pdf() returns (DocumentResult, text_status)
    instead of just DocumentResult, to enable text status tracking.
    """
    
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.pdf_extractor = PDFTextExtractor(config)
        self.ai_detector = AIReferenceDetector(config)
        self.filename_parser = FilenameParser()
    
    def process_pdf(self, pdf_path: str, company_info: dict | None = None) -> tuple[DocumentResult | None, str]:
        """
        Process a single PDF and return (DocumentResult, text_status).
        Includes sector and country.
        """
        try:
            text, num_pages, text_status = self.pdf_extractor.extract_text_from_pdf(pdf_path)

            if not text or len(text) < 100:
                logger.warning(f"Insufficient text in {pdf_path}")
                return None, text_status if text_status else 'empty'

            filename = Path(pdf_path).name
            parsed = self.filename_parser.parse_filename(filename)

            # Supplement with company_info if available (includes sector, country)
            if company_info:
                parsed['position'] = company_info.get('position', parsed.get('position', 0))
                parsed['sector'] = company_info.get('sector', parsed.get('sector', 'Unknown'))
                parsed['industry'] = company_info.get('industry', parsed.get('industry', 'Unknown'))
                parsed['country'] = company_info.get('country', parsed.get('country', 'Unknown'))
                if not parsed.get('company'):
                    parsed['company'] = company_info.get('company', 'Unknown')
            
            # Create DocumentResult (includes sector, country)
            doc_result = DocumentResult(
                company=parsed.get('company', 'Unknown'),
                year=parsed.get('year', 2024),
                position=parsed.get('position', 0),
                sector=parsed.get('sector', 'Unknown'),
                industry=parsed.get('industry', 'Unknown'),
                country=parsed.get('country', 'Unknown'),
                doc_type=parsed.get('doc_type', 'annual report'),
                source=filename,
                total_pages=num_pages,
                text_length=len(text)
            )
            
            # Detect AI references (includes sector, country)
            references = self.ai_detector.detect_references(
                text=text,
                company=doc_result.company,
                year=doc_result.year,
                position=doc_result.position,
                sector=doc_result.sector,
                industry=doc_result.industry,
                country=doc_result.country,
                doc_type=doc_result.doc_type,
                source=doc_result.source
            )
            
            for ref in references:
                doc_result.add_reference(ref)
            
            logger.info(f"Processed {filename}: {len(references)} references found (text_status={text_status})")
            return doc_result, text_status

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return None, 'error'


# ═══════════════════════════════════════════════════════════════════════════════
# COMPANY METADATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_company_metadata(csv_path: str) -> dict[str, dict]:
    """Load company metadata from CSV (position/sector/industry/country)."""
    if not csv_path or not Path(csv_path).exists():
        logger.warning("Company metadata CSV not found; using filename-derived data")
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        
        company_col = None
        position_col = None
        sector_col = None
        industry_col = None
        country_col = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ['company', 'name']:
                company_col = col
            elif col_lower in ['rank', 'position']:
                position_col = col
            elif col_lower == 'sector':
                sector_col = col
            elif col_lower == 'industry':
                industry_col = col
            elif col_lower == 'country':
                country_col = col
        
        if not company_col:
            logger.warning("Company column not found in CSV")
            return {}

        data = {}
        for _, row in df.iterrows():
            company = str(row[company_col]).strip()
            data[company.lower()] = {
                'company': company,
                'position': int(row[position_col]) if position_col and pd.notna(row[position_col]) else 0,
                'sector': str(row[sector_col]).strip() if sector_col and pd.notna(row[sector_col]) else 'Unknown',
                'industry': str(row[industry_col]).strip() if industry_col and pd.notna(row[industry_col]) else 'Unknown',
                'country': str(row[country_col]).strip() if country_col and pd.notna(row[country_col]) else 'Unknown'
            }

        logger.info(f"✓ Loaded {len(data)} companies from metadata CSV (Sector, Industry, Country)")
        return data

    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return {}


def match_company(filename: str, company_metadata: dict[str, dict]) -> dict | None:
    """Find the company metadata matching the given filename."""
    if not company_metadata:
        return None

    base = Path(filename).stem.lower()

    # Direct lookup
    for company_key, company_data in company_metadata.items():
        if company_key in base:
            return company_data

    # Partial match
    base_parts = base.replace('-', ' ').replace('_', ' ').split()
    for company_key, company_data in company_metadata.items():
        company_parts = company_key.split()
        if any(part in base_parts for part in company_parts if len(part) > 3):
            return company_data
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MENU MANAGER - MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class MenuManager:
    """Manager for the interactive menu interface."""

    def __init__(self):
        self.config = None
        self.db_manager = None
        self.processor = None
        self.deduplicator = None
        self.index_calculator = None
        self.excel_exporter = None
        self.viz_generator = None
        self.industry_aggregator = None
        self.company_metadata = {}
        self._stats_cache = None
        self._session_processed = []
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def get_statistics(self, force_refresh: bool = False) -> dict:
        """Get current state statistics, including text_status breakdown."""
        if self._stats_cache and not force_refresh:
            return self._stats_cache
        
        stats = {
            'total_pdfs': 0, 'processed': 0, 'new': 0, 'chinese': 0,
            'by_year': {}, 'by_industry': {},
            'total_refs_raw': 0, 'total_refs_dedup': 0, 'companies_with_index': 0,
            'reprocessed_docs': 0, 'total_occurrences': 0,
            'docs_with_text_issues': 0, 'by_text_status': {}
        }
        
        if not self.config:
            return stats
        
        pdf_folder = Path(self.config.input_folder)
        if pdf_folder.exists():
            all_pdfs = list(pdf_folder.glob("*.pdf"))
            stats['total_pdfs'] = len(all_pdfs)
            for pdf in all_pdfs:
                if MetadataExtractor.is_chinese_document(pdf.stem):
                    stats['chinese'] += 1
        
        if self.db_manager:
            try:
                cursor = self.db_manager.conn.cursor()
                
                # Statistics from processed_documents
                cursor.execute("SELECT COUNT(*) FROM processed_documents")
                stats['processed'] = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(*) FROM processed_documents WHERE process_count > 1")
                stats['reprocessed_docs'] = cursor.fetchone()[0] or 0
                
                # References
                cursor.execute("SELECT COUNT(*) FROM ai_references_raw")
                stats['total_refs_raw'] = cursor.fetchone()[0] or 0

                # Total occurrences
                try:
                    cursor.execute("SELECT SUM(occurrence_count) FROM ai_references_raw")
                    stats['total_occurrences'] = cursor.fetchone()[0] or stats['total_refs_raw']
                except:
                    stats['total_occurrences'] = stats['total_refs_raw']
                
                cursor.execute("SELECT COUNT(*) FROM ai_references_deduplicated")
                stats['total_refs_dedup'] = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(DISTINCT company) FROM adoption_index")
                stats['companies_with_index'] = cursor.fetchone()[0] or 0
                
                cursor.execute("""
                    SELECT year, COUNT(DISTINCT company) FROM adoption_index 
                    GROUP BY year ORDER BY year
                """)
                stats['by_year'] = {row[0]: row[1] for row in cursor.fetchall()}
                
                cursor.execute("""
                    SELECT industry, COUNT(DISTINCT company) FROM adoption_index 
                    GROUP BY industry ORDER BY COUNT(*) DESC
                """)
                stats['by_industry'] = {row[0]: row[1] for row in cursor.fetchall()}
                
                # text_status statistics
                try:
                    cursor.execute("""
                        SELECT text_status, COUNT(*) FROM processed_documents
                        GROUP BY text_status
                    """)
                    stats['by_text_status'] = {row[0] or 'unknown': row[1] for row in cursor.fetchall()}

                    cursor.execute("""
                        SELECT COUNT(*) FROM processed_documents
                        WHERE text_status IN ('corrupted_ocr_failed', 'ocr_needed', 'empty', 'error')
                    """)
                    stats['docs_with_text_issues'] = cursor.fetchone()[0] or 0
                except:
                    pass

            except Exception as e:
                logger.warning(f"Error retrieving statistics: {e}")
        
        stats['new'] = stats['total_pdfs'] - stats['processed'] - stats['chinese']
        if stats['new'] < 0:
            stats['new'] = 0
            
        self._stats_cache = stats
        return stats
    
    def display_header(self):
        """Display the application header."""
        stats = self.get_statistics()

        print("\n╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║                         AI SEMANTIC ANALYZER                                  ║")
        print("║              AI Adoption Analysis (classic + EU_Semantics)                   ║")
        print("║      15 classic categories | 12 EU subdomains | GenAI 2025 | Thr 0.60        ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"📊 STATUS: {stats['total_pdfs']} total | {stats['processed']} processed | "
              f"{stats['new']} new | {stats['chinese']} CN (skipped)")
        if stats['reprocessed_docs'] > 0:
            print(f"   🔄 {stats['reprocessed_docs']} documents re-analyzed")
        if stats['docs_with_text_issues'] > 0:
            print(f"   ⚠️  {stats['docs_with_text_issues']} documents with text issues (option 1.6)")
        if stats['total_refs_raw'] > 0:
            print(f"🔍 REFS: {stats['total_refs_raw']:,} unique | {stats['total_occurrences']:,} total occurrences | "
                  f"{stats['total_refs_dedup']:,} deduplicated | {stats['companies_with_index']} companies with index")
    
    def display_file_format_info(self):
        """Display information about the accepted file format."""
        print("\n┌─────────────────────────────────────────────────────────────────────────────────┐")
        print("│ 📄 ACCEPTED PDF FILE FORMAT:                                                  │")
        print("├─────────────────────────────────────────────────────────────────────────────────┤")
        print("│  Format: \"POSITION. Company Name - YEAR - document type.pdf\"                  │")
        print("│                                                                               │")
        print("│  Examples:                                                                    │")
        print("│    ✓ 38. Bank of America - 2024 - annual report.pdf                          │")
        print("│    ✓ 1. Walmart - 2023 - sustainability report.pdf                           │")
        print("│    ✓ 150. Microsoft - 2025 - proxy.pdf                                       │")
        print("│                                                                               │")
        print("│  Document types: annual report, sustainability, proxy, 10-K, ESG             │")
        print("│  CN (Chinese) documents are automatically skipped                            │")
        print("│  Supported years: 2020-2025                                                  │")
        print("└─────────────────────────────────────────────────────────────────────────────────┘")
    
    def display_main_menu(self):
        """Display the main menu."""
        self.display_header()
        self.display_file_format_info()

        stats = self.get_statistics()

        # Calculate documents with issues for display
        docs_with_issues = stats.get('docs_with_text_issues', 0)

        print("\n═══════════════════════════════════════════════════════════════════════════════")
        print("📥 1. EXTRACTION (Document Processing)")
        print("═══════════════════════════════════════════════════════════════════════════════")
        print(f"   1.1  ALL documents (new + re-analysis) [{stats['total_pdfs'] - stats['chinese']} docs]")
        print(f"   1.2  BATCH by positions (new + re-analysis in range)")
        print(f"   1.3  NEW documents only - all [{stats['new']} available]")
        print(f"   1.4  NEW documents only - batch of N")
        print(f"   1.5  Re-check documents WITHOUT AI references")
        if docs_with_issues > 0:
            print(f"   1.6  Re-process documents with CORRUPTED TEXT (OCR) [{docs_with_issues} docs] ⚠️")
        else:
            print(f"   1.6  Re-process documents with CORRUPTED TEXT (OCR)")

        print("\n═══════════════════════════════════════════════════════════════════════════════")
        print("📊 2. ANALYSIS & DEDUPLICATION")
        print("═══════════════════════════════════════════════════════════════════════════════")
        print("   2.1  BATCH analysis (specific positions)")
        print("   2.2  Analyze ALL companies")
        print("   2.3  Analyze only NEW ones (from this session)")

        print("\n═══════════════════════════════════════════════════════════════════════════════")
        print("📈 3. AI ADOPTION INDEX CALCULATION (7 dimensions)")
        print("═══════════════════════════════════════════════════════════════════════════════")
        print("   3.1  BATCH calculation (specific positions)")
        print("   3.2  Calculate ALL companies")
        print("   3.3  Calculate only the NEW ones")

        print("\n═══════════════════════════════════════════════════════════════════════════════")
        print("📋 4. REPORTS")
        print("═══════════════════════════════════════════════════════════════════════════════")
        print("   4.1  Export RAW references (ai_references_raw.xlsx)")
        print("   4.2  Export DEDUPLICATED references")
        print("   4.3  Export company AI INDEX (ai_adoption_index.xlsx)")
        print("   4.4  Export full JSON (results.json)")
        print("   4.5  TEXT summary report")
        print("   4.6  ALL reports")
        print("   4.7  Export EU_Semantics classification (eu_classification.xlsx)")

        print("\n═══════════════════════════════════════════════════════════════════════════════")
        print("📊 5. CHARTS (Plotly Interactive)")
        print("═══════════════════════════════════════════════════════════════════════════════")
        print("   5.1  Company ranking (bar chart) - per year")
        print("   5.2  Top companies evolution (line chart)")
        print("   5.3  Dimension radar - top 10 companies")
        print("   5.4  Heatmap companies × years")
        print("   5.5  AI category distribution (pie chart)")
        print("   5.6  ALL available charts")
        print("   5.7  INDUSTRY charts")
        print("   5.8  SECTOR charts")
        print("   5.9  COUNTRY charts")
        print("   5.10 EU_Semantics sunburst (domains × subdomains)")
        print("   5.11 Taxonomy comparison (classic vs EU)")

        print("\n═══════════════════════════════════════════════════════════════════════════════")
        print("🔧 6. UTILITIES")
        print("═══════════════════════════════════════════════════════════════════════════════")
        print("   6.1  Statistics and summary")
        print("   6.2  Database integrity check")
        print("   6.3  Configure paths and settings")
        print("   6.4  Import industry mapping (CSV)")
        print("   6.5  Taxonomy info (classic + EU_Semantics)")
        print("   6.6  Document coverage matrix")

        print("\n═══════════════════════════════════════════════════════════════════════════════")
        print("🚀 7. FULL UPDATE (Automatic Pipeline)")
        print("═══════════════════════════════════════════════════════════════════════════════")
        print("   Automatically processes NEW documents and generates ALL outputs")

        print("\n═══════════════════════════════════════════════════════════════════════════════")
        print("   0.  EXIT")
        print("═══════════════════════════════════════════════════════════════════════════════")
    
    def get_position_range(self) -> tuple[int, int]:
        """Prompt the user for a position range."""
        print("\nEnter the company ranking range:")
        while True:
            try:
                start = input("  START position (1-500) [1]: ").strip()
                start = int(start) if start else 1
                end = input("  END position (1-500) [500]: ").strip()
                end = int(end) if end else 500
                if 1 <= start <= 500 and 1 <= end <= 500 and start <= end:
                    return (start, end)
                print("  ✗ Invalid positions. Please re-enter.")
            except ValueError:
                print("  ✗ Please enter valid numbers.")
    
    def get_pdfs_by_position_range(self, start: int, end: int,
                                   skip_processed: bool = False,
                                   skip_chinese: bool = True) -> list[str]:
        """Get PDFs within the specified position range."""
        folder_path = Path(self.config.input_folder)
        all_pdfs = list(folder_path.glob("*.pdf"))
        filtered = []

        for pdf in all_pdfs:
            # Skip Chinese documents
            if skip_chinese and MetadataExtractor.is_chinese_document(pdf.stem):
                continue

            # Extract position from filename
            match = re.match(r'^(\d{1,3})\.', pdf.stem)
            if match:
                position = int(match.group(1))
                if start <= position <= end:
                    filtered.append(pdf)

        # Skip already-processed documents if requested
        if skip_processed and self.db_manager:
            processed_sources = self.db_manager.get_processed_documents()
            filtered = [pdf for pdf in filtered if pdf.name not in processed_sources]

        filtered.sort()
        return [str(pdf) for pdf in filtered]

    def get_all_pdfs(self, skip_processed: bool = False, skip_chinese: bool = True) -> list[str]:
        """Get all PDFs from the input folder."""
        folder_path = Path(self.config.input_folder)
        if not folder_path.exists():
            return []

        all_pdfs = list(folder_path.glob("*.pdf"))

        # Skip Chinese documents
        if skip_chinese:
            all_pdfs = [pdf for pdf in all_pdfs
                        if not MetadataExtractor.is_chinese_document(pdf.stem)]

        # Skip already-processed documents if requested
        if skip_processed and self.db_manager:
            processed_sources = self.db_manager.get_processed_documents()
            all_pdfs = [pdf for pdf in all_pdfs if pdf.name not in processed_sources]
        
        all_pdfs.sort()
        return [str(pdf) for pdf in all_pdfs]
    
    def initialize_components(self):
        """Initialize all required components."""
        if self.db_manager is None:
            print("\n⏳ Initializing components...")

            # Load or create the configuration
            saved_config = load_saved_config()
            if saved_config:
                self.config = AnalyzerConfig.from_dict(saved_config)
            else:
                self.config = AnalyzerConfig.from_interactive()

            # Initialize the extended Database Manager
            db_path = str(Path(self.config.output_folder) / self.config.database_name)
            self.db_manager = ExtendedDatabaseManager(db_path)
            self.db_manager.create_tables()

            # Initialize the remaining components
            self.processor = DocumentProcessor(self.config)
            self.deduplicator = SemanticDeduplicator(self.config)
            self.index_calculator = AIAdoptionIndexCalculator(self.config)
            self.excel_exporter = ExcelExporter(self.config)
            self.viz_generator = VisualizationGenerator(self.config)
            self.group_aggregator = GroupAggregator(self.db_manager)

            # Load company metadata
            if self.config.company_metadata_csv:
                self.company_metadata = load_company_metadata(self.config.company_metadata_csv)

            print("✓ Components initialized!")


# ═══════════════════════════════════════════════════════════════════════════════
# END PART 1/3 - Continued in part 2
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# AI SEMANTIC ANALYZER - PART 2/3
# MenuManager continuation - processing methods and handlers 1.x, 2.x, 3.x
# ═══════════════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN BATCH PROCESSING METHOD
    # ═══════════════════════════════════════════════════════════════════════════

    def _process_batch(self, pdf_files: list[str]) -> tuple[list[DocumentResult], dict]:
        """
        Process a batch of PDFs in parallel.

        PDF extraction and AI detection run concurrently (ThreadPoolExecutor).
        All SQLite writes are serialized on the main thread via wait(FIRST_COMPLETED).
        The degree of parallelism is controlled by config.parallel_threads (default PARALLEL_THREADS).

        Returns:
            tuple[list[DocumentResult], dict]: Results and processing statistics
        """
        from tqdm import tqdm

        results = []
        stats = {
            'new_refs': 0,           # Completely new references
            'updated_refs': 0,       # Existing references with occurrence_count incremented
            'docs_new': 0,           # Documents processed for the first time
            'docs_reanalyzed': 0,    # Documents re-analyzed
            'docs_no_refs': 0,       # Documents with no AI references
            'errors': 0              # Processing errors
        }

        # Read the set of already-processed documents once (read-only, no lock needed)
        processed_sources = self.db_manager.get_processed_documents()
        n_threads = getattr(self.config, 'parallel_threads', PARALLEL_THREADS)

        def _process_one(pdf_path: str):
            """Worker: pure PDF extraction + detection — no DB access."""
            fname = Path(pdf_path).name
            reanalysis = fname in processed_sources
            company_info = match_company(pdf_path, self.company_metadata)
            result, text_status = self.processor.process_pdf(pdf_path, company_info)
            return fname, result, text_status, reanalysis

        # Bounded prime-and-top-up pattern: keep at most MAX_INFLIGHT futures alive at once.
        # Submitting all 180 PDFs at once let result buffers + queued task args accumulate
        # past available RAM, swapping the process to death. Capping in-flight tasks bounds
        # peak memory to roughly MAX_INFLIGHT × per-PDF result size.
        import gc
        import psutil
        MAX_INFLIGHT = n_threads + 4     # workers + minimal drain buffer
        RAM_HIGH = 80.0                  # pause new submissions above this %
        RAM_PANIC = 90.0                 # force gc + skip submission above this %
        SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

        # Process smaller PDFs first: they complete fast, freeing slots and warming
        # caches before the giants land. Avoids the failure mode where several
        # 500-page reports happen to land in the pool simultaneously.
        try:
            pdf_files = sorted(pdf_files, key=lambda p: Path(p).stat().st_size)
        except OSError:
            pass

        iter_files = iter(pdf_files)
        in_flight: dict = {}
        executor = ThreadPoolExecutor(max_workers=n_threads)

        def _submit_one() -> bool:
            """Submit the next PDF from iter_files. Returns False when none remain."""
            try:
                p = next(iter_files)
            except StopIteration:
                return False
            in_flight[executor.submit(_process_one, p)] = p
            return True

        def _try_submit_with_throttle() -> bool:
            """Submit only if RAM headroom allows; otherwise let in-flight drain."""
            pct = psutil.virtual_memory().percent
            if pct >= RAM_PANIC:
                gc.collect()
                return False
            if pct >= RAM_HIGH and len(in_flight) >= n_threads:
                return False
            return _submit_one()

        # Prime the pool — at most MAX_INFLIGHT outstanding tasks at any time
        for _ in range(MAX_INFLIGHT):
            if not _submit_one():
                break

        # Sticky progress bar pinned at the bottom (position=0, leave=True).
        # bar_format keeps {postfix} at the end so the spinner is always visible.
        pbar = tqdm(
            total=len(pdf_files),
            desc="Processing PDFs",
            position=0,
            leave=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
        )

        # Route stderr log output through tqdm.write() so log lines clear the bar
        # before printing, keeping the bar pinned at the bottom of the terminal.
        # The file handler is left alone — only the console handler is swapped.
        class _TqdmLoggingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    tqdm.write(self.format(record), end="\n")
                except Exception:
                    self.handleError(record)

        _tqdm_handler = _TqdmLoggingHandler()
        _saved_stream_handlers: list[logging.Handler] = []
        for h in list(logger.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                _tqdm_handler.setLevel(h.level)
                _tqdm_handler.setFormatter(h.formatter)
                logger.removeHandler(h)
                _saved_stream_handlers.append(h)
        logger.addHandler(_tqdm_handler)

        # Background spinner thread updates a braille frame + active-worker count every 1s.
        # If the bar pauses but the spinner keeps moving, the script is alive (DB write is slow).
        # If the spinner ALSO freezes, the main thread itself has hung.
        stop_spin = threading.Event()

        def _spinner() -> None:
            i = 0
            while not stop_spin.is_set():
                active = sum(1 for f in in_flight if not f.done())
                pct = psutil.virtual_memory().percent
                pbar.set_postfix_str(
                    f"{SPINNER_FRAMES[i % len(SPINNER_FRAMES)]} {active} active | RAM {pct:.0f}%",
                    refresh=True,
                )
                i += 1
                stop_spin.wait(1.0)

        spin_thread = threading.Thread(target=_spinner, daemon=True)
        spin_thread.start()

        _gc_tick = 0
        try:
            while in_flight:
                # Wait for at least one future to complete, with a 15-minute hard timeout
                # so a single hanging PDF cannot freeze the entire pipeline.
                done, _pending = wait(in_flight, return_when=FIRST_COMPLETED, timeout=900)
                if not done:
                    logger.error("Hard timeout: no PDF completed in 15 minutes; aborting batch")
                    for f in list(in_flight):
                        f.cancel()
                    break

                future = next(iter(done))
                pdf_path = in_flight.pop(future)
                pbar.update(1)
                _try_submit_with_throttle()  # top up only if RAM headroom allows

                try:
                    filename, result, text_status, is_reanalysis = future.result()
                except Exception as e:
                    filename = Path(pdf_path).name
                    logger.error(f"Error processing {pdf_path}: {e}")
                    stats['errors'] += 1
                    # Attempt to mark the document even in case of error
                    try:
                        cursor = self.db_manager.conn.cursor()
                        cursor.execute("""
                            INSERT OR IGNORE INTO processed_documents
                            (source, company, year, refs_found, process_count, text_status)
                            VALUES (?, 'ERROR', 0, 0, 1, 'error')
                        """, (filename,))
                        self.db_manager.conn.commit()
                        logger.info(f"  ⚠ Marked as processed with error: {filename}")
                    except:
                        pass
                    continue

                if result:
                    refs_new = 0
                    refs_updated = 0

                    # Process each detected reference (DB writes — main thread only)
                    for ref in result.references:
                        try:
                            is_new, was_updated = self.db_manager.insert_or_update_reference(ref)
                            if is_new:
                                refs_new += 1
                            elif was_updated:
                                refs_updated += 1
                        except Exception as ref_error:
                            logger.warning(f"Error inserting reference: {ref_error}")
                            # Fallback: attempt direct update
                            try:
                                cursor = self.db_manager.conn.cursor()
                                cursor.execute("""
                                    UPDATE ai_references_raw
                                    SET occurrence_count = COALESCE(occurrence_count, 1) + 1
                                    WHERE company = ? AND year = ? AND doc_type = ?
                                    AND page = ? AND text = ?
                                """, (ref.company, ref.year, ref.doc_type, ref.page, ref.text))

                                if cursor.rowcount > 0:
                                    refs_updated += 1
                                    self.db_manager.conn.commit()
                                else:
                                    # If it doesn't exist, insert new
                                    cursor.execute("""
                                        INSERT OR IGNORE INTO ai_references_raw
                                        (company, year, position, industry, sector, country, doc_type, page,
                                         text, context, category_a, category_b, sentiment, sentiment_score,
                                         semantic_score, detection_method, source, occurrence_count,
                                         reference_strength, confidence_score, confidence_reasons)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                                    """, (ref.company, ref.year, ref.position, ref.industry,
                                          getattr(ref, "sector", "Unknown"), getattr(ref, "country", "Unknown"),
                                          ref.doc_type, ref.page, ref.text, ref.context,
                                          ref.category_a, ref.category_b, ref.sentiment, ref.sentiment_score,
                                          ref.semantic_score, ref.detection_method, ref.source,
                                          getattr(ref, "reference_strength", "unknown"),
                                          float(getattr(ref, "confidence_score", 0.0) or 0.0),
                                          getattr(ref, "confidence_reasons", "") or ""))
                                    if cursor.rowcount > 0:
                                        refs_new += 1
                                    self.db_manager.conn.commit()
                            except Exception as fallback_error:
                                logger.error(f"Fallback error: {fallback_error}")
                                stats['errors'] += 1

                    # IMPORTANT: Always mark the document as processed
                    ref_count = len(result.references)
                    try:
                        self.db_manager.mark_document_processed(
                            result, ref_count,
                            text_status=text_status
                        )
                        logger.debug(f"✓ Marked as processed: {filename}")
                    except Exception as doc_error:
                        logger.error(f"ERROR marking document {filename}: {doc_error}")

                    # Drop heavy fields now — references are persisted in the DB,
                    # so retaining them on the result for the rest of the batch
                    # is pure overhead (200–500 char context strings × N refs).
                    result.references = []

                    # Update statistics
                    stats['new_refs'] += refs_new
                    stats['updated_refs'] += refs_updated

                    if ref_count == 0:
                        stats['docs_no_refs'] += 1

                    if is_reanalysis:
                        stats['docs_reanalyzed'] += 1
                    else:
                        stats['docs_new'] += 1

                    results.append(result)

                    status = "🔄 RE-ANALYSIS" if is_reanalysis else "🆕 NEW"
                    logger.info(f"  {status} {result.company} ({result.year}): {refs_new} new, {refs_updated} updated")

                else:
                    # result is None — document could not be processed
                    # But it must still be marked to avoid reprocessing
                    logger.warning(f"⚠ Document with no result (insufficient text?): {filename}")

                    # Create a minimal DocumentResult for marking
                    try:
                        from _03_detection import FilenameParser
                        parser = FilenameParser()
                        parsed = parser.parse_filename(filename)

                        minimal_result = DocumentResult(
                            company=parsed.get('company', 'Unknown'),
                            year=parsed.get('year', 2024),
                            position=parsed.get('position', 0),
                            industry=parsed.get('industry', 'Unknown'),
                            sector=parsed.get('sector', 'Unknown'),
                            country=parsed.get('country', 'Unknown'),
                            doc_type=parsed.get('doc_type', 'Unknown'),
                            source=filename,
                            total_pages=0,
                            text_length=0
                        )
                        self.db_manager.mark_document_processed(
                            minimal_result, 0,
                            text_status=text_status if text_status else 'empty'
                        )
                        logger.info(f"  ⚠ Marked as processed (no content): {filename}")
                        stats['docs_no_refs'] += 1

                        if is_reanalysis:
                            stats['docs_reanalyzed'] += 1
                        else:
                            stats['docs_new'] += 1

                    except Exception as mark_error:
                        logger.error(f"Could not mark document {filename}: {mark_error}")
                        stats['errors'] += 1

                # Force GC every 5 documents to reclaim torch tensors and numpy
                # arrays that the reference counter may not catch immediately.
                _gc_tick += 1
                if _gc_tick % 5 == 0:
                    gc.collect()
        finally:
            stop_spin.set()
            spin_thread.join(timeout=2)
            executor.shutdown(wait=True)
            pbar.close()
            # Restore the original console log handler now that the bar is gone.
            logger.removeHandler(_tqdm_handler)
            for h in _saved_stream_handlers:
                logger.addHandler(h)

        # Final commit for safety
        try:
            self.db_manager.conn.commit()
        except:
            pass

        if stats['errors'] > 0:
            print(f"\n⚠️  {stats['errors']} errors during processing")
        if stats['docs_no_refs'] > 0:
            print(f"📄 {stats['docs_no_refs']} documents with no AI references")

        return results, stats
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HANDLERS FOR OPTIONS 1.x - EXTRACTION
    # ═══════════════════════════════════════════════════════════════════════════

    def handle_option_1_1(self):
        """
        1.1 - Process ALL documents (new + re-analysis of existing ones).

        This option processes every PDF in the input folder:
        - New documents are analyzed for the first time
        - Already-processed documents are RE-ANALYZED (occurrence_count++)
        """
        self.initialize_components()

        # Get ALL documents (do NOT skip processed)
        pdf_files = self.get_all_pdfs(skip_processed=False, skip_chinese=True)

        if not pdf_files:
            print("\n✓ No documents to process in the folder.")
            return

        # Calculate statistics for display
        processed_sources = self.db_manager.get_processed_documents()
        new_count = sum(1 for pdf in pdf_files if Path(pdf).name not in processed_sources)
        reanalysis_count = len(pdf_files) - new_count

        print(f"\n{'═'*70}")
        print(f"📄 PROCESSING ALL DOCUMENTS")
        print(f"{'═'*70}")
        print(f"   Total documents: {len(pdf_files)}")
        print(f"   🆕 {new_count} NEW documents (first analysis)")
        print(f"   🔄 {reanalysis_count} documents for RE-ANALYSIS")
        print(f"\n   ℹ️  On re-analysis:")
        print(f"      • Existing references will have occurrence_count incremented")
        print(f"      • Newly discovered references will be added")

        proceed = input("\nProceed? (yes/no) [yes]: ").strip().lower()
        if proceed not in ['', 'yes', 'y']:
            print("Processing cancelled.")
            return

        # Process all documents
        results, stats = self._process_batch(pdf_files)
        self._session_processed = [(r.company, r.year) for r in results]

        # Display results
        print(f"\n{'═'*70}")
        print(f"✓ PROCESSING COMPLETE!")
        print(f"{'═'*70}")
        print(f"   📄 Documents processed:")
        print(f"      • {stats['docs_new']} NEW documents")
        print(f"      • {stats['docs_reanalyzed']} RE-ANALYZED documents")
        print(f"   📊 References:")
        print(f"      • {stats['new_refs']} NEW references found")
        print(f"      • {stats['updated_refs']} EXISTING references (occurrence++)")

        self._stats_cache = None
    
    def handle_option_1_2(self):
        """
        1.2 - Process BATCH by positions (new + re-analysis in range).

        Processes documents within the specified position range:
        - New documents are analyzed for the first time
        - Already-processed documents are RE-ANALYZED
        """
        self.initialize_components()

        # Prompt for position range
        start, end = self.get_position_range()

        # Get documents in range (do NOT skip processed)
        pdf_files = self.get_pdfs_by_position_range(start, end, skip_processed=False)

        if not pdf_files:
            print(f"\n✓ No documents found in range {start}-{end}.")
            return

        # Calculate statistics
        processed_sources = self.db_manager.get_processed_documents()
        new_count = sum(1 for pdf in pdf_files if Path(pdf).name not in processed_sources)
        reanalysis_count = len(pdf_files) - new_count

        print(f"\n{'═'*70}")
        print(f"📄 BATCH PROCESSING POSITIONS {start}-{end}")
        print(f"{'═'*70}")
        print(f"   Total documents in range: {len(pdf_files)}")
        print(f"   🆕 {new_count} NEW documents")
        print(f"   🔄 {reanalysis_count} documents for RE-ANALYSIS")

        proceed = input("\nProceed? (yes/no) [yes]: ").strip().lower()
        if proceed not in ['', 'yes', 'y']:
            print("Processing cancelled.")
            return

        # Process the batch
        results, stats = self._process_batch(pdf_files)
        self._session_processed = [(r.company, r.year) for r in results]

        # Display results
        print(f"\n{'═'*70}")
        print(f"✓ BATCH PROCESSING COMPLETE!")
        print(f"{'═'*70}")
        print(f"   📄 {stats['docs_new']} NEW + {stats['docs_reanalyzed']} RE-ANALYZED")
        print(f"   📊 {stats['new_refs']} new refs + {stats['updated_refs']} updated refs")

        self._stats_cache = None
    
    def handle_option_1_3(self):
        """
        1.3 - Process NEW documents only (all unprocessed).

        This option processes ONLY documents that have never been analyzed.
        Does not perform re-analysis.
        """
        self.initialize_components()

        # Get ONLY unprocessed documents
        pdf_files = self.get_all_pdfs(skip_processed=True, skip_chinese=True)

        if not pdf_files:
            print(f"\n{'═'*70}")
            print(f"✓ No NEW documents to process.")
            print(f"{'═'*70}")
            print(f"   All documents have already been analyzed.")
            print(f"   💡 Use option 1.1 or 1.2 for re-analysis.")
            return

        print(f"\n{'═'*70}")
        print(f"🆕 PROCESSING NEW DOCUMENTS")
        print(f"{'═'*70}")
        print(f"   {len(pdf_files)} NEW documents to process")
        print(f"   These have not been analyzed yet.")

        # Show first documents
        print(f"\n   First documents:")
        for i, pdf in enumerate(pdf_files[:5], 1):
            print(f"      {i}. {Path(pdf).name}")
        if len(pdf_files) > 5:
            print(f"      ... and {len(pdf_files) - 5} more documents")

        proceed = input("\nProceed? (yes/no) [yes]: ").strip().lower()
        if proceed not in ['', 'yes', 'y']:
            print("Processing cancelled.")
            return

        # Process new documents
        results, stats = self._process_batch(pdf_files)
        self._session_processed = [(r.company, r.year) for r in results]

        # Display results
        print(f"\n{'═'*70}")
        print(f"✓ NEW DOCUMENTS PROCESSING COMPLETE!")
        print(f"{'═'*70}")
        print(f"   📄 {len(results)} documents processed")
        print(f"   📊 {stats['new_refs']} references found")

        self._stats_cache = None
    
    def handle_option_1_4(self):
        """
        1.4 - Process NEW documents only — batch of N.

        Processes the first N documents that have never been analyzed.
        Useful for incremental processing.
        """
        self.initialize_components()

        # Get ONLY unprocessed documents
        new_pdfs = self.get_all_pdfs(skip_processed=True, skip_chinese=True)

        if not new_pdfs:
            print(f"\n{'═'*70}")
            print(f"✓ No NEW documents to process.")
            print(f"{'═'*70}")
            print(f"   All documents have already been analyzed.")
            print(f"   💡 Use option 1.1 or 1.2 for re-analysis.")
            return

        print(f"\n{'═'*70}")
        print(f"🆕 BATCH PROCESSING NEW DOCUMENTS")
        print(f"{'═'*70}")
        print(f"   {len(new_pdfs)} NEW documents available")

        # Prompt for number of documents
        while True:
            try:
                n_input = input(f"\n   How many documents to process? (1-{len(new_pdfs)}) [10]: ").strip()
                n = int(n_input) if n_input else 10
                if 1 <= n <= len(new_pdfs):
                    break
                print(f"   ✗ Enter a number between 1 and {len(new_pdfs)}")
            except ValueError:
                print("   ✗ Please enter a valid number.")

        # Select the first N documents
        batch_files = new_pdfs[:n]

        print(f"\n   Will process {n} NEW documents:")
        for i, pdf in enumerate(batch_files[:5], 1):
            print(f"      {i}. 🆕 {Path(pdf).name}")
        if n > 5:
            print(f"      ... and {n - 5} more documents")

        proceed = input("\nProceed? (yes/no) [yes]: ").strip().lower()
        if proceed not in ['', 'yes', 'y']:
            print("Processing cancelled.")
            return

        # Process the batch
        results, stats = self._process_batch(batch_files)
        self._session_processed = [(r.company, r.year) for r in results]

        # Calculate how many remain
        remaining = len(new_pdfs) - n

        # Display results
        print(f"\n{'═'*70}")
        print(f"✓ BATCH PROCESSING COMPLETE!")
        print(f"{'═'*70}")
        print(f"   📄 {len(results)} NEW documents processed")
        print(f"   📊 {stats['new_refs']} references found")
        if remaining > 0:
            print(f"   📌 {remaining} NEW documents still remaining to process")
        else:
            print(f"   ✓ All NEW documents have been processed!")

        self._stats_cache = None
    
    def handle_option_1_5(self):
        """
        1.5 - Re-check documents WITHOUT AI references.

        Re-processes documents that were analyzed but found no AI references
        (refs_found = 0). Useful for:
        - Verification with different settings/thresholds
        - Re-analysis after updating patterns
        """
        self.initialize_components()

        # Find documents with no references
        cursor = self.db_manager.conn.cursor()
        cursor.execute("SELECT source FROM processed_documents WHERE refs_found = 0")
        no_refs_docs = [row[0] for row in cursor.fetchall()]

        if not no_refs_docs:
            print(f"\n{'═'*70}")
            print(f"✓ No documents without AI references.")
            print(f"{'═'*70}")
            return

        # Build the full list of paths
        pdf_folder = Path(self.config.input_folder)
        pdf_files = []
        missing = []

        for source in no_refs_docs:
            pdf_path = pdf_folder / source
            if pdf_path.exists():
                pdf_files.append(str(pdf_path))
            else:
                missing.append(source)

        if missing:
            print(f"\n⚠ {len(missing)} files no longer exist in the folder")

        if not pdf_files:
            print(f"\n✓ No files to re-check.")
            return

        print(f"\n{'═'*70}")
        print(f"🔄 RE-CHECKING DOCUMENTS WITHOUT AI REFERENCES")
        print(f"{'═'*70}")
        print(f"   {len(pdf_files)} documents to re-check")
        print(f"   These were previously processed but found no AI references.")
        print(f"\n   On re-check:")
        print(f"      • AI references will be searched again")
        print(f"      • Any found will be added to the database")
        print(f"      • The document will be updated with the new reference count")

        # Show first documents
        print(f"\n   First documents:")
        for i, pdf in enumerate(pdf_files[:5], 1):
            print(f"      {i}. {Path(pdf).name}")
        if len(pdf_files) > 5:
            print(f"      ... and {len(pdf_files) - 5} more documents")

        proceed = input("\nProceed? (yes/no) [yes]: ").strip().lower()
        if proceed not in ['', 'yes', 'y']:
            print("Re-check cancelled.")
            return

        # Process documents
        results, stats = self._process_batch(pdf_files)
        self._session_processed = [(r.company, r.year) for r in results]

        # Display results
        print(f"\n{'═'*70}")
        print(f"✓ RE-CHECK COMPLETE!")
        print(f"{'═'*70}")
        print(f"   📄 {len(results)} documents re-checked")
        print(f"   📊 {stats['new_refs']} NEW references found")
        print(f"   🔍 {stats['updated_refs']} references updated")

        # How many still have no references
        cursor.execute("SELECT COUNT(*) FROM processed_documents WHERE refs_found = 0")
        still_no_refs = cursor.fetchone()[0] or 0
        if still_no_refs > 0:
            print(f"   🔭 {still_no_refs} documents still without AI references")

        self._stats_cache = None
    
    def handle_option_1_6(self):
        """
        1.6 - Re-process documents with corrupted text (OCR needed).

        Re-processes documents with text_status = 'corrupted_ocr_failed'
        or 'ocr_needed'. Useful for:
        - Retrying OCR after installing Tesseract
        - Verification with different OCR settings
        """
        self.initialize_components()

        # Find documents with corrupted text
        cursor = self.db_manager.conn.cursor()
        cursor.execute("""
            SELECT source, text_status, company, year
            FROM processed_documents
            WHERE text_status IN ('corrupted_ocr_failed', 'ocr_needed', 'empty')
        """)
        corrupted_docs = cursor.fetchall()

        if not corrupted_docs:
            print(f"\n{'═'*70}")
            print(f"✓ No documents with corrupted text.")
            print(f"{'═'*70}")
            print(f"   All documents have valid extracted text.")
            return

        print(f"\n{'═'*70}")
        print(f"🔧 DOCUMENTS WITH CORRUPTED / INCOMPLETE TEXT")
        print(f"{'═'*70}")
        print(f"   {len(corrupted_docs)} documents require re-processing")

        # Statistics per status
        status_counts = {}
        for _, status, _, _ in corrupted_docs:
            status_counts[status] = status_counts.get(status, 0) + 1

        print(f"\n   By status:")
        for status, count in status_counts.items():
            icon = "🔴" if status == 'corrupted_ocr_failed' else "🟡" if status == 'ocr_needed' else "⚪"
            print(f"      {icon} {status}: {count}")

        # Show first documents
        print(f"\n   First documents:")
        for i, (source, status, company, year) in enumerate(corrupted_docs[:5], 1):
            print(f"      {i}. [{status}] {company} ({year})")
        if len(corrupted_docs) > 5:
            print(f"      ... and {len(corrupted_docs) - 5} more documents")

        # Check if OCR is available
        ocr_available = self.processor.pdf_extractor.ocr_available
        if not ocr_available:
            print(f"\n   ⚠️  OCR is NOT available!")
            print(f"   To enable OCR, install:")
            print(f"      1. pip install pytesseract pillow")
            print(f"      2. Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
            print(f"\n   You can continue without OCR, but results may be limited.")
        else:
            print(f"\n   ✓ OCR available (Tesseract)")

        proceed = input("\nRe-process these documents? (yes/no) [yes]: ").strip().lower()
        if proceed not in ['', 'yes', 'y']:
            print("Re-processing cancelled.")
            return

        # Build the list of PDFs
        pdf_folder = Path(self.config.input_folder)
        pdf_files = []
        missing = []

        for source, _, _, _ in corrupted_docs:
            pdf_path = pdf_folder / source
            if pdf_path.exists():
                pdf_files.append(str(pdf_path))
            else:
                missing.append(source)

        if missing:
            print(f"\n⚠️  {len(missing)} files no longer exist in the folder")

        if not pdf_files:
            print(f"\n✗ No files to re-process.")
            return

        # Re-process the documents
        print(f"\n⏳ Re-processing {len(pdf_files)} documents...")
        results, stats = self._process_batch(pdf_files)
        self._session_processed = [(r.company, r.year) for r in results]

        # Check how many were recovered
        cursor.execute("""
            SELECT COUNT(*) FROM processed_documents
            WHERE text_status = 'valid' AND source IN ({})
        """.format(','.join('?' * len([Path(p).name for p in pdf_files])),
                   [Path(p).name for p in pdf_files]))

        # Display results
        print(f"\n{'═'*70}")
        print(f"✓ RE-PROCESSING COMPLETE!")
        print(f"{'═'*70}")
        print(f"   📄 {len(results)} documents re-processed")
        print(f"   📊 {stats['new_refs']} NEW references found")
        print(f"   🔄 {stats['updated_refs']} references updated")

        # Check final status
        cursor.execute("""
            SELECT text_status, COUNT(*) FROM processed_documents
            WHERE text_status IN ('corrupted_ocr_failed', 'ocr_needed', 'empty')
            GROUP BY text_status
        """)
        remaining = cursor.fetchall()
        if remaining:
            print(f"\n   📌 Documents still problematic:")
            for status, count in remaining:
                print(f"      • {status}: {count}")
        else:
            print(f"\n   ✓ All documents now have valid text!")

        self._stats_cache = None

    # ═══════════════════════════════════════════════════════════════════════════
    # HANDLERS FOR OPTIONS 2.x - ANALYSIS & DEDUPLICATION
    # ═══════════════════════════════════════════════════════════════════════════

    def handle_option_2_1(self):
        """2.1 - Analysis and deduplication batch by positions."""
        self.initialize_components()
        start, end = self.get_position_range()
        self._run_deduplication(position_range=(start, end))

    def handle_option_2_2(self):
        """2.2 - Analysis and deduplication for all companies."""
        self.initialize_components()
        self._run_deduplication(position_range=None)

    def handle_option_2_3(self):
        """2.3 - Analyze only new companies (from this session or not yet deduplicated)."""
        self.initialize_components()
        cursor = self.db_manager.conn.cursor()

        # Find company-year pairs that have not been deduplicated yet
        cursor.execute("""
            SELECT DISTINCT company, year FROM ai_references_raw
            WHERE (company, year) NOT IN (
                SELECT DISTINCT company, year FROM ai_references_deduplicated
            )
        """)
        new_pairs = cursor.fetchall()

        if not new_pairs:
            print("\n✓ No new companies to analyze.")
            print("   All references have already been deduplicated.")
            return

        print(f"\n📊 {len(new_pairs)} company-year pairs to analyze.")
        self._run_deduplication(company_year_pairs=new_pairs)

    def _run_deduplication(self, position_range: tuple[int, int] | None = None,
                           company_year_pairs: list[tuple] | None = None):
        """
        Run the semantic deduplication process.

        Args:
            position_range: Tuple (start, end) for position-based filtering
            company_year_pairs: List of (company, year) tuples for targeted processing
        """
        from tqdm import tqdm
        cursor = self.db_manager.conn.cursor()

        # Determine which pairs to process
        if company_year_pairs:
            pairs = company_year_pairs
        else:
            query = "SELECT DISTINCT company, year, position FROM ai_references_raw"
            if position_range:
                query += f" WHERE position >= {position_range[0]} AND position <= {position_range[1]}"
            cursor.execute(query)
            pairs = [(row[0], row[1]) for row in cursor.fetchall()]
        
        if not pairs:
            print("\n✓ No data to process.")
            return

        print(f"\n⏳ Deduplication for {len(pairs)} company-year pairs...")

        total_original = 0
        total_dedup = 0

        for company, year in tqdm(pairs, desc="Deduplication"):
            # Fetch raw references for this company-year
            cursor.execute("""
                SELECT * FROM ai_references_raw WHERE company = ? AND year = ?
            """, (company, year))
            rows = cursor.fetchall()
            
            if not rows:
                continue
            
            # Build AIReference objects (name-based access — robust to column order).
            refs = []
            for row in rows:
                keys = row.keys()
                refs.append(AIReference(
                    company=row['company'], year=row['year'], position=row['position'],
                    industry=row['industry'],
                    sector=row['sector'] if 'sector' in keys else 'Unknown',
                    country=row['country'] if 'country' in keys else 'Unknown',
                    doc_type=row['doc_type'] if 'doc_type' in keys else 'Unknown',
                    page=row['page'] if 'page' in keys else 0,
                    text=row['text'] if 'text' in keys else '',
                    context=row['context'] if 'context' in keys else '',
                    category_a=row['category_a'] if 'category_a' in keys else 'none',
                    category_b=row['category_b'] if 'category_b' in keys else 'none',
                    sentiment=row['sentiment'] if 'sentiment' in keys else 'neutral',
                    sentiment_score=row['sentiment_score'] if 'sentiment_score' in keys else 0.0,
                    semantic_score=row['semantic_score'] if 'semantic_score' in keys else 0.0,
                    detection_method=row['detection_method'] if 'detection_method' in keys else '',
                    source=row['source'] if 'source' in keys else '',
                    eu_domain=(row['eu_domain'] if 'eu_domain' in keys else 'Unclassified'),
                    eu_subdomain=(row['eu_subdomain'] if 'eu_subdomain' in keys else 'Unclassified'),
                    eu_confidence=(row['eu_confidence'] if 'eu_confidence' in keys else 0.0),
                ))
            
            total_original += len(refs)

            # Apply semantic deduplication
            dedup_refs = self.deduplicator.deduplicate_references(refs)
            total_dedup += len(dedup_refs)

            # Save deduplicated references
            for dedup_ref in dedup_refs:
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO ai_references_deduplicated
                        (company, year, position, industry, sector, country, text, context,
                         category_a, category_b,
                         sources, doc_count, total_occurrences, avg_sentiment_score,
                         avg_semantic_score, original_refs, eu_domain, eu_subdomain)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        dedup_ref['company'], dedup_ref['year'], dedup_ref['position'],
                        dedup_ref['industry'], dedup_ref.get('sector', 'Unknown'),
                        dedup_ref.get('country', 'Unknown'),
                        dedup_ref['text'], dedup_ref['context'],
                        dedup_ref.get('category_a', 'none'), dedup_ref.get('category_b', 'none'),
                        dedup_ref['sources'], dedup_ref['doc_count'],
                        dedup_ref['total_occurrences'], dedup_ref['avg_sentiment_score'],
                        dedup_ref['avg_semantic_score'], dedup_ref['original_refs'],
                        dedup_ref.get('eu_domain', 'Unclassified'),
                        dedup_ref.get('eu_subdomain', 'Unclassified')
                    ))
                except Exception as e:
                    logger.error(f"Error saving deduplicated reference for {company}: {e}")

        self.db_manager.conn.commit()

        # Auto-export
        self._export_deduplicated_references()

        # Display results
        reduction = (1 - total_dedup / total_original) * 100 if total_original > 0 else 0
        print(f"\n{'═'*70}")
        print(f"✓ DEDUPLICATION COMPLETE!")
        print(f"{'═'*70}")
        print(f"   📊 {total_original} original references → {total_dedup} deduplicated")
        print(f"   📉 Reduction: {reduction:.1f}%")

        self._stats_cache = None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HANDLERS FOR OPTIONS 3.x - INDEX CALCULATION
    # ═══════════════════════════════════════════════════════════════════════════

    def handle_option_3_1(self):
        """3.1 - Calculate AI Adoption Index for a batch by positions."""
        self.initialize_components()
        start, end = self.get_position_range()
        self._calculate_indices(position_range=(start, end))

    def handle_option_3_2(self):
        """3.2 - Calculate AI Adoption Index for all companies."""
        self.initialize_components()
        self._calculate_indices(position_range=None)

    def handle_option_3_3(self):
        """3.3 - Calculate index only for new companies."""
        self.initialize_components()
        cursor = self.db_manager.conn.cursor()

        # Find pairs that have deduplicated references but no index yet
        cursor.execute("""
            SELECT DISTINCT company, year FROM ai_references_deduplicated
            WHERE (company, year) NOT IN (
                SELECT DISTINCT company, year FROM adoption_index
            )
        """)
        new_pairs = cursor.fetchall()

        if not new_pairs:
            print("\n✓ No new companies for index calculation.")
            print("   All companies with deduplicated references already have an index.")
            return

        print(f"\n📈 {len(new_pairs)} new companies for index calculation.")
        self._calculate_indices(company_year_pairs=new_pairs)

    def _calculate_indices(self, position_range: tuple[int, int] | None = None,
                           company_year_pairs: list[tuple] | None = None):
        """
        Calculate AI Adoption Index for the specified companies.

        Args:
            position_range: Tuple (start, end) for position-based filtering
            company_year_pairs: List of (company, year) tuples for targeted processing
        """
        from tqdm import tqdm
        cursor = self.db_manager.conn.cursor()

        # Determine which companies to process
        if company_year_pairs:
            pairs = company_year_pairs
        else:
            query = """
                SELECT DISTINCT company, year, position, industry
                FROM ai_references_deduplicated
            """
            if position_range:
                query += f" WHERE position >= {position_range[0]} AND position <= {position_range[1]}"
            cursor.execute(query)
            pairs = [(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()]

        if not pairs:
            print("\n✓ No data available for index calculation.")
            return

        print(f"\n⏳ Calculating index for {len(pairs)} companies...")
        
        for item in tqdm(pairs, desc="Calculating Index"):
            # Extract item information
            if len(item) == 2:
                company, year = item
                cursor.execute("""
                    SELECT position, industry FROM ai_references_deduplicated 
                    WHERE company = ? AND year = ? LIMIT 1
                """, (company, year))
                row = cursor.fetchone()
                position, industry = (row[0], row[1]) if row else (0, 'Unknown')
            else:
                company, year, position, industry = item
            
            # Get deduplicated references
            cursor.execute("""
                SELECT * FROM ai_references_deduplicated WHERE company = ? AND year = ?
            """, (company, year))
            rows = cursor.fetchall()
            
            if not rows:
                continue
            
            # Extract sector and country from the first row (columns 5 and 6 in the new schema)
            first_row = rows[0]
            sector = first_row[5] if len(first_row) > 5 and first_row[5] else 'Unknown'
            country = first_row[6] if len(first_row) > 6 and first_row[6] else 'Unknown'

            # Create DocumentResult for the calculator
            doc_result = DocumentResult(
                company=company, year=year, position=position, industry=industry,
                sector=sector,
                country=country,
                doc_type='Mixed', source='aggregated',
                total_pages=sum((row['doc_count'] if 'doc_count' in row.keys() else 0) or 0 for row in rows),
                text_length=0
            )

            # Add references (name-based access — robust to column order).
            for row in rows:
                keys = row.keys()
                row_sector = row['sector'] if 'sector' in keys and row['sector'] else 'Unknown'
                row_country = row['country'] if 'country' in keys and row['country'] else 'Unknown'
                ref = AIReference(
                    company=row['company'], year=row['year'], position=row['position'],
                    industry=row['industry'],
                    sector=row_sector, country=row_country,
                    doc_type='Mixed', page=1,
                    text=row['text'] if 'text' in keys else '',
                    context=row['context'] if 'context' in keys else '',
                    category_a=row['category_a'] if 'category_a' in keys else 'none',
                    category_b=row['category_b'] if 'category_b' in keys else 'none',
                    sentiment='neutral',
                    sentiment_score=row['avg_sentiment_score'] if 'avg_sentiment_score' in keys else 0.0,
                    semantic_score=row['avg_semantic_score'] if 'avg_semantic_score' in keys else 0.0,
                    detection_method='aggregated',
                    source=row['sources'] if 'sources' in keys else '',
                    eu_domain=(row['eu_domain'] if 'eu_domain' in keys else 'Unclassified'),
                    eu_subdomain=(row['eu_subdomain'] if 'eu_subdomain' in keys else 'Unclassified'),
                )
                doc_result.add_reference(ref)
            
            # Calculate the index
            adoption_index = self.index_calculator.calculate_index(doc_result)

            # Save to database
            cursor.execute("""
                INSERT OR REPLACE INTO adoption_index
                (company, year, position, industry, sector, country, intensity_index, semantic_index,
                 diversity_index, sentiment_index, maturity_index, future_index,
                 commitment_index, ai_adoption_index, total_refs, total_pages, categories_used,
                 diversity_index_eu, ai_adoption_index_eu, categories_used_eu)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                adoption_index.company, adoption_index.year, adoption_index.position,
                adoption_index.industry,
                getattr(adoption_index, 'sector', sector),
                getattr(adoption_index, 'country', country),
                adoption_index.intensity_index,
                adoption_index.semantic_index, adoption_index.diversity_index,
                adoption_index.sentiment_index, adoption_index.maturity_index,
                adoption_index.future_index, adoption_index.commitment_index,
                adoption_index.ai_adoption_index, adoption_index.total_refs,
                adoption_index.total_pages, adoption_index.categories_used,
                getattr(adoption_index, 'diversity_index_eu', 0.0),
                getattr(adoption_index, 'ai_adoption_index_eu', 0.0),
                getattr(adoption_index, 'categories_used_eu', 0)
            ))

            # Per-doc-type adoption indices. Robustness check that lets
            # the paper report the index pooled AND disaggregated. Reads from
            # ai_references_raw (which preserves doc_type) rather than from
            # ai_references_deduplicated (which pools doc_types).
            cursor.execute("""
                SELECT DISTINCT doc_type FROM ai_references_raw
                WHERE company = ? AND year = ? AND doc_type IS NOT NULL AND doc_type != ''
            """, (company, year))
            doc_types = [r[0] for r in cursor.fetchall()]

            for dt in doc_types:
                cursor.execute("""
                    SELECT company, year, position, industry, sector, country, doc_type,
                           page, text, context, category_a, category_b, sentiment, sentiment_score,
                           semantic_score, detection_method, source, robotics_type, rpa_type,
                           sentiment_confidence, category_a_confidence, category_b_confidence,
                           reference_strength, confidence_score, confidence_reasons,
                           eu_domain, eu_subdomain, eu_confidence
                      FROM ai_references_raw
                     WHERE company = ? AND year = ? AND doc_type = ?
                """, (company, year, dt))
                raw_rows = cursor.fetchall()
                if not raw_rows:
                    continue

                cursor.execute("""
                    SELECT COALESCE(SUM(total_pages), 0) FROM processed_documents
                    WHERE company = ? AND year = ? AND doc_type = ?
                """, (company, year, dt))
                dt_pages = cursor.fetchone()[0] or 0

                dt_first = raw_rows[0]
                dt_doc = DocumentResult(
                    company=dt_first[0], year=dt_first[1], position=dt_first[2],
                    industry=dt_first[3] or 'Unknown',
                    sector=dt_first[4] or 'Unknown',
                    country=dt_first[5] or 'Unknown',
                    doc_type=dt, source=f'per_doctype_{dt}',
                    total_pages=dt_pages,
                    text_length=0
                )
                for r in raw_rows:
                    dt_doc.add_reference(AIReference(
                        company=r[0], year=r[1], position=r[2],
                        industry=r[3] or 'Unknown',
                        sector=r[4] or 'Unknown',
                        country=r[5] or 'Unknown',
                        doc_type=r[6] or dt,
                        page=r[7] or 1,
                        text=r[8] or '',
                        context=r[9] or '',
                        category_a=r[10] or 'none',
                        category_b=r[11] or 'none',
                        sentiment=r[12] or 'neutral',
                        sentiment_score=r[13] or 0.0,
                        semantic_score=r[14] or 0.0,
                        detection_method=r[15] or '',
                        source=r[16] or '',
                        robotics_type=r[17] or 'not_robotics',
                        rpa_type=r[18] or 'not_rpa',
                        sentiment_confidence=r[19] or 'standard',
                        category_a_confidence=r[20] or 0.0,
                        category_b_confidence=r[21] or 0.0,
                        reference_strength=r[22] or 'unknown',
                        confidence_score=r[23] or 0.0,
                        confidence_reasons=r[24] or '',
                        eu_domain=r[25] if len(r) > 25 else 'Unclassified',
                        eu_subdomain=r[26] if len(r) > 26 else 'Unclassified',
                        eu_confidence=r[27] if len(r) > 27 else 0.0,
                    ))
                dt_idx = self.index_calculator.calculate_index(dt_doc)
                cursor.execute("""
                    INSERT OR REPLACE INTO adoption_index_by_doctype
                    (company, year, doc_type, position, industry, sector, country,
                     intensity_index, semantic_index, diversity_index, sentiment_index,
                     maturity_index, future_index, commitment_index, ai_adoption_index,
                     total_refs, total_pages, categories_used,
                     diversity_index_eu, ai_adoption_index_eu, categories_used_eu)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    dt_idx.company, dt_idx.year, dt, dt_idx.position,
                    dt_idx.industry,
                    getattr(dt_idx, 'sector', 'Unknown'),
                    getattr(dt_idx, 'country', 'Unknown'),
                    dt_idx.intensity_index, dt_idx.semantic_index,
                    dt_idx.diversity_index, dt_idx.sentiment_index,
                    dt_idx.maturity_index, dt_idx.future_index,
                    dt_idx.commitment_index, dt_idx.ai_adoption_index,
                    dt_idx.total_refs, dt_idx.total_pages, dt_idx.categories_used,
                    getattr(dt_idx, 'diversity_index_eu', 0.0),
                    getattr(dt_idx, 'ai_adoption_index_eu', 0.0),
                    getattr(dt_idx, 'categories_used_eu', 0)
                ))

        self.db_manager.conn.commit()

        # Update rankings
        self._update_rankings()

        # Auto-export
        self._export_adoption_index()

        print(f"\n{'═'*70}")
        print(f"✓ INDEX CALCULATION COMPLETE!")
        print(f"{'═'*70}")

        self._stats_cache = None

    def _update_rankings(self):
        """Update year-based rankings in the database."""
        cursor = self.db_manager.conn.cursor()

        # Get available years
        cursor.execute("SELECT DISTINCT year FROM adoption_index")
        years = [row[0] for row in cursor.fetchall()]
        
        for year in years:
            cursor.execute(f"""
                WITH ranked AS (
                    SELECT company,
                           ROW_NUMBER() OVER (ORDER BY ai_adoption_index DESC) as rank
                    FROM adoption_index WHERE year = {year}
                )
                UPDATE adoption_index
                SET rank_in_year = (
                    SELECT rank FROM ranked 
                    WHERE ranked.company = adoption_index.company
                )
                WHERE year = {year}
            """)
        
        self.db_manager.conn.commit()
        logger.info("Rankings updated for all years")

# ═══════════════════════════════════════════════════════════════════════════════
# END PART 2/3 - Continued in part 3
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# AI SEMANTIC ANALYZER - PART 3/3
# MenuManager continuation - Handlers 4.x, 5.x, 6.x, 7 and main loop
# THIS CODE IS ADDED TO THE MenuManager CLASS AFTER PART 2
# ═══════════════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════════
    # HANDLERS FOR OPTIONS 4.x - REPORTS & EXPORT
    # ═══════════════════════════════════════════════════════════════════════════

    def handle_option_4_1(self):
        """4.1 - Export RAW references."""
        self.initialize_components()
        self._export_raw_references()
        print("\n✓ RAW references export complete!")

    def handle_option_4_2(self):
        """4.2 - Export deduplicated references."""
        self.initialize_components()
        self._export_deduplicated_references()
        print("\n✓ Deduplicated references export complete!")

    def handle_option_4_3(self):
        """4.3 - Export AI Adoption Index."""
        self.initialize_components()
        self._export_adoption_index()
        print("\n✓ AI Adoption Index export complete!")

    def handle_option_4_4(self):
        """4.4 - Export full JSON."""
        self.initialize_components()
        self._export_json()
        print("\n✓ JSON export complete!")

    def handle_option_4_5(self):
        """4.5 - Generate TEXT summary report."""
        self.initialize_components()
        self._export_text_report()
        print("\n✓ TEXT report generated!")

    def handle_option_4_6(self):
        """4.6 - Generate ALL reports."""
        self.initialize_components()
        print("\n⏳ Generare toate rapoartele...")
        self._export_raw_references()
        self._export_deduplicated_references()
        self._export_adoption_index()
        self._export_json()
        self._export_text_report()
        print("\n✓ All reports generated!")

    def handle_option_4_7(self):
        """4.7 - Export EU_Semantics (JRC AI Watch) classification distribution."""
        self.initialize_components()
        cursor = self.db_manager.conn.cursor()
        cursor.execute("SELECT * FROM ai_references_deduplicated")
        rows = cursor.fetchall()
        if not rows:
            print("\n⚠ No deduplicated references available. Run analysis (option 2) first.")
            return
        refs = [{
            'eu_domain': (row['eu_domain'] if 'eu_domain' in row.keys() else 'Unclassified'),
            'eu_subdomain': (row['eu_subdomain'] if 'eu_subdomain' in row.keys() else 'Unclassified'),
        } for row in rows]
        path = self.excel_exporter.export_eu_classification(refs)
        if path:
            print(f"\n✓ EU_Semantics classification exported: {path}")
        else:
            print("\n⚠ No EU classification data to export.")

    def _export_raw_references(self):
        """Export raw references to Excel."""
        cursor = self.db_manager.conn.cursor()
        cursor.execute("SELECT * FROM ai_references_raw")
        rows = cursor.fetchall()
        
        if not rows:
            print("  ⚠ No raw references available for export.")
            return

        # Name-based access — robust to column order across fresh vs migrated DBs.
        def _g(row, key, default):
            return row[key] if key in row.keys() else default

        refs = []
        for row in rows:
            refs.append({
                'company': _g(row, 'company', ''),
                'year': _g(row, 'year', 0),
                'position': _g(row, 'position', 0),
                'industry': _g(row, 'industry', ''),
                'sector': _g(row, 'sector', ''),
                'country': _g(row, 'country', ''),
                'doc_type': _g(row, 'doc_type', ''),
                'page': _g(row, 'page', 0),
                'text': _g(row, 'text', ''),
                'context': _g(row, 'context', ''),
                'category_a': _g(row, 'category_a', 'none'),
                'category_b': _g(row, 'category_b', 'none'),
                'sentiment': _g(row, 'sentiment', ''),
                'sentiment_score': _g(row, 'sentiment_score', 0),
                'semantic_score': _g(row, 'semantic_score', 0),
                'detection_method': _g(row, 'detection_method', ''),
                'source': _g(row, 'source', ''),
                'robotics_type': _g(row, 'robotics_type', ''),
                'rpa_type': _g(row, 'rpa_type', ''),
                # FP-scoring fields
                'reference_strength': _g(row, 'reference_strength', 'unknown'),
                'confidence_score': _g(row, 'confidence_score', 0.0),
                'confidence_reasons': _g(row, 'confidence_reasons', ''),
                'eu_domain': _g(row, 'eu_domain', ''),
                'eu_subdomain': _g(row, 'eu_subdomain', ''),
                'eu_confidence': _g(row, 'eu_confidence', 0.0),
                # For compatibility with export_references()
                'avg_sentiment_score': _g(row, 'sentiment_score', 0),
                'avg_semantic_score': _g(row, 'semantic_score', 0),
                'doc_count': 1,
                'total_occurrences': 1
            })
        
        self.excel_exporter.export_references(refs, "ai_references_raw.xlsx")
        print(f"  ✓ Exported {len(refs)} raw references")

    def _export_deduplicated_references(self):
        """Export deduplicated references to Excel."""
        cursor = self.db_manager.conn.cursor()
        cursor.execute("SELECT * FROM ai_references_deduplicated")
        rows = cursor.fetchall()

        if not rows:
            print("  ⚠ No deduplicated references available for export.")
            return

        # Name-based access — robust to column order across fresh vs migrated DBs.
        def _g(row, key, default):
            return row[key] if key in row.keys() else default

        refs = []
        for row in rows:
            refs.append({
                'company': _g(row, 'company', ''),
                'year': _g(row, 'year', 0),
                'position': _g(row, 'position', 0),
                'industry': _g(row, 'industry', ''),
                'sector': _g(row, 'sector', ''),
                'country': _g(row, 'country', ''),
                'text': _g(row, 'text', ''),
                'context': _g(row, 'context', ''),
                'category_a': _g(row, 'category_a', 'none'),
                'category_b': _g(row, 'category_b', 'none'),
                'sources': _g(row, 'sources', ''),
                'source': _g(row, 'sources', ''),  # Alias for export
                'doc_count': _g(row, 'doc_count', 0),
                'total_occurrences': _g(row, 'total_occurrences', 0),
                'avg_sentiment_score': _g(row, 'avg_sentiment_score', 0),
                'avg_semantic_score': _g(row, 'avg_semantic_score', 0),
                'eu_domain': _g(row, 'eu_domain', ''),
                'eu_subdomain': _g(row, 'eu_subdomain', ''),
                # Placeholder for fields absent from dedup table (but present in Excel header)
                'reference_strength': 'aggregated',
                'confidence_score': 0.0,
                'confidence_reasons': 'Deduplicated reference'
            })
        
        self.excel_exporter.export_references(refs, "ai_references_deduplicated.xlsx")
        print(f"  ✓ Exported {len(refs)} deduplicated references")

    def _export_adoption_index(self):
        """Export AI Adoption Index to Excel."""
        cursor = self.db_manager.conn.cursor()
        cursor.execute("""
            SELECT * FROM adoption_index
            ORDER BY year DESC, ai_adoption_index DESC
        """)
        rows = cursor.fetchall()

        if not rows:
            print("  ⚠ No indices available for export.")
            return
        
        indices = []
        for row in rows:
            indices.append(AIAdoptionIndex(
                company=row[1], year=row[2], position=row[3], industry=row[4],
                sector=row[5] if len(row) > 5 else '',
                country=row[6] if len(row) > 6 else '',
                intensity_index=row[7] if len(row) > 7 else 0,
                semantic_index=row[8] if len(row) > 8 else 0,
                diversity_index=row[9] if len(row) > 9 else 0,
                sentiment_index=row[10] if len(row) > 10 else 0,
                maturity_index=row[11] if len(row) > 11 else 0,
                future_index=row[12] if len(row) > 12 else 0,
                commitment_index=row[13] if len(row) > 13 else 0,
                ai_adoption_index=row[14] if len(row) > 14 else 0,
                total_refs=row[16] if len(row) > 16 else 0,
                total_pages=row[17] if len(row) > 17 else 0,
                categories_used=row[18] if len(row) > 18 else 0,
                diversity_index_eu=(row['diversity_index_eu'] if 'diversity_index_eu' in row.keys() else 0.0),
                ai_adoption_index_eu=(row['ai_adoption_index_eu'] if 'ai_adoption_index_eu' in row.keys() else 0.0),
                categories_used_eu=(row['categories_used_eu'] if 'categories_used_eu' in row.keys() else 0)
            ))
        
        self.excel_exporter.export_adoption_index(indices)
        print(f"  ✓ Exported index for {len(indices)} companies")

    def _export_json(self):
        """Export all data to JSON format."""
        cursor = self.db_manager.conn.cursor()

        data = {
            'metadata': {
                'analysis_types': 'classic+eu_semantics',
                'generated_at': datetime.now().isoformat(),
                'categories': list(AI_CATEGORIES.keys()),
                'semantic_threshold': self.config.semantic_threshold
            },
            'companies': [],
            'industries': []
        }
        
        # Export companies
        cursor.execute("SELECT * FROM adoption_index")
        for row in cursor.fetchall():
            data['companies'].append({
                'company': row['company'], 'year': row['year'], 'position': row['position'],
                'industry': row['industry'],
                'ai_adoption_index': row['ai_adoption_index'],
                'ai_adoption_index_eu': (row['ai_adoption_index_eu'] if 'ai_adoption_index_eu' in row.keys() else 0.0),
                'dimensions': {
                    'intensity': row['intensity_index'], 'semantic': row['semantic_index'],
                    'diversity': row['diversity_index'], 'sentiment': row['sentiment_index'],
                    'maturity': row['maturity_index'], 'future': row['future_index'],
                    'commitment': row['commitment_index'],
                    'diversity_eu': (row['diversity_index_eu'] if 'diversity_index_eu' in row.keys() else 0.0)
                },
                'total_refs': row['total_refs'], 'categories_used': row['categories_used'],
                'categories_used_eu': (row['categories_used_eu'] if 'categories_used_eu' in row.keys() else 0)
            })
        
        # Export industries (if available)
        try:
            cursor.execute("SELECT * FROM adoption_index_industry")
            for row in cursor.fetchall():
                data['industries'].append({
                    'industry': row[1], 'year': row[2],
                    'ai_adoption_index_industry': row[10],
                    'num_companies': row[11]
                })
        except:
            pass
        
        output_path = Path(self.config.output_folder) / "results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"  ✓ Export JSON: {output_path}")
    
    def _export_text_report(self):
        """Generate a text summary report."""
        cursor = self.db_manager.conn.cursor()
        output_path = Path(self.config.output_folder) / "report.txt"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("AI ADOPTION ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")

            # Features
            f.write("FEATURES:\n")
            f.write(f"  - 15 AI Categories\n")
            f.write(f"  - 80+ AI Patterns\n")
            f.write(f"  - Semantic Threshold: {self.config.semantic_threshold}\n")
            f.write(f"  - Robotics/RPA filtering: enabled\n")
            f.write(f"  - Document tracking with re-analysis support\n\n")
            
            # Summary
            cursor.execute("""
                SELECT COUNT(DISTINCT company), MIN(year), MAX(year), SUM(total_refs) 
                FROM adoption_index
            """)
            row = cursor.fetchone()
            f.write(f"SUMMARY:\n")
            f.write(f"  Companies analyzed: {row[0]}\n")
            f.write(f"  Years covered: {row[1]}-{row[2]}\n")
            f.write(f"  Total references: {row[3]}\n\n")
            
            # Top 20
            cursor.execute("""
                SELECT company, ai_adoption_index, industry, total_refs FROM adoption_index
                WHERE year = (SELECT MAX(year) FROM adoption_index)
                ORDER BY ai_adoption_index DESC LIMIT 20
            """)
            f.write("TOP 20 COMPANIES (Latest Year):\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Rank':<5} {'Company':<35} {'Index':<8} {'Refs':<8} {'Industry'}\n")
            f.write("-" * 70 + "\n")
            for i, (company, idx, industry, refs) in enumerate(cursor.fetchall(), 1):
                f.write(f"{i:<5} {company[:33]:<35} {idx:<8.1f} {refs:<8} {industry[:20]}\n")
            
            # Processing statistics
            f.write("\n" + "=" * 70 + "\n")
            f.write("PROCESSING STATISTICS:\n")
            f.write("-" * 70 + "\n")
            try:
                cursor.execute("SELECT COUNT(*), SUM(process_count) FROM processed_documents")
                docs_row = cursor.fetchone()
                f.write(f"  Documents processed: {docs_row[0]}\n")
                f.write(f"  Total processing runs: {docs_row[1]}\n")
                
                cursor.execute("SELECT COUNT(*) FROM processed_documents WHERE process_count > 1")
                f.write(f"  Documents re-analyzed: {cursor.fetchone()[0]}\n")
                
                cursor.execute("SELECT COUNT(*), SUM(COALESCE(occurrence_count, 1)) FROM ai_references_raw")
                refs_row = cursor.fetchone()
                f.write(f"  Unique references: {refs_row[0]}\n")
                f.write(f"  Total occurrences: {refs_row[1]}\n")
            except Exception as e:
                f.write(f"  (Statistics unavailable: {e})\n")
            
            # Category distribution
            f.write("\n" + "=" * 70 + "\n")
            f.write("CATEGORY DISTRIBUTION:\n")
            f.write("-" * 70 + "\n")
            try:
                cursor.execute("""
                    SELECT cat, COUNT(*) as cnt FROM (
                        SELECT category_a AS cat FROM ai_references_deduplicated WHERE category_a != 'none'
                        UNION ALL
                        SELECT category_b AS cat FROM ai_references_deduplicated WHERE category_b != 'none'
                    ) GROUP BY cat ORDER BY cnt DESC
                """)
                for cat, cnt in cursor.fetchall():
                    f.write(f"  {cat}: {cnt}\n")
            except:
                pass
        
        print(f"  ✓ Report generated: {output_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # HANDLERS FOR OPTIONS 5.x - CHARTS
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_visualization_data(self):
        """Get the data needed for visualizations."""
        cursor = self.db_manager.conn.cursor()

        # Indices
        cursor.execute("SELECT * FROM adoption_index")
        rows = cursor.fetchall()
        
        indices = []
        for row in rows:
            indices.append(AIAdoptionIndex(
                company=row[1], year=row[2], position=row[3], industry=row[4],
                sector=row[5] if len(row) > 5 else '',
                country=row[6] if len(row) > 6 else '',
                intensity_index=row[7] if len(row) > 7 else 0,
                semantic_index=row[8] if len(row) > 8 else 0,
                diversity_index=row[9] if len(row) > 9 else 0,
                sentiment_index=row[10] if len(row) > 10 else 0,
                maturity_index=row[11] if len(row) > 11 else 0,
                future_index=row[12] if len(row) > 12 else 0,
                commitment_index=row[13] if len(row) > 13 else 0,
                ai_adoption_index=row[14] if len(row) > 14 else 0,
                total_refs=row[16] if len(row) > 16 else 0,
                total_pages=row[17] if len(row) > 17 else 0,
                categories_used=row[18] if len(row) > 18 else 0,
                diversity_index_eu=(row['diversity_index_eu'] if 'diversity_index_eu' in row.keys() else 0.0),
                ai_adoption_index_eu=(row['ai_adoption_index_eu'] if 'ai_adoption_index_eu' in row.keys() else 0.0),
                categories_used_eu=(row['categories_used_eu'] if 'categories_used_eu' in row.keys() else 0)
            ))
        
        # References for categories
        cursor.execute("SELECT company, year, category_a, category_b FROM ai_references_deduplicated")
        refs = [{'company': row[0], 'year': row[1], 'category_a': row[2], 'category_b': row[3]}
                for row in cursor.fetchall()]

        return indices, refs

    def handle_option_5_1(self):
        """5.1 - Company ranking chart."""
        self.initialize_components()
        indices, refs = self._get_visualization_data()
        if indices:
            self.viz_generator.generate_ranking_chart(indices)
            print("\n✓ Ranking chart generated!")
        else:
            print("\n⚠ No data available for chart.")

    def handle_option_5_2(self):
        """5.2 - Top companies evolution chart."""
        self.initialize_components()
        indices, refs = self._get_visualization_data()
        if indices:
            self.viz_generator.generate_timeline_chart(indices)
            print("\n✓ Timeline chart generated!")
        else:
            print("\n⚠ No data available for chart.")

    def handle_option_5_3(self):
        """5.3 - Dimension radar chart."""
        self.initialize_components()
        indices, refs = self._get_visualization_data()
        if indices:
            self.viz_generator.generate_radar_chart(indices)
            print("\n✓ Radar chart generated!")
        else:
            print("\n⚠ No data available for chart.")

    def handle_option_5_4(self):
        """5.4 - Heatmap companies × years."""
        self.initialize_components()
        indices, refs = self._get_visualization_data()
        if indices:
            self.viz_generator.generate_heatmap(indices)
            print("\n✓ Heatmap generated!")
        else:
            print("\n⚠ No data available for chart.")

    def handle_option_5_5(self):
        """5.5 - AI category distribution."""
        self.initialize_components()
        indices, refs = self._get_visualization_data()
        if refs:
            self.viz_generator.generate_category_distribution(refs)
            print("\n✓ Category chart generated!")
        else:
            print("\n⚠ No data available for chart.")

    def handle_option_5_6(self):
        """5.6 - All charts."""
        self.initialize_components()
        indices, refs = self._get_visualization_data()
        if indices:
            print("\n⏳ Generating all charts...")
            self.viz_generator.generate_all_visualizations(indices, refs)
            print("\n✓ All charts generated!")
        else:
            print("\n⚠ No data available for charts.")

    def handle_option_5_7(self):
        """5.7 - Industry charts."""
        self.initialize_components()
        cursor = self.db_manager.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM adoption_index_industry")

        if cursor.fetchone()[0] == 0:
            print("\n⚠ No industry-level aggregated data available.")
            print("   Run the full pipeline first (option 7).")
            return

        indices, refs = self._get_visualization_data()
        self.viz_generator.generate_industry_charts(indices)
        print("\n✓ Industry charts generated!")

    def handle_option_5_8(self):
        """5.8 - Sector charts."""
        self.initialize_components()
        indices, refs = self._get_visualization_data()
        if indices:
            self.viz_generator.generate_group_charts(indices, group_by='sector')
            print("\n✓ Sector charts generated!")
        else:
            print("\n⚠ No data available for chart.")

    def handle_option_5_9(self):
        """5.9 - Country charts."""
        self.initialize_components()
        indices, refs = self._get_visualization_data()
        if indices:
            self.viz_generator.generate_group_charts(indices, group_by='country')
            print("\n✓ Country charts generated!")
        else:
            print("\n⚠ No data available for chart.")    


    def handle_option_5_10(self):
        """5.10 - EU_Semantics sunburst (domains × subdomains)."""
        self.initialize_components()
        cursor = self.db_manager.conn.cursor()
        cursor.execute("SELECT * FROM ai_references_deduplicated")
        rows = cursor.fetchall()
        if not rows:
            print("\n⚠ No deduplicated references available. Run analysis (option 2) first.")
            return
        refs = [{
            'eu_domain': (row['eu_domain'] if 'eu_domain' in row.keys() else 'Unclassified'),
            'eu_subdomain': (row['eu_subdomain'] if 'eu_subdomain' in row.keys() else 'Unclassified'),
        } for row in rows]
        path = self.viz_generator.generate_eu_sunburst(refs)
        if path:
            print(f"\n✓ EU_Semantics sunburst generated: {path}")
        else:
            print("\n⚠ No EU classification data to visualize.")

    def handle_option_5_11(self):
        """5.11 - Taxonomy comparison (classic vs EU_Semantics composite index)."""
        self.initialize_components()
        indices, _refs = self._get_visualization_data()
        if indices:
            self.viz_generator.generate_taxonomy_comparison(indices)
            print("\n✓ Taxonomy comparison chart generated!")
        else:
            print("\n⚠ No data available for chart.")

    # ═══════════════════════════════════════════════════════════════════════════
    # HANDLERS FOR OPTIONS 6.x - UTILITIES
    # ═══════════════════════════════════════════════════════════════════════════

    def handle_option_6_1(self):
        """6.1 - Detailed statistics and summary."""
        self.initialize_components()
        stats = self.get_statistics(force_refresh=True)

        print(f"\n{'═'*70}")
        print("📊 DETAILED STATISTICS")
        print(f"{'═'*70}")

        print(f"\n📁 PDF DOCUMENTS:")
        print(f"   Total in folder: {stats['total_pdfs']}")
        print(f"   Processed: {stats['processed']}")
        print(f"   New (unprocessed): {stats['new']}")
        print(f"   Chinese (skipped): {stats['chinese']}")
        if stats['reprocessed_docs'] > 0:
            print(f"   Re-analyzed: {stats['reprocessed_docs']}")

        print(f"\n🔍 AI REFERENCES:")
        print(f"   Unique (raw): {stats['total_refs_raw']:,}")
        print(f"   Total occurrences: {stats['total_occurrences']:,}")
        print(f"   Deduplicated: {stats['total_refs_dedup']:,}")
        if stats['total_refs_raw'] > 0:
            ratio = stats['total_refs_dedup'] / stats['total_refs_raw'] * 100
            print(f"   Deduplication ratio: {ratio:.1f}%")
            if stats['total_occurrences'] > stats['total_refs_raw']:
                avg_occ = stats['total_occurrences'] / stats['total_refs_raw']
                print(f"   Avg occurrences/reference: {avg_occ:.2f}")

        print(f"\n📈 AI ADOPTION INDEX:")
        print(f"   Companies with index: {stats['companies_with_index']}")
        if stats['by_year']:
            print(f"\n   By year:")
            for year, count in sorted(stats['by_year'].items()):
                print(f"      {year}: {count} companies")

        if stats['by_industry']:
            print(f"\n   By industry (top 10):")
            for industry, count in list(stats['by_industry'].items())[:10]:
                print(f"      {industry[:30]}: {count} companies")

        # Detailed stats from processed_documents
        try:
            proc_stats = self.db_manager.get_processing_stats()
            print(f"\n🔄 PROCESSING STATISTICS:")
            print(f"   Total processing runs: {proc_stats['total_processing_runs']}")
            print(f"   Documents without references: {proc_stats['documents_without_refs']}")
        except:
            pass

        print(f"\n⚙️ CONFIGURATION:")
        print(f"   Semantic threshold: {self.config.semantic_threshold}")
        print(f"   Parallel threads: {getattr(self.config, 'parallel_threads', PARALLEL_THREADS)}")
        print(f"   AI categories: {len(AI_CATEGORIES)}")
        print(f"   Input folder: {self.config.input_folder}")
        print(f"   Output folder: {self.config.output_folder}")
        print(f"{'═'*70}")

    def handle_option_6_2(self):
        """6.2 - Database integrity check."""
        self.initialize_components()
        print("\n⏳ Checking database integrity...")

        cursor = self.db_manager.conn.cursor()
        issues = []

        # Check processed_documents table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='processed_documents'
        """)
        if not cursor.fetchone():
            issues.append("⚠ Table processed_documents is missing")

        # Check references without deduplication
        cursor.execute("""
            SELECT COUNT(DISTINCT company || '-' || year) FROM ai_references_raw
            WHERE (company, year) NOT IN (
                SELECT company, year FROM ai_references_deduplicated
            )
        """)
        count = cursor.fetchone()[0]
        if count > 0:
            issues.append(f"⚠ {count} company-year pairs without deduplication")

        # Check deduplicated without index
        cursor.execute("""
            SELECT COUNT(DISTINCT company || '-' || year) FROM ai_references_deduplicated
            WHERE (company, year) NOT IN (
                SELECT company, year FROM adoption_index
            )
        """)
        count = cursor.fetchone()[0]
        if count > 0:
            issues.append(f"⚠ {count} company-year pairs without a calculated index")

        # Check Unknown industries
        cursor.execute("""
            SELECT COUNT(*) FROM adoption_index
            WHERE industry = 'Unknown' OR industry IS NULL
        """)
        count = cursor.fetchone()[0]
        if count > 0:
            issues.append(f"⚠ {count} companies without industry (Unknown)")

        # Check sync between processed_documents and ai_references_raw
        try:
            cursor.execute("""
                SELECT COUNT(DISTINCT source) FROM ai_references_raw
                WHERE source NOT IN (SELECT source FROM processed_documents)
            """)
            count = cursor.fetchone()[0]
            if count > 0:
                issues.append(f"⚠ {count} sources in references but not in processed_documents")
        except:
            pass

        # SQLite integrity check
        cursor.execute("PRAGMA integrity_check")
        integrity = cursor.fetchone()[0]
        if integrity != 'ok':
            issues.append(f"✗ SQLite integrity problem: {integrity}")

        # Display results
        if issues:
            print(f"\n🔍 Issues found ({len(issues)}):")
            for issue in issues:
                print(f"   {issue}")
        else:
            print("\n✓ Database integrity check passed!")

        # Display tables and sizes
        print(f"\n📊 Tables in database:")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for row in cursor.fetchall():
            table_name = row[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"   {table_name}: {count:,} records")

    def handle_option_6_3(self):
        """6.3 - Configure paths and settings."""
        print(f"\n{'═'*70}")
        print("🔧 CONFIGURATION")
        print(f"{'═'*70}")

        saved_config = load_saved_config()
        if saved_config:
            print("\nCurrent configuration:")
            print(f"   Input folder: {saved_config.get('input_folder', 'N/A')}")
            print(f"   Output folder: {saved_config.get('output_folder', 'N/A')}")
            print(f"   Company metadata CSV: {saved_config.get('company_metadata_csv', saved_config.get('fortune500_csv', 'N/A'))}")
            print(f"   Database: {saved_config.get('database_name', 'N/A')}")
            print(f"   Semantic threshold: {saved_config.get('semantic_threshold', 'N/A')}")
            print(f"   Parallel threads: {saved_config.get('parallel_threads', PARALLEL_THREADS)}")

        choice = input("\n1. Modify configuration | 2. Keep current [2]: ").strip()
        if choice == '1':
            self.config = AnalyzerConfig.from_interactive()
            threads_input = input(f"  Parallel threads [{getattr(self.config, 'parallel_threads', PARALLEL_THREADS)}]: ").strip()
            if threads_input.isdigit() and int(threads_input) > 0:
                self.config.parallel_threads = int(threads_input)
                updated_cfg = load_saved_config() or {}
                updated_cfg['parallel_threads'] = self.config.parallel_threads
                save_config(updated_cfg)
            self.db_manager = None
            self._stats_cache = None
            print("\n✓ Configuration updated!")

    def handle_option_6_4(self):
        """6.4 - Import industry mapping from CSV."""
        self.initialize_components()

        print(f"\n{'═'*70}")
        print("📥 IMPORT INDUSTRY MAPPING")
        print(f"{'═'*70}")
        print("\nExpected CSV format:")
        print("   Column 1: Company Name")
        print("   Column 2: Industry/Sector")

        csv_path = input("\nPath to CSV: ").strip()
        if not csv_path or not Path(csv_path).exists():
            print("  ✗ Invalid or non-existent file.")
            return

        try:
            df = pd.read_csv(csv_path)

            # Detect columns
            company_col = None
            industry_col = None

            for col in df.columns:
                col_lower = col.lower()
                if 'company' in col_lower or 'name' in col_lower:
                    company_col = col
                elif 'industry' in col_lower or 'sector' in col_lower:
                    industry_col = col

            if not company_col or not industry_col:
                print("  ✗ Required columns not found (company, industry).")
                return

            cursor = self.db_manager.conn.cursor()
            updated = 0

            for _, row in df.iterrows():
                company = str(row[company_col]).strip()
                industry = str(row[industry_col]).strip()

                if company and industry and industry != 'nan':
                    # Update across all tables
                    for table in ['ai_references_raw', 'ai_references_deduplicated',
                                  'adoption_index', 'processed_documents']:
                        try:
                            cursor.execute(f"""
                                UPDATE {table} SET industry = ?
                                WHERE LOWER(company) = LOWER(?)
                            """, (industry, company))
                            updated += cursor.rowcount
                        except:
                            pass

            self.db_manager.conn.commit()
            print(f"\n✓ Updated {updated} records with new industry values.")
            self._stats_cache = None

        except Exception as e:
            print(f"  ✗ Import error: {e}")

    def handle_option_6_5(self):
        """6.5 - Taxonomy info (classic + EU_Semantics)."""
        from _02_core import AI_APPLICATIONS, AI_TECHNOLOGIES, EU_CATEGORIES
        print(f"\n{'═'*80}")
        print("📋 SEMANTIC TAXONOMIES")
        print(f"{'═'*80}")
        print("\n[CLASSIC] Dimension 1 - AI Applications (7):")
        for _c, _i in AI_APPLICATIONS.items():
            print(f"   {_c}: {_i['name']}")
        print("\n[CLASSIC] Dimension 2 - AI Technologies (8):")
        for _c, _i in AI_TECHNOLOGIES.items():
            print(f"   {_c}: {_i['name']}")
        print("\n[EU_SEMANTICS] JRC AI Watch - 8 domains / 12 subdomains:")
        _last = None
        for _c, _leaf in EU_CATEGORIES.items():
            if _leaf['domain'] != _last:
                print(f"   * {_leaf['domain']}")
                _last = _leaf['domain']
            print(f"       - {_leaf['subdomain']}")
        print(f"\n{'═'*80}")
        return
        # --- legacy category info below is unreachable (superseded) ---
        print("📋 CATEGORII AI - 13 CATEGORII")
        print(f"{'═'*80}")
        
        categories_info = {
            'ML_ANALYTICS': ('Machine Learning & Analytics', 
                'Algoritmi ML, predictive analytics, data science, statistical modeling'),
            'GENAI_LLM': ('Generative AI & LLMs', 
                'ChatGPT, GPT-4, LLMs, generative AI, foundation models, Copilot'),
            'NLP': ('Natural Language Processing', 
                'Text processing, sentiment analysis, chatbots, language understanding'),
            'COMPUTER_VISION': ('Computer Vision', 
                'Image recognition, video analytics, visual AI, object detection'),
            'AUTOMATION': ('Intelligent Automation', 
                'AI-powered automation, intelligent process automation, smart workflows'),
            'AI_STRATEGY': ('AI Strategy & Investment', 
                'AI strategy, investments, digital transformation, AI initiatives'),
            'AI_GOVERNANCE': ('AI Governance & Ethics', 
                'AI ethics, responsible AI, governance, bias mitigation, transparency'),
            'AI_TALENT': ('AI Talent & Workforce', 
                'AI skills, training, workforce transformation, AI talent acquisition'),
            'AI_INFRASTRUCTURE': ('AI Infrastructure & Cloud', 
                'Cloud AI, MLOps, AI platforms, GPU infrastructure, model deployment'),
            'AI_PRODUCTS': ('AI Products & Services', 
                'AI-powered products, AI services for customers, AI features'),
            'AI_OPERATIONS': ('AI in Operations', 
                'AI in supply chain, manufacturing AI, operational efficiency'),
            'AI_CUSTOMER': ('AI Customer Experience', 
                'AI personalization, recommendation engines, customer AI'),
            'AI_GENERAL': ('General AI References', 
                'Generic AI mentions, artificial intelligence references'),
        }
        
        for i, (key, (name, desc)) in enumerate(categories_info.items(), 1):
            print(f"\n{i:2d}. {key}")
            print(f"    📌 {name}")
            print(f"    📝 {desc}")
        
        print(f"\n{'═'*80}")
    
    def handle_option_6_6(self):
        """6.6 - Generare matrice acoperire documente."""
        self.initialize_components()
        print("\n⏳ Generare matrice acoperire documente...")
        
        self._generate_document_coverage_matrix()
    
    def _generate_document_coverage_matrix(self):
        """Generate the document coverage matrix (company × year × doc type)."""
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        
        # 1. Load company metadata list
        if not self.company_metadata:
            print("  ⚠ No company metadata loaded. Load the CSV first.")
            return

        # 2. Scan all PDFs in the input folder
        pdf_folder = Path(self.config.input_folder)
        all_pdfs = list(pdf_folder.glob("*.pdf"))

        if not all_pdfs:
            print("  ⚠ No PDF files found in folder.")
            return

        # 3. Parse metadata from each PDF filename
        doc_data = []
        doc_types_found = set()
        years_found = set()
        
        for pdf in all_pdfs:
            parsed = self.processor.filename_parser.parse_filename(pdf.name)
            position = parsed.get('position', 0)
            company = parsed.get('company', 'Unknown')
            year = parsed.get('year', 2024)
            doc_type = parsed.get('doc_type', 'Unknown')
            
            doc_data.append({
                'position': position,
                'company': company,
                'year': year,
                'doc_type': doc_type,
                'filename': pdf.name
            })
            
            doc_types_found.add(doc_type)
            years_found.add(year)
        
        doc_types_sorted = sorted(doc_types_found)
        years_sorted = sorted(years_found)

        # 4. Build coverage structure: (position, company) -> {(doc_type, year): 1}
        coverage = defaultdict(dict)
        
        for doc in doc_data:
            key = (doc['position'], doc['company'])
            coverage[key][(doc['doc_type'], doc['year'])] = 1
        
        # 5. Build Excel workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Document Coverage"

        # Styles
        header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF", size=10)
        green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
        red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
        border = Border(
            left=Side(style='thin'), right=Side(style='thin'),
            top=Side(style='thin'), bottom=Side(style='thin')
        )
        
        # Headers
        headers = ['Position', 'Company']
        for doc_type in doc_types_sorted:
            for year in years_sorted:
                headers.append(f"{doc_type} {year}")
        
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.fill = header_fill
            cell.font = header_font
            cell.border = border
            cell.alignment = Alignment(horizontal='center', wrap_text=True)
        
        # Data rows — sorted by position
        companies_sorted = sorted(coverage.keys(), key=lambda x: (x[0], x[1]))
        
        row = 2
        totals = defaultdict(int)
        
        for position, company in companies_sorted:
            ws.cell(row=row, column=1, value=position).border = border
            ws.cell(row=row, column=2, value=company).border = border
            
            col = 3
            for doc_type in doc_types_sorted:
                for year in years_sorted:
                    has_doc = coverage[(position, company)].get((doc_type, year), 0)
                    cell = ws.cell(row=row, column=col, value=has_doc)
                    cell.border = border
                    cell.alignment = Alignment(horizontal='center')
                    
                    if has_doc == 1:
                        cell.fill = green_fill
                        totals[(doc_type, year)] += 1
                    else:
                        cell.fill = red_fill
                    
                    col += 1
            row += 1
        
        # Totals row
        ws.cell(row=row, column=1, value='').border = border
        total_cell = ws.cell(row=row, column=2, value='TOTAL')
        total_cell.border = border
        total_cell.font = Font(bold=True)
        
        col = 3
        for doc_type in doc_types_sorted:
            for year in years_sorted:
                cell = ws.cell(row=row, column=col, value=totals[(doc_type, year)])
                cell.border = border
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal='center')
                col += 1
        
        # Column widths
        ws.column_dimensions['A'].width = 10
        ws.column_dimensions['B'].width = 30
        for i in range(3, len(headers) + 1):
            ws.column_dimensions[chr(64 + i) if i <= 26 else f"{chr(64 + (i-1)//26)}{chr(65 + (i-1)%26)}"].width = 12
        
        # Freeze
        ws.freeze_panes = 'C2'
        
        # Save
        output_path = Path(self.config.output_folder) / "document_coverage_matrix.xlsx"
        wb.save(output_path)

        print(f"\n{'═'*70}")
        print(f"✓ DOCUMENT COVERAGE MATRIX GENERATED!")
        print(f"{'═'*70}")
        print(f"   Companies: {len(companies_sorted)}")
        print(f"   Document types: {len(doc_types_sorted)} ({', '.join(doc_types_sorted)})")
        print(f"   Years: {len(years_sorted)} ({', '.join(map(str, years_sorted))})")
        print(f"   Total PDFs: {len(all_pdfs)}")
        print(f"\n   Output: {output_path}")
        print(f"{'═'*70}")


    # ═══════════════════════════════════════════════════════════════════════════
    # HANDLER FOR OPTION 7 - FULL PIPELINE
    # ═══════════════════════════════════════════════════════════════════════════

    def handle_option_7(self):
        """7 - Fully automated pipeline for NEW documents."""
        self.initialize_components()

        print(f"\n{'═'*70}")
        print("🚀 FULL PIPELINE")
        print(f"{'═'*70}")

        stats = self.get_statistics(force_refresh=True)

        if stats['new'] == 0:
            print("\n✓ No NEW documents to process.")
            choice = input("Run full pipeline with re-analysis? (yes/no) [no]: ").strip().lower()
            if choice not in ['yes', 'y']:
                return
            process_all = True
        else:
            print(f"\n📄 Will process {stats['new']} NEW documents.")
            proceed = input("Proceed with full pipeline? (yes/no) [yes]: ").strip().lower()
            if proceed not in ['', 'yes', 'y']:
                return
            process_all = False

        import time
        pipeline_start = time.time()

        # STAGE 1: Extraction
        print(f"\n{'-'*70}")
        print("📥 STAGE 1/4: DOCUMENT EXTRACTION")
        print(f"{'-'*70}")

        if process_all:
            pdf_files = self.get_all_pdfs(skip_processed=False, skip_chinese=True)
        else:
            pdf_files = self.get_all_pdfs(skip_processed=True, skip_chinese=True)

        if pdf_files:
            results, proc_stats = self._process_batch(pdf_files)
            self._session_processed = [(r.company, r.year) for r in results]
            print(f"  ✓ Processed {len(results)} documents")
            print(f"    📊 {proc_stats['new_refs']} new refs, {proc_stats['updated_refs']} updated")
        else:
            print("  ✓ No documents to process")

        # STAGE 2: Deduplication
        print(f"\n{'-'*70}")
        print("📊 STAGE 2/4: ANALYSIS & DEDUPLICATION")
        print(f"{'-'*70}")

        cursor = self.db_manager.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT company, year FROM ai_references_raw
            WHERE (company, year) NOT IN (
                SELECT DISTINCT company, year FROM ai_references_deduplicated
            )
        """)
        new_pairs = cursor.fetchall()

        if new_pairs:
            self._run_deduplication(company_year_pairs=new_pairs)
        else:
            print("  ✓ No deduplication needed")

        # STAGE 3: Index calculation
        print(f"\n{'-'*70}")
        print("📈 STAGE 3/4: AI ADOPTION INDEX CALCULATION")
        print(f"{'-'*70}")

        cursor.execute("""
            SELECT DISTINCT company, year FROM ai_references_deduplicated
            WHERE (company, year) NOT IN (
                SELECT DISTINCT company, year FROM adoption_index
            )
        """)
        new_index_pairs = cursor.fetchall()

        if new_index_pairs:
            self._calculate_indices(company_year_pairs=new_index_pairs)
        else:
            print("  ✓ No new indices to calculate")

        # STAGE 4: Output generation
        print(f"\n{'-'*70}")
        print("📋 STAGE 4/4: REPORT & CHART GENERATION")
        print(f"{'-'*70}")

        print("  Generating Excel reports...")
        self._export_raw_references()
        self._export_deduplicated_references()
        self._export_adoption_index()

        print("  Generating JSON...")
        self._export_json()

        print("  Generating text report...")
        self._export_text_report()

        print("  Generating charts...")
        indices, refs = self._get_visualization_data()
        if indices:
            self.viz_generator.generate_all_visualizations(indices, refs)

        print("  Aggregating industry data...")
        self.industry_aggregator.aggregate_by_industry()

        pipeline_time = time.time() - pipeline_start

        # Final results
        print(f"\n{'═'*70}")
        print("🎉 FULL PIPELINE COMPLETE!")
        print(f"{'═'*70}")
        print(f"   ⏱️  Total time: {pipeline_time/60:.1f} minutes")

        stats = self.get_statistics(force_refresh=True)
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Documents processed: {stats['processed']}")
        print(f"   Unique references: {stats['total_refs_raw']:,}")
        print(f"   Deduplicated references: {stats['total_refs_dedup']:,}")
        print(f"   Companies with index: {stats['companies_with_index']}")

        print(f"\n📁 OUTPUTS in {self.config.output_folder}:")
        print("   • ai_references_raw.xlsx")
        print("   • ai_references_deduplicated.xlsx")
        print("   • ai_adoption_index.xlsx")
        print("   • eu_classification.xlsx")
        print("   • results.json")
        print("   • report.txt")
        print("   • viz_*.html (incl. viz_eu_sunburst, viz_taxonomy_comparison)")

        self._stats_cache = None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN MENU LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    
    def run(self):
        """Run the interactive main menu loop."""

        handlers = {
            # 1.x - Extraction
            '1.1': self.handle_option_1_1,
            '1.2': self.handle_option_1_2,
            '1.3': self.handle_option_1_3,
            '1.4': self.handle_option_1_4,
            '1.5': self.handle_option_1_5,
            '1.6': self.handle_option_1_6,
                
            # 2.x - Analysis
            '2.1': self.handle_option_2_1,
            '2.2': self.handle_option_2_2,
            '2.3': self.handle_option_2_3,
            
            # 3.x - Index calculation
            '3.1': self.handle_option_3_1,
            '3.2': self.handle_option_3_2,
            '3.3': self.handle_option_3_3,
            
            # 4.x - Reports
            '4.1': self.handle_option_4_1,
            '4.2': self.handle_option_4_2,
            '4.3': self.handle_option_4_3,
            '4.4': self.handle_option_4_4,
            '4.5': self.handle_option_4_5,
            '4.6': self.handle_option_4_6,
            '4.7': self.handle_option_4_7,

            # 5.x - Charts
            '5.1': self.handle_option_5_1,
            '5.2': self.handle_option_5_2,
            '5.3': self.handle_option_5_3,
            '5.4': self.handle_option_5_4,
            '5.5': self.handle_option_5_5,
            '5.6': self.handle_option_5_6,
            '5.7': self.handle_option_5_7,
            '5.8': self.handle_option_5_8,
            '5.9': self.handle_option_5_9,
            '5.10': self.handle_option_5_10,
            '5.11': self.handle_option_5_11,

            # 6.x - Utilities
            '6.1': self.handle_option_6_1,
            '6.2': self.handle_option_6_2,
            '6.3': self.handle_option_6_3,
            '6.4': self.handle_option_6_4,
            '6.5': self.handle_option_6_5,
            '6.6': self.handle_option_6_6,
            
            # 7 - Full pipeline
            '7': self.handle_option_7,
        }
        
        while True:
            try:
                self.clear_screen()
                self.display_main_menu()
                
                choice = input("\n🔹 Select option: ").strip()

                if choice == '0':
                    print("\n👋 Goodbye!")
                    break
                elif choice in handlers:
                    handlers[choice]()
                    input("\n⏎ Press Enter to continue...")
                else:
                    print(f"\n✗ Invalid option: '{choice}'")
                    input("⏎ Press Enter to continue...")

            except KeyboardInterrupt:
                print("\n\n⚠ Interrupted by user.")
                break
            except Exception as e:
                logger.error(f"Menu error: {e}")
                print(f"\n✗ Error: {e}")
                input("⏎ Press Enter to continue...")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    print(f"\n{'═'*80}")
    print("       AI SEMANTIC ANALYZER - CORPORATE AI ADOPTION ANALYSIS")
    print(f"{'═'*80}")
    print("\n  Features:")
    print("    • Dual taxonomy (classic + EU_Semantics)")
    print("    • Semantic analysis with SentenceTransformer")
    print("    • FinBERT sentiment analysis")
    print("    • 7-dimensional AI Adoption Index")
    print("    • Advanced semantic deduplication")
    print("    • Export to Excel, JSON, Plotly visualizations")
    print("    • Traditional robotics and RPA filtering")
    print("    • Full document processing tracking")
    print("    • Re-analysis with occurrence count increment")
    print(f"\n{'═'*80}")

    menu = MenuManager()
    menu.run()


if __name__ == "__main__":
    main()