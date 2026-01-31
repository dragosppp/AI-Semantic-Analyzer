"""
═══════════════════════════════════════════════════════════════════════════════
AI SEMANTIC ANALYZER v6.1.2 - MODUL MENIU INTERACTIV
═══════════════════════════════════════════════════════════════════════════════

CHANGELOG v6.1.2 (Ian 2026):
    - FIX: Eroare "cannot access local variable 'ref'" în detecția semantică
    - FIX: DocumentResult lipsea sector/country pentru documente fără rezultat
    - FIX: Export RAW acum include reference_strength, confidence_score, confidence_reasons
    - FIX: Export DEDUP cu placeholder pentru câmpurile de confidence
    - FIX: Calcul Index folosește schema corectă pentru ai_references_deduplicated
    - FIX: AIReference în _detect_by_semantics primește acum strength/conf/reasons

CHANGELOG v6.0.5:
    - NOU: Opțiunea 1.6 - Re-procesare documente cu TEXT CORUPT (OCR)
    - NOU: Coloana text_status în processed_documents pentru tracking status text
    - NOU: DocumentProcessor.process_pdf() returnează acum (result, text_status)
    - NOU: mark_document_processed() acceptă parametrul text_status
    - NOU: Detectare automată text corupt (encoding issues, OCR necesar)
    - ÎMBUNĂTĂȚIT: Statistici extinse cu documente corupte/OCR
    
CHANGELOG v6.0.4:
    - RESTRUCTURARE COMPLETĂ opțiuni 1.x:
      * 1.1 = Procesează TOATE documentele (noi + re-analiză existente)
      * 1.2 = Procesează BATCH după poziții (noi + re-analiză în range)
      * 1.3 = Procesează doar documente NOI - toate neprocesate
      * 1.4 = Procesează doar documente NOI - batch de N
      * 1.5 = Reverificare documente FĂRĂ referințe AI
      * 1.6 = Re-procesare documente cu TEXT CORUPT (OCR) [NOU în v6.0.5]
    - La re-analiză se incrementează occurrence_count pentru referințe existente
    - Referințele noi găsite la re-analiză se adaugă corect
    - Statistici separate pentru documente noi vs re-analizate
    - Mesaje clare despre ce face fiecare opțiune

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import re
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

# Import toate modulele v6.0.6
try:
    from ai_analyzer_v6_1_module1 import (
        logger, AnalyzerConfig, AIReference, DocumentResult,
        AIAdoptionIndex, DatabaseManager, AI_CATEGORIES,
        load_saved_config, save_config, CONFIG_FILE,
        ANALYZER_VERSION
    )
    from ai_analyzer_v6_1_module2 import (
        PDFTextExtractor, AIReferenceDetector, FilenameParser,
        SemanticModelLoader, get_ai_description_embeddings
    )
    from ai_analyzer_v6_1_module3 import (
        FinBERTSentimentAnalyzer, ImprovedSentimentAnalyzer,
        SemanticDeduplicator, AIAdoptionIndexCalculatorV6
    )
    from ai_analyzer_v6_1_module4 import (
        ExcelExporter, VisualizationGenerator, GroupAggregator,
        AnalysisPipeline
    )
except ImportError as e:
    print(f"EROARE: Nu s-au putut importa modulele necesare: {e}")
    print("Asigură-te că toate fișierele ai_analyzer_v6_1_module*.py sunt în același folder.")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# EXTENDED DATABASE MANAGER - cu tracking documente procesate și text_status
# ═══════════════════════════════════════════════════════════════════════════════

class ExtendedDatabaseManager(DatabaseManager):
    """Database Manager extins cu tracking pentru documente procesate și text_status."""
    
    def create_tables(self):
        """Creează toate tabelele, inclusiv processed_documents cu text_status."""
        # Apelează metoda din clasa părinte
        super().create_tables()
        
        cursor = self.conn.cursor()
        
        # Tabelă pentru tracking documente procesate (include text_status)
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
        
        # Adaugă coloana occurrence_count la ai_references_raw dacă nu există
        try:
            cursor.execute("ALTER TABLE ai_references_raw ADD COLUMN occurrence_count INTEGER DEFAULT 1")
            logger.info("Adăugat coloana occurrence_count la ai_references_raw")
        except:
            pass  # Coloana există deja
        
        # Adaugă coloana ref_hash pentru identificare unică
        try:
            cursor.execute("ALTER TABLE ai_references_raw ADD COLUMN ref_hash TEXT")
            logger.info("Adăugat coloana ref_hash la ai_references_raw")
        except:
            pass
        
                # v6.1.1: FP meta columns (reference_strength / confidence_score / confidence_reasons)
        try:
            cursor.execute("ALTER TABLE ai_references_raw ADD COLUMN reference_strength TEXT DEFAULT 'unknown'")
            logger.info("Adăugat coloana reference_strength la ai_references_raw")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE ai_references_raw ADD COLUMN confidence_score REAL DEFAULT 0.0")
            logger.info("Adăugat coloana confidence_score la ai_references_raw")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE ai_references_raw ADD COLUMN confidence_reasons TEXT DEFAULT ''")
            logger.info("Adăugat coloana confidence_reasons la ai_references_raw")
        except:
            pass

        # Adaugă coloana text_status dacă nu există (pentru DB existente)
        try:
            cursor.execute("ALTER TABLE processed_documents ADD COLUMN text_status TEXT DEFAULT 'valid'")
            logger.info("Adăugat coloana text_status la processed_documents")
        except:
            pass  # Coloana există deja
        
        # Index pentru căutare rapidă
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_source ON processed_documents(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ref_hash ON ai_references_raw(ref_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_text_status ON processed_documents(text_status)")

        self.conn.commit()
        logger.info("✓ Tabele extinse create/verificate (v6.0.6)")
    
    def is_document_processed(self, source: str) -> bool:
        """Verifică dacă un document a fost procesat."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM processed_documents WHERE source = ?", (source,))
        return cursor.fetchone() is not None
    
    def get_processed_documents(self) -> set:
        """Returnează set-ul de documente procesate."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT source FROM processed_documents")
        return set(row[0] for row in cursor.fetchall())
    
    def mark_document_processed(self, doc_result: 'DocumentResult', refs_found: int, 
                               file_hash: str = None, text_status: str = 'valid'):
        """
        Marchează un document ca procesat sau actualizează dacă există.
        
        Args:
            doc_result: Rezultatul procesării documentului
            refs_found: Numărul de referințe AI găsite
            file_hash: Hash-ul fișierului (opțional)
            text_status: Statusul textului extras - 'valid', 'corrupted_ocr_success', 
                        'corrupted_ocr_failed', 'ocr_needed', 'empty', 'error'
        """
        cursor = self.conn.cursor()
        
        # Verifică dacă documentul există deja
        cursor.execute("SELECT id, process_count FROM processed_documents WHERE source = ?", 
                      (doc_result.source,))
        existing = cursor.fetchone()
        
        if existing:
            # Actualizează - incrementează process_count
            cursor.execute("""
                UPDATE processed_documents 
                SET last_processed_at = CURRENT_TIMESTAMP,
                    process_count = process_count + 1,
                    refs_found = ?,
                    file_hash = COALESCE(?, file_hash),
                    text_status = ?
                WHERE source = ?
            """, (refs_found, file_hash, text_status, doc_result.source))
            logger.debug(f"Actualizat document: {doc_result.source} (procesare #{existing[1]+1}, status={text_status})")
        else:
            # Inserează nou
            cursor.execute("""
                INSERT INTO processed_documents 
                (source, company, year, position, industry, doc_type, 
                 total_pages, text_length, refs_found, file_hash, text_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (doc_result.source, doc_result.company, doc_result.year,
                  doc_result.position, doc_result.industry, doc_result.doc_type,
                  doc_result.total_pages, doc_result.text_length, refs_found, 
                  file_hash, text_status))
            logger.debug(f"Nou document procesat: {doc_result.source} (status={text_status})")
        
        self.conn.commit()
    
    def insert_or_update_reference(self, ref: 'AIReference') -> Tuple[bool, bool]:
        """
        Inserează sau actualizează o referință.
        Returnează: (is_new, was_updated)
        """
        cursor = self.conn.cursor()
        
        # Generează hash unic pentru referință
        ref_hash = self._generate_ref_hash(ref)
        
        # Verifică dacă referința există deja (prin UNIQUE constraint fields)
        cursor.execute("""
            SELECT id, occurrence_count FROM ai_references_raw 
            WHERE company = ? AND year = ? AND doc_type = ? AND page = ? AND text = ?
        """, (ref.company, ref.year, ref.doc_type, ref.page, ref.text))
        
        existing = cursor.fetchone()
        
        if existing:
            # Referința există - incrementează occurrence_count
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
            logger.debug(f"Referință actualizată (occurrence={new_count}): {ref.text[:50]}...")
            return (False, True)  # Nu e nouă, dar a fost actualizată
        else:
            # Referința nu există - inserează nouă
            try:
                cursor.execute("""
                    INSERT INTO ai_references_raw 
                    (company, year, position, industry, sector, country, doc_type, page, text, context,
                     category, sentiment, sentiment_score, semantic_score, detection_method,
                     source, occurrence_count, ref_hash, reference_strength, confidence_score, confidence_reasons)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                """, (ref.company, ref.year, ref.position, ref.industry, 
                      getattr(ref, "sector", "Unknown"), getattr(ref, "country", "Unknown"), 
                      ref.doc_type, ref.page, ref.text, ref.context, 
                      ref.category, ref.sentiment, ref.sentiment_score, ref.semantic_score, 
                      ref.detection_method, ref.source, ref_hash,
                      getattr(ref, "reference_strength", "unknown"), 
                      float(getattr(ref, "confidence_score", 0.0) or 0.0), 
                      getattr(ref, "confidence_reasons", "") or ""))
                self.conn.commit()
                logger.debug(f"Referință nouă inserată: {ref.text[:50]}...")
                return (True, False)  # Este nouă
                
            except Exception as e:
                # Dacă tot apare UNIQUE constraint (race condition), actualizează
                if "UNIQUE constraint failed" in str(e):
                    logger.warning(f"UNIQUE constraint - actualizez referința existentă")
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
        Generează hash SHA-256 unic pentru o referință.
        
        Hash-ul include:
        - Company
        - Year
        - Doc type
        - Page
        - Text (primele 200 caractere pentru unicitate maximă)
        
        Returns:
            Hash SHA-256 (32 caractere hex)
        """
        # Normalizează textul pentru consistență
        normalized_text = ' '.join(ref.text.split())[:200]
        # Construiește string-ul unic
        unique_str = f"{ref.company}|{ref.year}|{ref.doc_type}|{ref.page}|{normalized_text}"
        # Generează SHA-256 și returnează primele 32 caractere
        return hashlib.sha256(unique_str.encode('utf-8')).hexdigest()[:32]    
    def get_processing_stats(self) -> Dict:
        """Obține statistici detaliate despre procesare, inclusiv text_status."""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Documente procesate
        cursor.execute("SELECT COUNT(*) FROM processed_documents")
        stats['total_documents_processed'] = cursor.fetchone()[0]
        
        # Documente procesate de mai multe ori
        cursor.execute("SELECT COUNT(*) FROM processed_documents WHERE process_count > 1")
        stats['documents_reprocessed'] = cursor.fetchone()[0]
        
        # Total procesări (sumă process_count)
        cursor.execute("SELECT SUM(process_count) FROM processed_documents")
        stats['total_processing_runs'] = cursor.fetchone()[0] or 0
        
        # Referințe
        try:
            cursor.execute("SELECT COUNT(*), SUM(COALESCE(occurrence_count, 1)) FROM ai_references_raw")
            row = cursor.fetchone()
            stats['unique_references'] = row[0] or 0
            stats['total_occurrences'] = row[1] or 0
        except:
            cursor.execute("SELECT COUNT(*) FROM ai_references_raw")
            stats['unique_references'] = cursor.fetchone()[0] or 0
            stats['total_occurrences'] = stats['unique_references']
        
        # Documente fără referințe
        cursor.execute("SELECT COUNT(*) FROM processed_documents WHERE refs_found = 0")
        stats['documents_without_refs'] = cursor.fetchone()[0]
        
        # Statistici text_status (NOU în v6.0.6)
        try:
            cursor.execute("""
                SELECT text_status, COUNT(*) FROM processed_documents 
                GROUP BY text_status
            """)
            stats['by_text_status'] = {row[0] or 'unknown': row[1] for row in cursor.fetchall()}
        except:
            stats['by_text_status'] = {}
        
        # Documente cu probleme de text
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
        """
        Metodă de compatibilitate cu codul existent.
        Redirecțează către insert_or_update_reference.
        """
        return self.insert_or_update_reference(ref)


# ═══════════════════════════════════════════════════════════════════════════════
# METADATA EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class MetadataExtractor:
    """Extractor metadata din filename."""
    
    @staticmethod
    def is_chinese_document(filename: str) -> bool:
        """Verifică dacă documentul este în chineză."""
        filename_clean = filename.strip()
        return (filename_clean.endswith(' CN') or 
                filename_clean.endswith('_CN') or 
                filename_clean.endswith('-CN') or
                filename_clean.upper().endswith(' CN'))


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT PROCESSOR v6.0.6 - cu returnare text_status
# ═══════════════════════════════════════════════════════════════════════════════

class DocumentProcessor:
    """
    Procesor pentru documente PDF v6.0.6.
    
    MODIFICARE v6.0.5: process_pdf() returnează acum (DocumentResult, text_status)
    în loc de doar DocumentResult, pentru a permite tracking-ul statusului textului.
    """
    
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.pdf_extractor = PDFTextExtractor(config)
        self.ai_detector = AIReferenceDetector(config)
        self.filename_parser = FilenameParser()
    
    def process_pdf(self, pdf_path: str, company_info: Optional[Dict] = None) -> Tuple[Optional[DocumentResult], str]:
        """
        Procesează un singur PDF și returnează (DocumentResult, text_status).
        v6.0.6: Include sector și country.
        """
        try:
            text, num_pages, text_status = self.pdf_extractor.extract_text_from_pdf(pdf_path)
            
            if not text or len(text) < 100:
                logger.warning(f"Text insuficient în {pdf_path}")
                return None, text_status if text_status else 'empty'
            
            filename = Path(pdf_path).name
            parsed = self.filename_parser.parse_filename(filename)
            
            # Completează cu company_info dacă există - v6.0.6: include sector, country
            if company_info:
                parsed['position'] = company_info.get('position', parsed.get('position', 0))
                parsed['sector'] = company_info.get('sector', parsed.get('sector', 'Unknown'))
                parsed['industry'] = company_info.get('industry', parsed.get('industry', 'Unknown'))
                parsed['country'] = company_info.get('country', parsed.get('country', 'Unknown'))
                if not parsed.get('company'):
                    parsed['company'] = company_info.get('company', 'Unknown')
            
            # Creează DocumentResult - v6.0.6: include sector, country
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
            
            # Detectează referințe AI - v6.0.6: include sector, country
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
            
            logger.info(f"Procesat {filename}: {len(references)} referințe găsite (text_status={text_status})")
            return doc_result, text_status
            
        except Exception as e:
            logger.error(f"Eroare procesare {pdf_path}: {e}")
            return None, 'error'


# ═══════════════════════════════════════════════════════════════════════════════
# FORTUNE 500 LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_fortune500_data(csv_path: str) -> Dict[str, Dict]:
    """Încarcă date Fortune 500 din CSV. v6.0.6: Include Sector, Industry, Country."""
    if not csv_path or not Path(csv_path).exists():
        logger.warning("CSV Fortune 500 nu există, se folosesc date din filename")
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
            logger.warning("Coloană company negăsită în CSV")
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
        
        logger.info(f"✓ Încărcat {len(data)} companii din Fortune 500 (cu Sector, Industry, Country)")
        return data
        
    except Exception as e:
        logger.error(f"Eroare încărcare CSV: {e}")
        return {}


def match_company(filename: str, fortune500_data: Dict[str, Dict]) -> Optional[Dict]:
    """Găsește compania din Fortune 500 pe baza filename."""
    if not fortune500_data:
        return None
    
    base = Path(filename).stem.lower()
    
    # Căutare directă
    for company_key, company_data in fortune500_data.items():
        if company_key in base:
            return company_data
    
    # Căutare parțială
    base_parts = base.replace('-', ' ').replace('_', ' ').split()
    for company_key, company_data in fortune500_data.items():
        company_parts = company_key.split()
        if any(part in base_parts for part in company_parts if len(part) > 3):
            return company_data
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MENU MANAGER - CLASA PRINCIPALĂ v6.0.6
# ═══════════════════════════════════════════════════════════════════════════════

class MenuManager:
    """Manager pentru interfața de meniu interactiv."""
    
    VERSION = ANALYZER_VERSION
    
    def __init__(self):
        self.config = None
        self.db_manager = None
        self.processor = None
        self.deduplicator = None
        self.index_calculator = None
        self.excel_exporter = None
        self.viz_generator = None
        self.industry_aggregator = None
        self.fortune500_data = {}
        self._stats_cache = None
        self._session_processed = []
    
    def clear_screen(self):
        """Curăță ecranul."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def get_statistics(self, force_refresh: bool = False) -> Dict:
        """Obține statistici despre starea curentă, inclusiv text_status."""
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
                
                # Statistici din processed_documents
                cursor.execute("SELECT COUNT(*) FROM processed_documents")
                stats['processed'] = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(*) FROM processed_documents WHERE process_count > 1")
                stats['reprocessed_docs'] = cursor.fetchone()[0] or 0
                
                # Referințe
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
                
                # Statistici text_status (NOU în v6.0.6)
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
                logger.warning(f"Eroare la obținerea statisticilor: {e}")
        
        stats['new'] = stats['total_pdfs'] - stats['processed'] - stats['chinese']
        if stats['new'] < 0:
            stats['new'] = 0
            
        self._stats_cache = stats
        return stats
    
    def display_header(self):
        """Afișează header-ul aplicației."""
        stats = self.get_statistics()
        
        print("\n╔═══════════════════════════════════════════════════════════════════════════════╗")
        print(f"║                      AI SEMANTIC ANALYZER v{self.VERSION}                             ║")
        print("║                   Fortune 500 AI Adoption Analysis                           ║")
        print("║        13 categorii AI | 80+ patterns | GenAI 2025 | Threshold 0.60          ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"📊 STATUS: {stats['total_pdfs']} total | {stats['processed']} procesate | "
              f"{stats['new']} noi | {stats['chinese']} CN (ignorate)")
        if stats['reprocessed_docs'] > 0:
            print(f"   🔄 {stats['reprocessed_docs']} documente re-analizate")
        if stats['docs_with_text_issues'] > 0:
            print(f"   ⚠️  {stats['docs_with_text_issues']} documente cu probleme de text (opțiunea 1.6)")
        if stats['total_refs_raw'] > 0:
            print(f"🔍 REFS: {stats['total_refs_raw']:,} unice | {stats['total_occurrences']:,} total apariții | "
                  f"{stats['total_refs_dedup']:,} deduplicate | {stats['companies_with_index']} companii cu index")
    
    def display_file_format_info(self):
        """Afișează informații despre formatul fișierelor."""
        print("\n┌─────────────────────────────────────────────────────────────────────────────────┐")
        print("│ 📄 FORMAT FIȘIERE PDF ACCEPTAT:                                               │")
        print("├─────────────────────────────────────────────────────────────────────────────────┤")
        print("│  Format: \"POZIȚIE. Nume Companie - AN - tip document.pdf\"                     │")
        print("│                                                                               │")
        print("│  Exemple:                                                                     │")
        print("│    ✓ 38. Bank of America - 2024 - annual report.pdf                          │")
        print("│    ✓ 1. Walmart - 2023 - sustainability report.pdf                           │")
        print("│    ✓ 150. Microsoft - 2025 - proxy.pdf                                       │")
        print("│                                                                               │")
        print("│  Tipuri document: annual report, sustainability, proxy, 10-K, ESG            │")
        print("│  Documente CN (chinezești) sunt ignorate automat                             │")
        print("│  Ani suportați: 2020-2025                                                    │")
        print("└─────────────────────────────────────────────────────────────────────────────────┘")
    
    def display_main_menu(self):
        """Afișează meniul principal."""
        self.display_header()
        self.display_file_format_info()
        
        stats = self.get_statistics()
        
        # Calculează documente cu probleme pentru afișare
        docs_with_issues = stats.get('docs_with_text_issues', 0)
        
        print("\n═══════════════════════════════════════════════════════════════════════════════")
        print("📥 1. EXTRAGERE DATE (Document Processing)")
        print("═══════════════════════════════════════════════════════════════════════════════")
        print(f"   1.1  TOATE documentele (noi + re-analiză) [{stats['total_pdfs'] - stats['chinese']} docs]")
        print(f"   1.2  BATCH după poziții (noi + re-analiză în range)")
        print(f"   1.3  Doar documente NOI - toate [{stats['new']} disponibile]")
        print(f"   1.4  Doar documente NOI - batch de N")
        print(f"   1.5  Reverificare documente FĂRĂ referințe AI")
        if docs_with_issues > 0:
            print(f"   1.6  Re-procesare documente cu TEXT CORUPT (OCR) [{docs_with_issues} docs] ⚠️")
        else:
            print(f"   1.6  Re-procesare documente cu TEXT CORUPT (OCR)")

        print("\n═══════════════════════════════════════════════════════════════════════════════")
        print("📊 2. ANALIZĂ & DEDUPLICARE")
        print("═══════════════════════════════════════════════════════════════════════════════")
        print("   2.1  Analiză BATCH (poziții specifice)")
        print("   2.2  Analiză TOATE companiile")
        print("   2.3  Analiză doar cele NOI (din această sesiune)")
        
        print("\n═══════════════════════════════════════════════════════════════════════════════")
        print("📈 3. CALCUL AI ADOPTION INDEX (7 dimensiuni)")
        print("═══════════════════════════════════════════════════════════════════════════════")
        print("   3.1  Calcul BATCH (poziții specifice)")
        print("   3.2  Calcul TOATE companiile")
        print("   3.3  Calcul doar cele NOI")
        
        print("\n═══════════════════════════════════════════════════════════════════════════════")
        print("📋 4. RAPOARTE")
        print("═══════════════════════════════════════════════════════════════════════════════")
        print("   4.1  Export referințe RAW (ai_references_v6.xlsx)")
        print("   4.2  Export referințe DEDUPLICATE")
        print("   4.3  Export AI INDEX companii (ai_adoption_index_v6.xlsx)")
        print("   4.4  Export JSON complet (results_v6.json)")
        print("   4.5  Raport TEXT sumar")
        print("   4.6  TOATE rapoartele")
        
        print("\n═══════════════════════════════════════════════════════════════════════════════")
        print("📊 5. GRAFICE (Plotly Interactive)")
        print("═══════════════════════════════════════════════════════════════════════════════")
        print("   5.1  Ranking companii (bar chart) - per an")
        print("   5.2  Evoluție top companii (line chart)")
        print("   5.3  Radar dimensiuni top 10 companii")
        print("   5.4  Heatmap companii x ani")
        print("   5.5  Distribuție categorii AI (pie chart)")
        print("   5.6  TOATE graficele disponibile")
        print("   5.7  Grafice INDUSTRIE")
        print("   5.8  Grafice SECTOR")
        print("   5.9  Grafice COUNTRY (Țară)")
        
        print("\n═══════════════════════════════════════════════════════════════════════════════")
        print("🔧 6. UTILITĂȚI")
        print("═══════════════════════════════════════════════════════════════════════════════")
        print("   6.1  Statistici și sumar")
        print("   6.2  Verificare integritate DB")
        print("   6.3  Configurare căi și setări")
        print("   6.4  Import mapping industrii (CSV)")
        print("   6.5  Info categorii AI v6.0")
        print("   6.6  Matrice acoperire documente")
        
        print("\n═══════════════════════════════════════════════════════════════════════════════")
        print("🚀 7. UPDATE COMPLET (Pipeline Automat)")
        print("═══════════════════════════════════════════════════════════════════════════════")
        print("   Procesează automat documentele NOI și generează TOATE output-urile")
        
        print("\n═══════════════════════════════════════════════════════════════════════════════")
        print("   0.  IEȘIRE")
        print("═══════════════════════════════════════════════════════════════════════════════")
    
    def get_position_range(self) -> Tuple[int, int]:
        """Solicită intervalul de poziții de la utilizator."""
        print("\nIntrodu intervalul de poziții Fortune 500:")
        while True:
            try:
                start = input("  Poziție START (1-500) [1]: ").strip()
                start = int(start) if start else 1
                end = input("  Poziție END (1-500) [500]: ").strip()
                end = int(end) if end else 500
                if 1 <= start <= 500 and 1 <= end <= 500 and start <= end:
                    return (start, end)
                print("  ✗ Poziții invalide. Reintrodu.")
            except ValueError:
                print("  ✗ Introdu numere valide.")
    
    def get_pdfs_by_position_range(self, start: int, end: int, 
                                   skip_processed: bool = False,
                                   skip_chinese: bool = True) -> List[str]:
        """Obține PDF-urile din intervalul de poziții specificat."""
        folder_path = Path(self.config.input_folder)
        all_pdfs = list(folder_path.glob("*.pdf"))
        filtered = []
        
        for pdf in all_pdfs:
            # Skip documente chinezești
            if skip_chinese and MetadataExtractor.is_chinese_document(pdf.stem):
                continue
            
            # Extrage poziția din filename
            match = re.match(r'^(\d{1,3})\.', pdf.stem)
            if match:
                position = int(match.group(1))
                if start <= position <= end:
                    filtered.append(pdf)
        
        # Skip documente deja procesate dacă e cerut
        if skip_processed and self.db_manager:
            processed_sources = self.db_manager.get_processed_documents()
            filtered = [pdf for pdf in filtered if pdf.name not in processed_sources]
        
        filtered.sort()
        return [str(pdf) for pdf in filtered]
    
    def get_all_pdfs(self, skip_processed: bool = False, skip_chinese: bool = True) -> List[str]:
        """Obține toate PDF-urile din folder."""
        folder_path = Path(self.config.input_folder)
        if not folder_path.exists():
            return []
        
        all_pdfs = list(folder_path.glob("*.pdf"))
        
        # Skip documente chinezești
        if skip_chinese:
            all_pdfs = [pdf for pdf in all_pdfs 
                       if not MetadataExtractor.is_chinese_document(pdf.stem)]
        
        # Skip documente deja procesate dacă e cerut
        if skip_processed and self.db_manager:
            processed_sources = self.db_manager.get_processed_documents()
            all_pdfs = [pdf for pdf in all_pdfs if pdf.name not in processed_sources]
        
        all_pdfs.sort()
        return [str(pdf) for pdf in all_pdfs]
    
    def initialize_components(self):
        """Inițializează toate componentele necesare."""
        if self.db_manager is None:
            print(f"\n⏳ Inițializare componente v{self.VERSION}...")
            
            # Încarcă sau creează configurația
            saved_config = load_saved_config()
            if saved_config:
                self.config = AnalyzerConfig.from_dict(saved_config)
            else:
                self.config = AnalyzerConfig.from_interactive()
            
            # Inițializează Database Manager extins
            db_path = str(Path(self.config.output_folder) / self.config.database_name)
            self.db_manager = ExtendedDatabaseManager(db_path)
            self.db_manager.create_tables()
            
            # Inițializează celelalte componente
            self.processor = DocumentProcessor(self.config)
            self.deduplicator = SemanticDeduplicator(self.config)
            self.index_calculator = AIAdoptionIndexCalculatorV6(self.config)
            self.excel_exporter = ExcelExporter(self.config)
            self.viz_generator = VisualizationGenerator(self.config)
            self.group_aggregator = GroupAggregator(self.db_manager)
            
            # Încarcă date Fortune 500
            if self.config.fortune500_csv:
                self.fortune500_data = load_fortune500_data(self.config.fortune500_csv)
            
            print(f"✓ Componente v{self.VERSION} inițializate!")


# ═══════════════════════════════════════════════════════════════════════════════
# SFÂRȘIT PARTEA 1/3 - Continuă în partea 2
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# AI SEMANTIC ANALYZER v6.0.6 - PARTEA 2/3
# Continuare clasa MenuManager - Metode de procesare și handlers 1.x, 2.x, 3.x
# ═══════════════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════════
    # METODA PRINCIPALĂ DE PROCESARE BATCH
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _process_batch(self, pdf_files: List[str]) -> Tuple[List[DocumentResult], Dict]:
        """
        Procesează un batch de PDF-uri.
        Detectează automat dacă documentul e nou sau trebuie re-analizat.
        La re-analiză: referințele existente au occurrence_count incrementat.
        
        Returns:
            Tuple[List[DocumentResult], Dict]: Rezultatele și statisticile procesării
        """
        from tqdm import tqdm
        
        results = []
        stats = {
            'new_refs': 0,           # Referințe complet noi
            'updated_refs': 0,       # Referințe existente cu occurrence incrementat
            'docs_new': 0,           # Documente procesate prima dată
            'docs_reanalyzed': 0,    # Documente re-analizate
            'docs_no_refs': 0,       # Documente fără referințe AI
            'errors': 0              # Erori de procesare
        }
        
        # Obține lista documentelor deja procesate
        processed_sources = self.db_manager.get_processed_documents()
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            filename = Path(pdf_path).name
            is_reanalysis = filename in processed_sources
            
            try:
                # Obține info companie din Fortune 500
                company_info = match_company(pdf_path, self.fortune500_data)
                
                # Procesează PDF-ul (acum returnează și text_status)
                result, text_status = self.processor.process_pdf(pdf_path, company_info)
                
                if result:
                    refs_new = 0
                    refs_updated = 0
                    
                    # Procesează fiecare referință găsită
                    for ref in result.references:
                        try:
                            is_new, was_updated = self.db_manager.insert_or_update_reference(ref)
                            if is_new:
                                refs_new += 1
                            elif was_updated:
                                refs_updated += 1
                        except Exception as ref_error:
                            logger.warning(f"Eroare la inserare referință: {ref_error}")
                            # Fallback: încearcă update direct
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
                                    # Dacă nu există, inserează nou
                                    cursor.execute("""
                                        INSERT OR IGNORE INTO ai_references_raw 
                                        (company, year, position, industry, sector, country, doc_type, page, 
                                         text, context, category, sentiment, sentiment_score, 
                                         semantic_score, detection_method, source, occurrence_count, 
                                         reference_strength, confidence_score, confidence_reasons)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                                    """, (ref.company, ref.year, ref.position, ref.industry, 
                                          getattr(ref,"sector","Unknown"), getattr(ref,"country","Unknown"),
                                          ref.doc_type, ref.page, ref.text, ref.context,
                                          ref.category, ref.sentiment, ref.sentiment_score,
                                          ref.semantic_score, ref.detection_method, ref.source,
                                          getattr(ref,"reference_strength","unknown"), 
                                          float(getattr(ref,"confidence_score",0.0) or 0.0), 
                                          getattr(ref,"confidence_reasons","") or ""))
                                    if cursor.rowcount > 0:
                                        refs_new += 1
                                    self.db_manager.conn.commit()
                            except Exception as fallback_error:
                                logger.error(f"Eroare fallback: {fallback_error}")
                                stats['errors'] += 1
                    
                    # IMPORTANT: Marchează documentul ca procesat ÎNTOTDEAUNA
                    try:
                        self.db_manager.mark_document_processed(
                            result, len(result.references), 
                            text_status=text_status
                        )
                        logger.debug(f"✓ Marcat ca procesat: {filename}")
                    except Exception as doc_error:
                        logger.error(f"EROARE la marcarea documentului {filename}: {doc_error}")
                    
                    # Actualizează statisticile
                    stats['new_refs'] += refs_new
                    stats['updated_refs'] += refs_updated
                    
                    if len(result.references) == 0:
                        stats['docs_no_refs'] += 1
                    
                    if is_reanalysis:
                        stats['docs_reanalyzed'] += 1
                    else:
                        stats['docs_new'] += 1
                    
                    results.append(result)
                    
                    # Log pentru fiecare document procesat
                    status = "🔄 RE-ANALIZĂ" if is_reanalysis else "🆕 NOU"
                    logger.info(f"  {status} {result.company} ({result.year}): {refs_new} noi, {refs_updated} actualizate")
                
                else:
                    # result e None - documentul nu a putut fi procesat
                    # DAR tot trebuie marcat pentru a nu fi încercat din nou
                    logger.warning(f"⚠ Document fără rezultat (text insuficient?): {filename}")
                    
                    # Creează un DocumentResult minimal pentru marcare
                    try:
                        from ai_analyzer_v6_1_module2 import FilenameParser
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
                        logger.info(f"  ⚠ Marcat ca procesat (fără conținut): {filename}")
                        stats['docs_no_refs'] += 1
                        
                        if is_reanalysis:
                            stats['docs_reanalyzed'] += 1
                        else:
                            stats['docs_new'] += 1
                            
                    except Exception as mark_error:
                        logger.error(f"Nu s-a putut marca documentul {filename}: {mark_error}")
                        stats['errors'] += 1
                        
            except Exception as e:
                logger.error(f"Eroare procesare {pdf_path}: {e}")
                stats['errors'] += 1
                
                # Încearcă să marcheze documentul chiar și cu eroare
                try:
                    cursor = self.db_manager.conn.cursor()
                    cursor.execute("""
                        INSERT OR IGNORE INTO processed_documents 
                        (source, company, year, refs_found, process_count, text_status)
                        VALUES (?, 'ERROR', 0, 0, 1, 'error')
                    """, (filename,))
                    self.db_manager.conn.commit()
                    logger.info(f"  ⚠ Marcat ca procesat cu eroare: {filename}")
                except:
                    pass
                
                continue
        
        # Commit final pentru siguranță
        try:
            self.db_manager.conn.commit()
        except:
            pass
        
        if stats['errors'] > 0:
            print(f"\n⚠️  {stats['errors']} erori în timpul procesării")
        if stats['docs_no_refs'] > 0:
            print(f"📄 {stats['docs_no_refs']} documente fără referințe AI")
        
        return results, stats
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HANDLERS PENTRU OPȚIUNILE 1.x - EXTRAGERE DATE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def handle_option_1_1(self):
        """
        1.1 - Procesează TOATE documentele (noi + re-analiză existente).
        
        Această opțiune procesează toate PDF-urile din folder:
        - Documentele noi sunt analizate pentru prima dată
        - Documentele deja procesate sunt RE-ANALIZATE (occurrence_count++)
        """
        self.initialize_components()
        
        # Obține TOATE documentele (NU skip processed)
        pdf_files = self.get_all_pdfs(skip_processed=False, skip_chinese=True)
        
        if not pdf_files:
            print("\n✓ Nu există documente de procesat în folder.")
            return
        
        # Calculează statistici pentru afișare
        processed_sources = self.db_manager.get_processed_documents()
        new_count = sum(1 for pdf in pdf_files if Path(pdf).name not in processed_sources)
        reanalysis_count = len(pdf_files) - new_count
        
        print(f"\n{'═'*70}")
        print(f"📄 PROCESARE TOATE DOCUMENTELE")
        print(f"{'═'*70}")
        print(f"   Total documente: {len(pdf_files)}")
        print(f"   🆕 {new_count} documente NOI (prima analiză)")
        print(f"   🔄 {reanalysis_count} documente pentru RE-ANALIZĂ")
        print(f"\n   ℹ️  La re-analiză:")
        print(f"      • Referințele existente vor avea occurrence_count incrementat")
        print(f"      • Referințele noi descoperite vor fi adăugate")
        
        proceed = input("\nContinuăm? (da/nu) [da]: ").strip().lower()
        if proceed not in ['', 'da', 'd', 'yes', 'y']:
            print("Procesare anulată.")
            return
        
        # Procesează toate documentele
        results, stats = self._process_batch(pdf_files)
        self._session_processed = [(r.company, r.year) for r in results]
        
        # Afișează rezultatele
        print(f"\n{'═'*70}")
        print(f"✓ PROCESARE COMPLETĂ!")
        print(f"{'═'*70}")
        print(f"   📄 Documente procesate:")
        print(f"      • {stats['docs_new']} documente NOI")
        print(f"      • {stats['docs_reanalyzed']} documente RE-ANALIZATE")
        print(f"   📊 Referințe:")
        print(f"      • {stats['new_refs']} referințe NOI găsite")
        print(f"      • {stats['updated_refs']} referințe EXISTENTE (occurrence++)")
        
        self._stats_cache = None
    
    def handle_option_1_2(self):
        """
        1.2 - Procesează BATCH după poziții (noi + re-analiză în range).
        
        Procesează documentele din intervalul de poziții specificat:
        - Documentele noi sunt analizate pentru prima dată
        - Documentele deja procesate sunt RE-ANALIZATE
        """
        self.initialize_components()
        
        # Solicită intervalul de poziții
        start, end = self.get_position_range()
        
        # Obține documentele din range (NU skip processed)
        pdf_files = self.get_pdfs_by_position_range(start, end, skip_processed=False)
        
        if not pdf_files:
            print(f"\n✓ Nu există documente în intervalul {start}-{end}.")
            return
        
        # Calculează statistici
        processed_sources = self.db_manager.get_processed_documents()
        new_count = sum(1 for pdf in pdf_files if Path(pdf).name not in processed_sources)
        reanalysis_count = len(pdf_files) - new_count
        
        print(f"\n{'═'*70}")
        print(f"📄 PROCESARE BATCH POZIȚII {start}-{end}")
        print(f"{'═'*70}")
        print(f"   Total documente în range: {len(pdf_files)}")
        print(f"   🆕 {new_count} documente NOI")
        print(f"   🔄 {reanalysis_count} documente pentru RE-ANALIZĂ")
        
        proceed = input("\nContinuăm? (da/nu) [da]: ").strip().lower()
        if proceed not in ['', 'da', 'd', 'yes', 'y']:
            print("Procesare anulată.")
            return
        
        # Procesează batch-ul
        results, stats = self._process_batch(pdf_files)
        self._session_processed = [(r.company, r.year) for r in results]
        
        # Afișează rezultatele
        print(f"\n{'═'*70}")
        print(f"✓ PROCESARE BATCH COMPLETĂ!")
        print(f"{'═'*70}")
        print(f"   📄 {stats['docs_new']} NOI + {stats['docs_reanalyzed']} RE-ANALIZATE")
        print(f"   📊 {stats['new_refs']} ref. NOI + {stats['updated_refs']} ref. actualizate")
        
        self._stats_cache = None
    
    def handle_option_1_3(self):
        """
        1.3 - Procesează doar documente NOI (toate neprocesate).
        
        Această opțiune procesează DOAR documentele care nu au fost 
        niciodată analizate. Nu face re-analiză.
        """
        self.initialize_components()
        
        # Obține DOAR documentele neprocesate
        pdf_files = self.get_all_pdfs(skip_processed=True, skip_chinese=True)
        
        if not pdf_files:
            print(f"\n{'═'*70}")
            print(f"✓ Nu există documente NOI de procesat.")
            print(f"{'═'*70}")
            print(f"   Toate documentele au fost deja analizate.")
            print(f"   💡 Folosește opțiunea 1.1 sau 1.2 pentru re-analiză.")
            return
        
        print(f"\n{'═'*70}")
        print(f"🆕 PROCESARE DOCUMENTE NOI")
        print(f"{'═'*70}")
        print(f"   {len(pdf_files)} documente NOI de procesat")
        print(f"   Acestea nu au fost analizate până acum.")
        
        # Afișează primele documente
        print(f"\n   Primele documente:")
        for i, pdf in enumerate(pdf_files[:5], 1):
            print(f"      {i}. {Path(pdf).name}")
        if len(pdf_files) > 5:
            print(f"      ... și încă {len(pdf_files) - 5} documente")
        
        proceed = input("\nContinuăm? (da/nu) [da]: ").strip().lower()
        if proceed not in ['', 'da', 'd', 'yes', 'y']:
            print("Procesare anulată.")
            return
        
        # Procesează documentele noi
        results, stats = self._process_batch(pdf_files)
        self._session_processed = [(r.company, r.year) for r in results]
        
        # Afișează rezultatele
        print(f"\n{'═'*70}")
        print(f"✓ PROCESARE DOCUMENTE NOI COMPLETĂ!")
        print(f"{'═'*70}")
        print(f"   📄 {len(results)} documente procesate")
        print(f"   📊 {stats['new_refs']} referințe găsite")
        
        self._stats_cache = None
    
    def handle_option_1_4(self):
        """
        1.4 - Procesează doar documente NOI în batch de N.
        
        Procesează primele N documente care nu au fost niciodată analizate.
        Util pentru procesare incrementală.
        """
        self.initialize_components()
        
        # Obține DOAR documentele neprocesate
        new_pdfs = self.get_all_pdfs(skip_processed=True, skip_chinese=True)
        
        if not new_pdfs:
            print(f"\n{'═'*70}")
            print(f"✓ Nu există documente NOI de procesat.")
            print(f"{'═'*70}")
            print(f"   Toate documentele au fost deja analizate.")
            print(f"   💡 Folosește opțiunea 1.1 sau 1.2 pentru re-analiză.")
            return
        
        print(f"\n{'═'*70}")
        print(f"🆕 PROCESARE BATCH DOCUMENTE NOI")
        print(f"{'═'*70}")
        print(f"   {len(new_pdfs)} documente NOI disponibile")
        
        # Solicită numărul de documente
        while True:
            try:
                n_input = input(f"\n   Câte documente să procesez? (1-{len(new_pdfs)}) [10]: ").strip()
                n = int(n_input) if n_input else 10
                if 1 <= n <= len(new_pdfs):
                    break
                print(f"   ✗ Introdu un număr între 1 și {len(new_pdfs)}")
            except ValueError:
                print("   ✗ Introdu un număr valid.")
        
        # Selectează primele N documente
        batch_files = new_pdfs[:n]
        
        print(f"\n   Se vor procesa {n} documente NOI:")
        for i, pdf in enumerate(batch_files[:5], 1):
            print(f"      {i}. 🆕 {Path(pdf).name}")
        if n > 5:
            print(f"      ... și încă {n - 5} documente")
        
        proceed = input("\nContinuăm? (da/nu) [da]: ").strip().lower()
        if proceed not in ['', 'da', 'd', 'yes', 'y']:
            print("Procesare anulată.")
            return
        
        # Procesează batch-ul
        results, stats = self._process_batch(batch_files)
        self._session_processed = [(r.company, r.year) for r in results]
        
        # Calculează câte mai rămân
        remaining = len(new_pdfs) - n
        
        # Afișează rezultatele
        print(f"\n{'═'*70}")
        print(f"✓ PROCESARE BATCH COMPLETĂ!")
        print(f"{'═'*70}")
        print(f"   📄 {len(results)} documente NOI procesate")
        print(f"   📊 {stats['new_refs']} referințe găsite")
        if remaining > 0:
            print(f"   📌 Rămân încă {remaining} documente NOI de procesat")
        else:
            print(f"   ✓ Toate documentele NOI au fost procesate!")
        
        self._stats_cache = None
    
    def handle_option_1_5(self):
        """
        1.5 - Reverificare documente FĂRĂ referințe AI.
        
        Procesează din nou documentele care au fost analizate dar nu au găsit
        nicio referință AI (refs_found = 0). Util pentru:
        - Verificare cu setări/threshold diferite
        - Re-analiză după actualizarea patterns
        """
        self.initialize_components()
        
        # Găsește documentele fără referințe
        cursor = self.db_manager.conn.cursor()
        cursor.execute("SELECT source FROM processed_documents WHERE refs_found = 0")
        no_refs_docs = [row[0] for row in cursor.fetchall()]
        
        if not no_refs_docs:
            print(f"\n{'═'*70}")
            print(f"✓ Nu există documente fără referințe AI.")
            print(f"{'═'*70}")
            return
        
        # Construiește lista completă de căi
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
            print(f"\n⚠ {len(missing)} fișiere nu mai există în folder")
        
        if not pdf_files:
            print(f"\n✓ Nu există fișiere de reverificat.")
            return
        
        print(f"\n{'═'*70}")
        print(f"🔄 REVERIFICARE DOCUMENTE FĂRĂ REFERINȚE AI")
        print(f"{'═'*70}")
        print(f"   {len(pdf_files)} documente de reverificat")
        print(f"   Acestea au fost procesate anterior dar nu au găsit referințe AI.")
        print(f"\n   La reverificare:")
        print(f"      • Se vor căuta din nou referințe AI")
        print(f"      • Dacă se găsesc, vor fi adăugate în baza de date")
        print(f"      • Documentul va fi actualizat cu numărul de referințe găsite")
        
        # Afișează primele documente
        print(f"\n   Primele documente:")
        for i, pdf in enumerate(pdf_files[:5], 1):
            print(f"      {i}. {Path(pdf).name}")
        if len(pdf_files) > 5:
            print(f"      ... și încă {len(pdf_files) - 5} documente")
        
        proceed = input("\nContinuăm? (da/nu) [da]: ").strip().lower()
        if proceed not in ['', 'da', 'd', 'yes', 'y']:
            print("Reverificare anulată.")
            return
        
        # Procesează documentele
        results, stats = self._process_batch(pdf_files)
        self._session_processed = [(r.company, r.year) for r in results]
        
        # Afișează rezultatele
        print(f"\n{'═'*70}")
        print(f"✓ REVERIFICARE COMPLETĂ!")
        print(f"{'═'*70}")
        print(f"   📄 {len(results)} documente reverificate")
        print(f"   📊 {stats['new_refs']} referințe NOI găsite")
        print(f"   🔍 {stats['updated_refs']} referințe actualizate")
        
        # Câte încă nu au referințe
        cursor.execute("SELECT COUNT(*) FROM processed_documents WHERE refs_found = 0")
        still_no_refs = cursor.fetchone()[0] or 0
        if still_no_refs > 0:
            print(f"   🔭 {still_no_refs} documente încă fără referințe AI")
        
        self._stats_cache = None
    
    def handle_option_1_6(self):
        """
        1.6 - Re-procesare documente cu text corupt (necesită OCR).
        
        Procesează din nou documentele care au text_status = 'corrupted_ocr_failed' 
        sau 'ocr_needed'. Util pentru:
        - Re-încercare OCR după instalare Tesseract
        - Verificare cu setări OCR diferite
        """
        self.initialize_components()
        
        # Caută documente cu text corupt
        cursor = self.db_manager.conn.cursor()
        cursor.execute("""
            SELECT source, text_status, company, year 
            FROM processed_documents 
            WHERE text_status IN ('corrupted_ocr_failed', 'ocr_needed', 'empty')
        """)
        corrupted_docs = cursor.fetchall()
        
        if not corrupted_docs:
            print(f"\n{'═'*70}")
            print(f"✓ Nu există documente cu text corupt.")
            print(f"{'═'*70}")
            print(f"   Toate documentele au text valid extras.")
            return
        
        print(f"\n{'═'*70}")
        print(f"🔧 DOCUMENTE CU TEXT CORUPT / INCOMPLET")
        print(f"{'═'*70}")
        print(f"   {len(corrupted_docs)} documente necesită re-procesare")
        
        # Statistici per status
        status_counts = {}
        for _, status, _, _ in corrupted_docs:
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\n   Per status:")
        for status, count in status_counts.items():
            icon = "🔴" if status == 'corrupted_ocr_failed' else "🟡" if status == 'ocr_needed' else "⚪"
            print(f"      {icon} {status}: {count}")
        
        # Afișează primele documente
        print(f"\n   Primele documente:")
        for i, (source, status, company, year) in enumerate(corrupted_docs[:5], 1):
            print(f"      {i}. [{status}] {company} ({year})")
        if len(corrupted_docs) > 5:
            print(f"      ... și încă {len(corrupted_docs) - 5} documente")
        
        # Verifică dacă OCR e disponibil
        ocr_available = self.processor.pdf_extractor.ocr_available
        if not ocr_available:
            print(f"\n   ⚠️  OCR NU este disponibil!")
            print(f"   Pentru a activa OCR, instalează:")
            print(f"      1. pip install pytesseract pillow")
            print(f"      2. Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
            print(f"\n   Poți continua fără OCR, dar rezultatele pot fi limitate.")
        else:
            print(f"\n   ✓ OCR disponibil (Tesseract)")
        
        proceed = input("\nRe-procesezi documentele? (da/nu) [da]: ").strip().lower()
        if proceed not in ['', 'da', 'd', 'yes', 'y']:
            print("Re-procesare anulată.")
            return
        
        # Construiește lista de PDF-uri
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
            print(f"\n⚠️  {len(missing)} fișiere nu mai există în folder")
        
        if not pdf_files:
            print(f"\n✗ Nu există fișiere de re-procesat.")
            return
        
        # Re-procesează documentele
        print(f"\n⏳ Re-procesare {len(pdf_files)} documente...")
        results, stats = self._process_batch(pdf_files)
        self._session_processed = [(r.company, r.year) for r in results]
        
        # Verifică câte au fost recuperate
        cursor.execute("""
            SELECT COUNT(*) FROM processed_documents 
            WHERE text_status = 'valid' AND source IN ({})
        """.format(','.join('?' * len([Path(p).name for p in pdf_files])), 
                   [Path(p).name for p in pdf_files]))
        
        # Afișează rezultatele
        print(f"\n{'═'*70}")
        print(f"✓ RE-PROCESARE COMPLETĂ!")
        print(f"{'═'*70}")
        print(f"   📄 {len(results)} documente re-procesate")
        print(f"   📊 {stats['new_refs']} referințe NOI găsite")
        print(f"   🔄 {stats['updated_refs']} referințe actualizate")
        
        # Verifică statusul final
        cursor.execute("""
            SELECT text_status, COUNT(*) FROM processed_documents 
            WHERE text_status IN ('corrupted_ocr_failed', 'ocr_needed', 'empty')
            GROUP BY text_status
        """)
        remaining = cursor.fetchall()
        if remaining:
            print(f"\n   📌 Documente încă problematice:")
            for status, count in remaining:
                print(f"      • {status}: {count}")
        else:
            print(f"\n   ✓ Toate documentele au acum text valid!")
        
        self._stats_cache = None

    # ═══════════════════════════════════════════════════════════════════════════
    # HANDLERS PENTRU OPȚIUNILE 2.x - ANALIZĂ & DEDUPLICARE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def handle_option_2_1(self):
        """2.1 - Analiză și deduplicare batch după poziții."""
        self.initialize_components()
        start, end = self.get_position_range()
        self._run_deduplication(position_range=(start, end))
    
    def handle_option_2_2(self):
        """2.2 - Analiză și deduplicare toate companiile."""
        self.initialize_components()
        self._run_deduplication(position_range=None)
    
    def handle_option_2_3(self):
        """2.3 - Analiză doar companiile noi (din această sesiune sau fără deduplicare)."""
        self.initialize_components()
        cursor = self.db_manager.conn.cursor()
        
        # Găsește perechile companie-an care nu au fost deduplicate încă
        cursor.execute("""
            SELECT DISTINCT company, year FROM ai_references_raw
            WHERE (company, year) NOT IN (
                SELECT DISTINCT company, year FROM ai_references_deduplicated
            )
        """)
        new_pairs = cursor.fetchall()
        
        if not new_pairs:
            print("\n✓ Nu există companii noi de analizat.")
            print("   Toate referințele au fost deja deduplicate.")
            return
        
        print(f"\n📊 {len(new_pairs)} perechi companie-an de analizat.")
        self._run_deduplication(company_year_pairs=new_pairs)
    
    def _run_deduplication(self, position_range: Optional[Tuple[int, int]] = None,
                          company_year_pairs: Optional[List[Tuple]] = None):
        """
        Rulează procesul de deduplicare semantică.
        
        Args:
            position_range: Tuple (start, end) pentru filtrare după poziție
            company_year_pairs: Listă de tuple (company, year) pentru procesare specifică
        """
        from tqdm import tqdm
        cursor = self.db_manager.conn.cursor()
        
        # Determină ce perechi să proceseze
        if company_year_pairs:
            pairs = company_year_pairs
        else:
            query = "SELECT DISTINCT company, year, position FROM ai_references_raw"
            if position_range:
                query += f" WHERE position >= {position_range[0]} AND position <= {position_range[1]}"
            cursor.execute(query)
            pairs = [(row[0], row[1]) for row in cursor.fetchall()]
        
        if not pairs:
            print("\n✓ Nu există date de procesat.")
            return
        
        print(f"\n⏳ Deduplicare pentru {len(pairs)} perechi companie-an...")
        
        total_original = 0
        total_dedup = 0
        
        for company, year in tqdm(pairs, desc="Deduplicare"):
            # Obține referințele raw pentru această companie-an
            cursor.execute("""
                SELECT * FROM ai_references_raw WHERE company = ? AND year = ?
            """, (company, year))
            rows = cursor.fetchall()
            
            if not rows:
                continue
            
            # Creează obiecte AIReference
            # Schema ai_references_raw (v6.1.2):
            # 0:id, 1:company, 2:year, 3:position, 4:industry, 5:sector, 6:country,
            # 7:doc_type, 8:page, 9:text, 10:context, 11:category, 12:sentiment,
            # 13:sentiment_score, 14:semantic_score, 15:detection_method, 16:source
            refs = []
            for row in rows:
                refs.append(AIReference(
                    company=row[1], year=row[2], position=row[3], industry=row[4],
                    sector=row[5] if len(row) > 5 else 'Unknown',
                    country=row[6] if len(row) > 6 else 'Unknown',
                    doc_type=row[7] if len(row) > 7 else 'Unknown',
                    page=row[8] if len(row) > 8 else 0,
                    text=row[9] if len(row) > 9 else '',
                    context=row[10] if len(row) > 10 else '',
                    category=row[11] if len(row) > 11 else 'General AI',
                    sentiment=row[12] if len(row) > 12 else 'neutral',
                    sentiment_score=row[13] if len(row) > 13 else 0.0,
                    semantic_score=row[14] if len(row) > 14 else 0.0,
                    detection_method=row[15] if len(row) > 15 else '',
                    source=row[16] if len(row) > 16 else ''
                ))
            
            total_original += len(refs)
            
            # Aplică deduplicarea semantică
            dedup_refs = self.deduplicator.deduplicate_references(refs)
            total_dedup += len(dedup_refs)
            
            # Salvează referințele deduplicate
            for dedup_ref in dedup_refs:
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO ai_references_deduplicated
                        (company, year, position, industry, sector, country, text, context, category,
                         sources, doc_count, total_occurrences, avg_sentiment_score,
                         avg_semantic_score, original_refs)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        dedup_ref['company'], dedup_ref['year'], dedup_ref['position'],
                        dedup_ref['industry'], dedup_ref.get('sector', 'Unknown'), 
                        dedup_ref.get('country', 'Unknown'),
                        dedup_ref['text'], dedup_ref['context'],
                        dedup_ref['category'], dedup_ref['sources'], dedup_ref['doc_count'],
                        dedup_ref['total_occurrences'], dedup_ref['avg_sentiment_score'],
                        dedup_ref['avg_semantic_score'], dedup_ref['original_refs']
                    ))
                except Exception as e:
                    logger.error(f"Eroare salvare dedup pentru {company}: {e}")
        
        self.db_manager.conn.commit()
        
        # Export automat
        self._export_deduplicated_references()
        
        # Afișează rezultatele
        reduction = (1 - total_dedup / total_original) * 100 if total_original > 0 else 0
        print(f"\n{'═'*70}")
        print(f"✓ DEDUPLICARE COMPLETĂ!")
        print(f"{'═'*70}")
        print(f"   📊 {total_original} referințe originale → {total_dedup} deduplicate")
        print(f"   📉 Reducere: {reduction:.1f}%")
        
        self._stats_cache = None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HANDLERS PENTRU OPȚIUNILE 3.x - CALCUL INDEX
    # ═══════════════════════════════════════════════════════════════════════════
    
    def handle_option_3_1(self):
        """3.1 - Calcul AI Adoption Index pentru batch după poziții."""
        self.initialize_components()
        start, end = self.get_position_range()
        self._calculate_indices(position_range=(start, end))
    
    def handle_option_3_2(self):
        """3.2 - Calcul AI Adoption Index pentru toate companiile."""
        self.initialize_components()
        self._calculate_indices(position_range=None)
    
    def handle_option_3_3(self):
        """3.3 - Calcul index doar pentru companiile noi."""
        self.initialize_components()
        cursor = self.db_manager.conn.cursor()
        
        # Găsește perechile care au referințe deduplicate dar nu au index calculat
        cursor.execute("""
            SELECT DISTINCT company, year FROM ai_references_deduplicated
            WHERE (company, year) NOT IN (
                SELECT DISTINCT company, year FROM adoption_index
            )
        """)
        new_pairs = cursor.fetchall()
        
        if not new_pairs:
            print("\n✓ Nu există companii noi pentru calcul index.")
            print("   Toate companiile cu referințe deduplicate au deja index calculat.")
            return
        
        print(f"\n📈 {len(new_pairs)} companii noi pentru calcul index.")
        self._calculate_indices(company_year_pairs=new_pairs)
    
    def _calculate_indices(self, position_range: Optional[Tuple[int, int]] = None,
                          company_year_pairs: Optional[List[Tuple]] = None):
        """
        Calculează AI Adoption Index pentru companiile specificate.
        
        Args:
            position_range: Tuple (start, end) pentru filtrare după poziție
            company_year_pairs: Listă de tuple (company, year) pentru procesare specifică
        """
        from tqdm import tqdm
        cursor = self.db_manager.conn.cursor()
        
        # Determină ce companii să proceseze
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
            print("\n✓ Nu există date pentru calcul index.")
            return
        
        print(f"\n⏳ Calcul index pentru {len(pairs)} companii...")
        
        for item in tqdm(pairs, desc="Calcul Index"):
            # Extrage informațiile
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
            
            # Obține referințele deduplicate
            cursor.execute("""
                SELECT * FROM ai_references_deduplicated WHERE company = ? AND year = ?
            """, (company, year))
            rows = cursor.fetchall()
            
            if not rows:
                continue
            
            # Extrage sector și country din primul row (coloanele 5 și 6 în schema nouă)
            first_row = rows[0]
            sector = first_row[5] if len(first_row) > 5 and first_row[5] else 'Unknown'
            country = first_row[6] if len(first_row) > 6 and first_row[6] else 'Unknown'
            
            # Creează DocumentResult pentru calculator
            doc_result = DocumentResult(
                company=company, year=year, position=position, industry=industry,
                sector=sector,
                country=country,
                doc_type='Mixed', source='aggregated',
                total_pages=sum(row[11] if len(row) > 11 else 0 for row in rows),  # doc_count
                text_length=0
            )
            
            # Adaugă referințele
            for row in rows:
                # Schema ai_references_deduplicated: id(0), company(1), year(2), position(3), 
                # industry(4), sector(5), country(6), text(7), context(8), category(9),
                # sources(10), doc_count(11), total_occurrences(12), avg_sentiment(13), avg_semantic(14)
                row_sector = row[5] if len(row) > 5 and row[5] else 'Unknown'
                row_country = row[6] if len(row) > 6 and row[6] else 'Unknown'
                ref = AIReference(
                    company=row[1], year=row[2], position=row[3], industry=row[4],
                    sector=row_sector, country=row_country,
                    doc_type='Mixed', page=1, 
                    text=row[7] if len(row) > 7 else '', 
                    context=row[8] if len(row) > 8 else '',
                    category=row[9] if len(row) > 9 else 'General AI',
                    sentiment='neutral', 
                    sentiment_score=row[13] if len(row) > 13 else 0.0,
                    semantic_score=row[14] if len(row) > 14 else 0.0, 
                    detection_method='aggregated', 
                    source=row[10] if len(row) > 10 else ''
                )
                doc_result.add_reference(ref)
            
            # Calculează indexul
            adoption_index = self.index_calculator.calculate_index(doc_result)
            
            # Salvează în baza de date
            cursor.execute("""
                INSERT OR REPLACE INTO adoption_index
                (company, year, position, industry, sector, country, intensity_index, semantic_index,
                 diversity_index, sentiment_index, maturity_index, future_index,
                 commitment_index, ai_adoption_index, total_refs, total_pages, categories_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                adoption_index.total_pages, adoption_index.categories_used
            ))
        
        self.db_manager.conn.commit()
        
        # Actualizează rankings
        self._update_rankings()
        
        # Export automat
        self._export_adoption_index()
        
        print(f"\n{'═'*70}")
        print(f"✓ CALCUL INDEX COMPLET!")
        print(f"{'═'*70}")
        
        self._stats_cache = None
    
    def _update_rankings(self):
        """Actualizează rankings în baza de date pentru fiecare an."""
        cursor = self.db_manager.conn.cursor()
        
        # Obține anii disponibili
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
        logger.info("Rankings actualizate pentru toate anii")

# ═══════════════════════════════════════════════════════════════════════════════
# SFÂRȘIT PARTEA 2/3 - Continuă în partea 3
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# AI SEMANTIC ANALYZER v6.0.6 - PARTEA 3/3
# Continuare clasa MenuManager - Handlers 4.x, 5.x, 6.x, 7 și main loop
# ACEST COD SE ADAUGĂ ÎN CLASA MenuManager DUPĂ PARTEA 2
# ═══════════════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════════
    # HANDLERS PENTRU OPȚIUNILE 4.x - RAPOARTE & EXPORT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def handle_option_4_1(self):
        """4.1 - Export referințe RAW."""
        self.initialize_components()
        self._export_raw_references()
        print("\n✓ Export referințe RAW complet!")
    
    def handle_option_4_2(self):
        """4.2 - Export referințe deduplicate."""
        self.initialize_components()
        self._export_deduplicated_references()
        print("\n✓ Export referințe deduplicate complet!")
    
    def handle_option_4_3(self):
        """4.3 - Export AI Adoption Index."""
        self.initialize_components()
        self._export_adoption_index()
        print("\n✓ Export AI Adoption Index complet!")
    
    def handle_option_4_4(self):
        """4.4 - Export JSON complet."""
        self.initialize_components()
        self._export_json()
        print("\n✓ Export JSON complet!")
    
    def handle_option_4_5(self):
        """4.5 - Generare raport TEXT sumar."""
        self.initialize_components()
        self._export_text_report()
        print("\n✓ Raport TEXT generat!")
    
    def handle_option_4_6(self):
        """4.6 - Generare TOATE rapoartele."""
        self.initialize_components()
        print("\n⏳ Generare toate rapoartele...")
        self._export_raw_references()
        self._export_deduplicated_references()
        self._export_adoption_index()
        self._export_json()
        self._export_text_report()
        print("\n✓ Toate rapoartele generate!")
    
    def _export_raw_references(self):
        """Exportă referințele raw în Excel."""
        cursor = self.db_manager.conn.cursor()
        cursor.execute("SELECT * FROM ai_references_raw")
        rows = cursor.fetchall()
        
        if not rows:
            print("  ⚠ Nu există referințe raw pentru export.")
            return
        
        # Schema ai_references_raw (v6.1.1):
        # 0:id, 1:company, 2:year, 3:position, 4:industry, 5:sector, 6:country,
        # 7:doc_type, 8:page, 9:text, 10:context, 11:category, 12:sentiment,
        # 13:sentiment_score, 14:semantic_score, 15:detection_method, 16:source,
        # 17:robotics_type, 18:rpa_type, 19:sentiment_confidence, 20:category_confidence,
        # 21:reference_strength, 22:confidence_score, 23:confidence_reasons, 24:created_at
        refs = []
        for row in rows:
            refs.append({
                'company': row[1] if len(row) > 1 else '',
                'year': row[2] if len(row) > 2 else 0,
                'position': row[3] if len(row) > 3 else 0,
                'industry': row[4] if len(row) > 4 else '',
                'sector': row[5] if len(row) > 5 else '',
                'country': row[6] if len(row) > 6 else '',
                'doc_type': row[7] if len(row) > 7 else '',
                'page': row[8] if len(row) > 8 else 0,
                'text': row[9] if len(row) > 9 else '',
                'context': row[10] if len(row) > 10 else '',
                'category': row[11] if len(row) > 11 else '',
                'sentiment': row[12] if len(row) > 12 else '',
                'sentiment_score': row[13] if len(row) > 13 else 0,
                'semantic_score': row[14] if len(row) > 14 else 0,
                'detection_method': row[15] if len(row) > 15 else '',
                'source': row[16] if len(row) > 16 else '',
                'robotics_type': row[17] if len(row) > 17 else '',
                'rpa_type': row[18] if len(row) > 18 else '',
                # Câmpuri noi v6.1.1
                'reference_strength': row[21] if len(row) > 21 else 'unknown',
                'confidence_score': row[22] if len(row) > 22 else 0.0,
                'confidence_reasons': row[23] if len(row) > 23 else '',
                # Pentru compatibilitate cu export_references()
                'avg_sentiment_score': row[13] if len(row) > 13 else 0,
                'avg_semantic_score': row[14] if len(row) > 14 else 0,
                'doc_count': 1,
                'total_occurrences': 1
            })
        
        self.excel_exporter.export_references(refs, "ai_references_raw_v6.xlsx")
        print(f"  ✓ Exportat {len(refs)} referințe raw")
    
    def _export_deduplicated_references(self):
        """Exportă referințele deduplicate în Excel."""
        cursor = self.db_manager.conn.cursor()
        cursor.execute("SELECT * FROM ai_references_deduplicated")
        rows = cursor.fetchall()
        
        if not rows:
            print("  ⚠ Nu există referințe deduplicate pentru export.")
            return
        
        # Schema ai_references_deduplicated (v6.0.6):
        # 0:id, 1:company, 2:year, 3:position, 4:industry, 5:sector, 6:country,
        # 7:text, 8:context, 9:category, 10:sources, 11:doc_count, 
        # 12:total_occurrences, 13:avg_sentiment_score, 14:avg_semantic_score, 15:original_refs
        refs = []
        for row in rows:
            refs.append({
                'company': row[1] if len(row) > 1 else '',
                'year': row[2] if len(row) > 2 else 0,
                'position': row[3] if len(row) > 3 else 0,
                'industry': row[4] if len(row) > 4 else '',
                'sector': row[5] if len(row) > 5 else '',
                'country': row[6] if len(row) > 6 else '',
                'text': row[7] if len(row) > 7 else '',
                'context': row[8] if len(row) > 8 else '',
                'category': row[9] if len(row) > 9 else '',
                'sources': row[10] if len(row) > 10 else '',
                'source': row[10] if len(row) > 10 else '',  # Alias pentru export
                'doc_count': row[11] if len(row) > 11 else 0,
                'total_occurrences': row[12] if len(row) > 12 else 0,
                'avg_sentiment_score': row[13] if len(row) > 13 else 0,
                'avg_semantic_score': row[14] if len(row) > 14 else 0,
                # Placeholder pentru câmpuri care nu există în dedup (dar sunt în header Excel)
                'reference_strength': 'aggregated',
                'confidence_score': 0.0,
                'confidence_reasons': 'Deduplicated reference'
            })
        
        self.excel_exporter.export_references(refs, "ai_references_deduplicated_v6.xlsx")
        print(f"  ✓ Exportat {len(refs)} referințe deduplicate")
    
    def _export_adoption_index(self):
        """Exportă AI Adoption Index în Excel."""
        cursor = self.db_manager.conn.cursor()
        cursor.execute("""
            SELECT * FROM adoption_index 
            ORDER BY year DESC, ai_adoption_index DESC
        """)
        rows = cursor.fetchall()
        
        if not rows:
            print("  ⚠ Nu există indici pentru export.")
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
                categories_used=row[18] if len(row) > 18 else 0
            ))
        
        self.excel_exporter.export_adoption_index(indices)
        print(f"  ✓ Exportat index pentru {len(indices)} companii")
    
    def _export_json(self):
        """Exportă toate datele în format JSON."""
        cursor = self.db_manager.conn.cursor()
        
        data = {
            'metadata': {
                'version': '6.0.4',
                'generated_at': datetime.now().isoformat(),
                'categories': list(AI_CATEGORIES.keys()),
                'semantic_threshold': self.config.semantic_threshold
            },
            'companies': [],
            'industries': []
        }
        
        # Export companii
        cursor.execute("SELECT * FROM adoption_index")
        for row in cursor.fetchall():
            data['companies'].append({
                'company': row[1], 'year': row[2], 'position': row[3],
                'industry': row[4], 'ai_adoption_index': row[12],
                'dimensions': {
                    'intensity': row[5], 'semantic': row[6], 'diversity': row[7],
                    'sentiment': row[8], 'maturity': row[9], 'future': row[10],
                    'commitment': row[11]
                },
                'total_refs': row[14], 'categories_used': row[16]
            })
        
        # Export industrii (dacă există)
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
        
        output_path = Path(self.config.output_folder) / "results_v6.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"  ✓ Export JSON: {output_path}")
    
    def _export_text_report(self):
        """Generează raport text sumar."""
        cursor = self.db_manager.conn.cursor()
        output_path = Path(self.config.output_folder) / "report_v6.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("AI ADOPTION ANALYSIS REPORT v6.0.6\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
            
            # Features
            f.write("FEATURES v6.0.6:\n")
            f.write(f"  - 13 AI Categories\n")
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
                    SELECT category, COUNT(*) as cnt 
                    FROM ai_references_deduplicated 
                    GROUP BY category ORDER BY cnt DESC
                """)
                for cat, cnt in cursor.fetchall():
                    f.write(f"  {cat}: {cnt}\n")
            except:
                pass
        
        print(f"  ✓ Raport generat: {output_path}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HANDLERS PENTRU OPȚIUNILE 5.x - GRAFICE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _get_visualization_data(self):
        """Obține datele necesare pentru vizualizări."""
        cursor = self.db_manager.conn.cursor()
        
        # Indici
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
                categories_used=row[18] if len(row) > 18 else 0
            ))
        
        # Referințe pentru categorii
        cursor.execute("SELECT company, year, category FROM ai_references_deduplicated")
        refs = [{'company': row[0], 'year': row[1], 'category': row[2]} 
                for row in cursor.fetchall()]
        
        return indices, refs
    
    def handle_option_5_1(self):
        """5.1 - Grafic ranking companii."""
        self.initialize_components()
        indices, refs = self._get_visualization_data()
        if indices:
            self.viz_generator.generate_ranking_chart(indices)
            print("\n✓ Grafic ranking generat!")
        else:
            print("\n⚠ Nu există date pentru grafic.")
    
    def handle_option_5_2(self):
        """5.2 - Grafic evoluție top companii."""
        self.initialize_components()
        indices, refs = self._get_visualization_data()
        if indices:
            self.viz_generator.generate_timeline_chart(indices)
            print("\n✓ Grafic timeline generat!")
        else:
            print("\n⚠ Nu există date pentru grafic.")
    
    def handle_option_5_3(self):
        """5.3 - Grafic radar dimensiuni."""
        self.initialize_components()
        indices, refs = self._get_visualization_data()
        if indices:
            self.viz_generator.generate_radar_chart(indices)
            print("\n✓ Grafic radar generat!")
        else:
            print("\n⚠ Nu există date pentru grafic.")
    
    def handle_option_5_4(self):
        """5.4 - Heatmap companii x ani."""
        self.initialize_components()
        indices, refs = self._get_visualization_data()
        if indices:
            self.viz_generator.generate_heatmap(indices)
            print("\n✓ Heatmap generat!")
        else:
            print("\n⚠ Nu există date pentru grafic.")
    
    def handle_option_5_5(self):
        """5.5 - Distribuție categorii AI."""
        self.initialize_components()
        indices, refs = self._get_visualization_data()
        if refs:
            self.viz_generator.generate_category_distribution(refs)
            print("\n✓ Grafic categorii generat!")
        else:
            print("\n⚠ Nu există date pentru grafic.")
    
    def handle_option_5_6(self):
        """5.6 - Toate graficele."""
        self.initialize_components()
        indices, refs = self._get_visualization_data()
        if indices:
            print("\n⏳ Generare toate graficele...")
            self.viz_generator.generate_all_visualizations(indices, refs)
            print("\n✓ Toate graficele generate!")
        else:
            print("\n⚠ Nu există date pentru grafice.")
    
    def handle_option_5_7(self):
        """5.7 - Grafice industrie."""
        self.initialize_components()
        cursor = self.db_manager.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM adoption_index_industry")
        
        if cursor.fetchone()[0] == 0:
            print("\n⚠ Nu există date agregate la nivel de industrie.")
            print("   Rulează mai întâi agregarea (opțiunea 7).")
            return
        
        indices, refs = self._get_visualization_data()
        self.viz_generator.generate_industry_charts(indices)
        print("\n✓ Grafice industrie generate!")
    
    def handle_option_5_8(self):
        """5.8 - Grafice sector."""
        self.initialize_components()
        indices, refs = self._get_visualization_data()
        if indices:
            self.viz_generator.generate_group_charts(indices, group_by='sector')
            print("\n✓ Grafice sector generate!")
        else:
            print("\n⚠ Nu există date pentru grafic.")
    
    def handle_option_5_9(self):
        """5.9 - Grafice country."""
        self.initialize_components()
        indices, refs = self._get_visualization_data()
        if indices:
            self.viz_generator.generate_group_charts(indices, group_by='country')
            print("\n✓ Grafice country generate!")
        else:
            print("\n⚠ Nu există date pentru grafic.")    


    # ═══════════════════════════════════════════════════════════════════════════
    # HANDLERS PENTRU OPȚIUNILE 6.x - UTILITĂȚI
    # ═══════════════════════════════════════════════════════════════════════════
    
    def handle_option_6_1(self):
        """6.1 - Statistici și sumar detaliat."""
        self.initialize_components()
        stats = self.get_statistics(force_refresh=True)
        
        print(f"\n{'═'*70}")
        print("📊 STATISTICI DETALIATE v6.0.6")
        print(f"{'═'*70}")
        
        print(f"\n📁 DOCUMENTE PDF:")
        print(f"   Total în folder: {stats['total_pdfs']}")
        print(f"   Procesate: {stats['processed']}")
        print(f"   Noi (neprocesate): {stats['new']}")
        print(f"   Chinezești (ignorate): {stats['chinese']}")
        if stats['reprocessed_docs'] > 0:
            print(f"   Re-analizate: {stats['reprocessed_docs']}")
        
        print(f"\n🔍 REFERINȚE AI:")
        print(f"   Unice (raw): {stats['total_refs_raw']:,}")
        print(f"   Total apariții: {stats['total_occurrences']:,}")
        print(f"   Deduplicate: {stats['total_refs_dedup']:,}")
        if stats['total_refs_raw'] > 0:
            ratio = stats['total_refs_dedup'] / stats['total_refs_raw'] * 100
            print(f"   Ratio deduplicare: {ratio:.1f}%")
            if stats['total_occurrences'] > stats['total_refs_raw']:
                avg_occ = stats['total_occurrences'] / stats['total_refs_raw']
                print(f"   Media apariții/referință: {avg_occ:.2f}")
        
        print(f"\n📈 AI ADOPTION INDEX:")
        print(f"   Companii cu index: {stats['companies_with_index']}")
        if stats['by_year']:
            print(f"\n   Per an:")
            for year, count in sorted(stats['by_year'].items()):
                print(f"      {year}: {count} companii")
        
        if stats['by_industry']:
            print(f"\n   Per industrie (top 10):")
            for industry, count in list(stats['by_industry'].items())[:10]:
                print(f"      {industry[:30]}: {count} companii")
        
        # Statistici detaliate din processed_documents
        try:
            proc_stats = self.db_manager.get_processing_stats()
            print(f"\n🔄 STATISTICI PROCESARE:")
            print(f"   Total rulări procesare: {proc_stats['total_processing_runs']}")
            print(f"   Documente fără referințe: {proc_stats['documents_without_refs']}")
        except:
            pass
        
        print(f"\n⚙️ CONFIGURAȚIE:")
        print(f"   Semantic threshold: {self.config.semantic_threshold}")
        print(f"   Categorii AI: {len(AI_CATEGORIES)}")
        print(f"   Input folder: {self.config.input_folder}")
        print(f"   Output folder: {self.config.output_folder}")
        print(f"{'═'*70}")
    
    def handle_option_6_2(self):
        """6.2 - Verificare integritate bază de date."""
        self.initialize_components()
        print("\n⏳ Verificare integritate bază de date...")
        
        cursor = self.db_manager.conn.cursor()
        issues = []
        
        # Verificare tabelă processed_documents
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='processed_documents'
        """)
        if not cursor.fetchone():
            issues.append("⚠ Tabela processed_documents lipsește")
        
        # Verificare referințe fără deduplicare
        cursor.execute("""
            SELECT COUNT(DISTINCT company || '-' || year) FROM ai_references_raw 
            WHERE (company, year) NOT IN (
                SELECT company, year FROM ai_references_deduplicated
            )
        """)
        count = cursor.fetchone()[0]
        if count > 0:
            issues.append(f"⚠ {count} perechi companie-an fără deduplicare")
        
        # Verificare deduplicate fără index
        cursor.execute("""
            SELECT COUNT(DISTINCT company || '-' || year) FROM ai_references_deduplicated 
            WHERE (company, year) NOT IN (
                SELECT company, year FROM adoption_index
            )
        """)
        count = cursor.fetchone()[0]
        if count > 0:
            issues.append(f"⚠ {count} perechi companie-an fără index calculat")
        
        # Verificare industrii Unknown
        cursor.execute("""
            SELECT COUNT(*) FROM adoption_index 
            WHERE industry = 'Unknown' OR industry IS NULL
        """)
        count = cursor.fetchone()[0]
        if count > 0:
            issues.append(f"⚠ {count} companii fără industrie (Unknown)")
        
        # Verificare sincronizare processed_documents vs ai_references_raw
        try:
            cursor.execute("""
                SELECT COUNT(DISTINCT source) FROM ai_references_raw 
                WHERE source NOT IN (SELECT source FROM processed_documents)
            """)
            count = cursor.fetchone()[0]
            if count > 0:
                issues.append(f"⚠ {count} surse în references dar nu în processed_documents")
        except:
            pass
        
        # Verificare integritate SQLite
        cursor.execute("PRAGMA integrity_check")
        integrity = cursor.fetchone()[0]
        if integrity != 'ok':
            issues.append(f"✗ Problemă integritate SQLite: {integrity}")
        
        # Afișare rezultate
        if issues:
            print(f"\n🔍 Probleme găsite ({len(issues)}):")
            for issue in issues:
                print(f"   {issue}")
        else:
            print("\n✓ Baza de date este integră!")
        
        # Afișare tabele și dimensiuni
        print(f"\n📊 Tabele în baza de date:")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for row in cursor.fetchall():
            table_name = row[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"   {table_name}: {count:,} înregistrări")
    
    def handle_option_6_3(self):
        """6.3 - Configurare căi și setări."""
        print(f"\n{'═'*70}")
        print("🔧 CONFIGURARE")
        print(f"{'═'*70}")
        
        saved_config = load_saved_config()
        if saved_config:
            print("\nConfigurație curentă:")
            print(f"   Input folder: {saved_config.get('input_folder', 'N/A')}")
            print(f"   Output folder: {saved_config.get('output_folder', 'N/A')}")
            print(f"   Fortune 500 CSV: {saved_config.get('fortune500_csv', 'N/A')}")
            print(f"   Database: {saved_config.get('database_name', 'N/A')}")
            print(f"   Semantic threshold: {saved_config.get('semantic_threshold', 'N/A')}")
        
        choice = input("\n1. Modifică configurația | 2. Păstrează [2]: ").strip()
        if choice == '1':
            self.config = AnalyzerConfig.from_interactive()
            self.db_manager = None
            self._stats_cache = None
            print("\n✓ Configurație actualizată!")
    
    def handle_option_6_4(self):
        """6.4 - Import mapping industrii din CSV."""
        self.initialize_components()
        
        print(f"\n{'═'*70}")
        print("📥 IMPORT MAPPING INDUSTRII")
        print(f"{'═'*70}")
        print("\nFormatul CSV așteptat:")
        print("   Coloană 1: Company Name")
        print("   Coloană 2: Industry/Sector")
        
        csv_path = input("\nCale către CSV: ").strip()
        if not csv_path or not Path(csv_path).exists():
            print("  ✗ Fișier invalid sau inexistent.")
            return
        
        try:
            df = pd.read_csv(csv_path)
            
            # Detectează coloanele
            company_col = None
            industry_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'company' in col_lower or 'name' in col_lower:
                    company_col = col
                elif 'industry' in col_lower or 'sector' in col_lower:
                    industry_col = col
            
            if not company_col or not industry_col:
                print("  ✗ Nu s-au găsit coloanele necesare (company, industry).")
                return
            
            cursor = self.db_manager.conn.cursor()
            updated = 0
            
            for _, row in df.iterrows():
                company = str(row[company_col]).strip()
                industry = str(row[industry_col]).strip()
                
                if company and industry and industry != 'nan':
                    # Update în toate tabelele
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
            print(f"\n✓ Actualizate {updated} înregistrări cu industrii noi.")
            self._stats_cache = None
            
        except Exception as e:
            print(f"  ✗ Eroare import: {e}")
    
    def handle_option_6_5(self):
        """6.5 - Informații categorii AI."""
        print(f"\n{'═'*80}")
        print("📋 CATEGORII AI v6.0 - 13 CATEGORII")
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
        """
        Generează matricea de acoperire documente.
        Arată ce documente există pentru fiecare companie × an × tip document.
        """
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        
        # 1. Încarcă lista companiilor din Fortune 500
        if not self.fortune500_data:
            print("  ⚠ Nu există date Fortune 500. Încarcă mai întâi CSV-ul.")
            return
        
        # 2. Scanează toate PDF-urile
        pdf_folder = Path(self.config.input_folder)
        all_pdfs = list(pdf_folder.glob("*.pdf"))
        
        if not all_pdfs:
            print("  ⚠ Nu există PDF-uri în folder.")
            return
        
        # 3. Parsează metadata din fiecare PDF
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
        
        # Sortează
        doc_types_sorted = sorted(doc_types_found)
        years_sorted = sorted(years_found)
        
        # 4. Construiește structura de date
        # Key: (position, company) -> {(doc_type, year): 1}
        coverage = defaultdict(dict)
        
        for doc in doc_data:
            key = (doc['position'], doc['company'])
            coverage[key][(doc['doc_type'], doc['year'])] = 1
        
        # 5. Creează Excel
        wb = Workbook()
        ws = wb.active
        ws.title = "Document Coverage"
        
        # Stiluri
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
        
        # Date - sortate după poziție
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
        
        # Rând cu totaluri
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
        
        # Ajustare lățimi coloane
        ws.column_dimensions['A'].width = 10
        ws.column_dimensions['B'].width = 30
        for i in range(3, len(headers) + 1):
            ws.column_dimensions[chr(64 + i) if i <= 26 else f"{chr(64 + (i-1)//26)}{chr(65 + (i-1)%26)}"].width = 12
        
        # Freeze
        ws.freeze_panes = 'C2'
        
        # Salvare
        output_path = Path(self.config.output_folder) / "document_coverage_matrix.xlsx"
        wb.save(output_path)
        
        print(f"\n{'═'*70}")
        print(f"✓ MATRICE ACOPERIRE GENERATĂ!")
        print(f"{'═'*70}")
        print(f"   Companii: {len(companies_sorted)}")
        print(f"   Tipuri documente: {len(doc_types_sorted)} ({', '.join(doc_types_sorted)})")
        print(f"   Ani: {len(years_sorted)} ({', '.join(map(str, years_sorted))})")
        print(f"   Total PDF-uri: {len(all_pdfs)}")
        print(f"\n   Output: {output_path}")
        print(f"{'═'*70}")


    # ═══════════════════════════════════════════════════════════════════════════
    # HANDLER PENTRU OPȚIUNEA 7 - PIPELINE COMPLET
    # ═══════════════════════════════════════════════════════════════════════════
    
    def handle_option_7(self):
        """7 - Pipeline complet automat pentru documente NOI."""
        self.initialize_components()
        
        print(f"\n{'═'*70}")
        print("🚀 PIPELINE COMPLET v6.0.6")
        print(f"{'═'*70}")
        
        stats = self.get_statistics(force_refresh=True)
        
        if stats['new'] == 0:
            print("\n✓ Nu există documente NOI de procesat.")
            choice = input("Dorești să rulezi pipeline-ul complet cu re-analiză? (da/nu) [nu]: ").strip().lower()
            if choice not in ['da', 'd', 'yes', 'y']:
                return
            process_all = True
        else:
            print(f"\n📄 Se vor procesa {stats['new']} documente NOI.")
            proceed = input("Continuăm cu pipeline-ul complet? (da/nu) [da]: ").strip().lower()
            if proceed not in ['', 'da', 'd', 'yes', 'y']:
                return
            process_all = False
        
        import time
        pipeline_start = time.time()
        
        # ETAPA 1: Extragere
        print(f"\n{'-'*70}")
        print("📥 ETAPA 1/4: EXTRAGERE DOCUMENTE")
        print(f"{'-'*70}")
        
        if process_all:
            pdf_files = self.get_all_pdfs(skip_processed=False, skip_chinese=True)
        else:
            pdf_files = self.get_all_pdfs(skip_processed=True, skip_chinese=True)
        
        if pdf_files:
            results, proc_stats = self._process_batch(pdf_files)
            self._session_processed = [(r.company, r.year) for r in results]
            print(f"  ✓ Procesate {len(results)} documente")
            print(f"    📊 {proc_stats['new_refs']} ref. noi, {proc_stats['updated_refs']} actualizate")
        else:
            print("  ✓ Niciun document de procesat")
        
        # ETAPA 2: Deduplicare
        print(f"\n{'-'*70}")
        print("📊 ETAPA 2/4: ANALIZĂ & DEDUPLICARE")
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
            print("  ✓ Nicio deduplicare necesară")
        
        # ETAPA 3: Calcul Index
        print(f"\n{'-'*70}")
        print("📈 ETAPA 3/4: CALCUL AI ADOPTION INDEX")
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
            print("  ✓ Niciun index nou de calculat")
        
        # ETAPA 4: Generare Output-uri
        print(f"\n{'-'*70}")
        print("📋 ETAPA 4/4: GENERARE RAPOARTE & GRAFICE")
        print(f"{'-'*70}")
        
        print("  Generare rapoarte Excel...")
        self._export_raw_references()
        self._export_deduplicated_references()
        self._export_adoption_index()
        
        print("  Generare JSON...")
        self._export_json()
        
        print("  Generare raport text...")
        self._export_text_report()
        
        print("  Generare grafice...")
        indices, refs = self._get_visualization_data()
        if indices:
            self.viz_generator.generate_all_visualizations(indices, refs)
        
        print("  Agregare date industrie...")
        self.industry_aggregator.aggregate_by_industry()
        
        pipeline_time = time.time() - pipeline_start
        
        # Rezultate finale
        print(f"\n{'═'*70}")
        print("🎉 PIPELINE COMPLET FINALIZAT!")
        print(f"{'═'*70}")
        print(f"   ⏱️  Timp total: {pipeline_time/60:.1f} minute")
        
        stats = self.get_statistics(force_refresh=True)
        print(f"\n📊 REZULTATE FINALE:")
        print(f"   Documente procesate: {stats['processed']}")
        print(f"   Referințe unice: {stats['total_refs_raw']:,}")
        print(f"   Referințe deduplicate: {stats['total_refs_dedup']:,}")
        print(f"   Companii cu index: {stats['companies_with_index']}")
        
        print(f"\n📁 OUTPUT-URI în {self.config.output_folder}:")
        print("   • ai_references_raw_v6.xlsx")
        print("   • ai_references_deduplicated_v6.xlsx")
        print("   • ai_adoption_index_v6.xlsx")
        print("   • results_v6.json")
        print("   • report_v6.txt")
        print("   • viz_*.html (grafice interactive)")
        
        self._stats_cache = None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN MENU LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    
    def run(self):
        """Rulează meniul interactiv principal."""
        
        # Mapping opțiuni la handlers
        handlers = {
            # 1.x - Extragere
            '1.1': self.handle_option_1_1,
            '1.2': self.handle_option_1_2,
            '1.3': self.handle_option_1_3,
            '1.4': self.handle_option_1_4,
            '1.5': self.handle_option_1_5,
            '1.6': self.handle_option_1_6,
                
            # 2.x - Analiză
            '2.1': self.handle_option_2_1,
            '2.2': self.handle_option_2_2,
            '2.3': self.handle_option_2_3,
            
            # 3.x - Calcul Index
            '3.1': self.handle_option_3_1,
            '3.2': self.handle_option_3_2,
            '3.3': self.handle_option_3_3,
            
            # 4.x - Rapoarte
            '4.1': self.handle_option_4_1,
            '4.2': self.handle_option_4_2,
            '4.3': self.handle_option_4_3,
            '4.4': self.handle_option_4_4,
            '4.5': self.handle_option_4_5,
            '4.6': self.handle_option_4_6,
            
            # 5.x - Grafice
            '5.1': self.handle_option_5_1,
            '5.2': self.handle_option_5_2,
            '5.3': self.handle_option_5_3,
            '5.4': self.handle_option_5_4,
            '5.5': self.handle_option_5_5,
            '5.6': self.handle_option_5_6,
            '5.7': self.handle_option_5_7,
            '5.8': self.handle_option_5_8,
            '5.9': self.handle_option_5_9,
            
            # 6.x - Utilități
            '6.1': self.handle_option_6_1,
            '6.2': self.handle_option_6_2,
            '6.3': self.handle_option_6_3,
            '6.4': self.handle_option_6_4,
            '6.5': self.handle_option_6_5,
            '6.6': self.handle_option_6_6,
            
            # 7 - Pipeline complet
            '7': self.handle_option_7,
        }
        
        while True:
            try:
                self.clear_screen()
                self.display_main_menu()
                
                choice = input("\n🔹 Selectează opțiunea: ").strip()
                
                if choice == '0':
                    print("\n👋 La revedere!")
                    break
                elif choice in handlers:
                    handlers[choice]()
                    input("\n⏎ Apasă Enter pentru a continua...")
                else:
                    print(f"\n✗ Opțiune invalidă: '{choice}'")
                    input("⏎ Apasă Enter pentru a continua...")
                    
            except KeyboardInterrupt:
                print("\n\n⚠ Întrerupt de utilizator.")
                break
            except Exception as e:
                logger.error(f"Eroare în meniu: {e}")
                print(f"\n✗ Eroare: {e}")
                input("⏎ Apasă Enter pentru a continua...")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Punct de intrare principal."""
    print(f"\n{'═'*80}")
    print("       AI SEMANTIC ANALYZER v6.0.6 - FORTUNE 500 AI ADOPTION ANALYSIS")
    print(f"{'═'*80}")
    print("\n  Features:")
    print("    • 13 categorii AI cu 80+ patterns")
    print("    • Semantic analysis cu SentenceTransformer")
    print("    • FinBERT sentiment analysis")
    print("    • 7-dimensional AI Adoption Index")
    print("    • Deduplicare semantică avansată")
    print("    • Export Excel, JSON, vizualizări Plotly")
    print("    • Filtrare robotics tradițional și RPA")
    print("    • Tracking complet documente procesate")
    print("    • Re-analiză cu incrementare occurrence count")
    print(f"\n{'═'*80}")
    
    menu = MenuManager()
    menu.run()


if __name__ == "__main__":
    main()