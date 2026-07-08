"""
═══════════════════════════════════════════════════════════════════════════════
AI SEMANTIC ANALYZER v6.1.1 - MAIN SCRIPT
═══════════════════════════════════════════════════════════════════════════════

Autor: TeRa0
Versiune: 6.1.1
Data: Ianuarie 2026

Descriere:
    Sistem avansat de analiză semantică pentru identificarea și cuantificarea
    adoptării AI în rapoartele anuale ale companiilor Fortune 500.
    
    Proiectat pentru cercetare doctorală în domeniul adoptării AI în 
    companiile Fortune 500, cu prezentare la conferințe și publicare 
    în jurnale academice.

Rulare:
    python ai_analyzer_v6_main.py

Module necesare:
    - ai_analyzer_v6_0_6_module1.py (Core & Config)
    - ai_analyzer_v6_0_6_module2.py (PDF Processing & Detection)
    - ai_analyzer_v6_0_6_module3.py (Advanced Analysis & Deduplication)
    - ai_analyzer_v6_0_6_module4.py (Export & Visualization)
    - ai_analyzer_v6_0_6_menu.py (Meniu Interactiv)

═══════════════════════════════════════════════════════════════════════════════
FEATURES PRINCIPALE v6.0.6:
═══════════════════════════════════════════════════════════════════════════════

📊 ANALIZĂ AI:
    • 13 categorii AI specializate (ML, GenAI/LLM, NLP, Computer Vision, etc.)
    • 80+ termeni și patterns GenAI 2025 (ChatGPT, GPT-4, Copilot, Claude, etc.)
    • Semantic analysis cu SentenceTransformer (all-MiniLM-L6-v2)
    • FinBERT sentiment analysis pentru context financiar
    • Threshold semantic 0.60 pentru precizie ridicată
    • ~60 false positive patterns pentru filtrare îmbunătățită
    • Context validators pentru termeni scurți (AI, ML, DL)

📈 AI ADOPTION INDEX (7 dimensiuni):
    • Intensity Index - densitatea referințelor AI per 100 pagini
    • Semantic Index - relevanța semantică medie
    • Diversity Index - diversitatea categoriilor AI (din 13)
    • Sentiment Index - tonul comunicării despre AI
    • Maturity Index - nivelul de implementare (exploring→embedded)
    • Future Index - orientarea strategică 2025-2028
    • Commitment Index - angajamentul organizațional

🏢 CLASIFICARE ORGANIZAȚIONALĂ (NOU v6.0.6):
    • Industry - clasificare la nivel de industrie
    • Sector - clasificare la nivel de sector
    • Country - țara de origine a companiei
    • Agregări și vizualizări per Industry/Sector/Country

🔍 FILTRARE INTELIGENTĂ:
    • Robotics tradițional vs AI-powered robotics
    • RPA tradițional vs Intelligent Automation
    • Excludere referințe non-AI (industrial robots, assembly line, etc.)
    • Validare context pentru acronime scurte

📁 TRACKING DOCUMENTE:
    • Tabelă separată processed_documents pentru tracking complet
    • Detectare automată documente noi vs deja procesate
    • Suport re-analiză cu incrementare occurrence_count
    • Statistici separate: documente noi vs re-analizate
    • Text status tracking (valid/corrupted/ocr_needed)

🔧 OCR & TEXT RECOVERY (NOU v6.0.5+):
    • Detectare automată text corupt (encoding issues)
    • OCR fallback cu pytesseract pentru documente problematice
    • Re-procesare selectivă documente cu text corupt
    • Status tracking: valid, corrupted_ocr_success, corrupted_ocr_failed

═══════════════════════════════════════════════════════════════════════════════
CHANGELOG:
═══════════════════════════════════════════════════════════════════════════════

v6.0.6 (Ianuarie 2026):
    - NOU: Suport complet pentru Sector și Country
    - NOU: Clasificare organizațională pe 3 niveluri (Industry/Sector/Country)
    - NOU: GroupAggregator pentru agregări flexibile
    - NOU: Vizualizări per Sector și Country
    - NOU: Export Excel cu coloane Sector, Country
    - NOU: Tabele DB actualizate cu sector, country
    - ÎMBUNĂTĂȚIT: Meniu cu opțiuni 5.8, 5.9 pentru grafice Sector/Country

v6.0.5 (Ianuarie 2026):
    - NOU: Opțiunea 1.6 - Re-procesare documente cu TEXT CORUPT (OCR)
    - NOU: Coloana text_status în processed_documents
    - NOU: Detectare automată text corupt (encoding issues, OCR necesar)
    - NOU: Context validators pentru texte scurte (AI, ML, DL)
    - ÎMBUNĂTĂȚIT: False positives extinse (~60 patterns)
    - ÎMBUNĂTĂȚIT: Statistici extinse cu documente corupte/OCR

v6.0.4 (Ianuarie 2026):
    - RESTRUCTURARE opțiuni meniu 1.x:
        * 1.1 = TOATE documentele (noi + re-analiză existente)
        * 1.2 = BATCH după poziții (noi + re-analiză în range)
        * 1.3 = Doar documente NOI - toate neprocesate
        * 1.4 = Doar documente NOI - batch de N
        * 1.5 = Reverificare documente FĂRĂ referințe AI
        * 1.6 = Re-procesare documente cu TEXT CORUPT (OCR)
    - Tracking complet documente în tabelă separată processed_documents
    - La re-analiză: occurrence_count incrementat pentru referințe existente
    - Referințele noi descoperite la re-analiză se adaugă corect
    - Statistici detaliate: docs noi vs re-analizate, refs noi vs actualizate

v6.0.3 (Ianuarie 2026):
    - Tabelă separată processed_documents pentru tracking
    - Fix UNIQUE constraint la inserare referințe
    - Statistici separate documente vs referințe

v6.0.2 (Ianuarie 2026):
    - Fix duplicate class definitions
    - Fix method indentation issues
    - Improved error handling

v6.0.0 (Ianuarie 2026):
    - 13 categorii AI (de la 12)
    - 80+ termeni GenAI 2025
    - Filtrare Robotics tradițional vs AI-robotics
    - Filtrare RPA tradițional vs Intelligent Automation
    - Threshold semantic 0.55 → 0.60
    - False positives extinse (~40 patterns)
    - Suport 2020-2025
    - Meniu interactiv complet

═══════════════════════════════════════════════════════════════════════════════
STRUCTURA MENIU v6.0.6:
═══════════════════════════════════════════════════════════════════════════════

1. EXTRAGERE DATE
   1.1 TOATE documentele (noi + re-analiză)
   1.2 BATCH după poziții (noi + re-analiză)
   1.3 Doar documente NOI - toate
   1.4 Doar documente NOI - batch de N
   1.5 Reverificare documente FĂRĂ referințe AI
   1.6 Re-procesare documente cu TEXT CORUPT (OCR)

2. ANALIZĂ & DEDUPLICARE
   2.1 Deduplicare semantică
   2.2 Analiză TOATE
   2.3 Analiză doar cele NOI

3. CALCUL AI ADOPTION INDEX
   3.1 Calcul BATCH
   3.2 Calcul TOATE
   3.3 Calcul doar cele NOI

4. RAPOARTE
   4.1 Export ai_references_raw.xlsx
   4.2 Export ai_references_deduplicated.xlsx
   4.3 Export ai_adoption_index.xlsx
   4.4 Export results.json
   4.5 Export report.txt
   4.6 Export TOATE rapoartele

5. GRAFICE (Plotly Interactive)
   5.1 Ranking companii per an
   5.2 Timeline evoluție
   5.3 Radar chart dimensiuni
   5.4 Heatmap companii/ani
   5.5 Distribuție categorii AI
   5.6 Analiză sentiment
   5.7 TOATE graficele
   5.8 Grafice per INDUSTRIE
   5.9 Grafice per SECTOR/COUNTRY

6. UTILITĂȚI
   6.1 Statistici complete
   6.2 Listare documente procesate
   6.3 Verificare integritate DB
   6.4 Backup bază de date
   6.5 Curățare date vechi
   6.6 Reconfigurare căi

7. PIPELINE COMPLET
   Procesare automată end-to-end (Extragere → Deduplicare → Index → Export)

0. IEȘIRE

═══════════════════════════════════════════════════════════════════════════════
CATEGORII AI (13):
═══════════════════════════════════════════════════════════════════════════════

    1. machine_learning          - ML, deep learning, neural networks
    2. natural_language_processing - NLP, text analysis, language models
    3. computer_vision           - image recognition, video analysis
    4. generative_ai_llm         - GenAI, GPT, ChatGPT, LLMs, Claude
    5. agentic_ai_systems        - AI agents, autonomous systems, multi-agent
    6. ai_infrastructure_mlops   - MLOps, AIOps, AI platforms
    7. predictive_analytics      - forecasting, predictive modeling
    8. automation_robotics       - AI-powered automation (nu traditional RPA)
    9. ai_applications           - chatbots, virtual assistants, recommendations
   10. ai_governance_ethics      - responsible AI, ethics, compliance
   11. ai_strategy_adoption      - AI transformation, digital strategy
   12. recommendation_systems    - personalization, recommendation engines
   13. data_ai_general           - general AI/data references

═══════════════════════════════════════════════════════════════════════════════
AI ADOPTION INDEX - FORMULA:
═══════════════════════════════════════════════════════════════════════════════

    AI_Adoption_Index = Σ (Dimension_i × Weight_i)
    
    ┌──────────────────────┬────────┬─────────────────────────────────┐
    │     Dimensiune       │ Weight │         Calcul                  │
    ├──────────────────────┼────────┼─────────────────────────────────┤
    │ Intensity Index      │  1.5   │ refs / 100 pagini (max 3.0)     │
    │ Semantic Index       │  1.2   │ avg semantic score (0-1)        │
    │ Diversity Index      │  1.3   │ unique_categories / 13 (0-1)    │
    │ Sentiment Index      │  1.0   │ % positive normalized (0-1)     │
    │ Maturity Index       │  1.5   │ implementation stage (0-3)      │
    │ Future Index         │  1.2   │ future plans 2025+ (0-2)        │
    │ Commitment Index     │  1.3   │ investment patterns (0-3)       │
    ├──────────────────────┼────────┼─────────────────────────────────┤
    │ MAX POSSIBLE SCORE   │        │ 12.0                            │
    └──────────────────────┴────────┴─────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FILES:
═══════════════════════════════════════════════════════════════════════════════

    📊 Excel Reports:
        • ai_references_raw_v6.xlsx - Toate referințele identificate
        • ai_references_deduplicated_v6.xlsx - Referințe după deduplicare
        • ai_adoption_index_v6.xlsx - AI Adoption Index per companie/an
        • industry_aggregation_v6.xlsx - Agregare per industrie
        • sector_aggregation_v6.xlsx - Agregare per sector (NOU)
        • country_aggregation_v6.xlsx - Agregare per țară (NOU)
    
    📈 Visualizations (Plotly HTML):
        • viz_ranking_*.html - Ranking companii
        • viz_trend_*.html - Evoluție temporală
        • viz_radar_*.html - Radar chart dimensiuni
        • viz_heatmap_*.html - Heatmap companii/ani
        • viz_category_*.html - Distribuție categorii
        • viz_sentiment_*.html - Analiză sentiment
        • viz_industry_*.html - Comparație industrii
        • viz_sector_*.html - Comparație sectoare (NOU)
        • viz_country_*.html - Comparație țări (NOU)
    
    📝 Other:
        • results_v6.json - Export complet JSON
        • report_v6.txt - Raport text sumar
    
    💾 Database:
        • ai_semantic_analyzer_v6.db - Baza de date SQLite

═══════════════════════════════════════════════════════════════════════════════
STRUCTURA BAZĂ DE DATE v6.0.6:
═══════════════════════════════════════════════════════════════════════════════

    📋 ai_references_raw
        - Toate referințele AI identificate
        - Coloane: company, year, position, industry, sector, country,
                   doc_type, page, text, context, category, sentiment,
                   semantic_score, robotics_type, rpa_type, occurrence_count
    
    📋 ai_references_deduplicated
        - Referințe unice după deduplicare semantică
        - Coloane: company, year, position, industry, sector, country,
                   text, context, category, sources, doc_count,
                   avg_sentiment_score, avg_semantic_score
    
    📋 adoption_index
        - AI Adoption Index per companie/an
        - Coloane: company, year, position, industry, sector, country,
                   intensity_index, semantic_index, diversity_index,
                   sentiment_index, maturity_index, future_index,
                   commitment_index, ai_adoption_index, rank_in_year
    
    📋 adoption_index_industry
        - Agregare per industrie
        - Statistici: avg, min, max, std_deviation, num_companies
    
    📋 processed_documents
        - Tracking documente procesate
        - Coloane: source, company, year, position, industry, doc_type,
                   total_pages, text_length, refs_found, process_count,
                   text_status (valid/corrupted/ocr_needed/ocr_success)

═══════════════════════════════════════════════════════════════════════════════
DEPENDENȚE:
═══════════════════════════════════════════════════════════════════════════════

    Core:
        • Python 3.9+
        • pandas, numpy
        • sqlite3 (built-in)
    
    PDF Processing:
        • pdfplumber
        • PyMuPDF (fitz)
        • pytesseract (optional, pentru OCR)
        • Pillow (pentru OCR)
    
    NLP & ML:
        • sentence-transformers (all-MiniLM-L6-v2)
        • transformers (pentru FinBERT)
        • torch
        • textblob (fallback sentiment)
        • scikit-learn
    
    Export & Visualization:
        • openpyxl (Excel cu formatare)
        • plotly (grafice interactive)
    
    Utilities:
        • tqdm (progress bars)
        • wordsegment (optional, text segmentation)

═══════════════════════════════════════════════════════════════════════════════
UTILIZARE:
═══════════════════════════════════════════════════════════════════════════════

    # Rulare standard (meniu interactiv)
    python ai_analyzer_v6_main.py
    
    # La prima rulare, configurează:
    # 1. Calea către folderul cu PDF-uri
    # 2. Calea către Fortune500.csv (opțional)
    # 3. Folderul de output (default: Results_v6/)
    
    # Format nume fișiere PDF:
    # {position}_{CompanyName}_{Year}_{DocType}.pdf
    # Exemplu: 001_Walmart_2024_Annual Report.pdf

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os

# Verifică versiunea Python
if sys.version_info < (3, 9):
    print("EROARE: Python 3.9+ necesar")
    print(f"Versiune curentă: {sys.version}")
    sys.exit(1)

# Import meniu (preferă v6.1.x)
try:
    from ai_analyzer_v6_1_menu import MenuManager, main as menu_main
    MENU_VERSION = "6.1.x"
except ImportError:
    # Fallback: meniu v6.0.6
    try:
        from ai_analyzer_v6_0_6_menu import MenuManager, main as menu_main
        MENU_VERSION = "6.0.6"
        print("⚠ ATENȚIE: Se folosește meniul v6.0.6 (fallback).")
        print("  Pentru noul scoring/mention_only, folosește ai_analyzer_v6_1_menu.py")
    except ImportError:
        # Fallback pentru versiuni mai vechi
        try:
            from ai_analyzer_v6_menu import MenuManager, main as menu_main
            MENU_VERSION = "6.0.x"
            print("⚠ ATENȚIE: Se folosește versiunea mai veche a meniului")
            print("  Pentru funcționalități complete, folosește ai_analyzer_v6_1_menu.py")
        except ImportError as e:
            print(f"EROARE: Nu s-a putut importa meniul: {e}")
            print("\nAsigură-te că unul din următoarele fișiere există în același folder:")
            print("  - ai_analyzer_v6_1_menu.py (recomandat)")
            print("  - ai_analyzer_v6_0_6_menu.py (fallback)")
            print("  - ai_analyzer_v6_menu.py (legacy)")
            sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# VERSION INFO
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "6.1.1"
__author__ = "TeRa0"
__date__ = "Ianuarie 2026"


def print_banner():
    """Afișează banner-ul de start."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     █████╗ ██╗    ███████╗███████╗███╗   ███╗ █████╗ ███╗   ██╗████████╗     ║
║    ██╔══██╗██║    ██╔════╝██╔════╝████╗ ████║██╔══██╗████╗  ██║╚══██╔══╝     ║
║    ███████║██║    ███████╗█████╗  ██╔████╔██║███████║██╔██╗ ██║   ██║        ║
║    ██╔══██║██║    ╚════██║██╔══╝  ██║╚██╔╝██║██╔══██║██║╚██╗██║   ██║        ║
║    ██║  ██║██║    ███████║███████╗██║ ╚═╝ ██║██║  ██║██║ ╚████║   ██║        ║
║    ╚═╝  ╚═╝╚═╝    ╚══════╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝        ║
║                                                                               ║
║                    AI SEMANTIC ANALYZER v6.1.1                                ║
║              Fortune 500 AI Adoption Analysis System                          ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_features():
    """Afișează lista de features."""
    print("""
┌───────────────────────────────────────────────────────────────────────────────┐
│                              FEATURES v6.1.1                                  │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  📊 ANALIZĂ AI                          │  🏢 CLASIFICARE ORGANIZAȚIONALĂ     │
│  • 13 categorii AI specializate         │  • Industry / Sector / Country      │
│  • 80+ patterns GenAI 2025              │  • Agregări multi-nivel              │
│  • Semantic matching (threshold 0.60 / strict 0.68)   │  • Vizualizări comparative           │
│  • FinBERT sentiment analysis           │                                      │
│  • ~60 false positive filters           │  🔧 OCR & RECOVERY                   │
│                                         │  • Detectare text corupt             │
│  📈 AI ADOPTION INDEX                   │  • OCR fallback (pytesseract)        │
│  • 7 dimensiuni ponderate               │  • Re-procesare selectivă            │
│  • Max score: 12.0                      │                                      │
│  • Ranking per an                       │  📁 OUTPUT                           │
│                                         │  • Excel cu Rich Text formatting     │
│  🔍 FILTRARE INTELIGENTĂ                │  • Grafice interactive Plotly        │
│  • AI-robotics vs traditional           │  • JSON, TXT, SQLite                 │
│  • Intelligent Automation vs RPA        │  • Statistici complete               │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
""")


def check_dependencies():
    """Verifică dependențele necesare."""
    missing = []
    optional_missing = []
    
    # Dependențe obligatorii
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'pdfplumber': 'pdfplumber',
        'fitz': 'PyMuPDF',
        'sentence_transformers': 'sentence-transformers',
        'openpyxl': 'openpyxl',
        'plotly': 'plotly',
        'tqdm': 'tqdm',
        'textblob': 'textblob',
        'sklearn': 'scikit-learn',
    }
    
    # Dependențe opționale
    optional = {
        'transformers': 'transformers (pentru FinBERT)',
        'torch': 'torch (pentru FinBERT)',
        'pytesseract': 'pytesseract (pentru OCR)',
        'PIL': 'Pillow (pentru OCR)',
        'wordsegment': 'wordsegment (pentru text segmentation)',
    }
    
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    for module, package in optional.items():
        try:
            __import__(module)
        except ImportError:
            optional_missing.append(package)
    
    if missing:
        print("\n⚠ DEPENDENȚE LIPSĂ (obligatorii):")
        for pkg in missing:
            print(f"   pip install {pkg}")
        print("\nInstalează cu: pip install " + " ".join(missing))
        return False
    
    if optional_missing:
        print("\n💡 Dependențe opționale lipsă (funcționalitate redusă):")
        for pkg in optional_missing:
            print(f"   • {pkg}")
    
    return True


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Punct de intrare principal."""
    
    # Banner
    print_banner()
    
    # Verifică dependențe
    if not check_dependencies():
        print("\n❌ Instalează dependențele lipsă și încearcă din nou.")
        sys.exit(1)
    
    # Features
    print_features()
    
    print(f"  Menu version: {MENU_VERSION}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Working directory: {os.getcwd()}")
    
    # Rulează meniul
    try:
        menu_main()
    except KeyboardInterrupt:
        print("\n\n⚠ Anulat de utilizator")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Eroare fatală: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
