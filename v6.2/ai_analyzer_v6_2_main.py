"""
═══════════════════════════════════════════════════════════════════════════════
AI SEMANTIC ANALYZER v6.2.0 - MAIN ORCHESTRATOR
═══════════════════════════════════════════════════════════════════════════════

NLP-powered semantic analysis system for quantifying AI adoption in Fortune 500 
annual reports (2020-2025). 

MAJOR UPDATE v6.2.0 (February 2026):
    🆕 Dual Taxonomy Framework v7.0.1
        - Dimension 1: AI Applications (8 categories) - "WHAT FOR?"
        - Dimension 2: AI Technologies (8 categories) - "WHAT TECHNOLOGY?"
    
    🔒 Enhanced Safety & Accuracy
        - Removed risky standalone keywords (ML, research, agent, etc.)
        - Mandatory AI context in patterns
        - Keyword confidence tiers (high/medium/low)
        - Reduced false positives by ~30%
    
    📊 Expanded Detection Capabilities
        - 413 keywords total (vs 150 in v6.1, +176%)
        - 99 patterns total (vs 45 in v6.1, +120%)
        - 25+ vendor-specific keywords (ChatGPT, Claude, Gemini, AWS, etc.)
    
    🏗️ Modular Architecture
        - Externalized taxonomy to ai_taxonomy_v7.py
        - Backward compatible with v6.1 data
        - Ready for v7.0 plugin-based refactoring

WORKFLOW:
    1. Interactive configuration (paths, Fortune 500 list, years)
    2. Multi-threaded PDF processing (pdfplumber + PyMuPDF)
    3. Dual detection: Pattern matching + Semantic analysis (SentenceTransformer)
    4. Category classification with confidence scoring
    5. False positive filtering + Context extraction
    6. Semantic deduplication (cosine similarity)
    7. FinBERT sentiment analysis (optional)
    8. Statistical analysis + Plotly visualizations
    9. Multi-format export (Excel, JSON, SQLite)

TECHNICAL STACK:
    - NLP: SentenceTransformer (all-MiniLM-L6-v2), FinBERT
    - PDF: pdfplumber, PyMuPDF, Tesseract OCR
    - Analysis: pandas, numpy, scikit-learn
    - Visualization: Plotly
    - Database: SQLite

USAGE:
    python ai_analyzer_v6_2_main.py
    
    # Or with saved config:
    python ai_analyzer_v6_2_main.py --config analyzer_config_v6.json

OUTPUTS:
    - ai_references_raw_v6.xlsx           # Raw detections
    - ai_references_deduplicated_v6.xlsx  # Cleaned dataset
    - ai_analysis_results_v6.db           # SQLite database
    - ai_analysis_plots_*.html            # Interactive visualizations
    - ai_adoption_report_*.json           # JSON export

MIGRATION FROM v6.1:
    See MIGRATION_v6.2_README.md for full upgrade guide.
    v6.1 categories automatically mapped to v7.0 codes.

CHANGELOG v6.2.0:
    - Dual taxonomy: 16 categories (was 13)
    - External taxonomy module (ai_taxonomy_v7.py)
    - Enhanced patterns with mandatory AI context
    - Vendor detection: OpenAI, Anthropic, Google, Microsoft, AWS, etc.
    - Keyword tiers for confidence scoring
    - Improved false positive filtering
    - Backward compatible category mapping

Author: TeRa0
Version: 6.2.0
Date: February 2026
GitHub: https://github.com/SerbanGalani/AI-Semantic-Analyzer
License: MIT

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
    from ai_analyzer_v6_2_menu import MenuManager, main as menu_main
    MENU_VERSION = "6.1.x"
except ImportError:
    # Fallback: meniu v6.0.6
    try:
        from ai_analyzer_v6_0_6_menu import MenuManager, main as menu_main
        MENU_VERSION = "6.0.6"
        print("⚠ ATENȚIE: Se folosește meniul v6.0.6 (fallback).")
        print("  Pentru noul scoring/mention_only, folosește ai_analyzer_v6_2_menu.py")
    except ImportError:
        # Fallback pentru versiuni mai vechi
        try:
            from ai_analyzer_v6_menu import MenuManager, main as menu_main
            MENU_VERSION = "6.0.x"
            print("⚠ ATENȚIE: Se folosește versiunea mai veche a meniului")
            print("  Pentru funcționalități complete, folosește ai_analyzer_v6_2_menu.py")
        except ImportError as e:
            print(f"EROARE: Nu s-a putut importa meniul: {e}")
            print("\nAsigură-te că unul din următoarele fișiere există în același folder:")
            print("  - ai_analyzer_v6_2_menu.py (recomandat)")
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
