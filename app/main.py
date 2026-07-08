"""
═══════════════════════════════════════════════════════════════════════════════
AI SEMANTIC ANALYZER - MAIN ORCHESTRATOR
═══════════════════════════════════════════════════════════════════════════════

NLP-powered semantic analysis system for quantifying AI adoption in corporate
ESG and annual reports.

KEY CAPABILITIES:
    1. Page-normalized future + commitment dimensions (intensity-over-time
       not confounded by filing-length inflation)
    2. Negation filter ("AI may pose risks" no longer counts as adoption)
    3. Frequency-weighted maturity (smooth progression, not a max-stage step)
    4. Confidence-weighted index aggregation (substantive mentions dominate
       boilerplate in every dimension)
    5. Deterministic seeds for torch / numpy / random (bit-identical re-runs)
    6. Provenance stamping in every Excel + SQLite output (seed, git commit,
       timestamp)
    7. Per-doc-type adoption index disaggregation (Annual Report vs ESG can be
       reported separately)

SEMANTIC TAXONOMIES (both classified in a single run):
    - CLASSIC: AI Applications (7, A1–A7) + AI Technologies (8, B1–B8)
    - EU_SEMANTICS: European Commission JRC "AI Watch" (8 domains / 12 subdomains)

USAGE:
    cd app
    python main.py

OUTPUTS (under app/RESULT/):
    - ai_references_raw.xlsx
    - ai_references_deduplicated.xlsx
    - ai_adoption_index.xlsx
    - eu_classification.xlsx
    - ai_adoption_analysis.db
    - charts/*.html (incl. viz_eu_sunburst, viz_taxonomy_comparison)

Author: TeRa0
License: MIT
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# DETERMINISTIC SEED SETUP — runs BEFORE any heavy import
# ═══════════════════════════════════════════════════════════════════════════
# All RNG-bearing libraries (torch, numpy, random) must be seeded before they
# are imported anywhere downstream. This block keeps the analyzer's outputs
# bit-identical across re-runs on identical input — a peer-review requirement.

import os
import random
import sys

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)

try:
    import numpy as np
    np.random.seed(SEED)
except ImportError:
    pass

try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    # Force deterministic CUDA / cuDNN where possible
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except ImportError:
    pass


# Python version guard — requires 3.11+
if sys.version_info < (3, 11):
    print("ERROR: Python 3.11+ is required.")
    print(f"Current version: {sys.version}")
    sys.exit(1)

# Import the interactive menu module.
try:
    from _06_menu import main as menu_main
except ImportError as exc:
    print(f"ERROR: failed to import menu module: {exc}")
    print("\nMake sure this file is executed from inside the app/ folder,")
    print("and that _06_menu.py (and the other modules) live next to it.")
    sys.exit(1)


__author__ = "TeRa0"


def print_banner() -> None:
    """Print the startup banner."""
    banner = r"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     █████╗ ██╗    ███████╗███████╗███╗   ███╗ █████╗ ███╗   ██╗████████╗      ║
║    ██╔══██╗██║    ██╔════╝██╔════╝████╗ ████║██╔══██╗████╗  ██║╚══██╔══╝      ║
║    ███████║██║    ███████╗█████╗  ██╔████╔██║███████║██╔██╗ ██║   ██║         ║
║    ██╔══██║██║    ╚════██║██╔══╝  ██║╚██╔╝██║██╔══██║██║╚██╗██║   ██║         ║
║    ██║  ██║██║    ███████║███████╗██║ ╚═╝ ██║██║  ██║██║ ╚████║   ██║         ║
║    ╚═╝  ╚═╝╚═╝    ╚══════╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝         ║
║                                                                               ║
║                         AI SEMANTIC ANALYZER                                  ║
║                     Corporate AI Adoption Analysis                            ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_features() -> None:
    """Print the feature list."""
    print("""
┌───────────────────────────────────────────────────────────────────────────────┐
│                                  FEATURES                                     │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  📊 AI ANALYSIS                         │  🏢 ORGANIZATIONAL CLASSIFICATION   │
│  • Dual taxonomy (classic + EU)         │  • Industry / Sector / Country      │
│  • GenAI-aware detection patterns       │  • Multi-level aggregation          │
│  • Semantic matching (0.60 / 0.68)      │  • Comparative visualizations       │
│  • FinBERT sentiment analysis           │                                     │
│  • Word-boundary keyword matching       │  🔧 OCR & RECOVERY                  │
│  • Negation filter                      │  • Corrupted text detection         │
│                                         │  • OCR fallback (pytesseract)       │
│  📈 AI ADOPTION INDEX                   │  • Selective reprocessing           │
│  • 7 page-normalized dimensions         │                                     │
│  • Confidence-weighted aggregation      │  📁 OUTPUT                          │
│  • Frequency-weighted maturity          │  • Excel with Metadata sheet        │
│  • Per-doc-type disaggregation          │  • Interactive Plotly charts        │
│                                         │  • JSON, TXT, SQLite                │
│  🔬 REPRODUCIBILITY                     │  • Provenance-stamped artifacts     │
│  • Deterministic seed=42                │                                     │
│  • Seed + git commit in every artifact  │                                     │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
""")


def check_dependencies() -> bool:
    """Verify that required and optional dependencies are installed."""
    missing: list[str] = []
    optional_missing: list[str] = []

    required = {
        "pandas": "pandas",
        "numpy": "numpy",
        "pdfplumber": "pdfplumber",
        "fitz": "PyMuPDF",
        "sentence_transformers": "sentence-transformers",
        "openpyxl": "openpyxl",
        "plotly": "plotly",
        "tqdm": "tqdm",
        "textblob": "textblob",
        "sklearn": "scikit-learn",
    }

    optional = {
        "transformers": "transformers (for FinBERT)",
        "torch": "torch (for FinBERT)",
        "pytesseract": "pytesseract (for OCR)",
        "PIL": "Pillow (for OCR)",
        "wordsegment": "wordsegment (for text segmentation)",
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
        print("\n⚠ MISSING REQUIRED DEPENDENCIES:")
        for pkg in missing:
            print(f"   pip install {pkg}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False

    if optional_missing:
        print("\n💡 Optional dependencies missing (reduced functionality):")
        for pkg in optional_missing:
            print(f"   • {pkg}")

    return True


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Primary entry point."""
    print_banner()

    if not check_dependencies():
        print("\n❌ Install the missing dependencies and try again.")
        sys.exit(1)

    # Surface the seed so the user / log captures it.
    try:
        from _02_core import logger
        logger.info(f"Deterministic seed: {SEED}")
    except Exception:
        pass

    print_features()

    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Deterministic seed: {SEED}")
    print(f"  Working directory: {os.getcwd()}")

    try:
        menu_main()
    except KeyboardInterrupt:
        print("\n\n⚠ Cancelled by user")
        sys.exit(0)
    except Exception as exc:
        print(f"\n❌ Fatal error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
