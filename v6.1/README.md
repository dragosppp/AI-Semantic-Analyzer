# AI Semantic Analyzer v6.1

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/Version-6.1.1-green.svg" alt="Version 6.1.1">
  <img src="https://img.shields.io/badge/License-Academic-orange.svg" alt="Academic License">
</p>

An advanced semantic analysis system designed for doctoral research to identify and quantify AI adoption patterns in Fortune 500 company annual reports (2020-2025).

## 🎯 Overview

AI Semantic Analyzer is a comprehensive NLP-powered tool that extracts, categorizes, and analyzes artificial intelligence references from corporate documents. It combines pattern matching with semantic similarity analysis using transformer models to produce a multi-dimensional AI Adoption Index.

**Designed for:** Academic research, doctoral dissertations, conference presentations, and journal publications on corporate AI adoption.

---

## 📁 Project Structure

```
ai_semantic_analyzer/
│
├── ai_analyzer_v6_1_main.py      # Entry point - run this file
├── ai_analyzer_v6_1_menu.py      # Interactive menu system
├── ai_analyzer_v6_1_module1.py   # Core configuration, data structures, database
├── ai_analyzer_v6_1_module2.py   # PDF processing, AI detection, text extraction
├── ai_analyzer_v6_1_module3.py   # Deduplication, sentiment analysis, index calculation
├── ai_analyzer_v6_1_module4.py   # Export (Excel, JSON), visualization (Plotly)
│
├── analyzer_config_v6.json       # Auto-generated configuration file
├── ai_analyzer_v6.log            # Runtime logs
│
├── Results_v6/                   # Default output folder
│   ├── ai_analysis_v6.db         # SQLite database
│   ├── ai_references_raw.xlsx    # Raw extracted references
│   ├── ai_references_deduplicated.xlsx
│   ├── ai_adoption_index.xlsx    # Final AI Adoption Index
│   ├── results.json
│   ├── report.txt
│   └── charts/                   # Interactive Plotly visualizations
│
└── PDFs/                         # Your input folder (configurable)
    └── {position}_{Company}_{Year}_{DocType}.pdf
```

### File Naming Convention

PDF files **must** follow this naming pattern for proper parsing:

```
{Position}. {Company Name} - {Year} - {Document Type}.pdf
```

**Examples:**
- `1. Walmart - 2024 - Annual Report.pdf`
- `38. Bank of America - 2024 - annual report.pdf`
- `125. Microsoft - 2023 - Sustainability.pdf`

**Supported Document Types:**
- `Annual Report` / `10K`
- `Sustainability` / `ESG` / `CSR`
- `Proxy` / `DEF14A`
- `Quarterly Report` / `10Q`

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- 8GB+ RAM recommended (for transformer models)
- GPU optional but recommended for faster processing

### Required Dependencies

```bash
pip install pandas numpy pdfplumber PyMuPDF sentence-transformers openpyxl plotly tqdm textblob scikit-learn
```

### Optional Dependencies (Enhanced Features)

```bash
# FinBERT for financial sentiment analysis (89% accuracy)
pip install transformers torch

# OCR for corrupted/scanned PDFs
pip install pytesseract Pillow
# Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract

# Word segmentation for concatenated text
pip install wordsegment
```

### Quick Start

```bash
# Clone or download all module files to the same directory
# Then run:
python ai_analyzer_v6_1_main.py
```

On first run, you'll be prompted to configure:
1. Path to PDF folder
2. Path to Fortune 500 CSV (optional, for industry/sector mapping)
3. Output folder (default: `Results_v6/`)

---

## 🔍 Text Identification Methodology

### Detection Pipeline

The system uses a **hybrid detection approach** combining three methods:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. PATTERN MATCHING (Fast, High Precision)                     │
│     └─> 80+ regex patterns for AI terms                         │
│     └─> Exact match for specific technologies                   │
│                                                                 │
│  2. SEMANTIC ANALYSIS (Deep Understanding)                      │
│     └─> SentenceTransformer (all-MiniLM-L6-v2)                  │
│     └─> Cosine similarity against 65+ canonical descriptions    │
│     └─> Threshold: 0.60 (standard) / 0.68 (strict)              │
│                                                                 │
│  3. CONTEXT VALIDATION (False Positive Reduction)               │
│     └─> 60+ false positive patterns                             │
│     └─> Context validators for short terms (AI, ML, DL)         │
│     └─> Robotics/RPA classification (traditional vs AI-powered) │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Pattern Matching Details

The system recognizes **80+ AI-related patterns** across categories:

| Category | Example Patterns |
|----------|------------------|
| Core AI/ML | `artificial intelligence`, `machine learning`, `deep learning`, `neural network` |
| Generative AI (2023-2025) | `generative AI`, `GenAI`, `LLM`, `GPT-4`, `ChatGPT`, `Claude`, `Copilot` |
| Agentic AI (2025) | `AI agent`, `agentic AI`, `autonomous system`, `multi-agent` |
| RAG & Knowledge | `RAG`, `retrieval-augmented generation`, `vector database`, `embeddings` |
| Infrastructure | `MLOps`, `LLMOps`, `AIOps`, `edge AI`, `model serving` |
| AI Safety | `responsible AI`, `AI ethics`, `AI governance`, `explainable AI` |
| Applications | `chatbot`, `virtual assistant`, `recommendation system`, `fraud detection` |

### Semantic Analysis

For paragraphs that don't match explicit patterns, semantic similarity is computed:

1. **Encoding**: Each paragraph is encoded using `all-MiniLM-L6-v2` (384-dimensional embeddings)
2. **Comparison**: Cosine similarity computed against 65+ canonical AI descriptions
3. **Thresholds**:
   - **0.60** - Standard threshold (when hard indicators or implementation evidence present)
   - **0.68** - Strict threshold (semantic-only detection, no pattern confirmation)

**Canonical Description Examples:**
```python
"Company is implementing machine learning models for predictive analytics"
"Organization deploying large language models and generative AI solutions"
"Enterprise adopting AI-powered automation and intelligent process optimization"
```

### False Positive Filtering

The system employs **60+ false positive patterns** to exclude non-AI references:

```python
FALSE_POSITIVE_PATTERNS = [
    # Generic business terms
    r'\b(?:AI|ML)\s*(?:segment|division|unit)\b',          # "AI segment" as business unit
    r'\bAl\b',                                              # Person name "Al"
    r'\bML\s*(?:Bill|Act|Law)\b',                          # "ML Bill" (legislation)
    
    # Industry-specific false positives
    r'\b(?:assembly|production)\s+(?:line|robot)\b',        # Traditional manufacturing
    r'\bindustrial\s+robot(?:s|ics)?\b',                    # Non-AI robotics
    r'\brobotic\s+process\s+automation\b(?!.*intelligent)', # Traditional RPA
    
    # Financial document noise
    r'\bAI\s+(?:insurance|insurer)\b',                      # American International
    r'\bfiscal\s+year\s+\d{4}.*AI\b',                       # False context
]
```

### Robotics & RPA Classification

The system distinguishes between traditional and AI-powered automation:

| Classification | Triggers AI Detection | Example |
|----------------|----------------------|---------|
| `ai_robotics` | ✅ Yes | "AI-powered robotic arms", "ML-enhanced automation" |
| `traditional_robotics` | ❌ No | "assembly line robots", "industrial automation" |
| `intelligent_rpa` | ✅ Yes | "intelligent process automation", "cognitive RPA" |
| `traditional_rpa` | ❌ No | "rule-based automation", "RPA bots" |

### Context Validators

For ambiguous short terms (AI, ML, DL), the system validates context:

```python
# Valid AI context indicators:
- "powered by AI"
- "AI-driven insights"
- "implementing machine learning"
- "deep learning models"

# Invalid context (filtered out):
- "AI segment reported revenue"  # Business unit
- "ML requirements"              # Generic abbreviation
- "DL deadline"                  # Non-AI meaning
```

### Confidence Scoring (v6.1+)

Each reference receives a **confidence score** and **strength classification**:

| Strength | Confidence | Criteria |
|----------|------------|----------|
| `strong` | 0.80+ | Hard indicator + high semantic score + implementation evidence |
| `medium` | 0.50-0.79 | Pattern match + moderate semantic score |
| `mention_only` | <0.50 | Weak signal, excluded from AI Adoption Index |

---

## 📊 AI Adoption Index

The **AI Adoption Index** is a composite score (0-100+) calculated from 7 weighted dimensions:

```
┌─────────────────────────────────────────────────────────────────┐
│               AI ADOPTION INDEX FORMULA                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  AI_Index = (                                                   │
│      Intensity   × 0.20  +   # References per 100 pages         │
│      Semantic    × 0.15  +   # Average semantic relevance       │
│      Diversity   × 0.15  +   # Shannon entropy (13 categories)  │
│      Sentiment   × 0.10  +   # Positive/negative tone           │
│      Maturity    × 0.15  +   # Implementation stage             │
│      Future      × 0.10  +   # Forward-looking statements       │
│      Commitment  × 0.15      # Investment signals               │
│  ) × 100                                                        │
│                                                                 │
│  Maximum Possible Score: ~12.0 (scaled to 100 = 1200 points)    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Dimension Definitions

| Dimension | Formula | Description |
|-----------|---------|-------------|
| **Intensity** | `min(1.0, refs / pages × 10)` | AI reference density |
| **Semantic** | `mean(scores > 0.60)` | Relevance quality |
| **Diversity** | `entropy / log₂(13)` | Category breadth |
| **Sentiment** | `(mean + 1) / 2` | Communication tone |
| **Maturity** | Pattern detection | `exploring` → `experimenting` → `scaling` → `embedded` |
| **Future** | Pattern detection | 2025-2028 roadmap mentions |
| **Commitment** | Weighted patterns | Investment, hiring, GenAI-specific signals |

---

## 🏢 AI Categories (13)

| # | Category | Description |
|---|----------|-------------|
| 1 | `machine_learning` | ML, deep learning, neural networks |
| 2 | `natural_language_processing` | NLP, text analysis, language models |
| 3 | `computer_vision` | Image recognition, video analysis |
| 4 | `generative_ai_llm` | GenAI, GPT, ChatGPT, LLMs, Claude |
| 5 | `agentic_ai_systems` | AI agents, autonomous systems |
| 6 | `ai_infrastructure_mlops` | MLOps, AIOps, AI platforms |
| 7 | `predictive_analytics` | Forecasting, predictive modeling |
| 8 | `automation_robotics` | AI-powered automation (not traditional RPA) |
| 9 | `ai_applications` | Chatbots, virtual assistants, recommendations |
| 10 | `ai_governance_ethics` | Responsible AI, ethics, compliance |
| 11 | `ai_strategy_adoption` | AI transformation, digital strategy |
| 12 | `recommendation_systems` | Personalization engines |
| 13 | `data_ai_general` | General AI/data references |

---

## 📈 Outputs

### Excel Reports

| File | Content |
|------|---------|
| `ai_references_raw.xlsx` | All extracted references with context, detection method, confidence |
| `ai_references_deduplicated.xlsx` | Semantically deduplicated references |
| `ai_adoption_index.xlsx` | Company rankings with 7-dimension breakdown |

**Excel Features:**
- Rich text formatting (AI terms in **bold red**)
- Summary sheets by Industry, Sector, Country
- Separate sheet for `mention_only` candidates (v6.1+)

### Interactive Visualizations (Plotly)

- Company ranking charts by year
- Timeline evolution charts
- Radar charts (7 dimensions)
- Heatmaps (companies × years)
- Category distribution charts
- Industry/Sector/Country comparisons

### Database (SQLite)

Tables:
- `ai_references_raw` - All extracted references
- `ai_references_deduplicated` - Deduplicated references
- `adoption_index` - Company-level AI Adoption Index
- `adoption_index_industry` - Industry aggregations
- `processed_documents` - Document tracking

---

## 🖥️ Menu Structure

```
1. DATA EXTRACTION
   1.1 Process ALL documents (new + re-analyze)
   1.2 Process BATCH by positions
   1.3 Process only NEW documents - all
   1.4 Process only NEW documents - batch of N
   1.5 Re-verify documents WITHOUT AI references
   1.6 Re-process documents with CORRUPTED TEXT (OCR)

2. ANALYSIS & DEDUPLICATION
   2.1 Semantic deduplication
   2.2 Analyze ALL
   2.3 Analyze only NEW

3. CALCULATE AI ADOPTION INDEX
   3.1 Calculate BATCH
   3.2 Calculate ALL
   3.3 Calculate only NEW

4. REPORTS
   4.1 Export ai_references_raw.xlsx
   4.2 Export ai_references_deduplicated.xlsx
   4.3 Export ai_adoption_index.xlsx
   4.4 Export results.json
   4.5 Export report.txt
   4.6 Export ALL reports

5. CHARTS (Plotly Interactive)
   5.1-5.7 Various visualizations
   5.8 Charts by INDUSTRY
   5.9 Charts by SECTOR/COUNTRY

6. UTILITIES
   6.1 Complete statistics
   6.2 List processed documents
   6.3 Verify database integrity
   6.4 Backup database
   6.5 Clean old data
   6.6 Reconfigure paths

7. COMPLETE PIPELINE
   Automated end-to-end processing

0. EXIT
```

---

## ⚙️ Configuration

The system auto-generates `analyzer_config_v6.json`:

```json
{
  "input_folder": "/path/to/PDFs",
  "fortune500_csv": "/path/to/Fortune500.csv",
  "output_folder": "Results_v6",
  "semantic_threshold": 0.60,
  "semantic_threshold_strict": 0.68,
  "filter_traditional_robotics": true,
  "filter_traditional_rpa": true,
  "weights": {
    "intensity": 0.20,
    "semantic": 0.15,
    "diversity": 0.15,
    "sentiment": 0.10,
    "maturity": 0.15,
    "future": 0.10,
    "commitment": 0.15
  }
}
```

---

## 🧪 Technical Details

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Semantic Embedding | `all-MiniLM-L6-v2` | 384-dim sentence embeddings |
| Sentiment Analysis | `FinBERT-tone` | Financial sentiment (pos/neg/neutral) |
| OCR (optional) | Tesseract | Scanned/corrupted PDF recovery |

### Performance

- **Processing speed**: ~2-5 seconds per PDF (depending on length)
- **Memory usage**: ~2-4 GB (with transformer models loaded)
- **Accuracy**: Estimated 85%+ precision on AI reference detection

---

## 📚 Citation

If you use this tool in academic research, please cite:

```bibtex
@software{ai_semantic_analyzer,
  author = {Galani, Serban},
  title = {AI Semantic Analyzer: Fortune 500 AI Adoption Analysis System},
  version = {6.1.1},
  year = {2026},
  url = {https://github.com/SerbanGalani/AI-Semantic-Analyzer}
}
```

---

## 📄 License

This software is developed for academic research purposes. Please contact the author for commercial use inquiries.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request for:
- New AI pattern suggestions
- False positive pattern improvements
- Bug fixes
- Documentation improvements

---

## 📧 Contact

For questions or collaboration inquiries related to doctoral research on AI adoption in Fortune 500 companies, please open an issue on this repository.
