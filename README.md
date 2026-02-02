# AI Semantic Analyzer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/Version-6.2.0-green.svg" alt="Version 6.2.0">
  <img src="https://img.shields.io/badge/License-Academic-orange.svg" alt="Academic License">
</p>

**NLP-powered semantic analysis system for quantifying AI adoption in Fortune 500 annual reports (2020-2025)**

---

## 🚀 Latest Release: v6.2.0 (February 2026)

### Major Update: Dual Taxonomy Implementation

**What's New:**
- ✅ **Dual Taxonomy Framework**: 8 AI Applications + 8 AI Technologies (16 categories total)
- ✅ **Enhanced Detection**: 413 keywords (+176%), 99 patterns (+120%)
- ✅ **Vendor Tracking**: Detects 25+ AI vendors (ChatGPT, Claude, Gemini, AWS, Azure, etc.)
- ✅ **Improved Safety**: Removed risky keywords, enhanced false positive filtering
- ✅ **Confidence Scoring**: Keyword tiers (high/medium/low confidence)
- ✅ **Backward Compatible**: Full mapping from v6.1 categories

📖 [v6.2 Documentation](v6.2/README.md) | 📖 [Migration Guide](docs/MIGRATION_v6.1_to_v6.2.md) | 📊 [Taxonomy Comparison](docs/taxonomy_comparison.md)

---

## 🎯 Overview

AI Semantic Analyzer is a comprehensive NLP-powered tool that extracts, categorizes, and analyzes artificial intelligence references from corporate documents. It combines pattern matching with semantic similarity analysis using transformer models to produce a multi-dimensional AI Adoption Index.

**Designed for:** Academic research, doctoral dissertations, conference presentations, and journal publications on corporate AI adoption.

---

## 🗂️ Versions

| Version | Status | Location | Description |
|---------|--------|----------|-------------|
| **v6.2.0** | ⭐ **Current** | [v6.2/](v6.2/) | Dual taxonomy (16 categories), 413 keywords, vendor tracking |
| v6.1.0 | Legacy | [v6.1/](v6.1/) | Mixed taxonomy (13 categories), 150 keywords |

**Using v6.1?** It remains fully supported. See [v6.1/README.md](v6.1/README.md) for documentation.

---

## 📦 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/SerbanGalani/AI-Semantic-Analyzer.git
cd AI-Semantic-Analyzer

# Install dependencies
pip install pandas numpy pdfplumber PyMuPDF sentence-transformers openpyxl plotly tqdm textblob scikit-learn

# Optional: FinBERT for sentiment analysis
pip install transformers torch
```

### Run v6.2 (Recommended)

```bash
cd v6.2
python ai_analyzer_v6_2_main.py
```

On first run, configure:
1. Path to PDF folder
2. Path to Fortune 500 CSV (optional)
3. Output folder (default: `Results_v6/`)

---

## 📊 v6.2 Dual Taxonomy

### Dimension 1: AI Applications (8 categories)

What companies **use AI for**:

| Code | Category | Description |
|------|----------|-------------|
| **A1** | Product & Service Innovation | AI-enhanced products, R&D acceleration |
| **A2** | Operational Excellence & Automation | Process automation, supply chain optimization |
| **A3** | Customer Experience & Engagement | Chatbots, personalization, marketing |
| **A4** | Risk Management & Compliance | Fraud detection, cybersecurity, regulatory compliance |
| **A5** | Data Analytics & Business Intelligence | Predictive analytics, forecasting, insights |
| **A6** | AI Strategy & Investment | Strategic initiatives, budget allocation, partnerships |
| **A7** | AI Governance & Ethics | Responsible AI, bias mitigation, explainability |
| **A8** | AI Talent & Workforce Development | Training, upskilling, hiring |

### Dimension 2: AI Technologies (8 categories)

What **AI technologies** companies deploy:

| Code | Category | Description |
|------|----------|-------------|
| **B1** | Traditional Machine Learning | Supervised/unsupervised learning, ensemble methods |
| **B2** | Deep Learning & Neural Networks | DNN, CNN, RNN, reinforcement learning |
| **B3** | Natural Language Processing (Non-LLM) | Text classification, NER, sentiment analysis |
| **B4** | Generative AI & Large Language Models | ChatGPT, Claude, Gemini, RAG, text/image generation |
| **B5** | Computer Vision | Image recognition, object detection, video analytics |
| **B6** | Robotics & Autonomous Systems | Autonomous vehicles, AI robotics, agentic AI |
| **B7** | AI Infrastructure & Platforms | MLOps, cloud AI, GPU infrastructure, model deployment |
| **B8** | AI (General/Unspecified) | Generic AI references without technical specificity |

**Total: 16 categories** (vs 13 in v6.1)

---

## 🔍 Detection Methodology

### Hybrid Detection Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION PIPELINE v6.2                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. PATTERN MATCHING (99 patterns)                             │
│     └─> Mandatory AI context for generic terms                  │
│     └─> Vendor-specific patterns (ChatGPT, Claude, AWS, etc.)   │
│                                                                 │
│  2. SEMANTIC ANALYSIS (SentenceTransformer)                     │
│     └─> all-MiniLM-L6-v2 embeddings                            │
│     └─> Cosine similarity: 0.60 (standard) / 0.68 (strict)     │
│                                                                 │
│  3. ENHANCED FALSE POSITIVE FILTERING                          │
│     └─> 60+ exclusion patterns                                  │
│     └─> ML/DL measurement unit detection                        │
│     └─> Keyword tier-based confidence scoring                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### v6.2 Improvements

**Removed risky keywords:**
- ❌ Standalone "ML" (conflicts with "ML of water")
- ❌ Standalone "research", "lab", "innovation" (too generic)
- ❌ Standalone "agent", "workflow" (confused with non-AI)

**Enhanced patterns:**
- ✅ Mandatory AI context: `research.*(?:AI|ML)` vs just `research`
- ✅ Better exclusions: `\bML\b(?!\s*(of|metric|tons?))` 
- ✅ Vendor detection: `ChatGPT|Claude|Gemini|Copilot`

**Confidence tiers:**
- **Tier 1**: High confidence (AI-specific compounds: "machine learning", "ChatGPT")
- **Tier 2**: Medium confidence (needs validation: "classification", "optimization")
- **Tier 3**: Low confidence (supportive: "algorithm", "model training")

---

## 📁 Project Structure

```
AI-Semantic-Analyzer/
│
├── README.md                          # This file
│
├── v6.2/                              # 🚀 CURRENT VERSION
│   ├── README.md                      # v6.2 specific docs
│   ├── ai_taxonomy_v7.py              # 🆕 External taxonomy module
│   ├── ai_analyzer_v6_2_main.py       # Entry point
│   ├── ai_analyzer_v6_2_menu.py       # Interactive menu
│   ├── ai_analyzer_v6_2_module1.py    # Core & config
│   ├── ai_analyzer_v6_2_module2.py    # Detection & extraction
│   ├── ai_analyzer_v6_2_module3.py    # Analysis & deduplication
│   └── ai_analyzer_v6_2_module4.py    # Export & visualization
│
├── v6.1/                              # 📦 LEGACY VERSION
│   ├── README.md                      # v6.1 specific docs
│   └── ai_analyzer_v6_1_*.py          # v6.1 modules
│
├── docs/                              # 📚 DOCUMENTATION
│   ├── MIGRATION_v6.1_to_v6.2.md     # Upgrade guide
│   └── taxonomy_comparison.md         # Detailed taxonomy comparison
│
└── Results_v6/                        # Output folder (auto-created)
    ├── ai_analysis_v6.db              # SQLite database
    ├── ai_references_raw.xlsx         # Raw extractions
    ├── ai_references_deduplicated.xlsx
    └── charts/                        # Plotly visualizations
```

### File Naming Convention

PDF files **must** follow this pattern:

```
{Position}. {Company Name} - {Year} - {Document Type}.pdf
```

**Examples:**
- `1. Walmart - 2024 - Annual Report.pdf`
- `38. Microsoft - 2023 - 10K.pdf`

---

## 📈 Outputs

### Excel Reports

| File | Content |
|------|---------|
| `ai_references_raw.xlsx` | All extracted references with context, detection method, confidence |
| `ai_references_deduplicated.xlsx` | Semantically deduplicated references |
| `ai_adoption_index.xlsx` | Company rankings with 7-dimension breakdown |

**v6.2 Enhancements:**
- 16-category distribution sheets
- Technology adoption matrices
- Vendor transparency analysis

### Interactive Visualizations (Plotly)

- **Temporal Analysis**: AI references over time (2020-2025)
- **Category Analysis**: 16-category distribution (Applications + Technologies)
- **Technology Heatmaps**: Adoption patterns by technology type (NEW v6.2)
- **Vendor Analysis**: Transparency tracking (NEW v6.2)
- **Company Rankings**: Top adopters by year
- **Sector Comparisons**: Industry benchmarking

### Database (SQLite)

Tables:
- `ai_references_raw` - All extracted references
- `ai_references_deduplicated` - Deduplicated references
- `adoption_index` - Company-level AI Adoption Index
- `processed_documents` - Document tracking

---

## 🧪 Technical Details

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Semantic Embedding | `all-MiniLM-L6-v2` | 384-dim sentence embeddings |
| Sentiment Analysis | `FinBERT-tone` | Financial sentiment (pos/neg/neutral) |
| OCR (optional) | Tesseract | Scanned PDF recovery |

### Performance

- **Processing speed**: ~2-5 seconds per PDF
- **Memory usage**: ~2-4 GB (with transformers)
- **Accuracy**: 85%+ precision (reduced false positives by ~30% in v6.2)

### Dependencies

**Required:**
```bash
pip install pandas numpy pdfplumber PyMuPDF sentence-transformers openpyxl plotly tqdm textblob scikit-learn
```

**Optional (Enhanced Features):**
```bash
pip install transformers torch          # FinBERT sentiment
pip install pytesseract Pillow          # OCR for scanned PDFs
pip install wordsegment                 # Text segmentation
```

---

## 📚 Citation

If you use this tool in academic research:

```bibtex
@software{ai_semantic_analyzer_2026,
  author = {Galani, Serban},
  title = {AI Semantic Analyzer: Dual Taxonomy Framework for Corporate AI Adoption Analysis},
  version = {6.2.0},
  year = {2026},
  url = {https://github.com/SerbanGalani/AI-Semantic-Analyzer}
}
```

---

## 📄 License

This software is developed for academic research purposes. Please contact the author for commercial use inquiries.

---

## 🤝 Contributing

Contributions welcome! Please open an issue or PR for:
- New AI pattern suggestions
- False positive improvements
- Vendor keyword additions
- Documentation improvements

---

## 📧 Contact

For questions or collaboration inquiries, please open an issue on this repository.

---

## 🔗 Resources

- **[v6.2 Documentation](v6.2/README.md)** - Detailed v6.2 guide
- **[v6.1 Documentation](v6.1/README.md)** - Legacy version docs
- **[Migration Guide](docs/MIGRATION_v6.1_to_v6.2.md)** - Upgrade from v6.1
- **[Taxonomy Comparison](docs/taxonomy_comparison.md)** - v6.1 vs v6.2 detailed comparison

---

**⭐ Star this repo if you find it useful for your research!**
