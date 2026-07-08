# AI Semantic Analyzer v6.2.0

**Dual Taxonomy Implementation - February 2026**

## 🚀 What's New

- ✅ **Dual Taxonomy**: 8 Applications + 8 Technologies (16 categories)
- ✅ **413 keywords** (+176% vs v6.1)
- ✅ **99 patterns** (+120% vs v6.1)
- ✅ **Vendor detection**: ChatGPT, Claude, Gemini, AWS, Azure, etc.
- ✅ **Enhanced safety**: Removed risky keywords
- ✅ **Keyword tiers**: High/medium/low confidence scoring

## 📦 Quick Start
```bash
# Ensure you have all dependencies
pip install -r ../requirements.txt

# Run analyzer
python ai_analyzer_v6_2_main.py
```

## 📊 Taxonomy Structure

### Dimension 1: AI Applications (8 categories)
- A1: Product & Service Innovation
- A2: Operational Excellence & Automation
- A3: Customer Experience & Engagement
- A4: Risk Management & Compliance
- A5: Data Analytics & Business Intelligence
- A6: AI Strategy & Investment
- A7: AI Governance & Ethics
- A8: AI Talent & Workforce Development

### Dimension 2: AI Technologies (8 categories)
- B1: Traditional Machine Learning
- B2: Deep Learning & Neural Networks
- B3: Natural Language Processing (Non-LLM)
- B4: Generative AI & Large Language Models
- B5: Computer Vision
- B6: Robotics & Autonomous Systems
- B7: AI Infrastructure & Platforms
- B8: AI (General/Unspecified)

## 🔄 Migration from v6.1

See [Migration Guide](../docs/MIGRATION_v6.1_to_v6.2.md)

## 📁 Files

- `ai_taxonomy_v7.py` - External taxonomy module (NEW)
- `ai_analyzer_v6_2_main.py` - Main orchestrator
- `ai_analyzer_v6_2_menu.py` - Interactive menu
- `ai_analyzer_v6_2_module1.py` - Core & configuration
- `ai_analyzer_v6_2_module2.py` - Detection & extraction
- `ai_analyzer_v6_2_module3.py` - Analysis & deduplication
- `ai_analyzer_v6_2_module4.py` - Export & visualization

## 📖 Documentation

Full documentation in [main README](../README.md)

---

**Version:** 6.2.0  
**Date:** February 2026  
**Author:** TeRa0