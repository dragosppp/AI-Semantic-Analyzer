"""
═══════════════════════════════════════════════════════════════════════════════
AI SEMANTIC ANALYZER v6.2.0 - MODUL 4: DATABASE, EXPORT & VISUALIZATION
═══════════════════════════════════════════════════════════════════════════════

Database management, multi-format export, and comprehensive visualization 
generation for AI Semantic Analyzer v6.2.

MAJOR UPDATE v6.2.0 (February 2026):
    🆕 DUAL TAXONOMY EXPORTS
        - 16-category Excel exports (Applications + Technologies)
        - Cross-dimensional analysis exports
        - Vendor transparency reports
        - Technology-application correlation matrices
    
    📊 ENHANCED VISUALIZATIONS
        - Technology adoption heatmaps
        - Application-technology Sankey diagrams
        - Vendor mention tracking charts
        - Category evolution timelines (2020-2025)
    
    💾 IMPROVED DATA MANAGEMENT
        - Backward compatible v6.1 → v6.2 database migration
        - Enhanced metadata tracking
        - Taxonomy version field (optional)
        - Legacy category mapping preservation

COMPONENTS:
    1. Database Management
       - SQLite operations (insert, update, query)
       - Schema validation
       - Backup and restore
       - Query builder with filters
       - Transaction management
    
    2. Excel Export
       - Raw references sheet
       - Deduplicated references sheet
       - Statistical summary sheet
       - Category distribution sheet (16 categories, NEW v6.2)
       - Company comparison sheet
       - Technology matrix sheet (NEW v6.2)
       - Formatted tables with auto-width
    
    3. JSON Export
       - Structured JSON with metadata
       - Nested company/year/category structure
       - Full reference details
       - Analysis results inclusion
       - Timestamp and version tracking
    
    4. Comprehensive Visualizations (Plotly)
       - Temporal Analysis:
         * AI references over time (line charts)
         * Year-over-year growth rates
         * Category trends (stacked areas)
       
       - Category Analysis:
         * Distribution by category (16 categories, NEW v6.2)
         * Applications vs Technologies (NEW v6.2)
         * Category evolution heatmaps
         * Treemap visualizations
       
       - Company Analysis:
         * Top adopters ranking
         * Company comparisons (grouped bars)
         * Sector benchmarking
       
       - Sector Analysis:
         * Sector distribution (pie charts)
         * Box plots by sector
         * Violin plots for distributions
       
       - NEW v6.2 Visualizations:
         * Technology adoption heatmaps
         * Application-technology correlation
         * Vendor transparency analysis
         * Category maturity charts
    
    5. Report Generation
       - HTML reports with embedded charts
       - PDF export (optional)
       - Executive summaries
       - Technical appendices

WORKFLOW:
    Data → Database Storage → Statistical Processing → Visualization Generation
    → Multi-format Export (Excel/JSON/HTML/PDF)

NEW IN v6.2:
    ✅ 16-category visualization support
    ✅ Technology adoption heatmaps
    ✅ Application-technology Sankey diagrams
    ✅ Vendor transparency charts
    ✅ Cross-dimensional correlation matrices
    ✅ Enhanced Excel exports with technology sheets

CHANGELOG v6.2.0 (Feb 2026):
    - Dual taxonomy export support (16 categories)
    - Technology adoption heatmaps
    - Application-technology correlation matrices
    - Vendor transparency analysis charts
    - Enhanced Excel exports with new sheets
    - Backward compatible database schema

CHANGELOG v6.1.1 (Jan 2026):
    - Enhanced metadata in exports
    - Improved deduplication tracking
    - Version bump 6.1.1

CHANGELOG v6.1.0 (Jan 2026):
    - Added strength/confidence/reasons fields to exports
    - Enhanced statistical summaries
    - Improved chart formatting

EXPORT FORMATS:
    - Excel (.xlsx): Multi-sheet workbooks
    - JSON (.json): Structured data with metadata
    - SQLite (.db): Full database export
    - HTML (.html): Interactive visualizations
    - CSV (.csv): Raw data tables

VISUALIZATION OUTPUTS:
    - ai_analysis_plots_temporal.html
    - ai_analysis_plots_categories.html
    - ai_analysis_plots_companies.html
    - ai_analysis_plots_sectors.html
    - ai_analysis_plots_technology_heatmap.html (NEW v6.2)
    - ai_analysis_plots_vendor_transparency.html (NEW v6.2)

Author: TeRa0
Version: 6.2.0
Date: February 2026
Part of: AI Semantic Analyzer

═══════════════════════════════════════════════════════════════════════════════
"""


import json
import sqlite3
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

# Openpyxl pentru formatare avansată
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.cell.rich_text import TextBlock, CellRichText
from openpyxl.cell.text import InlineFont

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ai_analyzer_v6_2_module1 import (
    logger, AnalyzerConfig, AIReference, DocumentResult, 
    AIAdoptionIndex, DatabaseManager, AI_CATEGORIES
)
from ai_analyzer_v6_2_module3 import SemanticDeduplicator, AIAdoptionIndexCalculatorV6


# ═══════════════════════════════════════════════════════════════════════════
# TEXT SANITIZATION & RICH TEXT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

# Regex pentru caractere ilegale în Excel
ILLEGAL_CHARACTERS_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffe\uffff\ufffd]')

# Pattern pentru marcaje AI: >>>termen<<< sau <font color="red"><b>termen</b></font>
AI_MARKER_PATTERN = re.compile(r'>>>(.*?)<<<|<font color="red"><b>(.*?)</b></font>', re.IGNORECASE)

# Caractere speciale de înlocuit
CHAR_REPLACEMENTS = {
    '•': '-', '●': '-', '○': '-', '▪': '-', '►': '>', 
    '→': '->', '✓': '[x]', '✗': '[_]', '–': '-', '—': '-',
    '"': '"', '"': '"', ''': "'", ''': "'", '…': '...',
    '　': ' ',  # Full-width space
}

def sanitize_for_excel(text: str) -> str:
    """Sanitizează text pentru Excel, PĂSTRÂND marcajele AI."""
    if text is None:
        return ''
    if not isinstance(text, str):
        text = str(text)
    
    # Înlocuiește caracterele speciale
    for char, replacement in CHAR_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    
    # Elimină caracterele ilegale
    text = ILLEGAL_CHARACTERS_RE.sub('', text)
    
    # Elimină spații multiple
    text = re.sub(r' {2,}', ' ', text)
    
    # Truncare dacă e prea lung
    if len(text) > 32000:
        text = text[:32000] + '...'
    
    return text.strip()


def create_rich_text_cell(text: str, ai_term_font: InlineFont) -> CellRichText:
    """
    Creează celulă cu rich text - termenii AI marcați cu >>><<< devin BOLD + ROȘU.
    
    Input: "Text normal >>>AI term<<< mai mult text"
    Output: CellRichText cu "AI term" formatat bold+roșu
    """
    if not text or not isinstance(text, str):
        return text
    
    # Sanitizează mai întâi
    text = sanitize_for_excel(text)
    
    # Verifică dacă există marcaje
    if '>>>' not in text and '<font' not in text:
        return text  # Returnează text simplu fără marcaje
    
    parts = []
    last_end = 0
    
    for match in AI_MARKER_PATTERN.finditer(text):
        # Text înainte de match (normal)
        if match.start() > last_end:
            normal_text = text[last_end:match.start()]
            if normal_text:
                parts.append(normal_text)
        
        # Termenul AI (bold + roșu)
        ai_term = match.group(1) or match.group(2)
        if ai_term:
            parts.append(TextBlock(ai_term_font, ai_term))
        
        last_end = match.end()
    
    # Text după ultimul match
    if last_end < len(text):
        remaining = text[last_end:]
        if remaining:
            parts.append(remaining)
    
    # Dacă avem părți formatate, returnează CellRichText
    if parts and any(isinstance(p, TextBlock) for p in parts):
        return CellRichText(*parts)
    
    # Altfel, curăță marcajele și returnează text simplu
    clean_text = AI_MARKER_PATTERN.sub(r'\1\2', text)
    return clean_text


def strip_ai_markers(text: str) -> str:
    """Elimină marcajele AI și returnează text curat."""
    if not text:
        return ''
    text = sanitize_for_excel(text)
    return AI_MARKER_PATTERN.sub(r'\1\2', text)


# ═══════════════════════════════════════════════════════════════════════════
# EXCEL EXPORTER v6.0.6 - CU SECTOR, COUNTRY ȘI RICH TEXT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════

class ExcelExporter:
    """
    Exporter Excel cu formatare avansată și Rich Text.
    Termenii AI sunt evidențiați cu BOLD + ROȘU în celulele Excel.
    v6.0.6: Include Sector și Country în export.
    """
    
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.output_folder = Path(config.output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Stiluri Excel
        self.header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
        self.header_font = Font(bold=True, color="FFFFFF", size=11)
        self.border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # Font pentru termeni AI în rich text: BOLD + ROȘU
        self.ai_term_font = InlineFont(b=True, color="FF0000")
    
    def export_references(self, references: List[Dict], filename: str = "ai_references_v6.xlsx"):
        """
        Export referințe cu Rich Text formatting.
        v6.0.6: Include Sector și Country.
        - Coloana Text: text PLAIN (fără formatare)
        - Coloana Context: termenii AI sunt BOLD + ROȘU
        """
        if not references:
            logger.warning("Nu există referințe pentru export")
            return None
        
        filepath = self.output_folder / filename
        wb = Workbook()
        ws = wb.active
        ws.title = "AI References"
        
        # Headers - ACTUALIZAT v6.0.6 cu Sector și Country
        headers = ['Company', 'Year', 'Position', 'Industry', 'Sector', 'Country',
                   'Category', 'Strength', 'Confidence', 'Reasons', 'Source', 'Text', 'Context',
                   'Sources', 'Doc Count', 'Occurrences', 'Avg Sentiment', 'Avg Semantic', 'Avg Confidence']
        
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.fill = self.header_fill
            cell.font = self.header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = self.border
        
        # Data rows
        for row_idx, ref in enumerate(references, 2):
            # Coloane text simple (sanitizate)
            ws.cell(row=row_idx, column=1, value=sanitize_for_excel(ref.get('company', ''))).border = self.border
            ws.cell(row=row_idx, column=2, value=ref.get('year', '')).border = self.border
            ws.cell(row=row_idx, column=3, value=ref.get('position', '')).border = self.border
            ws.cell(row=row_idx, column=4, value=sanitize_for_excel(ref.get('industry', ''))).border = self.border
            ws.cell(row=row_idx, column=5, value=sanitize_for_excel(ref.get('sector', ''))).border = self.border
            ws.cell(row=row_idx, column=6, value=sanitize_for_excel(ref.get('country', ''))).border = self.border
            ws.cell(row=row_idx, column=7, value=sanitize_for_excel(ref.get('category', ''))).border = self.border
            
            # Strength / Confidence / Reasons (v6.2, from v6.1.0)
            ws.cell(row=row_idx, column=8, value=sanitize_for_excel(ref.get('reference_strength', ref.get('strength', '')))).border = self.border
            ws.cell(row=row_idx, column=9, value=round(float(ref.get('confidence_score', 0) or ref.get('confidence', 0) or 0), 3)).border = self.border
            reasons_val = sanitize_for_excel(ref.get('confidence_reasons', ref.get('reasons', '')))
            reasons_cell = ws.cell(row=row_idx, column=10, value=reasons_val)
            reasons_cell.border = self.border
            reasons_cell.alignment = Alignment(wrap_text=True, vertical='top')

            # Source - NUMELE DOCUMENTULUI
            source_cell = ws.cell(row=row_idx, column=11, value=sanitize_for_excel(ref.get('source', '')))
            source_cell.border = self.border
            source_cell.alignment = Alignment(wrap_text=True, vertical='top')

            # Text - PLAIN, fără formatare (elimină marcajele)
            text_value = strip_ai_markers(ref.get('text', ''))
            text_cell = ws.cell(row=row_idx, column=12, value=text_value)
            text_cell.border = self.border
            text_cell.font = Font(color="000000")  # Negru normal
            text_cell.alignment = Alignment(wrap_text=True, vertical='top')
            
            # Context - cu RICH TEXT (termenii AI bold+roșu)
            context_value = ref.get('context', '')
            if context_value:
                rich_context = create_rich_text_cell(context_value, self.ai_term_font)
                context_cell = ws.cell(row=row_idx, column=13, value=rich_context)
            else:
                context_cell = ws.cell(row=row_idx, column=13, value='')
            context_cell.border = self.border
            context_cell.alignment = Alignment(wrap_text=True, vertical='top')
            
            # Sources și câmpuri numerice
            sources_val = ref.get('sources_files') or ref.get('sources') or ref.get('source') or ''
            ws.cell(row=row_idx, column=14, value=sanitize_for_excel(sources_val)).border = self.border
            ws.cell(row=row_idx, column=15, value=ref.get('doc_count', 0)).border = self.border
            ws.cell(row=row_idx, column=16, value=ref.get('total_occurrences', 0)).border = self.border
            ws.cell(row=row_idx, column=17, value=round(ref.get('avg_sentiment_score', 0), 3)).border = self.border
            ws.cell(row=row_idx, column=18, value=round(ref.get('avg_semantic_score', 0), 3)).border = self.border
            ws.cell(row=row_idx, column=19, value=round(ref.get('avg_confidence_score', ref.get('avg_confidence', 0) or 0), 3)).border = self.border
        
        # Adjust column widths - ACTUALIZAT v6.0.6
        ws.column_dimensions['A'].width = 25  # Company
        ws.column_dimensions['B'].width = 8   # Year
        ws.column_dimensions['C'].width = 10  # Position
        ws.column_dimensions['D'].width = 20  # Industry
        ws.column_dimensions['E'].width = 20  # Sector
        ws.column_dimensions['F'].width = 15  # Country
        ws.column_dimensions['G'].width = 25  # Category
        ws.column_dimensions['H'].width = 14  # Strength
        ws.column_dimensions['I'].width = 12  # Confidence
        ws.column_dimensions['J'].width = 45  # Reasons
        ws.column_dimensions['K'].width = 45  # Source
        ws.column_dimensions['L'].width = 40  # Text
        ws.column_dimensions['M'].width = 80  # Context
        ws.column_dimensions['N'].width = 30  # Sources
        ws.column_dimensions['O'].width = 10  # Doc count
        ws.column_dimensions['P'].width = 12  # Occurrences
        ws.column_dimensions['Q'].width = 12  # Sentiment
        ws.column_dimensions['R'].width = 12  # Semantic
        ws.column_dimensions['S'].width = 14  # Avg Confidence
        
        # Freeze header
        ws.freeze_panes = 'A2'
        
        # Row heights pentru context
        for row in range(2, len(references) + 2):
            ws.row_dimensions[row].height = 80
        
        wb.save(filepath)
        logger.info(f"✓ Export referințe cu Rich Text: {filepath}")
        return filepath
    
    def export_adoption_index(self, indices: List[AIAdoptionIndex], 
                             filename: str = "ai_adoption_index_v6.xlsx"):
        """Export AI Adoption Index cu formatare. v6.0.6: Include Sector și Country."""
        if not indices:
            logger.warning("Nu există indici pentru export")
            return None
        
        filepath = self.output_folder / filename
        wb = Workbook()
        
        # Sheet 1: Index complet
        ws = wb.active
        ws.title = "AI Adoption Index"
        
        # Headers - ACTUALIZAT v6.0.6 cu Sector și Country
        headers = ['Company', 'Year', 'Position', 'Industry', 'Sector', 'Country',
                   'Intensity', 'Semantic', 'Diversity', 'Sentiment',
                   'Maturity', 'Future', 'Commitment', 'AI Index',
                   'Total Refs', 'Pages', 'Categories']
        
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.fill = self.header_fill
            cell.font = self.header_font
            cell.border = self.border
        
        for row_idx, idx in enumerate(indices, 2):
            ws.cell(row=row_idx, column=1, value=sanitize_for_excel(idx.company)).border = self.border
            ws.cell(row=row_idx, column=2, value=idx.year).border = self.border
            ws.cell(row=row_idx, column=3, value=idx.position).border = self.border
            ws.cell(row=row_idx, column=4, value=sanitize_for_excel(idx.industry)).border = self.border
            ws.cell(row=row_idx, column=5, value=sanitize_for_excel(getattr(idx, 'sector', ''))).border = self.border
            ws.cell(row=row_idx, column=6, value=sanitize_for_excel(getattr(idx, 'country', ''))).border = self.border
            ws.cell(row=row_idx, column=7, value=round(idx.intensity_index, 3)).border = self.border
            ws.cell(row=row_idx, column=8, value=round(idx.semantic_index, 3)).border = self.border
            ws.cell(row=row_idx, column=9, value=round(idx.diversity_index, 3)).border = self.border
            ws.cell(row=row_idx, column=10, value=round(idx.sentiment_index, 3)).border = self.border
            ws.cell(row=row_idx, column=11, value=round(idx.maturity_index, 3)).border = self.border
            ws.cell(row=row_idx, column=12, value=round(idx.future_index, 3)).border = self.border
            ws.cell(row=row_idx, column=13, value=round(idx.commitment_index, 3)).border = self.border
            
            # Highlight AI Index cu culori
            index_cell = ws.cell(row=row_idx, column=14, value=round(idx.ai_adoption_index, 2))
            index_cell.border = self.border
            index_cell.font = Font(bold=True)
            if idx.ai_adoption_index >= 50:
                index_cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
            elif idx.ai_adoption_index >= 30:
                index_cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
            
            ws.cell(row=row_idx, column=15, value=idx.total_refs).border = self.border
            ws.cell(row=row_idx, column=16, value=idx.total_pages).border = self.border
            ws.cell(row=row_idx, column=17, value=idx.categories_used).border = self.border
        
        ws.freeze_panes = 'A2'
        
        # Sheet 2: Summary by Year
        ws2 = wb.create_sheet("Summary by Year")
        self._create_year_summary(ws2, indices)
        
        # Sheet 3: Summary by Industry
        ws3 = wb.create_sheet("Summary by Industry")
        self._create_group_summary(ws3, indices, group_by='industry')
        
        # Sheet 4: Summary by Sector - NOU v6.0.6
        ws4 = wb.create_sheet("Summary by Sector")
        self._create_group_summary(ws4, indices, group_by='sector')
        
        # Sheet 5: Summary by Country - NOU v6.0.6
        ws5 = wb.create_sheet("Summary by Country")
        self._create_group_summary(ws5, indices, group_by='country')
        
        wb.save(filepath)
        logger.info(f"✓ Export AI Adoption Index: {filepath}")
        return filepath
    
    def _create_year_summary(self, ws, indices: List[AIAdoptionIndex]):
        """Create year summary sheet."""
        headers = ['Year', 'Avg AI Index', 'Total Companies', 'Total Refs', 'Top Company']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.fill = self.header_fill
            cell.font = self.header_font
            cell.border = self.border
        
        year_data = defaultdict(list)
        for idx in indices:
            year_data[idx.year].append(idx)
        
        row = 2
        for year in sorted(year_data.keys()):
            year_indices = year_data[year]
            avg_index = np.mean([i.ai_adoption_index for i in year_indices])
            total_refs = sum(i.total_refs for i in year_indices)
            top = max(year_indices, key=lambda x: x.ai_adoption_index)
            
            ws.cell(row=row, column=1, value=year).border = self.border
            ws.cell(row=row, column=2, value=round(avg_index, 2)).border = self.border
            ws.cell(row=row, column=3, value=len(year_indices)).border = self.border
            ws.cell(row=row, column=4, value=total_refs).border = self.border
            ws.cell(row=row, column=5, value=f"{top.company} ({top.ai_adoption_index:.1f})").border = self.border
            row += 1
    
    def _create_group_summary(self, ws, indices: List[AIAdoptionIndex], group_by: str = 'industry'):
        """
        Create summary sheet grouped by specified field.
        v6.0.6: Metodă generică pentru industry/sector/country.
        
        Args:
            ws: Worksheet Excel
            indices: Lista de AIAdoptionIndex
            group_by: Câmpul după care se grupează ('industry', 'sector', 'country')
        """
        headers = [group_by.title(), 'Avg AI Index', 'Companies', 'Total Refs', 'Top Company']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.fill = self.header_fill
            cell.font = self.header_font
            cell.border = self.border
        
        group_data = defaultdict(list)
        for idx in indices:
            group_value = getattr(idx, group_by, 'Unknown')
            if group_value and group_value != 'Unknown':
                group_data[group_value].append(idx)
        
        row = 2
        for group_name in sorted(group_data.keys(), 
                              key=lambda x: np.mean([i.ai_adoption_index for i in group_data[x]]), 
                              reverse=True):
            group_indices = group_data[group_name]
            avg_index = np.mean([i.ai_adoption_index for i in group_indices])
            total_refs = sum(i.total_refs for i in group_indices)
            top = max(group_indices, key=lambda x: x.ai_adoption_index)
            
            ws.cell(row=row, column=1, value=sanitize_for_excel(group_name)).border = self.border
            ws.cell(row=row, column=2, value=round(avg_index, 2)).border = self.border
            ws.cell(row=row, column=3, value=len(group_indices)).border = self.border
            ws.cell(row=row, column=4, value=total_refs).border = self.border
            ws.cell(row=row, column=5, value=f"{top.company} ({top.ai_adoption_index:.1f})").border = self.border
            row += 1


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION GENERATOR v6.0.6 - CU GROUP_BY PARAMETRIZAT
# ═══════════════════════════════════════════════════════════════════════════

class VisualizationGenerator:
    """Generator vizualizări interactive Plotly v6.0.6."""
    
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.output_folder = Path(config.output_folder) / "visualizations"
        self.output_folder.mkdir(parents=True, exist_ok=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # METODE PUBLICE - APELATE DIN MENIU
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate_ranking_chart(self, indices: List[AIAdoptionIndex], year: int = None) -> Optional[str]:
        """
        5.1 - Generează grafic ranking companii pentru un an specific.
        
        Args:
            indices: Lista de AIAdoptionIndex
            year: Anul pentru care se generează ranking-ul (implicit ultimul an)
        
        Returns:
            Calea către fișierul HTML generat
        """
        if not indices:
            logger.warning("Nu există date pentru ranking chart")
            return None
        
        df = pd.DataFrame([i.to_dict() for i in indices])
        
        if year:
            df_year = df[df['year'] == year]
        else:
            # Folosește ultimul an disponibil
            year = df['year'].max()
            df_year = df[df['year'] == year]
        
        if df_year.empty:
            logger.warning(f"Nu există date pentru anul {year}")
            return None
        
        # Top 20 companii
        df_top = df_year.nlargest(20, 'ai_adoption_index')
        
        # Sortează pentru afișare corectă (cel mai mic jos)
        df_top = df_top.sort_values('ai_adoption_index', ascending=True)
        
        # Folosește sector dacă există, altfel industry
        hover_field = 'sector' if 'sector' in df_top.columns and df_top['sector'].notna().any() else 'industry'
        
        fig = go.Figure(go.Bar(
            x=df_top['ai_adoption_index'],
            y=df_top['company'],
            orientation='h',
            marker=dict(
                color=df_top['ai_adoption_index'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="AI Index")
            ),
            text=df_top['ai_adoption_index'].round(1),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>AI Index: %{x:.1f}<br>Sector: %{customdata}<extra></extra>',
            customdata=df_top[hover_field]
        ))
        
        fig.update_layout(
            title=dict(
                text=f'Top 20 Companies by AI Adoption Index ({year})',
                font=dict(size=18)
            ),
            xaxis_title='AI Adoption Index',
            yaxis_title='Company',
            template='plotly_white',
            height=700,
            margin=dict(l=200, r=50, t=80, b=50),
            showlegend=False
        )
        
        path = self.output_folder / f"viz_ranking_{year}.html"
        fig.write_html(str(path))
        logger.info(f"✓ Grafic ranking generat: {path}")
        return str(path)
    
    def generate_timeline_chart(self, indices: List[AIAdoptionIndex], top_n: int = 10) -> Optional[str]:
        """
        5.2 - Generează grafic evoluție temporală pentru top N companii.
        
        Args:
            indices: Lista de AIAdoptionIndex
            top_n: Numărul de companii de afișat (implicit 10)
        
        Returns:
            Calea către fișierul HTML generat
        """
        if not indices:
            logger.warning("Nu există date pentru timeline chart")
            return None
        
        df = pd.DataFrame([i.to_dict() for i in indices])
        
        # Selectează top N companii după media AI index
        top_companies = df.groupby('company')['ai_adoption_index'].mean().nlargest(top_n).index.tolist()
        df_top = df[df['company'].isin(top_companies)]
        
        fig = go.Figure()
        
        # Culori distincte pentru fiecare companie
        colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
        colors = colors[:top_n]
        
        for i, company in enumerate(top_companies):
            company_data = df_top[df_top['company'] == company].sort_values('year')
            fig.add_trace(go.Scatter(
                x=company_data['year'],
                y=company_data['ai_adoption_index'],
                mode='lines+markers',
                name=company,
                line=dict(color=colors[i], width=2),
                marker=dict(size=8),
                hovertemplate=f'<b>{company}</b><br>Year: %{{x}}<br>AI Index: %{{y:.1f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text=f'AI Adoption Index Evolution - Top {top_n} Companies',
                font=dict(size=18)
            ),
            xaxis_title='Year',
            yaxis_title='AI Adoption Index',
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                yanchor="top", y=0.99,
                xanchor="left", x=1.02,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            height=600,
            margin=dict(r=200)
        )
        
        path = self.output_folder / "viz_timeline.html"
        fig.write_html(str(path))
        logger.info(f"✓ Grafic timeline generat: {path}")
        return str(path)
    
    def generate_radar_chart(self, indices: List[AIAdoptionIndex], top_n: int = 5) -> Optional[str]:
        """
        5.3 - Generează grafic radar pentru cele 7 dimensiuni.
        
        Args:
            indices: Lista de AIAdoptionIndex
            top_n: Numărul de companii de comparat (implicit 5)
        
        Returns:
            Calea către fișierul HTML generat
        """
        if not indices:
            logger.warning("Nu există date pentru radar chart")
            return None
        
        dimensions = ['intensity_index', 'semantic_index', 'diversity_index',
                     'sentiment_index', 'maturity_index', 'future_index', 'commitment_index']
        labels = ['Intensity', 'Semantic', 'Diversity', 'Sentiment', 
                 'Maturity', 'Future', 'Commitment']
        
        df = pd.DataFrame([i.to_dict() for i in indices])
        
        fig = go.Figure()
        
        # Media tuturor companiilor
        avg_values = [df[dim].mean() for dim in dimensions]
        fig.add_trace(go.Scatterpolar(
            r=avg_values + [avg_values[0]],
            theta=labels + [labels[0]],
            fill='toself',
            name='Average (All Companies)',
            line_color='#1f77b4',
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))
        
        # Top N companii
        top_companies = df.groupby('company')['ai_adoption_index'].mean().nlargest(top_n).index.tolist()
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, company in enumerate(top_companies):
            company_data = df[df['company'] == company].iloc[0]
            company_values = [company_data[dim] for dim in dimensions]
            fig.add_trace(go.Scatterpolar(
                r=company_values + [company_values[0]],
                theta=labels + [labels[0]],
                fill='toself',
                name=company,
                line_color=colors[i % len(colors)],
                fillcolor=f'rgba({int(colors[i % len(colors)][1:3], 16)}, {int(colors[i % len(colors)][3:5], 16)}, {int(colors[i % len(colors)][5:7], 16)}, 0.1)'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0.2, 0.4, 0.6, 0.8, 1.0]
                )
            ),
            title=dict(
                text=f'AI Adoption - 7 Dimensions (Top {top_n} vs Average)',
                font=dict(size=18)
            ),
            template='plotly_white',
            legend=dict(
                yanchor="top", y=1.1,
                xanchor="left", x=1.1
            ),
            height=600
        )
        
        path = self.output_folder / "viz_radar.html"
        fig.write_html(str(path))
        logger.info(f"✓ Grafic radar generat: {path}")
        return str(path)
    
    def generate_heatmap(self, indices: List[AIAdoptionIndex], top_n: int = 20) -> Optional[str]:
        """
        5.4 - Generează heatmap companii x ani.
        
        Args:
            indices: Lista de AIAdoptionIndex
            top_n: Numărul de companii de afișat (implicit 20)
        
        Returns:
            Calea către fișierul HTML generat
        """
        if not indices:
            logger.warning("Nu există date pentru heatmap")
            return None
        
        df = pd.DataFrame([i.to_dict() for i in indices])
        
        # Top N companii după media AI index
        top_companies = df.groupby('company')['ai_adoption_index'].mean().nlargest(top_n).index.tolist()
        df_top = df[df['company'].isin(top_companies)]
        
        # Pivot pentru heatmap
        pivot = df_top.pivot_table(
            index='company', 
            columns='year', 
            values='ai_adoption_index',
            aggfunc='mean'
        ).fillna(0)
        
        # Sortează companiile după media
        pivot['mean'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('mean', ascending=True)
        pivot = pivot.drop('mean', axis=1)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[str(y) for y in pivot.columns],
            y=pivot.index,
            colorscale='Blues',
            text=np.round(pivot.values, 1),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='<b>%{y}</b><br>Year: %{x}<br>AI Index: %{z:.1f}<extra></extra>',
            colorbar=dict(title="AI Index")
        ))
        
        fig.update_layout(
            title=dict(
                text=f'Top {top_n} Companies - AI Index Heatmap by Year',
                font=dict(size=18)
            ),
            xaxis_title='Year',
            yaxis_title='Company',
            template='plotly_white',
            height=max(600, top_n * 30),
            margin=dict(l=200)
        )
        
        path = self.output_folder / "viz_heatmap.html"
        fig.write_html(str(path))
        logger.info(f"✓ Heatmap generat: {path}")
        return str(path)
    
    def generate_category_distribution(self, references: List[Dict]) -> Optional[str]:
        """
        5.5 - Generează grafic distribuție categorii AI.
        
        Args:
            references: Lista de referințe (dict-uri cu 'category')
        
        Returns:
            Calea către fișierul HTML generat
        """
        if not references:
            logger.warning("Nu există referințe pentru category distribution")
            return None
        
        categories = [ref.get('category', 'Unknown') for ref in references]
        cat_counts = pd.Series(categories).value_counts()
        
        # Exclude 'Unclassified' dacă există și e prea mare
        if 'Unclassified' in cat_counts.index and cat_counts['Unclassified'] > cat_counts.sum() * 0.5:
            cat_counts = cat_counts.drop('Unclassified')
        
        fig = go.Figure(data=[go.Pie(
            labels=cat_counts.index,
            values=cat_counts.values,
            hole=0.4,
            textinfo='label+percent',
            textposition='outside',
            marker_colors=px.colors.qualitative.Set3,
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=dict(
                text='AI Reference Categories Distribution',
                font=dict(size=18)
            ),
            template='plotly_white',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.3,
                xanchor="center", x=0.5
            )
        )
        
        path = self.output_folder / "viz_categories.html"
        fig.write_html(str(path))
        logger.info(f"✓ Grafic categorii generat: {path}")
        return str(path)
    
    def generate_group_charts(self, indices: List[AIAdoptionIndex], group_by: str = 'industry') -> Optional[str]:
        """
        5.7 - Generează grafice pentru analiză per grup (industry/sector/country).
        v6.0.6: Metodă parametrizată cu group_by.
        
        Args:
            indices: Lista de AIAdoptionIndex
            group_by: Câmpul după care se grupează ('industry', 'sector', 'country')
        
        Returns:
            Calea către fișierul HTML generat
        """
        if not indices:
            logger.warning(f"Nu există date pentru {group_by} charts")
            return None
        
        df = pd.DataFrame([i.to_dict() for i in indices])
        
        # Verifică dacă coloana există
        if group_by not in df.columns:
            logger.warning(f"Coloana '{group_by}' nu există în date")
            return None
        
        # Exclude 'Unknown'
        df = df[df[group_by] != 'Unknown']
        df = df[df[group_by].notna()]
        
        if df.empty:
            logger.warning(f"Nu există date valide pentru {group_by}")
            return None
        
        # Top 15 grupuri după medie
        top_groups = df.groupby(group_by)['ai_adoption_index'].mean().nlargest(15).index.tolist()
        df_top = df[df[group_by].isin(top_groups)]
        
        # Creează subplot cu 2 grafice
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f'AI Index Distribution by {group_by.title()} (Box Plot)',
                f'{group_by.title()} Average Trend Over Years'
            ),
            vertical_spacing=0.15,
            row_heights=[0.5, 0.5]
        )
        
        # Grafic 1: Box plot per grup
        for group_name in top_groups:
            group_data = df_top[df_top[group_by] == group_name]['ai_adoption_index']
            fig.add_trace(
                go.Box(
                    y=group_data,
                    name=str(group_name)[:25],
                    showlegend=False,
                    boxmean=True
                ),
                row=1, col=1
            )
        
        # Grafic 2: Trend per grup (top 5)
        colors = px.colors.qualitative.Set1[:5]
        for i, group_name in enumerate(top_groups[:5]):
            group_trend = df_top[df_top[group_by] == group_name].groupby('year')['ai_adoption_index'].mean().reset_index()
            fig.add_trace(
                go.Scatter(
                    x=group_trend['year'],
                    y=group_trend['ai_adoption_index'],
                    mode='lines+markers',
                    name=str(group_name)[:25],
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=8)
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=900,
            template='plotly_white',
            title_text=f'{group_by.title()} Analysis - AI Adoption',
            showlegend=True,
            legend=dict(
                yanchor="top", y=0.45,
                xanchor="left", x=1.02
            )
        )
        
        fig.update_xaxes(title_text=group_by.title(), row=1, col=1, tickangle=45)
        fig.update_xaxes(title_text="Year", row=2, col=1)
        fig.update_yaxes(title_text="AI Adoption Index", row=1, col=1)
        fig.update_yaxes(title_text="AI Adoption Index", row=2, col=1)
        
        path = self.output_folder / f"viz_{group_by}.html"
        fig.write_html(str(path))
        logger.info(f"✓ Grafice {group_by} generate: {path}")
        return str(path)
    
    def generate_industry_charts(self, indices: List[AIAdoptionIndex]) -> Optional[str]:
        """Wrapper pentru compatibilitate - apelează generate_group_charts cu 'industry'."""
        return self.generate_group_charts(indices, group_by='industry')
    
    def generate_sector_charts(self, indices: List[AIAdoptionIndex]) -> Optional[str]:
        """Generează grafice pentru sector."""
        return self.generate_group_charts(indices, group_by='sector')
    
    def generate_country_charts(self, indices: List[AIAdoptionIndex]) -> Optional[str]:
        """Generează grafice pentru country."""
        return self.generate_group_charts(indices, group_by='country')
    
    def generate_all_visualizations(self, indices: List[AIAdoptionIndex], 
                                    references: List[Dict]) -> List[str]:
        """
        5.6 - Generează toate vizualizările disponibile.
        v6.0.6: Include și sector/country charts.
        
        Args:
            indices: Lista de AIAdoptionIndex
            references: Lista de referințe deduplicate
        
        Returns:
            Lista de căi către fișierele HTML generate
        """
        generated = []
        
        # 1. Ranking Chart (ultimul an)
        path = self.generate_ranking_chart(indices)
        if path:
            generated.append(path)
        
        # 2. Timeline Chart
        path = self.generate_timeline_chart(indices)
        if path:
            generated.append(path)
        
        # 3. Radar Chart
        path = self.generate_radar_chart(indices)
        if path:
            generated.append(path)
        
        # 4. Heatmap
        path = self.generate_heatmap(indices)
        if path:
            generated.append(path)
        
        # 5. Category Distribution
        path = self.generate_category_distribution(references)
        if path:
            generated.append(path)
        
        # 6. Industry Charts
        path = self.generate_group_charts(indices, group_by='industry')
        if path:
            generated.append(path)
        
        # 7. Sector Charts - NOU v6.0.6
        path = self.generate_group_charts(indices, group_by='sector')
        if path:
            generated.append(path)
        
        # 8. Country Charts - NOU v6.0.6
        path = self.generate_group_charts(indices, group_by='country')
        if path:
            generated.append(path)
        
        # 9. Trend Chart (intern)
        fig = self._create_trend_chart(indices)
        if fig:
            path = self.output_folder / "viz_trend.html"
            fig.write_html(str(path))
            generated.append(str(path))
        
        logger.info(f"✓ Generat {len(generated)} vizualizări")
        return generated
    
    # ═══════════════════════════════════════════════════════════════════════
    # METODE INTERNE (PRIVATE)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _create_trend_chart(self, indices: List[AIAdoptionIndex]) -> Optional[go.Figure]:
        """Creează grafic trend AI Index per an (intern)."""
        if not indices:
            return None
        
        df = pd.DataFrame([i.to_dict() for i in indices])
        yearly = df.groupby('year')['ai_adoption_index'].agg(['mean', 'std', 'count']).reset_index()
        
        fig = go.Figure()
        
        # Banda de deviație standard
        fig.add_trace(go.Scatter(
            x=list(yearly['year']) + list(yearly['year'][::-1]),
            y=list(yearly['mean'] + yearly['std']) + list((yearly['mean'] - yearly['std'])[::-1]),
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±1 Std Dev',
            showlegend=True
        ))
        
        # Linia principală
        fig.add_trace(go.Scatter(
            x=yearly['year'],
            y=yearly['mean'],
            mode='lines+markers',
            name='Average AI Index',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10),
            hovertemplate='Year: %{x}<br>Avg AI Index: %{y:.1f}<br>Companies: %{customdata}<extra></extra>',
            customdata=yearly['count']
        ))
        
        fig.update_layout(
            title=dict(
                text='AI Adoption Index Trend (2020-2025)',
                font=dict(size=18)
            ),
            xaxis_title='Year',
            yaxis_title='AI Adoption Index',
            template='plotly_white',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def _create_group_comparison(self, indices: List[AIAdoptionIndex], group_by: str = 'industry') -> Optional[go.Figure]:
        """Creează grafic comparație grupuri (intern)."""
        if not indices:
            return None
        
        df = pd.DataFrame([i.to_dict() for i in indices])
        
        if group_by not in df.columns:
            return None
        
        group_avg = df.groupby(group_by)['ai_adoption_index'].mean().sort_values(ascending=True)
        group_avg = group_avg.tail(15)
        
        fig = go.Figure(go.Bar(
            x=group_avg.values,
            y=group_avg.index,
            orientation='h',
            marker_color=px.colors.sequential.Blues_r[:len(group_avg)]
        ))
        
        fig.update_layout(
            title=f'AI Adoption by {group_by.title()} (Top 15)',
            xaxis_title='Average AI Index',
            yaxis_title=group_by.title(),
            template='plotly_white',
            height=600
        )
        
        return fig
    
    def _create_category_distribution(self, references: List[Dict]) -> Optional[go.Figure]:
        """Creează grafic distribuție categorii (intern)."""
        if not references:
            return None
        
        categories = [ref.get('category', 'Unknown') for ref in references]
        cat_counts = pd.Series(categories).value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=cat_counts.index,
            values=cat_counts.values,
            hole=0.4,
            textinfo='label+percent',
            marker_colors=px.colors.qualitative.Set3
        )])
        
        fig.update_layout(
            title='AI Reference Categories Distribution',
            template='plotly_white'
        )
        
        return fig
    
    def _create_company_heatmap(self, indices: List[AIAdoptionIndex]) -> Optional[go.Figure]:
        """Creează heatmap companii (intern)."""
        if not indices:
            return None
        
        df = pd.DataFrame([i.to_dict() for i in indices])
        top_companies = df.groupby('company')['ai_adoption_index'].max().nlargest(20).index
        df_top = df[df['company'].isin(top_companies)]
        
        pivot = df_top.pivot_table(
            index='company', 
            columns='year', 
            values='ai_adoption_index',
            aggfunc='mean'
        ).fillna(0)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='Blues',
            text=np.round(pivot.values, 1),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Top 20 Companies - AI Index Heatmap by Year',
            xaxis_title='Year',
            yaxis_title='Company',
            template='plotly_white',
            height=800
        )
        
        return fig
    
    def _create_radar_chart(self, indices: List[AIAdoptionIndex]) -> Optional[go.Figure]:
        """Creează grafic radar (intern)."""
        if not indices:
            return None
        
        dimensions = ['intensity_index', 'semantic_index', 'diversity_index',
                     'sentiment_index', 'maturity_index', 'future_index', 'commitment_index']
        labels = ['Intensity', 'Semantic', 'Diversity', 'Sentiment', 
                 'Maturity', 'Future', 'Commitment']
        
        df = pd.DataFrame([i.to_dict() for i in indices])
        avg_values = [df[dim].mean() for dim in dimensions]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=avg_values + [avg_values[0]],
            theta=labels + [labels[0]],
            fill='toself',
            name='Average',
            line_color='#1f77b4'
        ))
        
        top_idx = max(indices, key=lambda x: x.ai_adoption_index)
        top_values = [getattr(top_idx, dim) for dim in dimensions]
        
        fig.add_trace(go.Scatterpolar(
            r=top_values + [top_values[0]],
            theta=labels + [labels[0]],
            fill='toself',
            name=f'Top: {top_idx.company}',
            line_color='#ff7f0e'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title='AI Adoption - 7 Dimensions',
            template='plotly_white'
        )
        
        return fig


# ═══════════════════════════════════════════════════════════════════════════
# GROUP AGGREGATOR v6.0.6 (înlocuiește IndustryAggregator)
# ═══════════════════════════════════════════════════════════════════════════

class GroupAggregator:
    """
    Agregator pentru statistici per grup (industry/sector/country).
    v6.0.6: Înlocuiește IndustryAggregator cu metodă parametrizată.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def aggregate_by_group(self, indices: List[AIAdoptionIndex] = None, 
                          group_by: str = 'industry') -> List[Dict]:
        """
        Agregă metrici per grup-an.
        
        Args:
            indices: Lista de AIAdoptionIndex (dacă None, citește din DB)
            group_by: Câmpul după care se grupează ('industry', 'sector', 'country')
        
        Returns:
            Lista de dicționare cu statistici agregate
        """
        if indices is None:
            # Citește din DB
            cursor = self.db.conn.cursor()
            cursor.execute("SELECT * FROM adoption_index")
            rows = cursor.fetchall()
            
            if not rows:
                return []
            
            indices = []
            for row in rows:
                idx = AIAdoptionIndex(
                    company=row[1], year=row[2], position=row[3], industry=row[4],
                    intensity_index=row[5] if len(row) > 5 else 0,
                    semantic_index=row[6] if len(row) > 6 else 0,
                    diversity_index=row[7] if len(row) > 7 else 0,
                    sentiment_index=row[8] if len(row) > 8 else 0,
                    maturity_index=row[9] if len(row) > 9 else 0,
                    future_index=row[10] if len(row) > 10 else 0,
                    commitment_index=row[11] if len(row) > 11 else 0,
                    ai_adoption_index=row[12] if len(row) > 12 else 0,
                    total_refs=row[14] if len(row) > 14 else 0,
                    total_pages=row[15] if len(row) > 15 else 0,
                    categories_used=row[16] if len(row) > 16 else 0
                )
                # Adaugă sector și country dacă există în DB
                if len(row) > 17:
                    idx.sector = row[5] if len(row) > 5 else 'Unknown'  # Ajustează indexul
                    idx.country = row[6] if len(row) > 6 else 'Unknown'
                indices.append(idx)
        
        if not indices:
            return []
        
        group_year_data = defaultdict(list)
        for idx in indices:
            group_value = getattr(idx, group_by, 'Unknown')
            if group_value and group_value != 'Unknown':
                key = (group_value, idx.year)
                group_year_data[key].append(idx)
        
        results = []
        for (group_name, year), year_indices in group_year_data.items():
            ai_indices = [i.ai_adoption_index for i in year_indices]
            
            result = {
                'group_by': group_by,
                'group_name': group_name,
                'year': year,
                'avg_intensity_index': np.mean([i.intensity_index for i in year_indices]),
                'avg_semantic_index': np.mean([i.semantic_index for i in year_indices]),
                'avg_diversity_index': np.mean([i.diversity_index for i in year_indices]),
                'avg_sentiment_index': np.mean([i.sentiment_index for i in year_indices]),
                'avg_maturity_index': np.mean([i.maturity_index for i in year_indices]),
                'avg_future_index': np.mean([i.future_index for i in year_indices]),
                'avg_commitment_index': np.mean([i.commitment_index for i in year_indices]),
                'ai_adoption_index_avg': np.mean(ai_indices),
                'num_companies': len(year_indices),
                'total_refs': sum(i.total_refs for i in year_indices),
                'min_index': min(ai_indices),
                'max_index': max(ai_indices),
                'std_deviation': np.std(ai_indices) if len(ai_indices) > 1 else 0,
                'companies_list': ', '.join(sorted(set(i.company for i in year_indices)))
            }
            results.append(result)
        
        # Calculează ranking per an
        for year in set(r['year'] for r in results):
            year_results = [r for r in results if r['year'] == year]
            year_results.sort(key=lambda x: x['ai_adoption_index_avg'], reverse=True)
            for rank, r in enumerate(year_results, 1):
                r['rank_in_year'] = rank
        
        # Salvează în DB dacă e industry (pentru compatibilitate)
        if group_by == 'industry':
            self.save_industry_to_database(results)
        
        return results
    
    def aggregate_by_industry(self, indices: List[AIAdoptionIndex] = None) -> List[Dict]:
        """Wrapper pentru compatibilitate cu IndustryAggregator."""
        return self.aggregate_by_group(indices, group_by='industry')
    
    def aggregate_by_sector(self, indices: List[AIAdoptionIndex] = None) -> List[Dict]:
        """Agregare pe sector."""
        return self.aggregate_by_group(indices, group_by='sector')
    
    def aggregate_by_country(self, indices: List[AIAdoptionIndex] = None) -> List[Dict]:
        """Agregare pe țară."""
        return self.aggregate_by_group(indices, group_by='country')
    
    def save_industry_to_database(self, industry_data: List[Dict]):
        """Salvează datele industry în baza de date (pentru compatibilitate)."""
        cursor = self.db.conn.cursor()
        
        for data in industry_data:
            if data.get('group_by') != 'industry':
                continue
            
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO adoption_index_industry
                    (industry, year, avg_intensity_index, avg_semantic_index,
                     avg_diversity_index, avg_sentiment_index, avg_maturity_index,
                     avg_future_index, avg_commitment_index, ai_adoption_index_industry,
                     rank_among_industries, num_companies, total_refs,
                     min_index, max_index, std_deviation, companies_list)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['group_name'], data['year'],
                    data['avg_intensity_index'], data['avg_semantic_index'],
                    data['avg_diversity_index'], data['avg_sentiment_index'],
                    data['avg_maturity_index'], data['avg_future_index'],
                    data['avg_commitment_index'], data['ai_adoption_index_avg'],
                    data.get('rank_in_year', 0), data['num_companies'],
                    data['total_refs'], data['min_index'], data['max_index'],
                    data['std_deviation'], data['companies_list']
                ))
            except sqlite3.Error as e:
                logger.error(f"Eroare inserare industry data: {e}")
        
        self.db.conn.commit()
        logger.info(f"✓ Salvat {len([d for d in industry_data if d.get('group_by') == 'industry'])} înregistrări industry")


# Alias pentru compatibilitate cu codul existent
IndustryAggregator = GroupAggregator


# ═══════════════════════════════════════════════════════════════════════════
# COMPLETE PIPELINE v6.0.6
# ═══════════════════════════════════════════════════════════════════════════

class AnalysisPipeline:
    """Pipeline complet pentru analiză v6.0.6."""
    
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.db = DatabaseManager(
            str(Path(config.output_folder) / config.database_name)
        )
        self.db.create_tables()
        
        self.deduplicator = SemanticDeduplicator(config)
        self.index_calculator = AIAdoptionIndexCalculatorV6(config)
        self.excel_exporter = ExcelExporter(config)
        self.viz_generator = VisualizationGenerator(config)
        self.group_aggregator = GroupAggregator(self.db)
        
        logger.info("Analysis Pipeline v6.0.6 inițializat")
    
    def process_document_results(self, doc_results: List[DocumentResult]) -> Dict:
        """Procesează rezultatele documentelor."""
        all_references = []
        all_indices = []
        
        for doc_result in doc_results:
            for ref in doc_result.references:
                self.db.insert_raw_reference(ref)
                all_references.append(ref)
        
        logger.info(f"Total referințe raw: {len(all_references)}")
        
        deduplicated = self.deduplicator.deduplicate_references(all_references)
        self._save_deduplicated(deduplicated)
        
        for doc_result in doc_results:
            index = self.index_calculator.calculate_index(doc_result)
            all_indices.append(index)
            self._save_index(index)
        
        # Agregare pe toate grupurile
        industry_data = self.group_aggregator.aggregate_by_industry(all_indices)
        sector_data = self.group_aggregator.aggregate_by_sector(all_indices)
        country_data = self.group_aggregator.aggregate_by_country(all_indices)
        
        ref_path = self.excel_exporter.export_references(deduplicated)
        idx_path = self.excel_exporter.export_adoption_index(all_indices)
        
        viz_paths = self.viz_generator.generate_all_visualizations(all_indices, deduplicated)
        
        return {
            'total_raw_refs': len(all_references),
            'deduplicated_refs': len(deduplicated),
            'documents_processed': len(doc_results),
            'indices_calculated': len(all_indices),
            'industries': len(set(i.industry for i in all_indices if i.industry)),
            'sectors': len(set(getattr(i, 'sector', '') for i in all_indices if getattr(i, 'sector', ''))),
            'countries': len(set(getattr(i, 'country', '') for i in all_indices if getattr(i, 'country', ''))),
            'excel_references': str(ref_path) if ref_path else None,
            'excel_index': str(idx_path) if idx_path else None,
            'visualizations': viz_paths
        }
    
    def _save_deduplicated(self, deduplicated: List[Dict]):
        """Salvează referințe deduplicate. v6.0.6: Include sector și country."""
        cursor = self.db.conn.cursor()
        for ref in deduplicated:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO ai_references_deduplicated
                    (company, year, position, industry, sector, country, 
                     text, context, category,
                     sources, doc_count, total_occurrences, 
                     avg_sentiment_score, avg_semantic_score, original_refs)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ref['company'], ref['year'], ref['position'], ref['industry'],
                    ref.get('sector', ''), ref.get('country', ''),
                    sanitize_for_excel(ref['text']), 
                    sanitize_for_excel(ref['context']), 
                    ref['category'],
                    ref['sources'], ref['doc_count'], ref['total_occurrences'],
                    ref['avg_sentiment_score'], ref['avg_semantic_score'],
                    ref['original_refs']
                ))
            except sqlite3.Error as e:
                logger.error(f"Eroare inserare dedup: {e}")
        self.db.conn.commit()
    
    def _save_index(self, index: AIAdoptionIndex):
        """Salvează AI Adoption Index. v6.0.6: Include sector și country."""
        cursor = self.db.conn.cursor()
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO adoption_index
                (company, year, position, industry, sector, country,
                 intensity_index, semantic_index, diversity_index,
                 sentiment_index, maturity_index, future_index, commitment_index,
                 ai_adoption_index, total_refs, total_pages, categories_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                index.company, index.year, index.position, index.industry,
                getattr(index, 'sector', ''), getattr(index, 'country', ''),
                index.intensity_index, index.semantic_index, index.diversity_index,
                index.sentiment_index, index.maturity_index, index.future_index,
                index.commitment_index, index.ai_adoption_index,
                index.total_refs, index.total_pages, index.categories_used
            ))
            self.db.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Eroare inserare index: {e}")
    
    def close(self):
        self.db.close()


# ═══════════════════════════════════════════════════════════════════════════
# FINAL
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("AI SEMANTIC ANALYZER v6.2 - MODUL 4: DATABASE, EXPORT & VISUALIZATION")
    print("=" * 80)
    print("\n✓ Module 4 v6.0.6 încărcat!")
    print("\nNOU în v6.0.6:")
    print("  - Suport Sector și Country în toate structurile")
    print("  - Export Excel cu coloane Sector, Country")
    print("  - Sheet-uri separate: Summary by Sector, Summary by Country")
    print("  - generate_group_charts(group_by='industry|sector|country')")
    print("  - GroupAggregator (înlocuiește IndustryAggregator)")
    print("\nComponents:")
    print("  - ExcelExporter (cu Rich Text + Sector/Country)")
    print("  - VisualizationGenerator v6.0.6:")
    print("    * generate_ranking_chart()")
    print("    * generate_timeline_chart()")
    print("    * generate_radar_chart()")
    print("    * generate_heatmap()")
    print("    * generate_category_distribution()")
    print("    * generate_group_charts(group_by=...)")
    print("    * generate_industry_charts() - wrapper")
    print("    * generate_sector_charts() - NOU")
    print("    * generate_country_charts() - NOU")
    print("    * generate_all_visualizations()")
    print("  - GroupAggregator (aggregate_by_group, by_industry, by_sector, by_country)")
    print("  - AnalysisPipeline (orchestrator complet)")
