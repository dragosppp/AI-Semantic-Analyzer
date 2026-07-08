"""
═══════════════════════════════════════════════════════════════════════════════
AI SEMANTIC ANALYZER - DATABASE, EXPORT & VISUALIZATION
═══════════════════════════════════════════════════════════════════════════════

Database management, multi-format export, and interactive visualization
generation.

COMPONENTS:
    1. Database Management
       - SQLite operations (insert, update, query)
       - Schema validation and transaction management

    2. Excel Export
       - Raw and deduplicated reference sheets
       - Statistical summary and category-distribution sheets
       - Company / sector / country comparison sheets
       - Provenance Metadata sheet on every workbook
       - Formatted tables with auto-width and AI-term highlighting

    3. JSON Export
       - Structured JSON with nested company/year/category data

    4. Interactive Visualizations (Plotly)
       - Temporal analysis (references over time, growth rates, trends)
       - Category analysis (distribution, Applications vs Technologies,
         EU_Semantics sunburst, taxonomy comparison)
       - Company analysis (rankings, comparisons, sector benchmarking)
       - Sector / country roll-ups and heatmaps

WORKFLOW:
    Data -> Database storage -> Statistical processing -> Visualization
    -> Multi-format export (Excel / JSON / HTML)

Author: TeRa0
Part of: AI Semantic Analyzer
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import sqlite3
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

# Openpyxl for advanced formatting
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.cell.rich_text import TextBlock, CellRichText
from openpyxl.cell.text import InlineFont

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from _02_core import (
    logger, AnalyzerConfig, AIReference, DocumentResult,
    AIAdoptionIndex, DatabaseManager, AI_CATEGORIES
)
from _04_analysis import SemanticDeduplicator, AIAdoptionIndexCalculator


# ═══════════════════════════════════════════════════════════════════════════
# PROVENANCE STAMPING
# ═══════════════════════════════════════════════════════════════════════════
# Every Excel workbook produced by ExcelExporter gets a Metadata sheet with the
# deterministic seed, generation timestamp, and (best-effort) git commit hash,
# so anyone can open an output file and verify how it was produced.

def _get_git_commit() -> str:
    """Best-effort git commit hash for the module directory. '' if not a repo."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).parent),
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return ""


def _build_metadata_rows() -> list[tuple[str, str]]:
    """Returns (key, value) rows for the Metadata sheet."""
    try:
        from main import SEED as _seed
    except ImportError:
        _seed = ""
    return [
        ("analysis_types",   "classic+eu_semantics"),
        ("seed",             str(_seed)),
        ("generated_at",     datetime.now().isoformat(timespec="seconds")),
        ("git_commit",       _get_git_commit()),
    ]


def _add_metadata_sheet(wb) -> None:
    """Add a Metadata sheet to a workbook with provenance stamps."""
    if "Metadata" in wb.sheetnames:
        del wb["Metadata"]
    ws = wb.create_sheet("Metadata")
    ws["A1"] = "Field"
    ws["B1"] = "Value"
    ws["A1"].font = Font(bold=True)
    ws["B1"].font = Font(bold=True)
    for i, (k, v) in enumerate(_build_metadata_rows(), start=2):
        ws[f"A{i}"] = k
        ws[f"B{i}"] = v
    ws.column_dimensions["A"].width = 20
    ws.column_dimensions["B"].width = 50


# ═══════════════════════════════════════════════════════════════════════════
# TEXT SANITIZATION & RICH TEXT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

# Regex for illegal characters in Excel
ILLEGAL_CHARACTERS_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f￾￿�]')

# Pattern for AI markers: >>>term<<< or <font color="red"><b>term</b></font>
AI_MARKER_PATTERN = re.compile(r'>>>(.*?)<<<|<font color="red"><b>(.*?)</b></font>', re.IGNORECASE)

# Special characters to replace
CHAR_REPLACEMENTS = {
    '•': '-', '●': '-', '○': '-', '▪': '-', '►': '>',
    '→': '->', '✓': '[x]', '✗': '[_]', '–': '-', '—': '-',
    '"': '"', '"': '"', ''': "'", ''': "'", '…': '...',
    '　': ' ',  # Full-width space
}

def sanitize_for_excel(text: str) -> str:
    """Sanitize text for Excel, PRESERVING AI markers."""
    if text is None:
        return ''
    if not isinstance(text, str):
        text = str(text)

    # Replace special characters
    for char, replacement in CHAR_REPLACEMENTS.items():
        text = text.replace(char, replacement)

    # Remove illegal characters
    text = ILLEGAL_CHARACTERS_RE.sub('', text)

    # Remove multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    # Truncate if too long
    if len(text) > 32000:
        text = text[:32000] + '...'

    return text.strip()


def create_rich_text_cell(text: str, ai_term_font: InlineFont) -> CellRichText:
    """
    Create a cell with rich text - AI terms marked with >>><<< become BOLD + RED.

    Input: "Normal text >>>AI term<<< more text"
    Output: CellRichText with "AI term" formatted bold+red
    """
    if not text or not isinstance(text, str):
        return text

    # Sanitize first
    text = sanitize_for_excel(text)

    # Check if markers exist
    if '>>>' not in text and '<font' not in text:
        return text  # Return plain text without markers

    parts = []
    last_end = 0

    for match in AI_MARKER_PATTERN.finditer(text):
        # Text before match (normal)
        if match.start() > last_end:
            normal_text = text[last_end:match.start()]
            if normal_text:
                parts.append(normal_text)

        # AI term (bold + red)
        ai_term = match.group(1) or match.group(2)
        if ai_term:
            parts.append(TextBlock(ai_term_font, ai_term))

        last_end = match.end()

    # Text after last match
    if last_end < len(text):
        remaining = text[last_end:]
        if remaining:
            parts.append(remaining)

    # If we have formatted parts, return CellRichText
    if parts and any(isinstance(p, TextBlock) for p in parts):
        return CellRichText(*parts)

    # Otherwise, clean markers and return plain text
    clean_text = AI_MARKER_PATTERN.sub(r'\1\2', text)
    return clean_text


def strip_ai_markers(text: str) -> str:
    """Remove AI markers and return clean text."""
    if not text:
        return ''
    text = sanitize_for_excel(text)
    return AI_MARKER_PATTERN.sub(r'\1\2', text)


# ═══════════════════════════════════════════════════════════════════════════
# EXCEL EXPORTER - WITH SECTOR, COUNTRY AND RICH TEXT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════

class ExcelExporter:
    """
    Excel exporter with advanced formatting and Rich Text.
    AI terms are highlighted with BOLD + RED in Excel cells.
    Includes Sector and Country in the export.
    """

    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.output_folder = Path(config.output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Excel styles
        self.header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
        self.header_font = Font(bold=True, color="FFFFFF", size=11)
        self.border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )

        # Font for AI terms in rich text: BOLD + RED
        self.ai_term_font = InlineFont(b=True, color="FF0000")

    def export_references(self, references: list[dict], filename: str = "ai_references.xlsx"):
        """
        Export references with Rich Text formatting.
        Includes Sector and Country.
        - Text column: PLAIN text (no formatting)
        - Context column: AI terms are BOLD + RED
        """
        if not references:
            logger.warning("No references to export")
            return None

        filepath = self.output_folder / filename
        wb = Workbook()
        ws = wb.active
        ws.title = "AI References"

        # Headers - with Sector and Country
        headers = ['Company', 'Year', 'Position', 'Industry', 'Sector', 'Country',
                   'Category A', 'Category B', 'Strength', 'Confidence', 'Reasons', 'Source', 'Text', 'Context',
                   'Sources', 'Doc Count', 'Occurrences', 'Avg Sentiment', 'Avg Semantic', 'Avg Confidence',
                   'EU Domain', 'EU Subdomain', 'EU Conf']

        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.fill = self.header_fill
            cell.font = self.header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = self.border

        # Data rows
        for row_idx, ref in enumerate(references, 2):
            # Simple text columns (sanitized)
            ws.cell(row=row_idx, column=1, value=sanitize_for_excel(ref.get('company', ''))).border = self.border
            ws.cell(row=row_idx, column=2, value=ref.get('year', '')).border = self.border
            ws.cell(row=row_idx, column=3, value=ref.get('position', '')).border = self.border
            ws.cell(row=row_idx, column=4, value=sanitize_for_excel(ref.get('industry', ''))).border = self.border
            ws.cell(row=row_idx, column=5, value=sanitize_for_excel(ref.get('sector', ''))).border = self.border
            ws.cell(row=row_idx, column=6, value=sanitize_for_excel(ref.get('country', ''))).border = self.border
            # Classic taxonomy — two independent axes (Applications / Technologies)
            ws.cell(row=row_idx, column=7, value=sanitize_for_excel(ref.get('category_a', 'none'))).border = self.border
            ws.cell(row=row_idx, column=8, value=sanitize_for_excel(ref.get('category_b', 'none'))).border = self.border

            # Strength / Confidence / Reasons
            ws.cell(row=row_idx, column=9, value=sanitize_for_excel(ref.get('reference_strength', ref.get('strength', '')))).border = self.border
            ws.cell(row=row_idx, column=10, value=round(float(ref.get('confidence_score', 0) or ref.get('confidence', 0) or 0), 3)).border = self.border
            reasons_val = sanitize_for_excel(ref.get('confidence_reasons', ref.get('reasons', '')))
            reasons_cell = ws.cell(row=row_idx, column=11, value=reasons_val)
            reasons_cell.border = self.border
            reasons_cell.alignment = Alignment(wrap_text=True, vertical='top')

            # Source - DOCUMENT NAME
            source_cell = ws.cell(row=row_idx, column=12, value=sanitize_for_excel(ref.get('source', '')))
            source_cell.border = self.border
            source_cell.alignment = Alignment(wrap_text=True, vertical='top')

            # Text - PLAIN, no formatting (strip markers)
            text_value = strip_ai_markers(ref.get('text', ''))
            text_cell = ws.cell(row=row_idx, column=13, value=text_value)
            text_cell.border = self.border
            text_cell.font = Font(color="000000")  # Plain black
            text_cell.alignment = Alignment(wrap_text=True, vertical='top')

            # Context - with RICH TEXT (AI terms bold+red)
            context_value = ref.get('context', '')
            if context_value:
                rich_context = create_rich_text_cell(context_value, self.ai_term_font)
                context_cell = ws.cell(row=row_idx, column=14, value=rich_context)
            else:
                context_cell = ws.cell(row=row_idx, column=14, value='')
            context_cell.border = self.border
            context_cell.alignment = Alignment(wrap_text=True, vertical='top')

            # Sources and numeric fields
            sources_val = ref.get('sources_files') or ref.get('sources') or ref.get('source') or ''
            ws.cell(row=row_idx, column=15, value=sanitize_for_excel(sources_val)).border = self.border
            ws.cell(row=row_idx, column=16, value=ref.get('doc_count', 0)).border = self.border
            ws.cell(row=row_idx, column=17, value=ref.get('total_occurrences', 0)).border = self.border
            ws.cell(row=row_idx, column=18, value=round(ref.get('avg_sentiment_score', 0), 3)).border = self.border
            ws.cell(row=row_idx, column=19, value=round(ref.get('avg_semantic_score', 0), 3)).border = self.border
            ws.cell(row=row_idx, column=20, value=round(ref.get('avg_confidence_score', ref.get('avg_confidence', 0) or 0), 3)).border = self.border

            # EU_Semantics classification
            ws.cell(row=row_idx, column=21, value=sanitize_for_excel(ref.get('eu_domain', ''))).border = self.border
            ws.cell(row=row_idx, column=22, value=sanitize_for_excel(ref.get('eu_subdomain', ''))).border = self.border
            ws.cell(row=row_idx, column=23, value=round(float(ref.get('eu_confidence', 0) or 0), 3)).border = self.border

        # Adjust column widths
        ws.column_dimensions['A'].width = 25  # Company
        ws.column_dimensions['B'].width = 8   # Year
        ws.column_dimensions['C'].width = 10  # Position
        ws.column_dimensions['D'].width = 20  # Industry
        ws.column_dimensions['E'].width = 20  # Sector
        ws.column_dimensions['F'].width = 15  # Country
        ws.column_dimensions['G'].width = 22  # Category A
        ws.column_dimensions['H'].width = 22  # Category B
        ws.column_dimensions['I'].width = 14  # Strength
        ws.column_dimensions['J'].width = 12  # Confidence
        ws.column_dimensions['K'].width = 45  # Reasons
        ws.column_dimensions['L'].width = 45  # Source
        ws.column_dimensions['M'].width = 40  # Text
        ws.column_dimensions['N'].width = 80  # Context
        ws.column_dimensions['O'].width = 30  # Sources
        ws.column_dimensions['P'].width = 10  # Doc count
        ws.column_dimensions['Q'].width = 12  # Occurrences
        ws.column_dimensions['R'].width = 12  # Sentiment
        ws.column_dimensions['S'].width = 12  # Semantic
        ws.column_dimensions['T'].width = 14  # Avg Confidence
        ws.column_dimensions['U'].width = 26  # EU Domain
        ws.column_dimensions['V'].width = 42  # EU Subdomain
        ws.column_dimensions['W'].width = 10  # EU Conf

        # Freeze header
        ws.freeze_panes = 'A2'

        # Row heights for context
        for row in range(2, len(references) + 2):
            ws.row_dimensions[row].height = 80

        _add_metadata_sheet(wb)
        wb.save(filepath)
        logger.info(f"References export with Rich Text: {filepath}")
        return filepath

    def export_adoption_index(self, indices: list[AIAdoptionIndex],
                             filename: str = "ai_adoption_index.xlsx"):
        """Export AI Adoption Index with formatting. Includes Sector and Country."""
        if not indices:
            logger.warning("No indices to export")
            return None

        filepath = self.output_folder / filename
        wb = Workbook()

        # Sheet 1: Full index
        ws = wb.active
        ws.title = "AI Adoption Index"

        # Headers - with Sector and Country
        headers = ['Company', 'Year', 'Position', 'Industry', 'Sector', 'Country',
                   'Intensity', 'Semantic', 'Diversity', 'Sentiment',
                   'Maturity', 'Future', 'Commitment', 'AI Index',
                   'Total Refs', 'Pages', 'Categories',
                   'Diversity (EU)', 'AI Index (EU)', 'Categories (EU)']

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

            # Highlight AI Index with colors
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

            # EU_Semantics parallel scoring
            ws.cell(row=row_idx, column=18, value=round(getattr(idx, 'diversity_index_eu', 0.0), 3)).border = self.border
            eu_index_cell = ws.cell(row=row_idx, column=19, value=round(getattr(idx, 'ai_adoption_index_eu', 0.0), 2))
            eu_index_cell.border = self.border
            eu_index_cell.font = Font(bold=True)
            ws.cell(row=row_idx, column=20, value=getattr(idx, 'categories_used_eu', 0)).border = self.border

        ws.freeze_panes = 'A2'

        # Sheet 2: Summary by Year
        ws2 = wb.create_sheet("Summary by Year")
        self._create_year_summary(ws2, indices)

        # Sheet 3: Summary by Industry
        ws3 = wb.create_sheet("Summary by Industry")
        self._create_group_summary(ws3, indices, group_by='industry')

        # Sheet 4: Summary by Sector
        ws4 = wb.create_sheet("Summary by Sector")
        self._create_group_summary(ws4, indices, group_by='sector')

        # Sheet 5: Summary by Country
        ws5 = wb.create_sheet("Summary by Country")
        self._create_group_summary(ws5, indices, group_by='country')

        _add_metadata_sheet(wb)
        wb.save(filepath)
        logger.info(f"AI Adoption Index export: {filepath}")
        return filepath

    def export_eu_classification(self, references: list[dict],
                                 filename: str = "eu_classification.xlsx"):
        """Export the EU_Semantics (JRC AI Watch) classification distribution:
        per-domain and per-subdomain reference counts and shares."""
        if not references:
            logger.warning("No references to export for EU classification")
            return None

        dom_counts: dict[str, int] = {}
        sub_counts: dict[tuple[str, str], int] = {}
        for ref in references:
            dom = ref.get('eu_domain', 'Unclassified') or 'Unclassified'
            sub = ref.get('eu_subdomain', 'Unclassified') or 'Unclassified'
            dom_counts[dom] = dom_counts.get(dom, 0) + 1
            sub_counts[(dom, sub)] = sub_counts.get((dom, sub), 0) + 1
        total = sum(dom_counts.values())

        filepath = self.output_folder / filename
        wb = Workbook()

        # Sheet 1: by domain
        ws = wb.active
        ws.title = "EU Domains"
        for col, header in enumerate(['EU Domain', 'References', 'Share %'], 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.fill = self.header_fill
            cell.font = self.header_font
            cell.border = self.border
        for r, (dom, cnt) in enumerate(
                sorted(dom_counts.items(), key=lambda kv: kv[1], reverse=True), 2):
            ws.cell(row=r, column=1, value=dom).border = self.border
            ws.cell(row=r, column=2, value=cnt).border = self.border
            ws.cell(row=r, column=3,
                    value=round(100 * cnt / total, 2) if total else 0).border = self.border
        ws.column_dimensions['A'].width = 32
        ws.column_dimensions['B'].width = 12
        ws.column_dimensions['C'].width = 10
        ws.freeze_panes = 'A2'

        # Sheet 2: by subdomain
        ws2 = wb.create_sheet("EU Subdomains")
        for col, header in enumerate(['EU Domain', 'EU Subdomain', 'References', 'Share %'], 1):
            cell = ws2.cell(row=1, column=col, value=header)
            cell.fill = self.header_fill
            cell.font = self.header_font
            cell.border = self.border
        for r, ((dom, sub), cnt) in enumerate(
                sorted(sub_counts.items(), key=lambda kv: kv[1], reverse=True), 2):
            ws2.cell(row=r, column=1, value=dom).border = self.border
            ws2.cell(row=r, column=2, value=sub).border = self.border
            ws2.cell(row=r, column=3, value=cnt).border = self.border
            ws2.cell(row=r, column=4,
                     value=round(100 * cnt / total, 2) if total else 0).border = self.border
        ws2.column_dimensions['A'].width = 32
        ws2.column_dimensions['B'].width = 45
        ws2.column_dimensions['C'].width = 12
        ws2.column_dimensions['D'].width = 10
        ws2.freeze_panes = 'A2'

        _add_metadata_sheet(wb)
        wb.save(filepath)
        logger.info(f"EU classification export: {filepath}")
        return filepath

    def _create_year_summary(self, ws, indices: list[AIAdoptionIndex]):
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

    def _create_group_summary(self, ws, indices: list[AIAdoptionIndex], group_by: str = 'industry'):
        """
        Create summary sheet grouped by specified field.
        Generic method for industry/sector/country.

        Args:
            ws: Excel worksheet
            indices: List of AIAdoptionIndex
            group_by: Field to group by ('industry', 'sector', 'country')
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
# VISUALIZATION GENERATOR - WITH PARAMETERIZED GROUP_BY
# ═══════════════════════════════════════════════════════════════════════════

class VisualizationGenerator:
    """Interactive Plotly visualization generator."""

    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.output_folder = Path(config.output_folder) / "visualizations"
        self.output_folder.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC METHODS - CALLED FROM THE MENU
    # ═══════════════════════════════════════════════════════════════════════

    def generate_ranking_chart(self, indices: list[AIAdoptionIndex], year: int = None) -> str | None:
        """
        5.1 - Generate company ranking chart for a specific year.

        Args:
            indices: List of AIAdoptionIndex
            year: Year for which to generate the ranking (defaults to latest year)

        Returns:
            Path to the generated HTML file
        """
        if not indices:
            logger.warning("No data for ranking chart")
            return None

        df = pd.DataFrame([i.to_dict() for i in indices])

        if year:
            df_year = df[df['year'] == year]
        else:
            # Use the latest available year
            year = df['year'].max()
            df_year = df[df['year'] == year]

        if df_year.empty:
            logger.warning(f"No data for year {year}")
            return None

        # Top 20 companies
        df_top = df_year.nlargest(20, 'ai_adoption_index')

        # Sort for proper display (smallest at the bottom)
        df_top = df_top.sort_values('ai_adoption_index', ascending=True)

        # Use sector if available, otherwise industry
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
        logger.info(f"Ranking chart generated: {path}")
        return str(path)

    def generate_timeline_chart(self, indices: list[AIAdoptionIndex], top_n: int = 10) -> str | None:
        """
        5.2 - Generate temporal evolution chart for top N companies.

        Args:
            indices: List of AIAdoptionIndex
            top_n: Number of companies to display (default 10)

        Returns:
            Path to the generated HTML file
        """
        if not indices:
            logger.warning("No data for timeline chart")
            return None

        df = pd.DataFrame([i.to_dict() for i in indices])

        # Select top N companies by average AI index
        top_companies = df.groupby('company')['ai_adoption_index'].mean().nlargest(top_n).index.tolist()
        df_top = df[df['company'].isin(top_companies)]

        fig = go.Figure()

        # Distinct colors for each company
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
        logger.info(f"Timeline chart generated: {path}")
        return str(path)

    def generate_radar_chart(self, indices: list[AIAdoptionIndex], top_n: int = 5) -> str | None:
        """
        5.3 - Generate radar chart for the 7 dimensions.

        Args:
            indices: List of AIAdoptionIndex
            top_n: Number of companies to compare (default 5)

        Returns:
            Path to the generated HTML file
        """
        if not indices:
            logger.warning("No data for radar chart")
            return None

        dimensions = ['intensity_index', 'semantic_index', 'diversity_index',
                     'sentiment_index', 'maturity_index', 'future_index', 'commitment_index']
        labels = ['Intensity', 'Semantic', 'Diversity', 'Sentiment',
                 'Maturity', 'Future', 'Commitment']

        df = pd.DataFrame([i.to_dict() for i in indices])

        fig = go.Figure()

        # Average across all companies
        avg_values = [df[dim].mean() for dim in dimensions]
        fig.add_trace(go.Scatterpolar(
            r=avg_values + [avg_values[0]],
            theta=labels + [labels[0]],
            fill='toself',
            name='Average (All Companies)',
            line_color='#1f77b4',
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))

        # Top N companies
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
        logger.info(f"Radar chart generated: {path}")
        return str(path)

    def generate_heatmap(self, indices: list[AIAdoptionIndex], top_n: int = 20) -> str | None:
        """
        5.4 - Generate companies x years heatmap.

        Args:
            indices: List of AIAdoptionIndex
            top_n: Number of companies to display (default 20)

        Returns:
            Path to the generated HTML file
        """
        if not indices:
            logger.warning("No data for heatmap")
            return None

        df = pd.DataFrame([i.to_dict() for i in indices])

        # Top N companies by average AI index
        top_companies = df.groupby('company')['ai_adoption_index'].mean().nlargest(top_n).index.tolist()
        df_top = df[df['company'].isin(top_companies)]

        # Pivot for heatmap
        pivot = df_top.pivot_table(
            index='company',
            columns='year',
            values='ai_adoption_index',
            aggfunc='mean'
        ).fillna(0)

        # Sort companies by average
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
        logger.info(f"Heatmap generated: {path}")
        return str(path)

    def generate_category_distribution(self, references: list[dict]) -> str | None:
        """
        5.5 - Generate AI category distribution chart.

        Args:
            references: List of references (dicts with 'category')

        Returns:
            Path to the generated HTML file
        """
        if not references:
            logger.warning("No references for category distribution")
            return None

        categories = [
            cat for ref in references
            for cat in (ref.get('category_a', 'none'), ref.get('category_b', 'none'))
            if cat and cat != 'none'
        ]
        cat_counts = pd.Series(categories).value_counts()

        # Exclude 'Unclassified' if it exists and is too large
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
        logger.info(f"Categories chart generated: {path}")
        return str(path)

    def generate_eu_sunburst(self, references: list[dict]) -> str | None:
        """5.10 - EU_Semantics (JRC AI Watch) distribution as a domain → subdomain sunburst."""
        if not references:
            logger.warning("No references for EU sunburst")
            return None

        rows = []
        for ref in references:
            dom = ref.get('eu_domain', 'Unclassified') or 'Unclassified'
            sub = ref.get('eu_subdomain', 'Unclassified') or 'Unclassified'
            if dom == 'Unclassified':
                continue
            rows.append({'domain': dom, 'subdomain': sub})

        if not rows:
            logger.warning("All references are EU-Unclassified; skipping sunburst")
            return None

        df = pd.DataFrame(rows)
        agg = df.groupby(['domain', 'subdomain']).size().reset_index(name='count')

        fig = px.sunburst(
            agg, path=['domain', 'subdomain'], values='count',
            title='EU_Semantics (JRC AI Watch) — Domains & Subdomains',
            color='count', color_continuous_scale='Blues'
        )
        fig.update_layout(height=700, template='plotly_white')

        path = self.output_folder / "viz_eu_sunburst.html"
        fig.write_html(str(path))
        logger.info(f"EU sunburst generated: {path}")
        return str(path)

    def generate_taxonomy_comparison(self, indices: list[AIAdoptionIndex]) -> str | None:
        """5.11 - Compare the classic vs EU_Semantics composite AI Adoption Index per company."""
        if not indices:
            logger.warning("No indices for taxonomy comparison")
            return None

        # Latest year per company for a clean grouped bar.
        latest: dict[str, AIAdoptionIndex] = {}
        for idx in indices:
            if idx.company not in latest or idx.year > latest[idx.company].year:
                latest[idx.company] = idx

        items = sorted(latest.values(), key=lambda i: i.ai_adoption_index, reverse=True)[:20]
        if not items:
            return None
        companies = [i.company for i in items]
        classic = [round(i.ai_adoption_index, 2) for i in items]
        eu = [round(getattr(i, 'ai_adoption_index_eu', 0.0), 2) for i in items]

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Classic', x=companies, y=classic, marker_color='#2c7fb8'))
        fig.add_trace(go.Bar(name='EU_Semantics', x=companies, y=eu, marker_color='#7fcdbb'))
        fig.update_layout(
            title='AI Adoption Index — Classic vs EU_Semantics (top 20, latest year)',
            barmode='group', template='plotly_white', height=600,
            xaxis_tickangle=-45, yaxis_title='AI Adoption Index'
        )

        path = self.output_folder / "viz_taxonomy_comparison.html"
        fig.write_html(str(path))
        logger.info(f"Taxonomy comparison generated: {path}")
        return str(path)

    def generate_group_charts(self, indices: list[AIAdoptionIndex], group_by: str = 'industry') -> str | None:
        """
        5.7 - Generate charts for per-group analysis (industry/sector/country).
        Method parameterized with group_by.

        Args:
            indices: List of AIAdoptionIndex
            group_by: Field to group by ('industry', 'sector', 'country')

        Returns:
            Path to the generated HTML file
        """
        if not indices:
            logger.warning(f"No data for {group_by} charts")
            return None

        df = pd.DataFrame([i.to_dict() for i in indices])

        # Check if column exists
        if group_by not in df.columns:
            logger.warning(f"Column '{group_by}' does not exist in data")
            return None

        # Exclude 'Unknown'
        df = df[df[group_by] != 'Unknown']
        df = df[df[group_by].notna()]

        if df.empty:
            logger.warning(f"No valid data for {group_by}")
            return None

        # Top 15 groups by average
        top_groups = df.groupby(group_by)['ai_adoption_index'].mean().nlargest(15).index.tolist()
        df_top = df[df[group_by].isin(top_groups)]

        # Create subplot with 2 charts
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f'AI Index Distribution by {group_by.title()} (Box Plot)',
                f'{group_by.title()} Average Trend Over Years'
            ),
            vertical_spacing=0.15,
            row_heights=[0.5, 0.5]
        )

        # Chart 1: Box plot per group
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

        # Chart 2: Trend per group (top 5)
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
        logger.info(f"{group_by} charts generated: {path}")
        return str(path)

    def generate_industry_charts(self, indices: list[AIAdoptionIndex]) -> str | None:
        """Compatibility wrapper - calls generate_group_charts with 'industry'."""
        return self.generate_group_charts(indices, group_by='industry')

    def generate_sector_charts(self, indices: list[AIAdoptionIndex]) -> str | None:
        """Generate charts for sector."""
        return self.generate_group_charts(indices, group_by='sector')

    def generate_country_charts(self, indices: list[AIAdoptionIndex]) -> str | None:
        """Generate charts for country."""
        return self.generate_group_charts(indices, group_by='country')

    def generate_all_visualizations(self, indices: list[AIAdoptionIndex],
                                    references: list[dict]) -> list[str]:
        """
        5.6 - Generate all available visualizations.
        Includes sector/country charts as well.

        Args:
            indices: List of AIAdoptionIndex
            references: List of deduplicated references

        Returns:
            List of paths to the generated HTML files
        """
        generated = []

        # 1. Ranking Chart (latest year)
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

        # 5. Category Distribution (classic taxonomy)
        path = self.generate_category_distribution(references)
        if path:
            generated.append(path)

        # 5b. EU_Semantics sunburst (domain → subdomain)
        path = self.generate_eu_sunburst(references)
        if path:
            generated.append(path)

        # 6. Industry Charts
        path = self.generate_group_charts(indices, group_by='industry')
        if path:
            generated.append(path)

        # 7. Sector Charts
        path = self.generate_group_charts(indices, group_by='sector')
        if path:
            generated.append(path)

        # 8. Country Charts
        path = self.generate_group_charts(indices, group_by='country')
        if path:
            generated.append(path)

        # 9. Trend Chart (internal)
        fig = self._create_trend_chart(indices)
        if fig:
            path = self.output_folder / "viz_trend.html"
            fig.write_html(str(path))
            generated.append(str(path))

        # 10. Taxonomy comparison (classic vs EU_Semantics composite)
        path = self.generate_taxonomy_comparison(indices)
        if path:
            generated.append(path)

        logger.info(f"Generated {len(generated)} visualizations")
        return generated

    # ═══════════════════════════════════════════════════════════════════════
    # INTERNAL (PRIVATE) METHODS
    # ═══════════════════════════════════════════════════════════════════════

    def _create_trend_chart(self, indices: list[AIAdoptionIndex]) -> go.Figure | None:
        """Create AI Index trend chart per year (internal)."""
        if not indices:
            return None

        df = pd.DataFrame([i.to_dict() for i in indices])
        yearly = df.groupby('year')['ai_adoption_index'].agg(['mean', 'std', 'count']).reset_index()

        fig = go.Figure()

        # Standard deviation band
        fig.add_trace(go.Scatter(
            x=list(yearly['year']) + list(yearly['year'][::-1]),
            y=list(yearly['mean'] + yearly['std']) + list((yearly['mean'] - yearly['std'])[::-1]),
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='+/-1 Std Dev',
            showlegend=True
        ))

        # Main line
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

    def _create_group_comparison(self, indices: list[AIAdoptionIndex], group_by: str = 'industry') -> go.Figure | None:
        """Create group comparison chart (internal)."""
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

    def _create_category_distribution(self, references: list[dict]) -> go.Figure | None:
        """Create category distribution chart (internal)."""
        if not references:
            return None

        categories = [
            cat for ref in references
            for cat in (ref.get('category_a', 'none'), ref.get('category_b', 'none'))
            if cat and cat != 'none'
        ]
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

    def _create_company_heatmap(self, indices: list[AIAdoptionIndex]) -> go.Figure | None:
        """Create company heatmap (internal)."""
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

    def _create_radar_chart(self, indices: list[AIAdoptionIndex]) -> go.Figure | None:
        """Create radar chart (internal)."""
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
# GROUP AGGREGATOR (parameterized industry/sector/country roll-ups)
# ═══════════════════════════════════════════════════════════════════════════

class GroupAggregator:
    """
    Aggregator for per-group statistics (industry/sector/country).
    Parameterized aggregation across industry / sector / country.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def aggregate_by_group(self, indices: list[AIAdoptionIndex] = None,
                          group_by: str = 'industry') -> list[dict]:
        """
        Aggregate metrics per group-year.

        Args:
            indices: List of AIAdoptionIndex (if None, read from DB)
            group_by: Field to group by ('industry', 'sector', 'country')

        Returns:
            List of dictionaries with aggregated statistics
        """
        if indices is None:
            # Read from DB
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
                # Add sector and country if present in DB
                if len(row) > 17:
                    idx.sector = row[5] if len(row) > 5 else 'Unknown'  # Adjust the index
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

        # Compute ranking per year
        for year in set(r['year'] for r in results):
            year_results = [r for r in results if r['year'] == year]
            year_results.sort(key=lambda x: x['ai_adoption_index_avg'], reverse=True)
            for rank, r in enumerate(year_results, 1):
                r['rank_in_year'] = rank

        # Save to DB if industry (for compatibility)
        if group_by == 'industry':
            self.save_industry_to_database(results)

        return results

    def aggregate_by_industry(self, indices: list[AIAdoptionIndex] = None) -> list[dict]:
        """Compatibility wrapper for IndustryAggregator."""
        return self.aggregate_by_group(indices, group_by='industry')

    def aggregate_by_sector(self, indices: list[AIAdoptionIndex] = None) -> list[dict]:
        """Aggregate by sector."""
        return self.aggregate_by_group(indices, group_by='sector')

    def aggregate_by_country(self, indices: list[AIAdoptionIndex] = None) -> list[dict]:
        """Aggregate by country."""
        return self.aggregate_by_group(indices, group_by='country')

    def save_industry_to_database(self, industry_data: list[dict]):
        """Save industry data to the database (for compatibility)."""
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
                logger.error(f"Error inserting industry data: {e}")

        self.db.conn.commit()
        logger.info(f"Saved {len([d for d in industry_data if d.get('group_by') == 'industry'])} industry records")


# Alias for compatibility with existing code
IndustryAggregator = GroupAggregator


# ═══════════════════════════════════════════════════════════════════════════
# COMPLETE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class AnalysisPipeline:
    """Complete analysis pipeline."""

    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.db = DatabaseManager(
            str(Path(config.output_folder) / config.database_name)
        )
        self.db.create_tables()

        self.deduplicator = SemanticDeduplicator(config)
        self.index_calculator = AIAdoptionIndexCalculator(config)
        self.excel_exporter = ExcelExporter(config)
        self.viz_generator = VisualizationGenerator(config)
        self.group_aggregator = GroupAggregator(self.db)

        logger.info("Analysis Pipeline initialized")

    def process_document_results(self, doc_results: list[DocumentResult]) -> dict:
        """Process document results."""
        all_references = []
        all_indices = []

        for doc_result in doc_results:
            for ref in doc_result.references:
                self.db.insert_raw_reference(ref)
                all_references.append(ref)

        logger.info(f"Total raw references: {len(all_references)}")

        deduplicated = self.deduplicator.deduplicate_references(all_references)
        self._save_deduplicated(deduplicated)

        for doc_result in doc_results:
            index = self.index_calculator.calculate_index(doc_result)
            all_indices.append(index)
            self._save_index(index)

        # Aggregate across all groups
        industry_data = self.group_aggregator.aggregate_by_industry(all_indices)
        sector_data = self.group_aggregator.aggregate_by_sector(all_indices)
        country_data = self.group_aggregator.aggregate_by_country(all_indices)

        ref_path = self.excel_exporter.export_references(deduplicated)
        idx_path = self.excel_exporter.export_adoption_index(all_indices)
        eu_path = self.excel_exporter.export_eu_classification(deduplicated)

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
            'excel_eu_classification': str(eu_path) if eu_path else None,
            'visualizations': viz_paths
        }

    def _save_deduplicated(self, deduplicated: list[dict]):
        """Save deduplicated references. Includes sector and country."""
        cursor = self.db.conn.cursor()
        for ref in deduplicated:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO ai_references_deduplicated
                    (company, year, position, industry, sector, country,
                     text, context, category_a, category_b, eu_domain, eu_subdomain,
                     sources, doc_count, total_occurrences,
                     avg_sentiment_score, avg_semantic_score, original_refs)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ref['company'], ref['year'], ref['position'], ref['industry'],
                    ref.get('sector', ''), ref.get('country', ''),
                    sanitize_for_excel(ref['text']),
                    sanitize_for_excel(ref['context']),
                    ref.get('category_a', 'none'), ref.get('category_b', 'none'),
                    ref.get('eu_domain', 'Unclassified'), ref.get('eu_subdomain', 'Unclassified'),
                    ref['sources'], ref['doc_count'], ref['total_occurrences'],
                    ref['avg_sentiment_score'], ref['avg_semantic_score'],
                    ref['original_refs']
                ))
            except sqlite3.Error as e:
                logger.error(f"Error inserting dedup: {e}")
        self.db.conn.commit()

    def _save_index(self, index: AIAdoptionIndex):
        """Save AI Adoption Index. Includes sector and country."""
        cursor = self.db.conn.cursor()
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO adoption_index
                (company, year, position, industry, sector, country,
                 intensity_index, semantic_index, diversity_index,
                 sentiment_index, maturity_index, future_index, commitment_index,
                 ai_adoption_index, total_refs, total_pages, categories_used,
                 diversity_index_eu, ai_adoption_index_eu, categories_used_eu)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                index.company, index.year, index.position, index.industry,
                getattr(index, 'sector', ''), getattr(index, 'country', ''),
                index.intensity_index, index.semantic_index, index.diversity_index,
                index.sentiment_index, index.maturity_index, index.future_index,
                index.commitment_index, index.ai_adoption_index,
                index.total_refs, index.total_pages, index.categories_used,
                getattr(index, 'diversity_index_eu', 0.0),
                getattr(index, 'ai_adoption_index_eu', 0.0),
                getattr(index, 'categories_used_eu', 0)
            ))
            self.db.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error inserting index: {e}")

    def close(self):
        self.db.close()


# ═══════════════════════════════════════════════════════════════════════════
# FINAL
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("AI SEMANTIC ANALYZER - DATABASE, EXPORT & VISUALIZATION")
    print("=" * 80)
    print("\nModule loaded!")
    print("\nFeatures:")
    print("  - Sector and Country support in all structures")
    print("  - Excel export with Sector, Country columns")
    print("  - Separate sheets: Summary by Sector, Summary by Country")
    print("  - generate_group_charts(group_by='industry|sector|country')")
    print("  - GroupAggregator (parameterized roll-ups)")
    print("\nComponents:")
    print("  - ExcelExporter (with Rich Text + Sector/Country)")
    print("  - VisualizationGenerator:")
    print("    * generate_ranking_chart()")
    print("    * generate_timeline_chart()")
    print("    * generate_radar_chart()")
    print("    * generate_heatmap()")
    print("    * generate_category_distribution()")
    print("    * generate_group_charts(group_by=...)")
    print("    * generate_industry_charts() - wrapper")
    print("    * generate_sector_charts() - NEW")
    print("    * generate_country_charts() - NEW")
    print("    * generate_all_visualizations()")
    print("  - GroupAggregator (aggregate_by_group, by_industry, by_sector, by_country)")
    print("  - AnalysisPipeline (full orchestrator)")
