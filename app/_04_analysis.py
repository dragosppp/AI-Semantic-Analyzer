"""
═══════════════════════════════════════════════════════════════════════════════
AI SEMANTIC ANALYZER - ADVANCED ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

Statistical analysis, semantic deduplication, sentiment analysis, and the
AI Adoption Index calculation.

COMPONENTS:
    1. Statistical Analysis
       - Company-level AI adoption metrics
       - Category distribution analysis
       - AI Adoption Index calculation (classic + EU_Semantics)

    2. Semantic Deduplication
       - SentenceTransformer embeddings
       - Cosine similarity clustering (default threshold: 0.85)
       - Smart merging: preserve highest confidence

    3. Sentiment Analysis
       - FinBERT-based sentiment scoring
       - Batch processing with progress bars
       - Fallback to TextBlob if FinBERT unavailable

WORKFLOW:
    Raw data -> statistical analysis -> deduplication -> sentiment analysis
    -> AI Adoption Index

TECHNICAL NOTES:
    - Deduplication threshold: 0.85 (cosine similarity)
    - FinBERT model: ProsusAI/finbert
    - Semantic model: all-MiniLM-L6-v2 (shared with detection)
    - Thread-safe: SemanticModelLoader singleton

Author: TeRa0
Part of: AI Semantic Analyzer
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import re
import json
from collections import Counter
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

from _02_core import (
    logger, AnalyzerConfig, AIReference, DocumentResult,
    AIAdoptionIndex, AI_CATEGORIES, EU_LEAF_UNITS, FINBERT_AVAILABLE
)
from _03_detection import SemanticModelLoader


# ═══════════════════════════════════════════════════════════════════════════
# PAGE-NORMALIZATION SCALES
# ═══════════════════════════════════════════════════════════════════════════
# Calibrated so a "typical" filing produces scores in the familiar 0-1
# dimensional range. Because reports have grown longer over time, the Future
# and Commitment dimensions are expressed as per-page rates (not raw counts)
# to keep the index comparable across years.
FUTURE_PAGE_SCALE = 30.0      # 1 future ref in ~30 pages → ~1.0
COMMITMENT_PAGE_SCALE = 2.5   # weighted pattern matches per page → cap at 1.0


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS - Strength/Confidence parsing
# ═══════════════════════════════════════════════════════════════════════════
_strength_re = re.compile(r'(?:^|\|)strength=([^|]+)')
_conf_re = re.compile(r'(?:^|\|)conf=([0-9]*\.?[0-9]+)')
_reasons_re = re.compile(r'(?:^|\|)reasons=([^|]+)')

def _parse_ref_meta_from_detection_method(ref: AIReference) -> tuple[str, float, str]:
    """Fallback: extract strength/confidence/reasons from ref.detection_method if not set."""
    dm = getattr(ref, 'detection_method', '') or ''
    strength = getattr(ref, 'reference_strength', '') or ''
    conf = float(getattr(ref, 'confidence_score', 0.0) or 0.0)
    reasons = getattr(ref, 'confidence_reasons', '') or ''

    if strength in ('strong', 'medium', 'mention_only') and conf > 0:
        return strength, conf, reasons

    m = _strength_re.search(dm)
    if m and strength not in ('strong', 'medium', 'mention_only'):
        strength = m.group(1).strip()
    m = _conf_re.search(dm)
    if m and conf <= 0:
        try:
            conf = float(m.group(1))
        except:
            conf = 0.0
    m = _reasons_re.search(dm)
    if m and not reasons:
        reasons = m.group(1).strip()
    return strength or 'unknown', conf, reasons

def _is_mention_only(ref: AIReference) -> bool:
    strength = getattr(ref, 'reference_strength', '') or ''
    if strength in ('strong', 'medium'):
        return False
    if strength == 'mention_only':
        return True
    strength2, _, _ = _parse_ref_meta_from_detection_method(ref)
    return strength2 == 'mention_only'


# ═══════════════════════════════════════════════════════════════════════════
# FINBERT SENTIMENT ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class FinBERTSentimentAnalyzer:
    """FinBERT for financial sentiment (89% accuracy)."""

    def __init__(self):
        self.finbert_available = FINBERT_AVAILABLE

        if self.finbert_available:
            logger.info("Loading FinBERT...")
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch

                self.tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
                self.model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
                self.model.eval()
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(self.device)
                self.torch = torch
                logger.info(f"OK FinBERT on {self.device}")
            except Exception as e:
                logger.error(f"FinBERT error: {e}")
                self.finbert_available = False

    def analyze_sentiment(self, text: str) -> tuple[str, float]:
        if not text or len(text.strip()) < 10:
            return 'neutral', 0.0

        if self.finbert_available:
            return self._analyze_with_finbert(text)
        return self._analyze_with_textblob(text)

    def _analyze_with_finbert(self, text: str) -> tuple[str, float]:
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=512, padding=True).to(self.device)

            with self.torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            probs = self.torch.nn.functional.softmax(logits, dim=-1)[0]
            neg_score = probs[0].item()
            pos_score = probs[2].item()

            sentiment_score = pos_score - neg_score

            if sentiment_score > 0.1:
                return 'positive', sentiment_score
            elif sentiment_score < -0.1:
                return 'negative', sentiment_score
            return 'neutral', sentiment_score
        except Exception as e:
            return self._analyze_with_textblob(text)

    def _analyze_with_textblob(self, text: str) -> tuple[str, float]:
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity

            if polarity > 0.1:
                return 'positive', polarity
            elif polarity < -0.1:
                return 'negative', polarity
            return 'neutral', polarity
        except:
            return 'neutral', 0.0


# ═══════════════════════════════════════════════════════════════════════════
# SENTIMENT ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class ImprovedSentimentAnalyzer:
    """
    Sentiment analyzer with governance-context correction.

    Fix: FinBERT incorrectly classifies AI governance as "negative".
    Solution: Adjust sentiment for governance contexts.
    """

    # Governance terms - these are NOT negative!
    AI_GOVERNANCE_TERMS = [
        'responsible ai', 'ai governance', 'ai ethics', 'ai risk management',
        'ai policy', 'ai compliance', 'ai oversight', 'ai principles',
        'ethical ai', 'trustworthy ai', 'ai framework', 'ai guidelines',
        'bias mitigation', 'fairness', 'transparency', 'accountability',
        'ai safety', 'guardrails', 'model risk', 'ai audit'
    ]

    # Truly negative terms
    TRULY_NEGATIVE_TERMS = [
        'failed', 'failure', 'lawsuit', 'litigation', 'breach', 'hack',
        'layoff', 'job loss', 'replace workers', 'eliminate jobs',
        'harmful', 'dangerous', 'threat to', 'concern about',
        'security incident', 'data breach', 'privacy violation'
    ]

    def __init__(self, base_analyzer: FinBERTSentimentAnalyzer):
        self.base_analyzer = base_analyzer
        # Pre-compile with \b boundaries so short tokens like "ai" inside
        # "responsible ai" don't substring-match unrelated words.
        self._governance_patterns = [
            re.compile(r'\b' + re.escape(t.lower()) + r'\b')
            for t in self.AI_GOVERNANCE_TERMS
        ]
        self._truly_negative_patterns = [
            re.compile(r'\b' + re.escape(t.lower()) + r'\b')
            for t in self.TRULY_NEGATIVE_TERMS
        ]

    def analyze_sentiment_improved(self, context: str) -> tuple[str, float, str]:
        """
        Returns: (sentiment_label, sentiment_score, confidence_flag)
        confidence_flag: 'standard', 'governance_adjusted', 'confirmed_negative'
        """
        context_lower = context.lower()

        is_governance = any(p.search(context_lower) for p in self._governance_patterns)
        is_truly_negative = any(p.search(context_lower) for p in self._truly_negative_patterns)

        label, score = self.base_analyzer.analyze_sentiment(context)

        # Adjustments
        if is_governance and label == 'negative' and not is_truly_negative:
            adjusted_score = score * 0.3
            if adjusted_score > -0.1:
                return 'neutral', adjusted_score, 'governance_adjusted'
            return label, adjusted_score, 'governance_adjusted'

        if is_truly_negative:
            return label, score, 'confirmed_negative'

        return label, score, 'standard'


# ═══════════════════════════════════════════════════════════════════════════
# SEMANTIC DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════

class SemanticDeduplicator:
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.semantic_loader = SemanticModelLoader.get_instance()
        self.model = self.semantic_loader.model
        logger.info(f"Deduplicator: threshold={config.deduplication_threshold}")

    def deduplicate_references(self, references: list[AIReference]) -> list[dict]:
        if not references:
            return []

        contexts = [ref.context for ref in references]
        embeddings = self.model.encode(contexts, convert_to_tensor=True, show_progress_bar=False)
        similarity_matrix = cosine_similarity(embeddings.cpu().numpy())
        clusters = self._find_clusters(similarity_matrix, self.config.deduplication_threshold)

        deduplicated = []
        for cluster_indices in clusters:
            merged_ref = self._merge_cluster([references[i] for i in cluster_indices], cluster_indices)
            deduplicated.append(merged_ref)

        logger.info(f"Deduplicate: {len(references)} -> {len(deduplicated)}")
        return deduplicated

    def _find_clusters(self, similarity_matrix: np.ndarray, threshold: float) -> list[list[int]]:
        n = len(similarity_matrix)
        assigned = [False] * n
        clusters = []

        for i in range(n):
            if assigned[i]:
                continue

            cluster = set([i])
            queue = [i]
            assigned[i] = True

            while queue:
                current = queue.pop(0)
                for j in range(n):
                    if not assigned[j] and similarity_matrix[current][j] >= threshold:
                        cluster.add(j)
                        queue.append(j)
                        assigned[j] = True

            clusters.append(list(cluster))

        return clusters

    def _merge_cluster(self, refs: list[AIReference], indices: list[int]) -> dict:
        representative = max(refs, key=lambda r: (getattr(r, 'confidence_score', 0.0) or _parse_ref_meta_from_detection_method(r)[1], r.semantic_score, len(r.context)))

        sources = sorted(list(set(ref.doc_type for ref in refs)))
        sentiment_scores = [ref.sentiment_score for ref in refs]
        semantic_scores = [ref.semantic_score for ref in refs if ref.semantic_score > 0]

        return {
            'company': representative.company,
            'year': representative.year,
            'position': representative.position,
            'industry': representative.industry,
            'sector': representative.sector, #new
            'country': representative.country, #new
            'text': representative.text,
            'context': representative.context,
            'category_a': getattr(representative, 'category_a', 'none'),
            'category_b': getattr(representative, 'category_b', 'none'),
            'eu_domain': getattr(representative, 'eu_domain', 'Unclassified'),
            'eu_subdomain': getattr(representative, 'eu_subdomain', 'Unclassified'),
            'sources': ', '.join(sources),
            'doc_count': len(sources),
            'total_occurrences': len(refs),
            'avg_sentiment_score': float(np.mean(sentiment_scores)),
            'avg_semantic_score': float(np.mean(semantic_scores)) if semantic_scores else 0.0,'reference_strength': getattr(representative, 'reference_strength', _parse_ref_meta_from_detection_method(representative)[0]),
            'confidence_score': float(getattr(representative, 'confidence_score', 0.0) or _parse_ref_meta_from_detection_method(representative)[1]),
            'confidence_reasons': getattr(representative, 'confidence_reasons', _parse_ref_meta_from_detection_method(representative)[2]),
            'avg_confidence_score': float(np.mean([(getattr(r,'confidence_score',0.0) or _parse_ref_meta_from_detection_method(r)[1]) for r in refs])),
            'source': representative.source,
            'sources_files': ', '.join(sorted(list(set([r.source for r in refs if getattr(r,'source',None)])))),
            'original_refs': json.dumps(indices)
        }


# ═══════════════════════════════════════════════════════════════════════════
# MATURITY STAGE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class MaturityStageDetector:
    """Maturity detector with GenAI in all stages."""

    MATURITY_PATTERNS = {
        'exploring': {
            'score': 0.2,
            'patterns': [
                r'\b(?:evaluat|consider|explor|investigat|assess|study)\w*\s+(?:AI|artificial intelligence|GenAI|LLM)\b',
                r'\b(?:AI|GenAI)\s+(?:evaluation|feasibility|assessment)\b',
                r'\b(?:considering|exploring)\s+(?:AI|generative AI)\s+(?:opportunities|capabilities|potential)\b',
                r'\b(?:experiment|test)(?:ing)?\s+with\s+(?:AI|LLM|GenAI)\b',
            ]
        },
        'piloting': {
            'score': 0.4,
            'patterns': [
                r'\b(?:pilot|POC|proof[\s-]of[\s-]concept|trial)\w*\s+(?:AI|artificial intelligence|GenAI|LLM)\b',
                r'\b(?:AI|GenAI)\s+(?:pilot|prototype|experiment|trial)\b',
                r'\b(?:testing|piloting)\s+(?:AI|LLM|generative AI)\s+(?:solutions|use cases)\b',
                r'\binitial\s+(?:AI|GenAI)\s+(?:deployment|implementation)\b',
            ]
        },
        'implementing': {
            'score': 0.6,
            'patterns': [
                r'\b(?:implement|deploy|roll[\s-]?out|integrat)\w*\s+(?:AI|artificial intelligence|GenAI|LLM)\b',
                r'\b(?:AI|GenAI)\s+(?:implementation|deployment|integration)\b',
                r'\b(?:deploying|implementing)\s+(?:AI|GenAI|LLM)\s+(?:across|in|throughout)\b',
                r'\bproduction\s+(?:AI|GenAI|LLM)\s+(?:deployment|system)\b',
                r'\b(?:launched|released)\s+(?:AI|GenAI)[\s-]powered\b',
            ]
        },
        'scaling': {
            'score': 0.8,
            'patterns': [
                r'\b(?:scal|expand)\w*\s+(?:AI|artificial intelligence|GenAI|LLM)\b',
                r'\benterprise[\s-]wide\s+(?:AI|GenAI)\b',
                r'\b(?:AI|GenAI)\s+(?:scaling|expansion|enterprise[\s-]deployment)\b',
                r'\b(?:AI|GenAI)\s+at\s+scale\b',
                r'\b(?:hundreds|thousands)\s+of\s+(?:AI|GenAI)\s+(?:use cases|applications)\b',
            ]
        },
        'optimizing': {
            'score': 1.0,
            'patterns': [
                r'\bAI[\s-]first\b',
                r'\bAI[\s-]native\b',
                r'\b(?:AI|GenAI)[\s-]driven\s+(?:organization|company|business|enterprise)\b',
                r'\b(?:AI|GenAI)\s+(?:transformation|optimization|excellence|maturity)\b',
                r'\b(?:continuous|ongoing)\s+(?:AI|GenAI)\s+(?:improvement|innovation|optimization)\b',
                r'\b(?:AI|GenAI)\s+center\s+of\s+excellence\b',
                r'\b(?:fully|completely)\s+(?:AI|GenAI)[\s-](?:powered|enabled|integrated)\b',
            ]
        }
    }

    def __init__(self):
        self.compiled_patterns = {}
        for stage, data in self.MATURITY_PATTERNS.items():
            self.compiled_patterns[stage] = {
                'score': data['score'],
                'patterns': [re.compile(p, re.IGNORECASE) for p in data['patterns']]
            }
        logger.info("Maturity detector initialized")

    def detect_maturity_stage(self, refs: list[AIReference]) -> float:
        """
        Frequency-weighted average across maturity stages, where each ref's
        pattern matches are weighted by ref.confidence_score.

        A weighted average (rather than taking the single highest stage)
        produces a smooth maturity progression suitable for time-series
        analysis.
        """
        if not refs:
            return 0.0

        stage_counts: dict[str, float] = {stage: 0.0 for stage in self.compiled_patterns}
        for ref in refs:
            w = ref.confidence_score if ref.confidence_score > 0 else 1.0
            ctx = ref.context or ""
            for stage, data in self.compiled_patterns.items():
                hits = sum(len(p.findall(ctx)) for p in data['patterns'])
                stage_counts[stage] += hits * w

        total = sum(stage_counts.values())
        if total == 0:
            return 0.0

        return sum(
            self.compiled_patterns[s]['score'] * (c / total)
            for s, c in stage_counts.items()
        )


# ═══════════════════════════════════════════════════════════════════════════
# FUTURE ORIENTATION ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class FutureOrientationAnalyzer:
    """Forward-looking analyzer with timeline detection."""

    FUTURE_PATTERNS = [
        # Future tense verbs
        r'\bwill\s+(?:implement|deploy|invest|launch|develop|adopt|integrate|scale|expand|build)\b',
        r'\b(?:planning|plan)\s+to\s+(?:implement|deploy|invest|launch|develop|adopt)\b',
        r'\bintend(?:s)?\s+to\b',
        r'\bexpect(?:s|ed)?\s+to\b',
        r'\baim(?:s)?\s+to\b',
        r'\bcommit(?:s|ted)?\s+to\b',

        # Timelines 2025-2028
        r'\bby\s+(?:20)?2[5-8]\b',
        r'\bin\s+(?:the\s+)?(?:next|coming|following)\s+(?:year|quarter|months?)\b',
        r'\bover\s+the\s+next\s+(?:\d+\s+)?(?:years?|months?)\b',
        r'\b(?:2025|2026|2027|2028)\s+(?:goals?|targets?|objectives?|priorities?)\b',

        # Strategic planning
        r'\b(?:AI|GenAI)\s+roadmap\b',
        r'\bstrategic\s+(?:AI|GenAI)\s+plan\b',
        r'\bfuture\s+(?:AI|GenAI)\s+(?:initiatives|investments|capabilities)\b',
        r'\b(?:multi[\s-]year|long[\s-]term)\s+(?:AI|GenAI)\s+(?:strategy|investment)\b',

        # GenAI specific forward-looking
        r'\b(?:expand|scale|accelerate)\s+(?:GenAI|generative\s+AI|LLM)\s+(?:adoption|deployment)\b',
        r'\b(?:will|plan\s+to)\s+(?:deploy|implement|integrate)\s+(?:GenAI|LLM|GPT|Copilot)\b',
        r'\b(?:future|upcoming|planned)\s+(?:GenAI|LLM)\s+(?:use\s+cases|applications)\b',
    ]

    def __init__(self):
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.FUTURE_PATTERNS]
        logger.info("Future analyzer initialized")

    def analyze_future_orientation(self, refs: list[AIReference], total_pages: int) -> float:
        """
        Confidence-weighted rate of forward-looking refs per page.

        Page-normalized so longer filings don't mechanically score higher,
        and confidence-weighted so substantive mentions count more than
        boilerplate.
        """
        if not refs or total_pages <= 0:
            return 0.0

        weighted = 0.0
        for ref in refs:
            ctx = ref.context or ""
            if any(p.search(ctx) for p in self.compiled_patterns):
                weighted += ref.confidence_score if ref.confidence_score > 0 else 1.0

        # FUTURE_PAGE_SCALE calibrated so a ~30-page filing with one strong
        # future-mention scores ~1.0; longer filings need proportionally more.
        rate = weighted / total_pages * FUTURE_PAGE_SCALE
        return min(1.0, rate)


# ═══════════════════════════════════════════════════════════════════════════
# COMMITMENT DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class CommitmentDetector:
    """Commitment detector with a genai_specific category."""

    COMMITMENT_PATTERNS = {
        'financial': {
            'weight': 2.0,
            'patterns': [
                r'\binvest(?:ed|ing|ment)\s+[\$€£][\d,.]+\s*(?:million|billion|M|B)\s+(?:in|on)\s+(?:AI|GenAI)\b',
                r'\b(?:AI|GenAI)\s+(?:budget|investment|spending)\s+of\s+[\$€£][\d,.]+\b',
                r'\ballocat(?:ed|ing)\s+[\$€£][\d,.]+.*(?:AI|GenAI)\b',
                r'\b(?:committed|pledged|dedicated)\s+[\$€£][\d,.]+.*(?:AI|GenAI)\b',
            ]
        },
        'strategic': {
            'weight': 1.5,
            'patterns': [
                r'\b(?:AI|GenAI)\s+(?:is|as)\s+(?:a\s+)?(?:core|key|strategic|top|critical)\s+priority\b',
                r'\b(?:strategic|key|critical)\s+(?:focus|initiative|priority).*(?:AI|GenAI)\b',
                r'\bcommitted\s+to\s+(?:AI|artificial intelligence|GenAI)\b',
                r'\b(?:board|executive|leadership)\s+(?:commitment|support)\s+(?:to|for)\s+(?:AI|GenAI)\b',
            ]
        },
        'resources': {
            'weight': 1.5,
            'patterns': [
                r'\bhir(?:e|ed|ing)\s+\d+.*(?:AI|ML|data|GenAI)\s+(?:engineers?|scientists?|experts?)\b',
                r'\b(?:building|established|expanding)\s+(?:AI|GenAI)\s+(?:team|lab|center|capability)\b',
                r'\bChief\s+AI\s+Officer\b',
                r'\b(?:AI|GenAI)\s+center\s+of\s+excellence\b',
            ]
        },
        'general': {
            'weight': 1.0,
            'patterns': [
                r'\bcommitted\s+to\s+(?:advancing|developing|deploying|embracing)\s+(?:AI|GenAI)\b',
                r'\b(?:AI|GenAI)\s+(?:commitment|dedication|focus)\b',
                r'\b(?:accelerat|prioritiz|embrac)\w+\s+(?:AI|GenAI)\s+(?:adoption|transformation)\b',
            ]
        },
        # GenAI-specific commitment
        'genai_specific': {
            'weight': 1.8,
            'patterns': [
                r'\b(?:deployed|launched|implemented)\s+(?:GenAI|generative\s+AI|LLM|Copilot|ChatGPT)\b',
                r'\b(?:enterprise[\s-]wide|company[\s-]wide)\s+(?:GenAI|LLM|Copilot)\s+(?:deployment|rollout)\b',
                r'\b\d+\s+(?:GenAI|LLM)\s+(?:use\s+cases|applications|projects)\s+(?:in\s+production|deployed)\b',
                r'\b(?:GenAI|LLM)\s+(?:production|live|operational)\s+(?:deployment|system)\b',
                r'\b(?:scaled|scaling)\s+(?:GenAI|LLM)\s+(?:across|throughout)\b',
            ]
        }
    }

    MAX_POSSIBLE_SCORE = 12.0  # Increased from 10.0 for genai_specific

    def __init__(self):
        self.compiled_patterns = {}
        for category, data in self.COMMITMENT_PATTERNS.items():
            self.compiled_patterns[category] = {
                'weight': data['weight'],
                'patterns': [re.compile(p, re.IGNORECASE) for p in data['patterns']]
            }
        logger.info("Commitment detector initialized")

    def detect_commitment(self, refs: list[AIReference], total_pages: int) -> float:
        """
        Page-normalized + confidence-weighted commitment score.

        Normalizing per page keeps the score comparable across filings of
        different lengths (a 300-page report otherwise has far more
        pattern-match opportunity than a 60-page filing of equivalent intent).
        """
        if not refs or total_pages <= 0:
            return 0.0

        total_weighted = 0.0
        for ref in refs:
            w = ref.confidence_score if ref.confidence_score > 0 else 1.0
            ctx = ref.context or ""
            for category, data in self.compiled_patterns.items():
                matches = sum(len(p.findall(ctx)) for p in data['patterns'])
                total_weighted += matches * data['weight'] * w

        rate = total_weighted / total_pages * COMMITMENT_PAGE_SCALE
        return min(1.0, rate)


# ═══════════════════════════════════════════════════════════════════════════
# AI ADOPTION INDEX CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════

class AIAdoptionIndexCalculator:
    """
    AI Adoption Index Calculator with 7 dimensions.

    Design principles:
      - All dimensions are confidence-weighted by AIReference.confidence_score
        (Tier 1 keywords dominate Tier 3 boilerplate in every aggregation).
      - Future + commitment are page-normalized rates instead of raw counts
        or count/refs ratios.
      - Maturity is frequency-weighted across stages (smooth) instead of a
        max-stage step function.
    """

    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.maturity_detector = MaturityStageDetector()
        self.future_analyzer = FutureOrientationAnalyzer()
        self.commitment_detector = CommitmentDetector()

        # Compute the number of categories dynamically
        self.num_categories = len(AI_CATEGORIES)
        self.num_eu_units = len(EU_LEAF_UNITS)
        logger.info(
            f"AI Adoption Index Calculator initialized "
            f"({self.num_categories} classic categories, {self.num_eu_units} EU leaves)"
        )

    def calculate_index(self, doc_result: DocumentResult) -> AIAdoptionIndex:
        refs_all = doc_result.references
        # Exclude mention_only from index calculations (kept for audit)
        refs = [r for r in refs_all if not _is_mention_only(r)]

        if not refs:
            return AIAdoptionIndex(
                company=doc_result.company, year=doc_result.year,
                position=doc_result.position, industry=doc_result.industry,
                sector=doc_result.sector, country=doc_result.country
            )

        intensity = self._calculate_intensity(refs, doc_result.total_pages)
        semantic = self._calculate_semantic(refs)
        diversity = self._calculate_diversity(refs)
        diversity_eu = self._calculate_diversity_eu(refs)
        sentiment = self._calculate_sentiment(refs)
        maturity = self._calculate_maturity(refs)
        future = self._calculate_future(refs, doc_result.total_pages)
        commitment = self._calculate_commitment(refs, doc_result.total_pages)

        weights = self.config.weights

        def _composite(div: float) -> float:
            # Only the diversity term differs between classic and EU_Semantics;
            # the other six dimensions are taxonomy-independent.
            return (
                intensity * weights['intensity'] +
                semantic * weights['semantic'] +
                div * weights['diversity'] +
                sentiment * weights['sentiment'] +
                maturity * weights['maturity'] +
                future * weights['future'] +
                commitment * weights['commitment']
            ) * 100

        final_index = _composite(diversity)
        final_index_eu = _composite(diversity_eu)

        categories_used = len({
            cat for ref in refs for cat in (ref.category_a, ref.category_b)
            if cat and cat != 'none'
        })
        categories_used_eu = len({
            r.eu_subdomain for r in refs
            if getattr(r, 'eu_subdomain', 'Unclassified') not in ('', 'Unclassified')
        })

        logger.info(
            f"AI Index (classic={final_index:.1f}, EU={final_index_eu:.1f}): "
            f"{doc_result.company} ({doc_result.year})"
        )

        return AIAdoptionIndex(
            company=doc_result.company, year=doc_result.year,
            position=doc_result.position, industry=doc_result.industry,
            sector=doc_result.sector, country=doc_result.country,
            intensity_index=intensity, semantic_index=semantic,
            diversity_index=diversity, sentiment_index=sentiment,
            maturity_index=maturity, future_index=future,
            commitment_index=commitment, ai_adoption_index=final_index,
            diversity_index_eu=diversity_eu, ai_adoption_index_eu=final_index_eu,
            total_refs=len(refs), total_pages=doc_result.total_pages,
            categories_used=categories_used, categories_used_eu=categories_used_eu
        )

    @staticmethod
    def _ref_weight(ref: AIReference) -> float:
        """Confidence-derived weight, ≥1.0 floor so unscored refs aren't zeroed."""
        return ref.confidence_score if ref.confidence_score and ref.confidence_score > 0 else 1.0

    def _calculate_intensity(self, refs: list[AIReference], total_pages: int) -> float:
        """
        Compute the intensity of AI references per page (confidence-weighted).

        Formula: min(1.0, sum(confidence) / total_pages * 10)
        """
        if total_pages == 0:
            return 0.0
        weighted = sum(self._ref_weight(r) for r in refs)
        return min(1.0, weighted / total_pages * 10)

    def _calculate_semantic(self, refs: list[AIReference]) -> float:
        """
        Confidence-weighted mean semantic_score for refs above threshold.
        """
        weighted = [
            (r.semantic_score, self._ref_weight(r))
            for r in refs
            if r.semantic_score > self.config.semantic_threshold
        ]
        if not weighted:
            return 0.0
        total_w = sum(w for _, w in weighted)
        if total_w == 0:
            return 0.0
        return sum(s * w for s, w in weighted) / total_w

    def _calculate_diversity(self, refs: list[AIReference]) -> float:
        """
        Shannon entropy where each category's probability is the
        confidence-weighted share of its refs (rather than raw count).

        Formula:
            entropy = -sum(p_i * log2(p_i)) for each category
            diversity = entropy / max_entropy

        Where max_entropy = log2(num_categories).

        Both classic axes contribute: a reference adds its confidence weight to
        its Applications (A) bucket AND its Technologies (B) bucket. 'none' on
        an axis contributes nothing for that axis.
        """
        # Categories are weighted by the sum of their refs' confidence scores
        # — a high-confidence A1 ref contributes more than a Tier-3 boilerplate ref.
        cat_weight: dict[str, float] = {}
        for r in refs:
            w = self._ref_weight(r)
            for cat in (r.category_a, r.category_b):
                if cat and cat != 'none':
                    cat_weight[cat] = cat_weight.get(cat, 0.0) + w
        total = sum(cat_weight.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for w in cat_weight.values():
            p = w / total
            if p > 0:
                entropy -= p * math.log2(p)

        max_entropy = math.log2(self.num_categories)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _calculate_diversity_eu(self, refs: list[AIReference]) -> float:
        """Shannon-entropy diversity over EU_Semantics subdomains.

        Mirrors _calculate_diversity but groups refs by their EU subdomain
        (leaf) instead of the classic category, normalising by
        log2(num_eu_units). EU-Unclassified refs are ignored.
        """
        eu_weight: dict[str, float] = {}
        for r in refs:
            leaf = getattr(r, 'eu_subdomain', 'Unclassified')
            if not leaf or leaf == 'Unclassified':
                continue
            eu_weight[leaf] = eu_weight.get(leaf, 0.0) + self._ref_weight(r)
        total = sum(eu_weight.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for w in eu_weight.values():
            p = w / total
            if p > 0:
                entropy -= p * math.log2(p)

        max_entropy = math.log2(self.num_eu_units)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _calculate_sentiment(self, refs: list[AIReference]) -> float:
        """
        Confidence-weighted mean sentiment, mapped from [-1, 1] → [0, 1].
        """
        if not refs:
            return 0.5
        weighted = [(r.sentiment_score, self._ref_weight(r)) for r in refs]
        total_w = sum(w for _, w in weighted)
        if total_w == 0:
            return 0.5
        weighted_mean = sum(s * w for s, w in weighted) / total_w
        return (weighted_mean + 1) / 2

    def _calculate_maturity(self, refs: list[AIReference]) -> float:
        """Frequency-weighted average across stages (see MaturityStageDetector)."""
        return self.maturity_detector.detect_maturity_stage(refs)

    def _calculate_future(self, refs: list[AIReference], total_pages: int) -> float:
        """Page-normalized, confidence-weighted future-orientation rate."""
        return self.future_analyzer.analyze_future_orientation(refs, total_pages)

    def _calculate_commitment(self, refs: list[AIReference], total_pages: int) -> float:
        """Page-normalized, confidence-weighted commitment rate."""
        return self.commitment_detector.detect_commitment(refs, total_pages)


# ═══════════════════════════════════════════════════════════════════════════
# FINAL
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("AI SEMANTIC ANALYZER - ADVANCED ANALYSIS")
    print("=" * 80)
    print("\nModule loaded!")
    print(f"\n_calculate_diversity uses {len(AI_CATEGORIES)} categories")
    print("\nFeatures:")
    print("  - ImprovedSentimentAnalyzer (governance adjustment)")
    print("  - MATURITY_PATTERNS with GenAI")
    print("  - FUTURE_PATTERNS with 2025-2028")
    print("  - COMMITMENT_PATTERNS + genai_specific (weight 1.8)")
    print("  - MAX_POSSIBLE_SCORE = 12.0")
    print(f"  - Diversity max_entropy = log2({len(AI_CATEGORIES)}) = {math.log2(len(AI_CATEGORIES)):.2f}")
