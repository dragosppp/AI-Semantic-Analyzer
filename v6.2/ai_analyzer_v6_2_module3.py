"""
═══════════════════════════════════════════════════════════════════════════════
AI SEMANTIC ANALYZER v6.2.0 - MODUL 3: ADVANCED ANALYSIS & VISUALIZATION
═══════════════════════════════════════════════════════════════════════════════

Advanced statistical analysis, semantic deduplication, sentiment analysis, and
interactive visualizations for AI Semantic Analyzer v6.2.

MAJOR UPDATE v6.2.0 (February 2026):
    🆕 DUAL TAXONOMY ANALYTICS
        - Cross-dimensional analysis (Applications × Technologies)
        - 16-category distribution analysis
        - Technology-application correlation matrices
        - Vendor transparency analysis
    
    📊 ENHANCED VISUALIZATIONS
        - Technology adoption heatmaps
        - Application-technology Sankey diagrams
        - Vendor mention tracking charts
        - Expanded temporal trend analysis (2020-2025)
    
    🎯 IMPROVED METRICS
        - AI Adoption Index adjusted for 16 categories
        - Category diversity scores
        - Technology maturity indicators
        - Vendor disclosure transparency metrics

COMPONENTS:
    1. Statistical Analysis
       - Company-level AI adoption metrics
       - Year-over-year trend analysis
       - Sector benchmarking
       - Category distribution analysis
       - AI Adoption Index calculation
    
    2. Semantic Deduplication
       - SentenceTransformer embeddings
       - Cosine similarity clustering
       - Configurable threshold (default: 0.85)
       - Smart merging: preserve highest confidence
       - Duplicate tracking and logging
    
    3. Sentiment Analysis
       - FinBERT-based sentiment scoring
       - Context-aware classification
       - Batch processing with progress bars
       - Fallback to TextBlob if FinBERT unavailable
    
    4. Interactive Visualizations (Plotly)
       - Temporal trends (line charts)
       - Category distributions (bar charts, treemaps)
       - Company comparisons (grouped bars)
       - Sector analysis (box plots, violin plots)
       - Technology heatmaps (NEW v6.2)
       - Vendor analysis (NEW v6.2)
       - All charts exportable as HTML

WORKFLOW:
    Raw Data → Statistical Analysis → Deduplication → Sentiment Analysis
    → Visualization Generation → Export Results

NEW IN v6.2:
    ✅ Dual taxonomy analysis (16 categories)
    ✅ Technology-application cross-analysis
    ✅ Vendor transparency metrics
    ✅ Enhanced category diversity scores
    ✅ Technology maturity tracking
    ✅ Application-technology correlation

CHANGELOG v6.2.0 (Feb 2026):
    - Dual taxonomy analytics (Applications + Technologies)
    - 16-category distribution analysis
    - Technology adoption heatmaps
    - Vendor transparency analysis
    - Enhanced AI Adoption Index for 16 categories
    - Application-technology correlation matrices

CHANGELOG v6.1.1 (Jan 2026):
    - Export compatibility for strength/confidence scores
    - Enhanced deduplication with confidence preservation
    - Version bump 6.1.1

CHANGELOG v6.1.0 (Jan 2026):
    - AI Adoption Index: exclude mention_only from calculations
    - Enhanced confidence scoring in deduplication
    - Improved semantic similarity thresholds

TECHNICAL NOTES:
    - Deduplication threshold: 0.85 (cosine similarity)
    - FinBERT model: ProsusAI/finbert
    - Semantic model: all-MiniLM-L6-v2 (shared with module2)
    - Memory efficient: Batch processing for large datasets
    - Thread-safe: SemanticModelLoader singleton

Author: TeRa0
Version: 6.2.0
Date: February 2026
Part of: AI Semantic Analyzer

═══════════════════════════════════════════════════════════════════════════════
"""


import math
import re
import json
from collections import Counter
from typing import List, Dict, Tuple, Optional
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

from ai_analyzer_v6_2_module1 import (
    logger, AnalyzerConfig, AIReference, DocumentResult, 
    AIAdoptionIndex, AI_CATEGORIES, FINBERT_AVAILABLE
)
from ai_analyzer_v6_2_module2 import SemanticModelLoader


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS - Strength/Confidence parsing (v6.2, from v6.1.0)
# ═══════════════════════════════════════════════════════════════════════════
_strength_re = re.compile(r'(?:^|\|)strength=([^|]+)')
_conf_re = re.compile(r'(?:^|\|)conf=([0-9]*\.?[0-9]+)')
_reasons_re = re.compile(r'(?:^|\|)reasons=([^|]+)')

def _parse_ref_meta_from_detection_method(ref: AIReference) -> Tuple[str, float, str]:
    """Fallback: extrage strength/confidence/reasons din ref.detection_method dacă nu sunt setate."""
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
    """FinBERT pentru sentiment financiar (89% accuracy)."""
    
    def __init__(self):
        self.finbert_available = FINBERT_AVAILABLE
        
        if self.finbert_available:
            logger.info("Încărcare FinBERT...")
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                
                self.tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
                self.model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
                self.model.eval()
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(self.device)
                self.torch = torch
                logger.info(f"✓ FinBERT pe {self.device}")
            except Exception as e:
                logger.error(f"Eroare FinBERT: {e}")
                self.finbert_available = False
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        if not text or len(text.strip()) < 10:
            return 'neutral', 0.0
        
        if self.finbert_available:
            return self._analyze_with_finbert(text)
        return self._analyze_with_textblob(text)
    
    def _analyze_with_finbert(self, text: str) -> Tuple[str, float]:
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
    
    def _analyze_with_textblob(self, text: str) -> Tuple[str, float]:
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
# IMPROVED SENTIMENT ANALYZER v6.0
# ═══════════════════════════════════════════════════════════════════════════

class ImprovedSentimentAnalyzer:
    """
    Sentiment analyzer îmbunătățit v6.0.
    
    Fix: FinBERT clasifică greșit AI governance ca "negative".
    Soluție: Ajustează sentiment pentru contexte de governance.
    """
    
    # Termeni de governance - NU sunt negative!
    AI_GOVERNANCE_TERMS = [
        'responsible ai', 'ai governance', 'ai ethics', 'ai risk management',
        'ai policy', 'ai compliance', 'ai oversight', 'ai principles',
        'ethical ai', 'trustworthy ai', 'ai framework', 'ai guidelines',
        'bias mitigation', 'fairness', 'transparency', 'accountability',
        'ai safety', 'guardrails', 'model risk', 'ai audit'
    ]
    
    # Termeni truly negative
    TRULY_NEGATIVE_TERMS = [
        'failed', 'failure', 'lawsuit', 'litigation', 'breach', 'hack',
        'layoff', 'job loss', 'replace workers', 'eliminate jobs',
        'harmful', 'dangerous', 'threat to', 'concern about',
        'security incident', 'data breach', 'privacy violation'
    ]
    
    def __init__(self, base_analyzer: FinBERTSentimentAnalyzer):
        self.base_analyzer = base_analyzer
    
    def analyze_sentiment_improved(self, context: str) -> Tuple[str, float, str]:
        """
        Returns: (sentiment_label, sentiment_score, confidence_flag)
        confidence_flag: 'standard', 'governance_adjusted', 'confirmed_negative'
        """
        context_lower = context.lower()
        
        is_governance = any(term in context_lower for term in self.AI_GOVERNANCE_TERMS)
        is_truly_negative = any(term in context_lower for term in self.TRULY_NEGATIVE_TERMS)
        
        label, score = self.base_analyzer.analyze_sentiment(context)
        
        # Ajustări
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
        logger.info(f"Deduplicator v6.0.5: threshold={config.deduplication_threshold}")
    
    def deduplicate_references(self, references: List[AIReference]) -> List[Dict]:
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
        
        logger.info(f"Deduplicate: {len(references)} → {len(deduplicated)}")
        return deduplicated
    
    def _find_clusters(self, similarity_matrix: np.ndarray, threshold: float) -> List[List[int]]:
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
    
    def _merge_cluster(self, refs: List[AIReference], indices: List[int]) -> Dict:
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
            'category': representative.category,
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
# MATURITY STAGE DETECTOR v6.0.5
# ═══════════════════════════════════════════════════════════════════════════

class MaturityStageDetector:
    """Detector maturitate v6.0.5 cu GenAI în toate stadiile."""
    
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
        logger.info("Maturity detector v6.0.5 inițializat")
    
    def detect_maturity_stage(self, contexts: List[str]) -> float:
        stage_counts = {stage: 0 for stage in self.compiled_patterns}
        all_text = ' '.join(contexts)
        
        for stage, data in self.compiled_patterns.items():
            for pattern in data['patterns']:
                matches = pattern.findall(all_text)
                stage_counts[stage] += len(matches)
        
        for stage in ['optimizing', 'scaling', 'implementing', 'piloting', 'exploring']:
            if stage_counts[stage] > 0:
                return self.compiled_patterns[stage]['score']
        
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# FUTURE ORIENTATION ANALYZER v6.0.5
# ═══════════════════════════════════════════════════════════════════════════

class FutureOrientationAnalyzer:
    """Analizor forward-looking v6.0.5 cu timelines 2025-2028."""
    
    FUTURE_PATTERNS = [
        # Verbe viitoare
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
        logger.info("Future analyzer v6.0.5 inițializat")
    
    def analyze_future_orientation(self, contexts: List[str]) -> float:
        if not contexts:
            return 0.0
        
        future_count = sum(
            1 for context in contexts 
            if any(pattern.search(context) for pattern in self.compiled_patterns)
        )
        
        return future_count / len(contexts)


# ═══════════════════════════════════════════════════════════════════════════
# COMMITMENT DETECTOR v6.0.5
# ═══════════════════════════════════════════════════════════════════════════

class CommitmentDetector:
    """Detector commitment v6.0.5 cu genai_specific category."""
    
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
        # NOU v6.0: GenAI specific commitment
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
    
    MAX_POSSIBLE_SCORE = 12.0  # Crescut de la 10.0 pentru genai_specific
    
    def __init__(self):
        self.compiled_patterns = {}
        for category, data in self.COMMITMENT_PATTERNS.items():
            self.compiled_patterns[category] = {
                'weight': data['weight'],
                'patterns': [re.compile(p, re.IGNORECASE) for p in data['patterns']]
            }
        logger.info("Commitment detector v6.0.5 inițializat")
    
    def detect_commitment(self, contexts: List[str]) -> float:
        if not contexts:
            return 0.0
        
        all_text = ' '.join(contexts)
        total_score = 0.0
        
        for category, data in self.compiled_patterns.items():
            category_matches = sum(len(p.findall(all_text)) for p in data['patterns'])
            total_score += category_matches * data['weight']
        
        return min(total_score / self.MAX_POSSIBLE_SCORE, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# AI ADOPTION INDEX CALCULATOR v6.0.5
# ═══════════════════════════════════════════════════════════════════════════

class AIAdoptionIndexCalculatorV6:
    """
    Calculator AI Adoption Index v6.0.5 cu 7 dimensiuni.
    
    V6.0.5 Adaugare sector si tara

    FIX v6.0.5: Corectat numărul de categorii în _calculate_diversity
    - Anterior: math.log2(17) - GREȘIT
    - Acum: math.log2(len(AI_CATEGORIES)) - CORECT (13 categorii)
    """
    
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.maturity_detector = MaturityStageDetector()
        self.future_analyzer = FutureOrientationAnalyzer()
        self.commitment_detector = CommitmentDetector()
        
        # Calculează numărul de categorii dinamic
        self.num_categories = len(AI_CATEGORIES)
        logger.info(f"AI Adoption Index Calculator v6.0.5 inițializat ({self.num_categories} categorii)")
    
    def calculate_index(self, doc_result: DocumentResult) -> AIAdoptionIndex:
        refs_all = doc_result.references
        # v6.2 (from v6.1.0): Exclude mention_only from index calculations (kept for audit)
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
        sentiment = self._calculate_sentiment(refs)
        maturity = self._calculate_maturity(refs)
        future = self._calculate_future(refs)
        commitment = self._calculate_commitment(refs)
        
        weights = self.config.weights
        final_index = (
            intensity * weights['intensity'] +
            semantic * weights['semantic'] +
            diversity * weights['diversity'] +
            sentiment * weights['sentiment'] +
            maturity * weights['maturity'] +
            future * weights['future'] +
            commitment * weights['commitment']
        ) * 100
        
        categories_used = len(set(ref.category for ref in refs))
        
        logger.info(f"AI Index v6.2 (Dual Taxonomy): {doc_result.company} ({doc_result.year}) = {final_index:.1f}")
        
        return AIAdoptionIndex(
            company=doc_result.company, year=doc_result.year,
            position=doc_result.position, industry=doc_result.industry,
            sector=doc_result.sector, country=doc_result.country,
            intensity_index=intensity, semantic_index=semantic,
            diversity_index=diversity, sentiment_index=sentiment,
            maturity_index=maturity, future_index=future,
            commitment_index=commitment, ai_adoption_index=final_index,
            total_refs=len(refs), total_pages=doc_result.total_pages,
            categories_used=categories_used
        )
    
    def _calculate_intensity(self, refs: List[AIReference], total_pages: int) -> float:
        """
        Calculează intensitatea referințelor AI per pagină.
        
        Formula: min(1.0, num_refs / total_pages * 10)
        """
        if total_pages == 0:
            return 0.0
        return min(1.0, len(refs) / total_pages * 10)
    
    def _calculate_semantic(self, refs: List[AIReference]) -> float:
        """
        Calculează scorul semantic mediu pentru referințele peste threshold.
        """
        scores = [r.semantic_score for r in refs if r.semantic_score > self.config.semantic_threshold]
        return np.mean(scores) if scores else 0.0
    
    def _calculate_diversity(self, refs: List[AIReference]) -> float:
        """
        Calculează diversitatea categoriilor folosind entropia Shannon.
        
        FIX v6.0.5: Folosește len(AI_CATEGORIES) = 13, nu 17!
        
        Formula:
            entropy = -Σ(p_i * log2(p_i)) pentru fiecare categorie
            diversity = entropy / max_entropy
            
        Unde max_entropy = log2(număr_categorii) = log2(13) ≈ 3.7
        """
        categories = [ref.category for ref in refs]
        category_counts = Counter(categories)
        total = len(categories)
        
        if total == 0:
            return 0.0
        
        # Calculează entropia Shannon
        entropy = 0.0
        for count in category_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # CORECTIE v6.0.5: Folosește numărul real de categorii (13)
        # Anterior era hardcodat 17 - GREȘIT!
        max_entropy = math.log2(self.num_categories)  # log2(13) ≈ 3.7
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_sentiment(self, refs: List[AIReference]) -> float:
        """
        Calculează scorul de sentiment normalizat.
        
        Formula: (mean(sentiment_scores) + 1) / 2
        Transformă din [-1, 1] în [0, 1]
        """
        scores = [ref.sentiment_score for ref in refs]
        if not scores:
            return 0.5
        return (np.mean(scores) + 1) / 2
    
    def _calculate_maturity(self, refs: List[AIReference]) -> float:
        """Calculează nivelul de maturitate AI."""
        contexts = [ref.context for ref in refs]
        return self.maturity_detector.detect_maturity_stage(contexts)
    
    def _calculate_future(self, refs: List[AIReference]) -> float:
        """Calculează orientarea spre viitor."""
        contexts = [ref.context for ref in refs]
        return self.future_analyzer.analyze_future_orientation(contexts)
    
    def _calculate_commitment(self, refs: List[AIReference]) -> float:
        """Calculează nivelul de commitment."""
        contexts = [ref.context for ref in refs]
        return self.commitment_detector.detect_commitment(contexts)


# ═══════════════════════════════════════════════════════════════════════════
# FINAL
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("AI SEMANTIC ANALYZER v6.0.5 - MODUL 3: ADVANCED ANALYSIS")
    print("=" * 80)
    print("\n✓ Module 3 v6.0.5 încărcat!")
    print(f"\n✓ FIX: _calculate_diversity folosește {len(AI_CATEGORIES)} categorii (nu 17)")
    print("\nFeatures:")
    print("  - ImprovedSentimentAnalyzer (governance adjustment)")
    print("  - MATURITY_PATTERNS cu GenAI")
    print("  - FUTURE_PATTERNS cu 2025-2028")
    print("  - COMMITMENT_PATTERNS + genai_specific (weight 1.8)")
    print("  - MAX_POSSIBLE_SCORE = 12.0")
    print(f"  - Diversity max_entropy = log2({len(AI_CATEGORIES)}) = {math.log2(len(AI_CATEGORIES)):.2f}")
