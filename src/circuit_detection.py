"""
Quantum Circuit Detection Module for the Quantum Circuit Dataset Pipeline.

This module uses NLP-based methods to determine whether an extracted figure
is a quantum circuit diagram. The approach combines:
1. Caption text analysis
2. Surrounding text context analysis
3. Keyword and pattern matching
4. Structural text features

This is the CORE NLP component - no computer vision models are used.

Author: [Your Name]
Exam ID: 37
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter

import sys
sys.path.append(str(Path(__file__).parent))

from config import CONFIG
from data_models import FigureInfo
from utils.logging_utils import setup_logger


# Module logger
logger = setup_logger(__name__)


@dataclass
class CircuitDetectionResult:
    """
    Result of quantum circuit detection analysis.
    
    Attributes
    ----------
    is_circuit : bool
        Whether the figure is classified as a quantum circuit.
    confidence : float
        Confidence score (0-1).
    evidence : Dict[str, List[str]]
        Evidence supporting the classification.
    keyword_scores : Dict[str, float]
        Individual keyword match scores.
    caption_score : float
        Score from caption analysis.
    context_score : float
        Score from surrounding text analysis.
    """
    is_circuit: bool
    confidence: float
    evidence: Dict[str, List[str]] = field(default_factory=dict)
    keyword_scores: Dict[str, float] = field(default_factory=dict)
    caption_score: float = 0.0
    context_score: float = 0.0


class QuantumCircuitKeywordMatcher:
    """
    Matches quantum circuit-related keywords in text.
    
    This class provides weighted keyword matching to identify
    text that describes quantum circuits.
    
    Attributes
    ----------
    primary_keywords : Set[str]
        High-confidence keywords strongly indicating circuits.
    secondary_keywords : Set[str]
        Supporting keywords that add confidence.
    negative_keywords : Set[str]
        Keywords that indicate non-circuit content.
    gate_keywords : Set[str]
        Quantum gate names.
    
    Examples
    --------
    >>> matcher = QuantumCircuitKeywordMatcher()
    >>> score, matches = matcher.score_text("Figure 1 shows the CNOT circuit")
    >>> print(score, matches)
    """
    
    def __init__(self):
        """Initialize keyword sets from configuration."""
        # Primary keywords - strong indicators of quantum circuits
        # These phrases strongly suggest a circuit diagram
        self.primary_keywords = {
            # Direct circuit mentions
            'quantum circuit', 'circuit diagram', 'gate sequence',
            'qubit register', 'quantum gate', 'circuit implementation',
            'gate decomposition', 'circuit depth', 'circuit width',
            'circuit construction', 'unitary circuit', 'circuit for',
            'circuit implementing', 'circuit that', 'circuit to',
            # Specific gates in context
            'hadamard gate', 'cnot gate', 'controlled-not', 'controlled not',
            'toffoli gate', 'pauli gate', 'rotation gate', 'phase gate',
            # Circuit elements
            'qubit line', 'quantum wire', 'measurement gate',
            'ancilla qubit', 'control qubit', 'target qubit',
            # Algorithm circuits
            'grover circuit', 'shor circuit', 'vqe circuit', 'qaoa circuit',
            'qft circuit', 'fourier transform circuit', 'teleportation circuit',
            # Circuit descriptions
            'applied to qubit', 'apply gate', 'applying gate', 'gate applied',
            'gates are applied', 'sequence of gates', 'series of gates'
        }
        
        # Secondary keywords - supporting evidence
        self.secondary_keywords = {
            'qubit', 'qubits', 'gate', 'gates', 'circuit', 'register',
            'ancilla', 'measurement', 'unitary', 'controlled', 'target',
            'wire', 'wires', 'quantum register', 'state preparation',
            'basis state', 'computational basis', 'oracle', 'rotation',
            'phase', 'amplitude', 'superposition', 'entanglement',
            'hadamard', 'cnot', 'pauli', 'toffoli', 'swap', 'cz',
            'rx', 'ry', 'rz', 'u1', 'u2', 'u3', 'cx', 'ccx',
            'bell state', 'ghz state', 'initial state', 'final state',
            'variational', 'ansatz', 'parameterized'
        }
        
        # Negative keywords - indicate this is probably NOT a circuit diagram
        self.negative_keywords = {
            'photograph', 'photo', 'experimental setup', 'lab equipment',
            'device image', 'chip image', 'hardware image', 'cryostat',
            'bar chart', 'histogram', 'scatter plot', 'line graph',
            'data plot', 'error bars', 'experimental results', 'experimental data',
            'phase diagram', 'energy diagram', 'band structure',
            'molecule', 'molecular structure', 'chemical', 'atomic structure',
            'lattice structure', 'crystal structure', 'bloch sphere',
            'error rate', 'fidelity vs', 'success probability',
            'noise model', 'decoherence', 'relaxation time',
            'table of', 'table showing', 'comparison of'
        }
        
        # Gate names for direct matching
        self.gate_keywords = set()
        for gate, aliases in CONFIG.nlp.quantum_gates.items():
            self.gate_keywords.add(gate.lower())
            for alias in aliases:
                self.gate_keywords.add(alias.lower())
    
    def score_text(self, text: str) -> Tuple[float, Dict[str, List[str]]]:
        """
        Score text for quantum circuit relevance.
        
        Parameters
        ----------
        text : str
            Text to analyze (caption or context).
        
        Returns
        -------
        Tuple[float, Dict[str, List[str]]]
            (score, matched_keywords_by_category)
        
        Notes
        -----
        Scoring weights:
        - Primary keywords: +3.0 each
        - Secondary keywords: +1.0 each
        - Gate keywords: +1.5 each
        - Negative keywords: -2.0 each
        
        Final score is normalized to [0, 1].
        """
        text_lower = text.lower()
        
        matches = {
            'primary': [],
            'secondary': [],
            'gates': [],
            'negative': []
        }
        
        raw_score = 0.0
        
        # Match primary keywords (highest weight)
        for kw in self.primary_keywords:
            if kw in text_lower:
                matches['primary'].append(kw)
                raw_score += 3.0
        
        # Match secondary keywords
        for kw in self.secondary_keywords:
            # Use word boundary matching for short keywords
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, text_lower):
                matches['secondary'].append(kw)
                raw_score += 1.0
        
        # Match gate keywords
        for kw in self.gate_keywords:
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, text_lower):
                matches['gates'].append(kw)
                raw_score += 1.5
        
        # Match negative keywords (reduce score)
        for kw in self.negative_keywords:
            if kw in text_lower:
                matches['negative'].append(kw)
                raw_score -= 2.0
        
        # Normalize score to [0, 1]
        # Assume max reasonable score is ~20 (several primary + secondary + gates)
        normalized_score = max(0.0, min(1.0, raw_score / 15.0))
        
        return normalized_score, matches
    
    def has_circuit_indicators(self, text: str) -> bool:
        """
        Quick check if text has any circuit indicators.
        
        Parameters
        ----------
        text : str
            Text to check.
        
        Returns
        -------
        bool
            True if any circuit-related keywords found.
        """
        text_lower = text.lower()
        
        # Check for any primary keyword
        for kw in self.primary_keywords:
            if kw in text_lower:
                return True
        
        # Check for "circuit" or "gate" as standalone words
        if re.search(r'\bcircuit\b', text_lower):
            return True
        if re.search(r'\bgate\b', text_lower):
            return True
        
        return False


class CaptionAnalyzer:
    """
    Analyzes figure captions to determine circuit relevance.
    
    This class provides detailed caption parsing to:
    1. Extract figure type from caption
    2. Identify mentioned components
    3. Score circuit probability
    
    Attributes
    ----------
    keyword_matcher : QuantumCircuitKeywordMatcher
        The keyword matching component.
    
    Examples
    --------
    >>> analyzer = CaptionAnalyzer()
    >>> result = analyzer.analyze("Figure 1: Quantum circuit for Grover's algorithm")
    >>> print(result['is_circuit'], result['confidence'])
    """
    
    def __init__(self):
        """Initialize the caption analyzer."""
        self.keyword_matcher = QuantumCircuitKeywordMatcher()
        
        # Patterns that directly indicate circuit diagrams
        self.circuit_patterns = [
            # Direct circuit figure patterns - these are high confidence
            r'(?:quantum\s+)?circuit\s+(?:for|implementing|that|to|used)',
            r'(?:the\s+)?circuit\s+(?:is\s+)?(?:shown|depicted|illustrated)',
            r'(?:shows?|depicts?|illustrates?)\s+(?:the\s+)?(?:a\s+)?(?:quantum\s+)?circuit',
            r'circuit\s+diagram',
            r'gate\s+(?:sequence|decomposition)',
            r'(?:\d+|one|two|three|four|five|n)\s*-?\s*qubit\s+circuit',
            # Gate diagram patterns
            r'(?:cnot|hadamard|toffoli|swap|cz|cx)\s+gate\s+(?:circuit|diagram|sequence)',
            r'(?:controlled|control)\s+(?:gate|not|z|phase)',
            # Algorithm circuit patterns
            r'(?:grover|shor|vqe|qaoa|qft|qpe|teleportation|bell)\s*[\'s]*\s*(?:circuit|algorithm)',
            r'variational\s+(?:circuit|ansatz)',
            r'parameterized\s+(?:circuit|quantum)',
        ]
        
        # Patterns that strongly indicate NON-circuit figures - be strict here
        self.non_circuit_patterns = [
            # Results/data patterns
            r'(?:real|imaginary)\s+parts?\s+of',
            r'eigenvalue',
            r'result(?:s)?\s+(?:of|from|for|showing)',
            r'performance\s+of',
            r'comparison\s+(?:of|between)',
            r'(?:plot|graph)\s+(?:of|showing)',
            r'experimental\s+(?:data|results|setup)',
            r'error\s+(?:rate|bar)',
            r'fidelity\s+(?:vs|versus|against|as)',
            r'success\s+(?:probability|rate)',
            r'(?:energy|phase)\s+(?:landscape|spectrum|diagram)',
            r'histogram|bar\s+chart',
            r'convergence',
            r'(?:training|learning)\s+curve',
            # Matrix/mathematical patterns
            r'matrix\s+element',
            r'(?:density|choi|pauli)\s+matrix',
            r'calibration',
            r'coefficient',
            r'(?:real|imaginary)\s+(?:part|component)',
            r'overlap|scalar\s+product',
            # Physical apparatus
            r'(?:schematic|diagram)\s+of\s+(?:the\s+)?(?:experiment|setup|device|chip)',
            r'(?:superconducting|transmon)\s+(?:qubit|device|chip)',
        ]
    
    def analyze(self, caption: str) -> Dict:
        """
        Analyze a figure caption.
        
        Parameters
        ----------
        caption : str
            The figure caption text.
        
        Returns
        -------
        Dict
            Analysis result with keys:
            - 'is_circuit': bool
            - 'confidence': float
            - 'keyword_score': float
            - 'pattern_score': float
            - 'evidence': list of matched patterns/keywords
        """
        if not caption:
            return {
                'is_circuit': False,
                'confidence': 0.0,
                'keyword_score': 0.0,
                'pattern_score': 0.0,
                'evidence': []
            }
        
        evidence = []
        
        # Get keyword score
        keyword_score, matches = self.keyword_matcher.score_text(caption)
        if matches['primary']:
            evidence.extend([f"primary:{kw}" for kw in matches['primary']])
        if matches['gates']:
            evidence.extend([f"gate:{kw}" for kw in matches['gates'][:3]])
        
        # Check circuit patterns
        pattern_score = 0.0
        caption_lower = caption.lower()
        
        # Check positive circuit patterns
        positive_matches = 0
        for pattern in self.circuit_patterns:
            if re.search(pattern, caption_lower):
                pattern_score += 0.35
                positive_matches += 1
                evidence.append(f"pattern:{pattern[:30]}...")
        
        # Check non-circuit patterns (strong negative signal)
        negative_matches = 0
        for pattern in self.non_circuit_patterns:
            if re.search(pattern, caption_lower):
                pattern_score -= 0.5  # Strong penalty for non-circuit patterns
                negative_matches += 1
                evidence.append(f"negative:{pattern[:20]}...")
        
        # Boost if caption mentions specific gate names together with "circuit"
        has_circuit_word = 'circuit' in caption_lower
        gate_names = ['hadamard', 'cnot', 'toffoli', 'pauli', 'swap', 'cx', 'cz', 
                      'rotation', 'phase gate', 'controlled']
        gate_count = sum(1 for g in gate_names if g in caption_lower)
        if gate_count >= 1 and has_circuit_word:
            pattern_score += 0.3
            evidence.append(f"gate+circuit:{gate_count}")
        
        # If we have strong negative signals, require strong positive signals
        if negative_matches > 0 and positive_matches == 0:
            pattern_score = min(pattern_score, -0.2)  # Cap low if only negatives
        
        # Combine scores - give more weight to pattern matching for captions
        pattern_score = max(-0.5, min(1.0, pattern_score + 0.3))
        combined_score = 0.4 * keyword_score + 0.6 * pattern_score
        
        # Decision threshold - raised for precision
        is_circuit = combined_score >= 0.35
        
        return {
            'is_circuit': is_circuit,
            'confidence': combined_score,
            'keyword_score': keyword_score,
            'pattern_score': pattern_score,
            'evidence': evidence
        }


class ContextAnalyzer:
    """
    Analyzes surrounding text context to support circuit detection.
    
    This class examines paragraphs and sentences that reference
    a figure to gather additional evidence.
    
    Attributes
    ----------
    keyword_matcher : QuantumCircuitKeywordMatcher
        The keyword matching component.
    
    Examples
    --------
    >>> analyzer = ContextAnalyzer()
    >>> contexts = ["In Fig. 1, we show the quantum circuit that...", ...]
    >>> result = analyzer.analyze(contexts)
    """
    
    def __init__(self):
        """Initialize the context analyzer."""
        self.keyword_matcher = QuantumCircuitKeywordMatcher()
        
        # Patterns indicating circuit description in context
        self.description_patterns = [
            r'(?:shows?|presents?|depicts?|illustrates?)\s+(?:the\s+)?(?:quantum\s+)?circuit',
            r'circuit\s+(?:consists?|contains?|comprises?)',
            r'(?:apply|applying|applied)\s+(?:the\s+)?(?:\w+\s+)?gate',
            r'(?:first|then|next|finally)\s+(?:we\s+)?apply',
            r'qubits?\s+(?:are|is)\s+(?:initialized|prepared|measured)',
            r'gate\s+sequence',
            r'unitary\s+(?:is|are)\s+(?:applied|implemented)',
            # Additional patterns for better detection
            r'(?:in|from)\s+(?:fig|figure)\s*\.?\s*\d+',
            r'(?:cnot|hadamard|toffoli|pauli|swap)\s+(?:gate|operation)',
            r'qubit\s+\d+',
            r'(?:single|two|multi)\s*-?\s*qubit',
            r'(?:control|target)\s+qubit',
            r'(?:input|output)\s+(?:state|qubit)',
            r'(?:initial|final)\s+state',
            r'measurement\s+(?:of|on)',
            r'(?:variational|parameterized)\s+(?:circuit|ansatz)',
            r'(?:grover|shor|vqe|qaoa|qft)\s*[\'s]*',
        ]
    
    def analyze(self, context_texts: List[str]) -> Dict:
        """
        Analyze context texts for circuit evidence.
        
        Parameters
        ----------
        context_texts : List[str]
            List of text snippets referencing the figure.
        
        Returns
        -------
        Dict
            Analysis result with keys:
            - 'is_circuit': bool
            - 'confidence': float
            - 'evidence': list of evidence found
            - 'context_count': number of contexts analyzed
        """
        if not context_texts:
            return {
                'is_circuit': False,
                'confidence': 0.0,
                'evidence': [],
                'context_count': 0
            }
        
        total_score = 0.0
        evidence = []
        
        for ctx in context_texts:
            # Get keyword score for this context
            kw_score, matches = self.keyword_matcher.score_text(ctx)
            total_score += kw_score
            
            if matches['primary']:
                evidence.extend(matches['primary'][:2])
            
            # Check description patterns
            ctx_lower = ctx.lower()
            for pattern in self.description_patterns:
                if re.search(pattern, ctx_lower):
                    total_score += 0.15
                    evidence.append(f"ctx_pattern:{pattern[:15]}...")
        
        # Average score across contexts
        avg_score = total_score / len(context_texts) if context_texts else 0.0
        
        # Normalize
        confidence = min(1.0, avg_score)
        is_circuit = confidence >= 0.3
        
        return {
            'is_circuit': is_circuit,
            'confidence': confidence,
            'evidence': evidence[:10],  # Limit evidence list
            'context_count': len(context_texts)
        }


class VisualStructureAnalyzer:
    """
    Placeholder for visual structure analysis of circuit diagrams.
    Currently returns neutral scores as visual analysis is not implemented.
    """
    
    def analyze(self, image_path: str = None) -> Dict:
        """
        Analyze visual structure of an image.
        
        Parameters
        ----------
        image_path : str
            Path to the image file.
            
        Returns
        -------
        Dict
            Analysis result with score and features.
        """
        return {
            'score': 0.5,
            'reason': 'Visual analysis not implemented',
            'features': {}
        }


class RasterCircuitDetector:
    """
    Placeholder for raster-based circuit detection.
    Currently returns neutral scores as raster analysis is not implemented.
    """
    
    def analyze(self, image_path: str = None) -> Dict:
        """
        Analyze raster image for circuit features.
        
        Parameters
        ----------
        image_path : str
            Path to the image file.
            
        Returns
        -------
        Dict
            Analysis result with score and features.
        """
        return {
            'score': 0.5,
            'reason': 'Raster analysis not implemented',
            'features': {}
        }


class QuantumCircuitDetector:
    """
    Main class for detecting quantum circuits from text analysis.
    
    This class combines caption and context analysis to make
    a final decision on whether a figure is a quantum circuit.
    
    Attributes
    ----------
    caption_analyzer : CaptionAnalyzer
        Caption analysis component.
    context_analyzer : ContextAnalyzer
        Context analysis component.
    confidence_threshold : float
        Minimum confidence for positive classification.
    
    Examples
    --------
    >>> detector = QuantumCircuitDetector()
    >>> result = detector.detect(caption="Figure 1: Circuit for QFT", 
    ...                          contexts=["shows the circuit..."])
    >>> if result.is_circuit:
    ...     print(f"Quantum circuit detected with confidence {result.confidence}")
    """
    
    def __init__(self, confidence_threshold: float = 0.35):
        """
        Initialize the circuit detector.
        
        Parameters
        ----------
        confidence_threshold : float
            Minimum confidence score to classify as circuit (balanced for precision/recall).
        """
        self.caption_analyzer = CaptionAnalyzer()
        self.context_analyzer = ContextAnalyzer()
        self.confidence_threshold = confidence_threshold
        self.visual_analyzer = VisualStructureAnalyzer() # Vector analyzer
        self.raster_analyzer = RasterCircuitDetector()
    
    def detect(
        self, 
        caption: str = "", 
        contexts: List[str] = None
    ) -> CircuitDetectionResult:
        """
        Detect if a figure is a quantum circuit.
        
        Parameters
        ----------
        caption : str
            The figure caption.
        contexts : List[str]
            List of text snippets referencing the figure.
        
        Returns
        -------
        CircuitDetectionResult
            Detection result with confidence and evidence.
        
        Notes
        -----
        The detection combines:
        - Caption analysis (60% weight if present)
        - Context analysis (40% weight if present)
        
        If only caption or only context is available,
        it's weighted at 100%.
        """
        if contexts is None:
            contexts = []
        
        evidence = {}
        
        # Analyze caption
        caption_result = self.caption_analyzer.analyze(caption)
        caption_score = caption_result['confidence']
        evidence['caption'] = caption_result['evidence']
        
        # Analyze contexts
        context_result = self.context_analyzer.analyze(contexts)
        context_score = context_result['confidence']
        evidence['context'] = context_result['evidence']
        
        # Combine scores
        if caption and contexts:
            # Both available - weighted combination
            combined_score = 0.6 * caption_score + 0.4 * context_score
        elif caption:
            # Only caption
            combined_score = caption_score
        elif contexts:
            # Only context
            combined_score = context_score
        else:
            # Nothing to analyze
            combined_score = 0.0
        
        # Make decision
        is_circuit = combined_score >= self.confidence_threshold
        
        return CircuitDetectionResult(
            is_circuit=is_circuit,
            confidence=combined_score,
            evidence=evidence,
            caption_score=caption_score,
            context_score=context_score
        )
    
    def batch_detect(
        self, 
        figures: List[Dict]
    ) -> List[Tuple[Dict, CircuitDetectionResult]]:
        """
        Detect circuits for multiple figures.
        
        Parameters
        ----------
        figures : List[Dict]
            List of figure dictionaries with 'caption' and 'contexts' keys.
        
        Returns
        -------
        List[Tuple[Dict, CircuitDetectionResult]]
            List of (figure, detection_result) tuples.
        """
        results = []
        
        for fig in figures:
            caption = fig.get('caption', '')
            contexts = fig.get('contexts', [])
            
            result = self.detect(caption=caption, contexts=contexts)
            results.append((fig, result))
        
        # Log summary
        detected = sum(1 for _, r in results if r.is_circuit)
        logger.info(f"Batch detection: {detected}/{len(figures)} classified as circuits")
        
        return results


def filter_circuit_figures(
    figures_with_text: List[Dict],
    confidence_threshold: float = 0.35
) -> List[Dict]:
    """
    Filter a list of figures to keep only quantum circuits.
    
    Parameters
    ----------
    figures_with_text : List[Dict]
        List of figure dictionaries, each with:
        - 'figure': ExtractedFigure object
        - 'caption': str
        - 'contexts': List[str]
    confidence_threshold : float
        Minimum confidence for keeping a figure (balanced for precision/recall).
    
    Returns
    -------
    List[Dict]
        Filtered list with additional 'detection_result' key.
    
    Examples
    --------
    >>> figures = [{'figure': fig, 'caption': 'Circuit for QFT', 'contexts': [...]}]
    >>> circuits = filter_circuit_figures(figures)
    """
    detector = QuantumCircuitDetector(confidence_threshold)
    
    filtered = []
    for fig_dict in figures_with_text:
        caption = fig_dict.get('caption', '')
        contexts = fig_dict.get('contexts', [])
        
        result = detector.detect(
            caption=caption,
            contexts=contexts
        )
        
        # Debug logging to see what's being analyzed
        logger.debug(f"Caption: {caption[:100]}..." if len(caption) > 100 else f"Caption: {caption}")
        logger.debug(f"Caption score: {result.caption_score:.3f}, Context score: {result.context_score:.3f}, Combined: {result.confidence:.3f}")
        
        if result.is_circuit:
            fig_dict['detection_result'] = result
            filtered.append(fig_dict)
            logger.info(f"✓ Kept figure with confidence {result.confidence:.2f}: {caption[:60]}...")
        else:
            logger.debug(f"✗ Filtered out figure with confidence {result.confidence:.2f}")
    
    logger.info(f"Kept {len(filtered)} out of {len(figures_with_text)} figures as circuits")
    return filtered


if __name__ == "__main__":
    # Test the circuit detector
    print("Testing Quantum Circuit Detector...")
    
    detector = QuantumCircuitDetector()
    
    # Test cases
    test_cases = [
        {
            'caption': "Figure 1: Quantum circuit implementing Grover's search algorithm",
            'contexts': ["The circuit applies Hadamard gates to initialize the qubits"]
        },
        {
            'caption': "Figure 2: Experimental results showing error rates",
            'contexts': ["The plot shows the fidelity versus circuit depth"]
        },
        {
            'caption': "Figure 3: CNOT gate sequence for Bell state preparation",
            'contexts': ["We apply a Hadamard gate followed by a CNOT"]
        }
    ]
    
    for i, tc in enumerate(test_cases):
        result = detector.detect(caption=tc['caption'], contexts=tc['contexts'])
        print(f"\nTest {i+1}:")
        print(f"  Caption: {tc['caption'][:50]}...")
        print(f"  Is Circuit: {result.is_circuit}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Evidence: {result.evidence}")
    
    print("\nTest completed!")
