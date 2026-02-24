"""
Quantum Gate Extraction Module for the Quantum Circuit Dataset Pipeline.

This module uses NLP techniques to extract quantum gate names from:
1. Figure captions
2. Surrounding text
3. Section content

The extraction uses a controlled vocabulary with fuzzy matching
and normalization to standardize gate names.

Author: [Your Name]
Exam ID: 37
"""

import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
from dataclasses import dataclass, field

import sys
sys.path.append(str(Path(__file__).parent))

from config import CONFIG
from utils.logging_utils import setup_logger


# Module logger
logger = setup_logger(__name__)


@dataclass
class GateExtractionResult:
    """
    Result of quantum gate extraction.
    
    Attributes
    ----------
    gates : List[str]
        List of normalized gate names found.
    gate_counts : Dict[str, int]
        Count of each gate type.
    raw_matches : List[str]
        Original matched strings before normalization.
    confidence : float
        Confidence score for the extraction.
    source_evidence : Dict[str, List[str]]
        Evidence organized by source (caption, context).
    """
    gates: List[str] = field(default_factory=list)
    gate_counts: Dict[str, int] = field(default_factory=dict)
    raw_matches: List[str] = field(default_factory=list)
    confidence: float = 0.0
    source_evidence: Dict[str, List[str]] = field(default_factory=dict)


class GateVocabulary:
    """
    Manages the controlled vocabulary of quantum gates.
    
    This class provides:
    1. Mapping from aliases to canonical gate names
    2. Pattern matching for gate detection
    3. Normalization of gate names
    
    The vocabulary is designed to be easily extensible.
    
    Attributes
    ----------
    canonical_to_aliases : Dict[str, List[str]]
        Mapping from canonical name to aliases.
    alias_to_canonical : Dict[str, str]
        Reverse mapping for normalization.
    gate_patterns : List[Tuple[str, re.Pattern]]
        Compiled regex patterns for matching.
    
    Examples
    --------
    >>> vocab = GateVocabulary()
    >>> print(vocab.normalize("hadamard gate"))
    'H'
    >>> print(vocab.find_gates("Apply CNOT and Hadamard"))
    ['CNOT', 'H']
    """
    
    def __init__(self):
        """Initialize the gate vocabulary from configuration."""
        self.canonical_to_aliases = CONFIG.nlp.quantum_gates.copy()
        self.alias_to_canonical = {}
        self.gate_patterns = []
        
        self._build_mappings()
        self._compile_patterns()
    
    def _build_mappings(self) -> None:
        """Build reverse mappings from aliases to canonical names."""
        for canonical, aliases in self.canonical_to_aliases.items():
            # Add canonical name itself
            self.alias_to_canonical[canonical.lower()] = canonical
            
            # Add all aliases
            for alias in aliases:
                self.alias_to_canonical[alias.lower()] = canonical
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for gate matching."""
        # Sort aliases by length (longer first) to match longer names first
        all_terms = []
        for canonical, aliases in self.canonical_to_aliases.items():
            all_terms.append((canonical, canonical))
            for alias in aliases:
                all_terms.append((alias, canonical))
        
        # Sort by length descending
        all_terms.sort(key=lambda x: len(x[0]), reverse=True)
        
        for term, canonical in all_terms:
            # Create pattern with word boundaries
            # Handle special characters in gate names
            escaped = re.escape(term)
            # Allow for optional "gate" suffix
            pattern = rf'\b{escaped}(?:\s+gate)?\b'
            
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                self.gate_patterns.append((canonical, compiled))
            except re.error:
                logger.warning(f"Failed to compile pattern for: {term}")
    
    def normalize(self, gate_text: str) -> Optional[str]:
        """
        Normalize a gate name to its canonical form.
        
        Parameters
        ----------
        gate_text : str
            Gate name or alias.
        
        Returns
        -------
        Optional[str]
            Canonical gate name, or None if not recognized.
        
        Examples
        --------
        >>> vocab.normalize("hadamard gate")
        'H'
        >>> vocab.normalize("controlled-not")
        'CNOT'
        """
        # Clean the input
        cleaned = gate_text.lower().strip()
        cleaned = re.sub(r'\s+gate$', '', cleaned)  # Remove "gate" suffix
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize spaces
        
        # Direct lookup
        if cleaned in self.alias_to_canonical:
            return self.alias_to_canonical[cleaned]
        
        # Try without hyphens
        cleaned_no_hyphen = cleaned.replace('-', ' ').replace('_', ' ')
        if cleaned_no_hyphen in self.alias_to_canonical:
            return self.alias_to_canonical[cleaned_no_hyphen]
        
        return None
    
    def find_gates(self, text: str) -> List[str]:
        """
        Find all gate names in a text.
        
        Parameters
        ----------
        text : str
            Text to search for gates.
        
        Returns
        -------
        List[str]
            List of canonical gate names found (unique, sorted).
        """
        found_gates = set()
        
        for canonical, pattern in self.gate_patterns:
            if pattern.search(text):
                found_gates.add(canonical)
        
        return sorted(list(found_gates))
    
    def find_gates_with_positions(
        self, 
        text: str
    ) -> List[Tuple[str, int, int, str]]:
        """
        Find gates with their positions in text.
        
        Parameters
        ----------
        text : str
            Text to search.
        
        Returns
        -------
        List[Tuple[str, int, int, str]]
            List of (canonical_name, start, end, matched_text).
        """
        results = []
        
        for canonical, pattern in self.gate_patterns:
            for match in pattern.finditer(text):
                results.append((
                    canonical,
                    match.start(),
                    match.end(),
                    match.group()
                ))
        
        # Sort by position
        results.sort(key=lambda x: x[1])
        
        return results
    
    def add_gate(self, canonical: str, aliases: List[str]) -> None:
        """
        Add a new gate to the vocabulary.
        
        Parameters
        ----------
        canonical : str
            Canonical gate name.
        aliases : List[str]
            List of aliases.
        
        Notes
        -----
        This method allows dynamic extension of the vocabulary.
        """
        self.canonical_to_aliases[canonical] = aliases
        self.alias_to_canonical[canonical.lower()] = canonical
        for alias in aliases:
            self.alias_to_canonical[alias.lower()] = canonical
        
        # Recompile patterns
        self._compile_patterns()
        logger.info(f"Added gate '{canonical}' with {len(aliases)} aliases")
    
    def get_all_gates(self) -> List[str]:
        """Get list of all canonical gate names."""
        return sorted(self.canonical_to_aliases.keys())


class GateExtractor:
    """
    Extracts quantum gates from text content.
    
    This class coordinates gate extraction from multiple
    text sources and provides aggregated results.
    
    Attributes
    ----------
    vocabulary : GateVocabulary
        Gate vocabulary for matching.
    
    Examples
    --------
    >>> extractor = GateExtractor()
    >>> result = extractor.extract(
    ...     caption="Circuit with H and CNOT gates",
    ...     contexts=["We apply Hadamard followed by controlled-X"]
    ... )
    >>> print(result.gates)
    ['CNOT', 'H']
    """
    
    def __init__(self, vocabulary: GateVocabulary = None):
        """
        Initialize the gate extractor.
        
        Parameters
        ----------
        vocabulary : GateVocabulary
            Gate vocabulary. If None, creates default.
        """
        self.vocabulary = vocabulary or GateVocabulary()
    
    def extract(
        self,
        caption: str = "",
        contexts: List[str] = None,
        additional_text: str = ""
    ) -> GateExtractionResult:
        """
        Extract gates from figure-related text.
        
        Parameters
        ----------
        caption : str
            Figure caption.
        contexts : List[str]
            Surrounding context texts.
        additional_text : str
            Any additional text to analyze.
        
        Returns
        -------
        GateExtractionResult
            Extraction result with gates and metadata.
        """
        if contexts is None:
            contexts = []
        
        all_gates = []
        raw_matches = []
        source_evidence = {'caption': [], 'context': [], 'additional': []}
        
        # Extract from caption (highest priority)
        if caption:
            caption_gates = self.vocabulary.find_gates(caption)
            all_gates.extend(caption_gates)
            source_evidence['caption'] = caption_gates
            
            # Also get raw matches
            for g, start, end, matched in self.vocabulary.find_gates_with_positions(caption):
                raw_matches.append(matched)
        
        # Extract from contexts
        for ctx in contexts:
            ctx_gates = self.vocabulary.find_gates(ctx)
            all_gates.extend(ctx_gates)
            source_evidence['context'].extend(ctx_gates)
            
            for g, start, end, matched in self.vocabulary.find_gates_with_positions(ctx):
                raw_matches.append(matched)
        
        # Extract from additional text
        if additional_text:
            add_gates = self.vocabulary.find_gates(additional_text)
            all_gates.extend(add_gates)
            source_evidence['additional'] = add_gates
        
        # Count gates
        gate_counts = Counter(all_gates)
        
        # Get unique sorted list
        unique_gates = sorted(set(all_gates))
        
        # Calculate confidence based on evidence
        confidence = self._calculate_confidence(gate_counts, source_evidence)
        
        return GateExtractionResult(
            gates=unique_gates,
            gate_counts=dict(gate_counts),
            raw_matches=raw_matches,
            confidence=confidence,
            source_evidence=source_evidence
        )
    
    def _calculate_confidence(
        self,
        gate_counts: Counter,
        source_evidence: Dict[str, List[str]]
    ) -> float:
        """
        Calculate confidence score for the extraction.
        
        Parameters
        ----------
        gate_counts : Counter
            Count of each gate.
        source_evidence : Dict
            Evidence by source.
        
        Returns
        -------
        float
            Confidence score (0-1).
        """
        if not gate_counts:
            return 0.0
        
        score = 0.0
        
        # More gates = higher confidence
        score += min(0.3, len(gate_counts) * 0.05)
        
        # Gates found in caption = higher confidence
        if source_evidence.get('caption'):
            score += 0.3
        
        # Gates found in multiple contexts = higher confidence
        if len(source_evidence.get('context', [])) > 0:
            score += 0.2
        
        # Common quantum gates increase confidence
        common_gates = {'H', 'CNOT', 'X', 'Z', 'RX', 'RY', 'RZ', 'CZ', 'MEASURE'}
        found_common = set(gate_counts.keys()) & common_gates
        score += min(0.2, len(found_common) * 0.05)
        
        return min(1.0, score)
    
    def extract_from_descriptions(
        self,
        descriptions: List[str]
    ) -> GateExtractionResult:
        """
        Extract gates from a list of descriptions.
        
        Parameters
        ----------
        descriptions : List[str]
            List of description texts.
        
        Returns
        -------
        GateExtractionResult
            Extraction result.
        """
        if not descriptions:
            return GateExtractionResult()
        
        # Use first description as "caption", rest as contexts
        caption = descriptions[0] if descriptions else ""
        contexts = descriptions[1:] if len(descriptions) > 1 else []
        
        return self.extract(caption=caption, contexts=contexts)


class GateCleaner:
    """
    Cleans and normalizes extracted gate lists.
    
    This class provides post-processing for gate lists to:
    1. Remove duplicates
    2. Fix common errors
    3. Standardize formatting
    
    Examples
    --------
    >>> cleaner = GateCleaner()
    >>> cleaned = cleaner.clean(['H', 'hadamard', 'CNOT', 'CX'])
    >>> print(cleaned)
    ['CNOT', 'H']
    """
    
    def __init__(self):
        """Initialize the gate cleaner."""
        self.vocabulary = GateVocabulary()
    
    def clean(self, gates: List[str]) -> List[str]:
        """
        Clean a list of gate names.
        
        Parameters
        ----------
        gates : List[str]
            List of gate names (possibly with duplicates/variants).
        
        Returns
        -------
        List[str]
            Cleaned, unique, sorted list of canonical names.
        """
        cleaned = set()
        
        for gate in gates:
            normalized = self.vocabulary.normalize(gate)
            if normalized:
                cleaned.add(normalized)
            elif gate.upper() in self.vocabulary.canonical_to_aliases:
                cleaned.add(gate.upper())
        
        return sorted(list(cleaned))
    
    def validate_gate(self, gate: str) -> bool:
        """
        Check if a gate name is valid.
        
        Parameters
        ----------
        gate : str
            Gate name to validate.
        
        Returns
        -------
        bool
            True if valid gate name.
        """
        return self.vocabulary.normalize(gate) is not None


def extract_gates_from_text(
    text: str,
    vocabulary: GateVocabulary = None
) -> List[str]:
    """
    Simple function to extract gates from text.
    
    Parameters
    ----------
    text : str
        Text to analyze.
    vocabulary : GateVocabulary
        Vocabulary to use. If None, uses default.
    
    Returns
    -------
    List[str]
        List of canonical gate names found.
    
    Examples
    --------
    >>> gates = extract_gates_from_text("Apply H then CNOT")
    >>> print(gates)
    ['CNOT', 'H']
    """
    vocab = vocabulary or GateVocabulary()
    return vocab.find_gates(text)


def clean_gate_name(gate: str) -> Optional[str]:
    """
    Clean and normalize a single gate name.
    
    Parameters
    ----------
    gate : str
        Gate name to clean.
    
    Returns
    -------
    Optional[str]
        Canonical gate name, or None if not recognized.
    
    Notes
    -----
    This function is referenced in the documentation as:
    "see method clean_gate_name() in file gate_extraction.py"
    
    It handles common variations like:
    - "Hadamard" -> "H"
    - "CNOT gate" -> "CNOT"
    - "controlled-X" -> "CNOT"
    
    Examples
    --------
    >>> clean_gate_name("hadamard gate")
    'H'
    >>> clean_gate_name("CX")
    'CNOT'
    """
    vocab = GateVocabulary()
    return vocab.normalize(gate)


if __name__ == "__main__":
    # Test the gate extraction
    print("Testing Gate Extraction...")
    
    extractor = GateExtractor()
    
    test_texts = [
        "Apply a Hadamard gate followed by a CNOT gate",
        "The circuit uses X, Y, and Z Pauli gates",
        "Controlled-Z and SWAP operations are applied",
        "Toffoli gate for the AND operation",
        "RX(π/2) and RZ(θ) rotations"
    ]
    
    for text in test_texts:
        result = extractor.extract(caption=text)
        print(f"\nText: {text}")
        print(f"Gates: {result.gates}")
        print(f"Confidence: {result.confidence:.2f}")
    
    print("\nTest completed!")
