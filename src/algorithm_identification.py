"""
Quantum Algorithm/Problem Identification Module.

This module uses NLP techniques to identify which quantum
algorithm or problem is being demonstrated by a circuit.

The identification uses:
1. Keyword matching with a curated vocabulary
2. Context analysis
3. Section title extraction
4. Reference to related algorithms

Author: [Your Name]
Exam ID: 37
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
from dataclasses import dataclass, field

import sys
sys.path.append(str(Path(__file__).parent))

from config import CONFIG
from utils.logging_utils import setup_logger


# Module logger
logger = setup_logger(__name__)


@dataclass
class AlgorithmIdentificationResult:
    """
    Result of quantum algorithm identification.
    
    Attributes
    ----------
    algorithm : str
        Primary identified algorithm/problem.
    confidence : float
        Confidence score (0-1).
    alternatives : List[str]
        Alternative possible algorithms.
    evidence : List[str]
        Text evidence supporting the identification.
    source : str
        Source of the identification (caption, context, title).
    """
    algorithm: str = "Unspecified quantum circuit"
    confidence: float = 0.0
    alternatives: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    source: str = ""


class AlgorithmVocabulary:
    """
    Manages the comprehensive vocabulary of quantum algorithms and problems.
    
    This class provides pattern matching and normalization for a wide
    range of quantum computing concepts, from textbook algorithms to
    NISQ applications and error correction.
    """
    
    def __init__(self):
        """Initialize with a comprehensive hardcoded vocabulary."""
        self.algorithms = self._get_comprehensive_vocabulary()
        self.alias_to_algorithm = {}
        self.patterns = []
        
        self._build_mappings()
        self._compile_patterns()
    
    def _get_comprehensive_vocabulary(self) -> Dict[str, List[str]]:
        """
        Returns the master dictionary of quantum circuit problems.
        Categorized for maintenance, but returns a flat dict for the matcher.
        """
        vocab = {}

       
        # 1. FOUNDATIONAL & TEXTBOOK ALGORITHMS
       
        vocab.update({
            "Deutsch-Jozsa Algorithm": ["deutsch jozsa", "deutsch-jozsa", "constant vs balanced", "DJ algorithm"],
            "Bernstein-Vazirani Algorithm": ["bernstein vazirani", "bernstein-vazirani", "secret string finding", "BV algorithm"],
            "Simon's Algorithm": ["simon's algorithm", "simon algorithm", "hidden subgroup", "period finding"],
            "Grover's Algorithm": ["grover", "amplitude amplification", "unstructured search", "grover search", "oracle search"],
            "Shor's Algorithm": ["shor", "prime factorization", "order finding", "modular exponentiation", "period finding"],
            "Quantum Phase Estimation (QPE)": ["phase estimation", "QPE", "eigenvalue estimation", "phase kickback", "iterative phase estimation", "IPE"],
            "Quantum Fourier Transform (QFT)": ["quantum fourier transform", "QFT", "inverse QFT", "IQFT", "approximate QFT"],
            "HHL Algorithm": ["HHL", "harrow hassidim lloyd", "linear systems", "matrix inversion", "quantum linear system"],
            "Quantum Counting": ["quantum counting", "counting algorithm", "approximate counting"],
            "Quantum Amplitude Estimation (QAE)": ["amplitude estimation", "QAE", "canonical amplitude estimation", "maximum likelihood amplitude estimation"],
        })

       
        # 2. VARIATIONAL & NISQ (Modern Era)
       
        vocab.update({
            "Variational Quantum Eigensolver (VQE)": ["VQE", "variational quantum eigensolver", "ground state energy", "UCCSD", "unitary coupled cluster", "adaptive VQE", "ADAPT-VQE"],
            "Quantum Approximate Optimization (QAOA)": ["QAOA", "quantum approximate optimization", "maxcut", "max-cut", "combinatorial optimization", "mixer hamiltonian", "driver hamiltonian"],
            "Variational Quantum Classifier (VQC)": ["VQC", "variational classifier", "quantum classifier", "circuit centric classifier"],
            "Quantum Neural Network (QNN)": ["QNN", "quantum neural network", "parameterized quantum circuit", "PQC", "quantum perceptron", "dissipative QNN"],
            "Hardware Efficient Ansatz": ["hardware efficient", "HEA", "entangling layers", "ry-cnot", "alternating layer"],
            "Quantum Kernel Method": ["quantum kernel", "kernel estimation", "QSVM", "quantum support vector machine", "fidelity kernel"],
            "Variational Autoencoder (QVAE)": ["QVAE", "quantum autoencoder", "variational autoencoder", "compression circuit"],
        })

       
        # 3. QUANTUM ERROR CORRECTION (QEC) & FAULT TOLERANCE
       
        vocab.update({
            "Quantum Error Correction (General)": ["error correction", "QEC", "syndrome measurement", "stabilizer measurement", "parity check"],
            "Surface Code": ["surface code", "toric code", "lattice surgery", "plaquette", "star operator", "rotated surface code"],
            "Repetition Code": ["repetition code", "bit flip code", "phase flip code", "majority vote", "3-qubit code"],
            "Shor Code": ["shor code", "9-qubit code", "nine qubit code"],
            "Steane Code": ["steane code", "7-qubit code", "seven qubit code", "color code", "triangular code"],
            "Bacon-Shor Code": ["bacon-shor", "bacon shor", "subsystem code"],
            "Heavy Hex Code": ["heavy hex", "heavy-hex", "ibm quantum code"],
            "Bosonic Codes": ["cat code", "gkp code", "gottesman-kitaev-preskill", "binomial code"],
            "Magic State Distillation": ["magic state distillation", "magic state", "distillation circuit", "T-state distillation"],
            "Dynamical Decoupling": ["dynamical decoupling", "spin echo", "CPMG", "XY4", "pulse sequence", "DD sequence"],
        })

       
        # 4. SUBROUTINES & PRIMITIVES (The Building Blocks)
       
        vocab.update({
            "Swap Test": ["swap test", "overlap measurement", "inner product measurement", "fidelity test"],
            "Hadamard Test": ["hadamard test", "expectation value estimation", "imaginary part estimation"],
            "Quantum Teleportation": ["teleportation", "teleport", "bell measurement", "ebit"],
            "Superdense Coding": ["superdense coding", "dense coding", "information transfer"],
            "Entanglement Swapping": ["entanglement swapping", "repeater", "swap operation"],
            "State Preparation": ["state preparation", "loading data", "amplitude encoding", "basis encoding"],
            "Oracle Implementation": ["oracle", "black box", "marking unitary", "phase oracle"],
        })

       
        # 5. ARITHMETIC & LOGIC CIRCUITS
       
        vocab.update({
            "Quantum Adder": ["quantum adder", "ripple carry", "draper qft adder", "addition circuit", "CDKM ripple carry"],
            "Quantum Multiplier": ["quantum multiplier", "multiplication circuit", "karatsuba", "arithmetic circuit"],
            "Quantum Comparator": ["comparator", "compare", "greater than", "less than"],
            "Quantum RAM (QRAM)": ["QRAM", "quantum ram", "quantum memory", "bucket brigade", "fanout"],
            "Modular Exponentiation": ["modular exponentiation", "mod exp", "modular multiplier"],
        })

       
        # 6. SIMULATION & CHEMISTRY
       
        vocab.update({
            "Hamiltonian Simulation": ["hamiltonian simulation", "time evolution", "trotter", "trotterization", "suzuki-trotter", "product formula"],
            "Qubitization": ["qubitization", "quantum signal processing", "QSP", "block encoding"],
            "Boson Sampling": ["boson sampling", "gaussian boson sampling", "photon network"],
            "Quantum Random Walk": ["quantum walk", "random walk", "walker", "coin operator", "szegedy walk"],
        })

       
        # 7. BENCHMARKING & TOMOGRAPHY
       
        vocab.update({
            "Randomized Benchmarking": ["randomized benchmarking", "RB", "clifford sequences", "interleaved RB", "direct RB"],
            "Quantum Volume": ["quantum volume", "QV", "heavy output", "square circuit"],
            "Tomography": ["state tomography", "process tomography", "QST", "QPT", "maximum likelihood estimation", "shadow tomography"],
            "Cross-Entropy Benchmarking": ["cross entropy", "XEB", "supremacy", "sycamore"],
        })

       
        # 8. CRYPTOGRAPHY & SECURITY
       
        vocab.update({
            "Quantum Key Distribution (QKD)": ["QKD", "BB84", "E91", "B92", "key distribution", "quantum cryptography", "key exchange"],
            "Quantum Money": ["quantum money", "unforgeable", "verification"],
            "Blind Quantum Computing": ["blind quantum computing", "delegated quantum computing", "blind computing"],
        })

        return vocab
    
    def _build_mappings(self) -> None:
        """Build reverse mappings."""
        for algo, aliases in self.algorithms.items():
            self.alias_to_algorithm[algo.lower()] = algo
            for alias in aliases:
                self.alias_to_algorithm[alias.lower()] = algo
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for matching."""
        # Sort by length (longer patterns first to match specific phrases before general words)
        all_terms = []
        for algo, aliases in self.algorithms.items():
            all_terms.append((algo, algo))
            for alias in aliases:
                all_terms.append((alias, algo))
        
        all_terms.sort(key=lambda x: len(x[0]), reverse=True)
        
        for term, algo in all_terms:
            escaped = re.escape(term)
            # Match word boundaries to avoid partial matches inside other words
            pattern = rf'\b{escaped}\b'
            
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                self.patterns.append((algo, compiled))
            except re.error:
                pass
    
    def find_algorithms(self, text: str) -> List[str]:
        """Find algorithms mentioned in text."""
        found = set()
        for algo, pattern in self.patterns:
            if pattern.search(text):
                found.add(algo)
        return list(found)
    
    def normalize(self, algo_text: str) -> Optional[str]:
        """Normalize an algorithm name."""
        cleaned = algo_text.lower().strip()
        return self.alias_to_algorithm.get(cleaned)
    
    def get_all_algorithms(self) -> List[str]:
        """Get all algorithm names."""
        return sorted(self.algorithms.keys())


class AlgorithmIdentifier:
    """
    Identifies quantum algorithms from figure context.
    
    This class combines multiple signals to identify
    which algorithm a circuit implements.
    
    Attributes
    ----------
    vocabulary : AlgorithmVocabulary
        Algorithm vocabulary.
    
    Examples
    --------
    >>> identifier = AlgorithmIdentifier()
    >>> result = identifier.identify(
    ...     caption="Grover's algorithm circuit",
    ...     contexts=["for amplitude amplification"]
    ... )
    >>> print(result.algorithm)
    "Grover's algorithm"
    """
    
    def __init__(self):
        """Initialize the identifier."""
        self.vocabulary = AlgorithmVocabulary()
        
        # Patterns for generic circuit types
        self.generic_patterns = [
            (r'variational\s+(?:quantum\s+)?circuit', 'Variational circuit'),
            (r'parameterized\s+(?:quantum\s+)?circuit', 'Parameterized circuit'),
            (r'ansatz\s+circuit', 'Variational ansatz'),
            (r'encoder\s+circuit', 'Quantum encoder circuit'),
            (r'oracle\s+(?:circuit|implementation)', 'Oracle circuit'),
            (r'state\s+preparation\s+circuit', 'State preparation circuit'),
            (r'measurement\s+circuit', 'Measurement circuit'),
            (r'error\s+correction\s+circuit', 'Quantum Error Correction'),
            (r'entangling\s+(?:layer|circuit)', 'Entangling circuit'),
        ]
    
    def identify(
        self,
        caption: str = "",
        contexts: List[str] = None,
        section_title: str = ""
    ) -> AlgorithmIdentificationResult:
        """
        Identify the quantum algorithm for a circuit.
        
        Parameters
        ----------
        caption : str
            Figure caption.
        contexts : List[str]
            Surrounding context texts.
        section_title : str
            Title of the section containing the figure.
        
        Returns
        -------
        AlgorithmIdentificationResult
            Identification result.
        """
        if contexts is None:
            contexts = []
        
        candidates = []
        evidence = []
        
        # Check section title first (highest priority)
        if section_title:
            algos = self.vocabulary.find_algorithms(section_title)
            for algo in algos:
                candidates.append((algo, 1.0, 'section_title'))
                evidence.append(f"Section: {section_title[:50]}")
        
        # Check caption (high priority)
        if caption:
            algos = self.vocabulary.find_algorithms(caption)
            for algo in algos:
                candidates.append((algo, 0.9, 'caption'))
                evidence.append(f"Caption: {caption[:50]}")
            
            # Check generic patterns in caption
            for pattern, name in self.generic_patterns:
                if re.search(pattern, caption, re.IGNORECASE):
                    candidates.append((name, 0.7, 'caption_generic'))
        
        # Check contexts
        for ctx in contexts:
            algos = self.vocabulary.find_algorithms(ctx)
            for algo in algos:
                candidates.append((algo, 0.6, 'context'))
                evidence.append(f"Context: {ctx[:40]}...")
        
        # Select best candidate
        if candidates:
            # Sort by confidence, then by how specific the algorithm is
            candidates.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
            
            best = candidates[0]
            alternatives = list(set(c[0] for c in candidates[1:4]))
            
            return AlgorithmIdentificationResult(
                algorithm=best[0],
                confidence=best[1],
                alternatives=alternatives,
                evidence=evidence[:5],
                source=best[2]
            )
        
        # Fallback: try to infer from gate patterns
        fallback = self._infer_from_text(caption, contexts)
        if fallback:
            return fallback
        
        return AlgorithmIdentificationResult(
            algorithm="Unspecified quantum circuit",
            confidence=0.1,
            evidence=[],
            source="fallback"
        )
    
    def _infer_from_text(
        self,
        caption: str,
        contexts: List[str]
    ) -> Optional[AlgorithmIdentificationResult]:
        """
        Try to infer algorithm from text patterns.
        
        Parameters
        ----------
        caption : str
            Caption text.
        contexts : List[str]
            Context texts.
        
        Returns
        -------
        Optional[AlgorithmIdentificationResult]
            Inferred result if possible.
        """
        combined_text = caption + " " + " ".join(contexts)
        combined_lower = combined_text.lower()
        
        # Look for specific patterns
        inferences = []
        
        # Expanded keyword matching
        keyword_patterns = {
            'Bell State Preparation': ['bell state', 'bell pair', 'entangled pair'],
            'GHZ State Preparation': ['ghz', 'greenberger-horne-zeilinger'],
            'Quantum Teleportation': ['teleport', 'quantum teleportation'],
            'Quantum Error Correction': ['error correction', 'fault tolerance', 'error detection'],
            'QAOA': ['optimization', 'maxcut', 'quantum approximate optimization'],
            'VQE': ['eigenvalue', 'eigensolver', 'variational quantum eigensolver'],
            'Quantum Fourier Transform': ['fourier', 'quantum fourier transform'],
            'Quantum Machine Learning': ['quantum machine learning', 'qml', 'quantum neural network'],
            'Quantum Cryptography': ['quantum cryptography', 'quantum key distribution', 'qkd']
        }
        
        for problem, keywords in keyword_patterns.items():
            if any(keyword in combined_lower for keyword in keywords):
                inferences.append(problem)
        
        if inferences:
            return AlgorithmIdentificationResult(
                algorithm=inferences[0],
                confidence=0.7,  # Increased confidence for expanded matching
                alternatives=inferences[1:],
                evidence=[f"Inferred from text patterns: {', '.join(inferences)}"],
                source="inference"
            )
        
        return None


class SectionTitleExtractor:
    """
    Extracts section titles from paper text.
    
    This class identifies section headings to provide
    context for algorithm identification.
    
    Examples
    --------
    >>> extractor = SectionTitleExtractor()
    >>> title = extractor.find_section_for_page(full_text, page_num=3)
    """
    
    def __init__(self):
        """Initialize the extractor."""
        # Patterns for section titles
        self.section_patterns = [
            r'^(\d+\.?\s+[A-Z][^.\n]{5,50})$',  # "1. Introduction"
            r'^([IVXLC]+\.?\s+[A-Z][^.\n]{5,50})$',  # "I. Background"
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})$',  # "Related Work"
        ]
    
    def extract_sections(self, full_text: str) -> List[Dict]:
        """
        Extract all section titles from text.
        
        Parameters
        ----------
        full_text : str
            Full text of the document.
        
        Returns
        -------
        List[Dict]
            List of sections with 'title' and 'position'.
        """
        sections = []
        
        for pattern in self.section_patterns:
            for match in re.finditer(pattern, full_text, re.MULTILINE):
                title = match.group(1).strip()
                if len(title) > 3 and len(title) < 100:
                    sections.append({
                        'title': title,
                        'position': match.start()
                    })
        
        # Sort by position
        sections.sort(key=lambda x: x['position'])
        
        return sections
    
    def find_section_for_position(
        self,
        full_text: str,
        char_position: int
    ) -> Optional[str]:
        """
        Find the section title for a given text position.
        
        Parameters
        ----------
        full_text : str
            Full document text.
        char_position : int
            Character position to find section for.
        
        Returns
        -------
        Optional[str]
            Section title if found.
        """
        sections = self.extract_sections(full_text)
        
        # Find the section that contains this position
        current_section = None
        for section in sections:
            if section['position'] <= char_position:
                current_section = section['title']
            else:
                break
        
        return current_section


def identify_quantum_problem(
    caption: str = "",
    contexts: List[str] = None,
    section_title: str = ""
) -> str:
    """
    Simple function to identify the quantum problem.
    
    Parameters
    ----------
    caption : str
        Figure caption.
    contexts : List[str]
        Context texts.
    section_title : str
        Section title.
    
    Returns
    -------
    str
        Identified algorithm/problem name.
    
    Examples
    --------
    >>> problem = identify_quantum_problem(
    ...     caption="Grover's search circuit"
    ... )
    >>> print(problem)
    "Grover's algorithm"
    """
    identifier = AlgorithmIdentifier()
    result = identifier.identify(caption, contexts, section_title)
    return result.algorithm


if __name__ == "__main__":
    # Test the algorithm identifier
    print("Testing Algorithm Identification...")
    
    identifier = AlgorithmIdentifier()
    
    test_cases = [
        {
            'caption': "Quantum circuit for Shor's factoring algorithm",
            'contexts': ["modular exponentiation", "period finding"],
        },
        {
            'caption': "VQE ansatz circuit for molecular simulation",
            'contexts': ["variational eigensolver", "ground state energy"],
        },
        {
            'caption': "Bell state preparation circuit",
            'contexts': ["entanglement generation"],
        },
        {
            'caption': "General quantum circuit implementation",
            'contexts': ["quantum gates are applied"],
        }
    ]
    
    for tc in test_cases:
        result = identifier.identify(
            caption=tc['caption'],
            contexts=tc.get('contexts', [])
        )
        print(f"\nCaption: {tc['caption'][:40]}...")
        print(f"Algorithm: {result.algorithm}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Alternatives: {result.alternatives}")
    
    print("\nTest completed!")
