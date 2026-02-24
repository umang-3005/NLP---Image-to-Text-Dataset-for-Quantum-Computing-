"""
Configuration module for the Quantum Circuit Dataset Pipeline.

This module contains all configuration parameters, thresholds, and paths
used throughout the pipeline. Using a centralized configuration ensures
reproducibility and makes the pipeline easily adjustable.

Author: [Your Name]
Exam ID: 37
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set


@dataclass
class PathConfig:
    """
    Configuration for all file and directory paths.
    
    Attributes
    ----------
    base_dir : Path
        Root directory of the project.
    paper_list_file : Path
        Path to the input paper list file.
    output_dir : Path
        Directory for all output files.
    images_dir : Path
        Directory for extracted PNG images.
    temp_dir : Path
        Directory for temporary files (PDFs, intermediate data).
    logs_dir : Path
        Directory for log files.
    """
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    @property
    def paper_list_file(self) -> Path:
        return self.base_dir / "paper_list_37.txt"
    
    @property
    def output_dir(self) -> Path:
        return self.base_dir / "output"
    
    @property
    def images_dir(self) -> Path:
        """Final directory for accepted quantum circuit images."""
        return self.base_dir / "images_37"
    
    @property
    def phase1_images_dir(self) -> Path:
        """Directory for Phase 1 raw figure extraction before filtering."""
        return self.base_dir / "phase1_raw_figures"
    
    @property
    def rejected_images_dir(self) -> Path:
        """Directory for rejected (non-circuit) figures."""
        return self.base_dir / "rejected_figures"
    
    @property
    def temp_dir(self) -> Path:
        return self.base_dir / "temp"
    
    @property
    def pdfs_dir(self) -> Path:
        return self.temp_dir / "pdfs"
    
    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"
    
    @property
    def dataset_json(self) -> Path:
        return self.output_dir / "dataset_37.json"
    
    @property
    def paper_counts_csv(self) -> Path:
        return self.output_dir / "paper_list_counts_37.csv"
    
    def create_directories(self) -> None:
        """Create all necessary directories including Phase 1 and rejected storage."""
        for dir_path in [self.output_dir, self.images_dir, self.phase1_images_dir,
                         self.rejected_images_dir, self.temp_dir, self.pdfs_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class ExtractionConfig:
    """
    Configuration for PDF and figure extraction.
    
    Attributes
    ----------
    target_image_count : int
        Number of quantum circuit images to collect before stopping.
    min_image_width : int
        Minimum width in pixels for valid images.
    min_image_height : int
        Minimum height in pixels for valid images.
    max_image_width : int
        Maximum width to avoid full-page extractions.
    max_image_height : int
        Maximum height to avoid full-page extractions.
    min_aspect_ratio : float
        Minimum width/height ratio (circuits are usually wider than tall).
    max_aspect_ratio : float
        Maximum width/height ratio.
    image_dpi : int
        DPI for rendering vector graphics to PNG.
    """
    target_image_count: int = 250
    min_image_width: int = 100
    min_image_height: int = 50
    max_image_width: int = 3000
    max_image_height: int = 2000
    min_aspect_ratio: float = 0.3
    max_aspect_ratio: float = 10.0
    image_dpi: int = 150


@dataclass
class NLPConfig:
    """
    Configuration for NLP-based text processing and extraction.
    
    Attributes
    ----------
    circuit_keywords : Set[str]
        Keywords indicating quantum circuit content in text.
    quantum_gates : Dict[str, List[str]]
        Mapping of normalized gate names to their aliases/variants.
    quantum_algorithms : Dict[str, List[str]]
        Mapping of algorithm names to their aliases/variants.
    figure_reference_patterns : List[str]
        Regex patterns for detecting figure references in text.
    """
    circuit_keywords: Set[str] = field(default_factory=lambda: {
        'circuit', 'quantum circuit', 'gate', 'qubit', 'qubits',
        'hadamard', 'cnot', 'controlled', 'unitary', 'measurement',
        'ancilla', 'register', 'quantum gate', 'quantum register',
        'pauli', 'rotation', 'phase', 'entanglement', 'superposition',
        'basis state', 'quantum operation', 'quantum channel'
    })
    
    quantum_gates: Dict[str, List[str]] = field(default_factory=lambda: {
        # Single-qubit gates
        'X': ['x', 'x gate', 'pauli-x', 'pauli x', 'not', 'bit flip', 'σx', 'sigma-x'],
        'Y': ['y', 'y gate', 'pauli-y', 'pauli y', 'σy', 'sigma-y'],
        'Z': ['z', 'z gate', 'pauli-z', 'pauli z', 'phase flip', 'σz', 'sigma-z'],
        'H': ['h', 'hadamard', 'hadamard gate', 'h gate'],
        'S': ['s', 's gate', 'phase gate', 'p gate', '√z', 'sqrt-z'],
        'T': ['t', 't gate', 'π/8', 'pi/8 gate'],
        'I': ['i', 'identity', 'identity gate', 'id'],
        'RX': ['rx', 'r_x', 'rotation-x', 'x rotation', 'rx(θ)', 'rx(theta)'],
        'RY': ['ry', 'r_y', 'rotation-y', 'y rotation', 'ry(θ)', 'ry(theta)'],
        'RZ': ['rz', 'r_z', 'rotation-z', 'z rotation', 'rz(θ)', 'rz(theta)'],
        'U': ['u', 'u gate', 'universal gate', 'u3', 'u2', 'u1'],
        'SX': ['sx', '√x', 'sqrt-x', 'v gate'],
        
        # Two-qubit gates
        'CNOT': ['cnot', 'cx', 'controlled-not', 'controlled not', 'c-not', 
                 'controlled-x', 'controlled x', 'feynman gate'],
        'CZ': ['cz', 'controlled-z', 'controlled z', 'c-z', 'cphase'],
        'CY': ['cy', 'controlled-y', 'controlled y', 'c-y'],
        'SWAP': ['swap', 'swap gate', 'exchange'],
        'iSWAP': ['iswap', 'i-swap'],
        'CSWAP': ['cswap', 'fredkin', 'fredkin gate', 'controlled swap'],
        'CRX': ['crx', 'controlled-rx', 'controlled rx'],
        'CRY': ['cry', 'controlled-ry', 'controlled ry'],
        'CRZ': ['crz', 'controlled-rz', 'controlled rz'],
        'CH': ['ch', 'controlled-h', 'controlled hadamard'],
        'CS': ['cs', 'controlled-s', 'controlled s'],
        'CT': ['ct', 'controlled-t', 'controlled t'],
        'DCX': ['dcx', 'double cnot'],
        'ECR': ['ecr', 'echoed cross-resonance'],
        
        # Three-qubit gates
        'Toffoli': ['toffoli', 'ccnot', 'ccx', 'controlled-controlled-not', 
                    'double controlled not', 'and gate'],
        'CCZ': ['ccz', 'controlled-controlled-z', 'double controlled z'],
        'CCCX': ['cccx', 'c3x', 'triple controlled x'],
        
        # Special gates
        'MEASURE': ['measure', 'measurement', 'readout', 'm'],
        'RESET': ['reset', 'initialize', '|0⟩'],
        'BARRIER': ['barrier', 'sync'],
    })
    
    quantum_algorithms: Dict[str, List[str]] = field(default_factory=lambda: {
        "Shor's algorithm": ['shor', "shor's", 'shor algorithm', 'integer factorization',
                             'period finding', 'order finding'],
        "Grover's algorithm": ['grover', "grover's", 'grover search', 'amplitude amplification',
                               'database search', 'unstructured search'],
        'QAOA': ['qaoa', 'quantum approximate optimization', 'approximate optimization algorithm'],
        'VQE': ['vqe', 'variational quantum eigensolver', 'variational eigensolver'],
        'Quantum Fourier Transform': ['qft', 'quantum fourier', 'fourier transform circuit'],
        'Quantum Phase Estimation': ['qpe', 'phase estimation', 'eigenvalue estimation'],
        'Quantum Teleportation': ['teleportation', 'quantum teleportation', 'state transfer'],
        'Superdense Coding': ['superdense', 'dense coding', 'superdense coding'],
        'Quantum Error Correction': ['error correction', 'qec', 'fault tolerant', 'fault-tolerant',
                                      'syndrome measurement', 'stabilizer code'],
        'Bernstein-Vazirani': ['bernstein-vazirani', 'bernstein vazirani', 'hidden string'],
        'Deutsch-Jozsa': ['deutsch-jozsa', 'deutsch jozsa', 'constant or balanced'],
        'Simon': ['simon algorithm', "simon's algorithm", 'simon problem'],
        'HHL': ['hhl', 'harrow-hassidim-lloyd', 'linear systems'],
        'Quantum Walk': ['quantum walk', 'quantum random walk', 'coined walk'],
        'Quantum Counting': ['quantum counting', 'counting algorithm'],
        'Quantum Simulation': ['quantum simulation', 'hamiltonian simulation', 
                               'trotter', 'trotterization', 'product formula'],
        'QRAM': ['qram', 'quantum ram', 'quantum memory'],
        'Quantum Machine Learning': ['qml', 'quantum machine learning', 'quantum classifier',
                                      'quantum neural network', 'qnn', 'variational circuit'],
        'Bell State Preparation': ['bell state', 'bell pair', 'epr pair', 'entanglement generation'],
        'GHZ State Preparation': ['ghz state', 'ghz circuit', 'greenberger-horne-zeilinger'],
        'W State Preparation': ['w state', 'w state preparation'],
    })
    
    figure_reference_patterns: List[str] = field(default_factory=lambda: [
        r'[Ff]ig(?:ure)?\.?\s*(\d+)',
        r'[Ff]igure\s+(\d+)',
        r'[Ff]ig\.\s*(\d+)',
        r'\([Ff]ig\.?\s*(\d+)\)',
        r'[Ss]chematic\s+(\d+)',
        r'[Cc]ircuit\s+(\d+)',
    ])
    
    # Confidence thresholds
    circuit_keyword_threshold: int = 2  # Min keywords to consider text relevant
    gate_confidence_threshold: float = 0.5  # Min confidence for gate extraction


@dataclass
class NetworkConfig:
    """
    Configuration for network requests and rate limiting.
    
    Attributes
    ----------
    arxiv_request_delay : float
        Delay in seconds between arXiv requests (as per robots.txt guidelines).
    max_retries : int
        Maximum number of retry attempts for failed requests.
    request_timeout : int
        Request timeout in seconds.
    arxiv_api_url : str
        Base URL for arXiv API.
    """
    arxiv_request_delay: float = 3.0  # 3 seconds delay when using arXiv API
    max_retries: int = 3
    request_timeout: int = 60
    arxiv_api_url: str = "http://export.arxiv.org/api/query"


@dataclass
class PipelineConfig:
    """
    Master configuration combining all sub-configurations.
    
    Attributes
    ----------
    exam_id : int
        Exam ID for file naming.
    paths : PathConfig
        Path configuration.
    extraction : ExtractionConfig
        Extraction configuration.
    nlp : NLPConfig
        NLP configuration.
    network : NetworkConfig
        Network/rate limiting configuration.
    random_seed : int
        Random seed for reproducibility.
    log_level : str
        Logging level.
    """
    exam_id: int = 37
    paths: PathConfig = field(default_factory=PathConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    random_seed: int = 42
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Initialize directories after configuration is created."""
        self.paths.create_directories()


# Global configuration instance
CONFIG = PipelineConfig()
