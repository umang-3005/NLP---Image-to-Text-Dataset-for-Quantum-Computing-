"""
Data models for the Quantum Circuit Dataset Pipeline.

This module defines all data structures used throughout the pipeline,
ensuring type safety and consistent data handling.

Author: [Your Name]
Exam ID: 37
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import json


class ProcessingStatus(Enum):
    """Status of paper processing."""
    NOT_VISITED = "not_visited"
    PROCESSED = "processed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PaperInfo:
    """
    Information about an arXiv paper.
    
    Attributes
    ----------
    arxiv_id : str
        The arXiv identifier (e.g., "2410.08073").
    full_id : str
        Full arXiv identifier with prefix (e.g., "arXiv:2410.08073").
    pdf_url : str
        URL to download the PDF.
    status : ProcessingStatus
        Current processing status.
    image_count : Optional[int]
        Number of valid images extracted (None if not processed).
    error_message : Optional[str]
        Error message if processing failed.
    
    Examples
    --------
    >>> paper = PaperInfo.from_line("arXiv:2410.08073")
    >>> print(paper.pdf_url)
    'https://arxiv.org/pdf/2410.08073.pdf'
    """
    arxiv_id: str
    full_id: str = ""
    pdf_url: str = ""
    status: ProcessingStatus = ProcessingStatus.NOT_VISITED
    image_count: Optional[int] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Generate derived fields."""
        if not self.full_id:
            self.full_id = f"arXiv:{self.arxiv_id}"
        if not self.pdf_url:
            self.pdf_url = f"https://arxiv.org/pdf/{self.arxiv_id}.pdf"
    
    @classmethod
    def from_line(cls, line: str) -> 'PaperInfo':
        """
        Create PaperInfo from a paper list line.
        
        Parameters
        ----------
        line : str
            Line from paper_list_37.txt (e.g., "arXiv:2410.08073").
        
        Returns
        -------
        PaperInfo
            Parsed paper information.
        """
        line = line.strip()
        if line.startswith("arXiv:"):
            arxiv_id = line[6:]
        else:
            arxiv_id = line
        return cls(arxiv_id=arxiv_id, full_id=line)


@dataclass
class TextPosition:
    """
    Position of extracted text in the source document.
    
    The position is defined as character offsets in the concatenated
    plain-text representation of the paper. This representation is
    created by extracting text page-by-page and section-by-section,
    preserving the reading order.
    
    Attributes
    ----------
    start : int
        Starting character offset (0-indexed, inclusive).
    end : int
        Ending character offset (0-indexed, exclusive).
    source : str
        Source type: 'caption', 'paragraph', 'section', 'abstract'.
    page : Optional[int]
        Page number where the text appears.
    
    Notes
    -----
    The text_positions field stores (start, end) tuples. The convention is:
    - All positions refer to the full extracted text of the paper
    - The full text is stored in a companion file for reproducibility
    - Offsets are 0-indexed, with 'end' being exclusive (Python slice convention)
    """
    start: int
    end: int
    source: str = "paragraph"
    page: Optional[int] = None
    
    def to_tuple(self) -> Tuple[int, int]:
        """Convert to simple tuple for JSON serialization."""
        return (self.start, self.end)
    
    @classmethod
    def from_tuple(cls, t: Tuple[int, int], source: str = "paragraph") -> 'TextPosition':
        """Create from tuple."""
        return cls(start=t[0], end=t[1], source=source)


@dataclass
class FigureInfo:
    """
    Information about an extracted figure.
    
    Attributes
    ----------
    filename : str
        Generated filename for the image (e.g., "2410.08073_p3_f1.png").
    arxiv_id : str
        arXiv identifier of the source paper.
    page_number : int
        Page number where the figure appears (1-indexed).
    figure_number : int
        Figure number/label from the paper (1-indexed).
    bbox : Tuple[float, float, float, float]
        Bounding box coordinates (x0, y0, x1, y1) on the page.
    caption : str
        Extracted figure caption.
    caption_position : Optional[TextPosition]
        Position of the caption in the full text.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    is_vector : bool
        Whether the figure is vector graphics (vs raster).
    """
    filename: str
    arxiv_id: str
    page_number: int
    figure_number: int
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    caption: str = ""
    caption_position: Optional[TextPosition] = None
    width: int = 0
    height: int = 0
    is_vector: bool = True


@dataclass
class QuantumCircuitImage:
    """
    Complete information about a quantum circuit image for the final dataset.
    
    This is the main data structure for the output JSON file.
    
    Attributes
    ----------
    filename : str
        Image filename (main key in the JSON).
    arxiv_id : str
        arXiv number of the source paper.
    page_number : int
        Page number where the image appears (1-indexed).
    figure_number : int
        Figure number in the paper (1-indexed).
    quantum_gates : List[str]
        List of quantum gates appearing in the circuit.
    quantum_problem : str
        The quantum problem/algorithm realized by the circuit.
    descriptions : List[str]
        Descriptive text parts from the paper.
    text_positions : List[Tuple[int, int]]
        Position tuples (start, end) for each description.
    confidence_score : float
        Confidence that this is a valid quantum circuit (0-1).
    metadata : Dict[str, Any]
        Additional metadata for quality control.
    
    Examples
    --------
    >>> circuit = QuantumCircuitImage(
    ...     filename="2410.08073_p3_f1.png",
    ...     arxiv_id="2410.08073",
    ...     page_number=3,
    ...     figure_number=1,
    ...     quantum_gates=["H", "CNOT", "MEASURE"],
    ...     quantum_problem="Bell State Preparation",
    ...     descriptions=["Figure 1 shows the Bell state preparation circuit."],
    ...     text_positions=[(1234, 1289)]
    ... )
    """
    filename: str
    arxiv_id: str
    page_number: int
    figure_number: int
    quantum_gates: List[str] = field(default_factory=list)
    quantum_problem: str = "Unspecified quantum circuit"
    descriptions: List[str] = field(default_factory=list)
    text_positions: List[Tuple[int, int]] = field(default_factory=list)
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    caption: str = ""  # Add caption field to store figure captions
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation matching the required JSON schema.
            
        Notes
        -----
        Only includes the required fields per the specification:
        - arxiv_id, page_number, figure_number
        - quantum_gates, quantum_problem
        - descriptions, text_positions
        
        Excludes internal fields: confidence_score, metadata
        """
        return {
            "arxiv_id": self.arxiv_id,
            "page_number": self.page_number,
            "figure_number": self.figure_number,
            "quantum_gates": self.quantum_gates,
            "quantum_problem": self.quantum_problem,
            "descriptions": self.descriptions,
            "text_positions": self.text_positions
        }
    
    @classmethod
    def from_dict(cls, filename: str, data: Dict[str, Any]) -> 'QuantumCircuitImage':
        """
        Create from dictionary (for loading from JSON).
        
        Parameters
        ----------
        filename : str
            The image filename (main key).
        data : Dict[str, Any]
            The dictionary data.
        
        Returns
        -------
        QuantumCircuitImage
            Reconstructed object.
        """
        return cls(
            filename=filename,
            arxiv_id=data.get("arxiv_id", ""),
            page_number=data.get("page_number", 0),
            figure_number=data.get("figure_number", 0),
            quantum_gates=data.get("quantum_gates", []),
            quantum_problem=data.get("quantum_problem", ""),
            descriptions=data.get("descriptions", []),
            text_positions=[tuple(p) for p in data.get("text_positions", [])],
            confidence_score=data.get("confidence_score", 0.0),
            metadata=data.get("metadata", {})
        )


@dataclass
class DatasetStatistics:
    """
    Statistics about the compiled dataset.
    
    Attributes
    ----------
    total_papers_processed : int
        Number of papers that were analyzed.
    total_images_extracted : int
        Total number of images extracted.
    total_circuits_found : int
        Number of valid quantum circuit images.
    papers_with_circuits : int
        Number of papers that contained circuits.
    gate_distribution : Dict[str, int]
        Count of each gate type across all circuits.
    algorithm_distribution : Dict[str, int]
        Count of each algorithm type.
    avg_gates_per_circuit : float
        Average number of gates per circuit.
    avg_descriptions_per_image : float
        Average number of description snippets per image.
    """
    total_papers_processed: int = 0
    total_images_extracted: int = 0
    total_circuits_found: int = 0
    papers_with_circuits: int = 0
    gate_distribution: Dict[str, int] = field(default_factory=dict)
    algorithm_distribution: Dict[str, int] = field(default_factory=dict)
    avg_gates_per_circuit: float = 0.0
    avg_descriptions_per_image: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
