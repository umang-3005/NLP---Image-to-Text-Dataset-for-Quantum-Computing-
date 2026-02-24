"""
Quantum Circuit Dataset Pipeline - Source Package.

This package provides a complete NLP-based pipeline for extracting
quantum circuit images and their metadata from scientific papers.

Modules
-------
config : Pipeline configuration
data_models : Data structures and models
paper_acquisition : arXiv paper download and tracking
pdf_extraction : PDF parsing and figure extraction
text_extraction : Text extraction and figure-text linking
circuit_detection : NLP-based quantum circuit detection
gate_extraction : Quantum gate name extraction
algorithm_identification : Quantum algorithm identification
quality_control : Dataset validation and cleaning
dataset_export : Final dataset export
main : Main pipeline orchestrator

Author: [Umang Dholakiya]
Exam ID: 37
"""

__version__ = "1.0.0"
__author__ = "[Umang Dholakiya]"
__exam_id__ = 37

from .config import CONFIG
from .data_models import (
    PaperInfo,
    QuantumCircuitImage,
    DatasetStatistics,
    TextPosition,
    FigureInfo
)
from .main import QuantumCircuitDatasetPipeline, main

__all__ = [
    'CONFIG',
    'PaperInfo',
    'QuantumCircuitImage',
    'DatasetStatistics',
    'TextPosition',
    'FigureInfo',
    'QuantumCircuitDatasetPipeline',
    'main'
]
