"""
Quality Control and Dataset Validation Module.

This module provides:
1. Data consistency validation
2. Schema validation
3. Quality metrics
4. Error detection and reporting
5. Automatic cleaning and fixing

Author: [Your Name]
Exam ID: 37
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter
import re

import sys
sys.path.append(str(Path(__file__).parent))

from config import CONFIG
from data_models import QuantumCircuitImage, DatasetStatistics
from gate_extraction import GateCleaner
from utils.logging_utils import setup_logger


# Module logger
logger = setup_logger(__name__)


@dataclass
class ValidationError:
    """
    Represents a validation error.
    
    Attributes
    ----------
    error_type : str
        Type of error (e.g., 'missing_file', 'invalid_field').
    severity : str
        Severity level ('error', 'warning', 'info').
    message : str
        Description of the error.
    filename : str
        Affected filename.
    field : str
        Affected field name.
    suggestion : str
        Suggested fix.
    """
    error_type: str
    severity: str
    message: str
    filename: str = ""
    field: str = ""
    suggestion: str = ""


@dataclass
class ValidationReport:
    """
    Complete validation report for the dataset.
    
    Attributes
    ----------
    is_valid : bool
        Overall validation result.
    total_images : int
        Total number of images.
    errors : List[ValidationError]
        List of errors found.
    warnings : List[ValidationError]
        List of warnings found.
    statistics : DatasetStatistics
        Dataset statistics.
    """
    is_valid: bool = True
    total_images: int = 0
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    statistics: Optional[DatasetStatistics] = None
    
    def add_error(self, error: ValidationError) -> None:
        """Add an error to the report."""
        if error.severity == 'error':
            self.errors.append(error)
            self.is_valid = False
        else:
            self.warnings.append(error)
    
    def summary(self) -> str:
        """Generate a summary string."""
        return (
            f"Validation Report:\n"
            f"  Valid: {self.is_valid}\n"
            f"  Total Images: {self.total_images}\n"
            f"  Errors: {len(self.errors)}\n"
            f"  Warnings: {len(self.warnings)}"
        )


class DatasetValidator:
    """
    Validates the quantum circuit dataset.
    
    This class checks:
    1. File existence (PNG images)
    2. JSON schema compliance
    3. Field value validity
    4. Cross-reference consistency
    
    Attributes
    ----------
    images_dir : Path
        Directory containing images.
    dataset_path : Path
        Path to the JSON dataset.
    
    Examples
    --------
    >>> validator = DatasetValidator()
    >>> report = validator.validate()
    >>> if not report.is_valid:
    ...     print("Validation failed!")
    ...     for error in report.errors:
    ...         print(f"  {error.message}")
    """
    
    def __init__(
        self,
        images_dir: Path = None,
        dataset_path: Path = None
    ):
        """
        Initialize the validator.
        
        Parameters
        ----------
        images_dir : Path
            Directory with PNG images.
        dataset_path : Path
            Path to JSON dataset file.
        """
        self.images_dir = images_dir or CONFIG.paths.images_dir
        self.dataset_path = dataset_path or CONFIG.paths.dataset_json
        self.gate_cleaner = GateCleaner()
    
    def validate(self) -> ValidationReport:
        """
        Perform full validation of the dataset.
        
        Returns
        -------
        ValidationReport
            Complete validation report.
        """
        logger.info("Starting dataset validation...")
        report = ValidationReport()
        
        # Load dataset
        try:
            dataset = self._load_dataset()
        except Exception as e:
            report.add_error(ValidationError(
                error_type='load_error',
                severity='error',
                message=f"Failed to load dataset: {e}"
            ))
            return report
        
        report.total_images = len(dataset)
        
        # Validate each entry
        for filename, data in dataset.items():
            self._validate_entry(filename, data, report)
        
        # Check for orphan files
        self._check_orphan_files(dataset, report)
        
        # Calculate statistics
        report.statistics = self._calculate_statistics(dataset)
        
        logger.info(report.summary())
        return report
    
    def _load_dataset(self) -> Dict[str, Any]:
        """Load the dataset from JSON."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _validate_entry(
        self,
        filename: str,
        data: Dict[str, Any],
        report: ValidationReport
    ) -> None:
        """
        Validate a single dataset entry.
        
        Parameters
        ----------
        filename : str
            Image filename (key).
        data : Dict
            Entry data.
        report : ValidationReport
            Report to add errors to.
        """
        # Check image file exists
        image_path = self.images_dir / filename
        if not image_path.exists():
            report.add_error(ValidationError(
                error_type='missing_file',
                severity='error',
                message=f"Image file not found: {filename}",
                filename=filename,
                suggestion="Check if the image was saved correctly"
            ))
        
        # Validate required fields
        required_fields = [
            ('arxiv_id', str),
            ('page_number', int),
            ('figure_number', int),
            ('quantum_gates', list),
            ('quantum_problem', str),
            ('descriptions', list),
            ('text_positions', list)
        ]
        
        for field_name, field_type in required_fields:
            if field_name not in data:
                report.add_error(ValidationError(
                    error_type='missing_field',
                    severity='error',
                    message=f"Missing required field: {field_name}",
                    filename=filename,
                    field=field_name
                ))
            elif not isinstance(data[field_name], field_type):
                report.add_error(ValidationError(
                    error_type='wrong_type',
                    severity='error',
                    message=f"Wrong type for {field_name}: expected {field_type.__name__}",
                    filename=filename,
                    field=field_name
                ))
        
        # Validate specific fields
        self._validate_arxiv_id(data.get('arxiv_id', ''), filename, report)
        self._validate_page_number(data.get('page_number', 0), filename, report)
        self._validate_gates(data.get('quantum_gates', []), filename, report)
        self._validate_descriptions(data.get('descriptions', []), 
                                    data.get('text_positions', []), 
                                    filename, report)
    
    def _validate_arxiv_id(
        self,
        arxiv_id: str,
        filename: str,
        report: ValidationReport
    ) -> None:
        """Validate arXiv ID format."""
        if not arxiv_id:
            return
        
        # arXiv ID patterns: YYMM.NNNNN or category/YYMMNNN
        pattern = r'^(\d{4}\.\d{4,5}|[a-z-]+/\d{7})$'
        if not re.match(pattern, arxiv_id):
            report.add_error(ValidationError(
                error_type='invalid_format',
                severity='warning',
                message=f"Invalid arXiv ID format: {arxiv_id}",
                filename=filename,
                field='arxiv_id'
            ))
        
        # Check if arxiv_id matches filename
        if arxiv_id not in filename:
            report.add_error(ValidationError(
                error_type='mismatch',
                severity='warning',
                message=f"arXiv ID doesn't match filename",
                filename=filename,
                field='arxiv_id'
            ))
    
    def _validate_page_number(
        self,
        page_number: int,
        filename: str,
        report: ValidationReport
    ) -> None:
        """Validate page number is reasonable."""
        if page_number < 1:
            report.add_error(ValidationError(
                error_type='invalid_value',
                severity='error',
                message=f"Invalid page number: {page_number}",
                filename=filename,
                field='page_number'
            ))
        elif page_number > 100:
            report.add_error(ValidationError(
                error_type='suspicious_value',
                severity='warning',
                message=f"Unusually high page number: {page_number}",
                filename=filename,
                field='page_number'
            ))
    
    def _validate_gates(
        self,
        gates: List[str],
        filename: str,
        report: ValidationReport
    ) -> List[str]:
        """Validate quantum gate names and return valid gates."""
        if not gates:
            report.add_error(ValidationError(
                error_type='empty_list',
                severity='warning',
                message="No quantum gates listed",
                filename=filename,
                field='quantum_gates',
                suggestion="Consider reviewing the circuit for gates"
            ))
            return []

        valid_gates = []
        for gate in gates:
            if self.gate_cleaner.validate_gate(gate):
                valid_gates.append(gate)
            else:
                report.add_error(ValidationError(
                    error_type='unknown_gate',
                    severity='warning',
                    message=f"Unknown gate name: {gate}",
                    filename=filename,
                    field='quantum_gates',
                    suggestion=f"Consider normalizing to a standard name"
                ))

        return valid_gates
    
    def _validate_descriptions(
        self,
        descriptions: List[str],
        positions: List[Tuple],
        filename: str,
        report: ValidationReport
    ) -> None:
        """Validate descriptions and their positions."""
        if not descriptions:
            report.add_error(ValidationError(
                error_type='empty_list',
                severity='warning',
                message="No descriptions provided",
                filename=filename,
                field='descriptions'
            ))
        
        if len(descriptions) != len(positions):
            report.add_error(ValidationError(
                error_type='length_mismatch',
                severity='error',
                message=f"Mismatch: {len(descriptions)} descriptions, {len(positions)} positions",
                filename=filename,
                field='text_positions'
            ))
        
        for i, pos in enumerate(positions):
            if not isinstance(pos, (list, tuple)) or len(pos) != 2:
                report.add_error(ValidationError(
                    error_type='invalid_format',
                    severity='error',
                    message=f"Invalid position format at index {i}",
                    filename=filename,
                    field='text_positions'
                ))
            elif pos[0] > pos[1]:
                report.add_error(ValidationError(
                    error_type='invalid_range',
                    severity='error',
                    message=f"Invalid position range: start > end at index {i}",
                    filename=filename,
                    field='text_positions'
                ))
    
    def _check_orphan_files(
        self,
        dataset: Dict,
        report: ValidationReport
    ) -> None:
        """Check for image files not in the dataset."""
        if not self.images_dir.exists():
            return
        
        dataset_files = set(dataset.keys())
        disk_files = {f.name for f in self.images_dir.glob("*.png")}
        
        orphans = disk_files - dataset_files
        for orphan in orphans:
            report.add_error(ValidationError(
                error_type='orphan_file',
                severity='warning',
                message=f"Image file not in dataset: {orphan}",
                filename=orphan,
                suggestion="Add to dataset or delete the file"
            ))
        
        missing = dataset_files - disk_files
        for miss in missing:
            report.add_error(ValidationError(
                error_type='missing_file',
                severity='error',
                message=f"Dataset entry without image file: {miss}",
                filename=miss
            ))
    
    def _calculate_statistics(self, dataset: Dict) -> DatasetStatistics:
        """Calculate dataset statistics."""
        stats = DatasetStatistics()
        stats.total_circuits_found = len(dataset)
        
        all_gates = []
        all_algorithms = []
        total_descriptions = 0
        
        arxiv_ids = set()
        
        for filename, data in dataset.items():
            gates = data.get('quantum_gates', [])
            all_gates.extend(gates)
            
            algo = data.get('quantum_problem', '')
            if algo:
                all_algorithms.append(algo)
            
            descriptions = data.get('descriptions', [])
            total_descriptions += len(descriptions)
            
            arxiv_ids.add(data.get('arxiv_id', ''))
        
        stats.gate_distribution = dict(Counter(all_gates))
        stats.algorithm_distribution = dict(Counter(all_algorithms))
        stats.papers_with_circuits = len(arxiv_ids)
        
        if dataset:
            stats.avg_gates_per_circuit = len(all_gates) / len(dataset)
            stats.avg_descriptions_per_image = total_descriptions / len(dataset)
        
        return stats


class DatasetCleaner:
    """
    Cleans and fixes dataset issues.
    
    This class provides automatic fixing for common issues:
    1. Gate name normalization
    2. Missing field defaults
    3. Format standardization
    
    Examples
    --------
    >>> cleaner = DatasetCleaner()
    >>> cleaned = cleaner.clean_dataset(dataset)
    >>> cleaner.save(cleaned)
    """
    
    def __init__(self):
        """Initialize the cleaner."""
        self.gate_cleaner = GateCleaner()
    
    def clean_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean the entire dataset.
        
        Parameters
        ----------
        dataset : Dict
            Original dataset.
        
        Returns
        -------
        Dict
            Cleaned dataset.
        """
        cleaned = {}
        
        for filename, data in dataset.items():
            cleaned[filename] = self.clean_entry(data)
        
        logger.info(f"Cleaned {len(cleaned)} entries")
        return cleaned
    
    def clean_entry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean a single dataset entry.
        
        Parameters
        ----------
        data : Dict
            Entry data.
        
        Returns
        -------
        Dict
            Cleaned entry.
        """
        cleaned = data.copy()
        
        # Clean gate names
        if 'quantum_gates' in cleaned:
            cleaned['quantum_gates'] = self.gate_cleaner.clean(
                cleaned['quantum_gates']
            )
        
        # Ensure required fields exist
        defaults = {
            'quantum_gates': [],
            'quantum_problem': 'Unspecified quantum circuit',
            'descriptions': [],
            'text_positions': [],
        }
        
        for field, default in defaults.items():
            if field not in cleaned or cleaned[field] is None:
                cleaned[field] = default
        
        # Ensure text_positions matches descriptions
        if len(cleaned['descriptions']) != len(cleaned['text_positions']):
            # Pad with (0, 0) if needed
            while len(cleaned['text_positions']) < len(cleaned['descriptions']):
                cleaned['text_positions'].append([0, 0])
            # Trim if too many
            cleaned['text_positions'] = cleaned['text_positions'][:len(cleaned['descriptions'])]
        
        # Clean descriptions
        cleaned['descriptions'] = [
            self._clean_text(d) for d in cleaned['descriptions']
        ]
        
        return cleaned
    
    def _clean_text(self, text: str) -> str:
        """Clean description text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page markers
        text = re.sub(r'\[PAGE \d+\]', '', text)
        
        return text.strip()
    
    def save(
        self,
        dataset: Dict[str, Any],
        output_path: Path = None
    ) -> None:
        """
        Save the cleaned dataset.
        
        Parameters
        ----------
        dataset : Dict
            Dataset to save.
        output_path : Path
            Output file path.
        """
        output_path = output_path or CONFIG.paths.dataset_json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved cleaned dataset to: {output_path}")


def validate_and_fix_dataset() -> Tuple[bool, ValidationReport]:
    """
    Validate the dataset and apply automatic fixes.
    
    Returns
    -------
    Tuple[bool, ValidationReport]
        (success, validation_report)
    
    Examples
    --------
    >>> success, report = validate_and_fix_dataset()
    >>> if success:
    ...     print("Dataset is valid!")
    """
    validator = DatasetValidator()
    cleaner = DatasetCleaner()
    
    # First validation
    report = validator.validate()
    logger.info(f"Initial validation: {len(report.errors)} errors, {len(report.warnings)} warnings")
    
    if report.errors:
        # Try to fix what we can
        try:
            dataset_path = CONFIG.paths.dataset_json
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            cleaned = cleaner.clean_dataset(dataset)
            cleaner.save(cleaned)
            
            # Re-validate
            report = validator.validate()
            logger.info(f"After cleaning: {len(report.errors)} errors, {len(report.warnings)} warnings")
            
        except Exception as e:
            logger.error(f"Failed to clean dataset: {e}")
    
    return report.is_valid, report


def clean_json_entry(entry: Dict[str, any]) -> Dict[str, any]:
    """
    Clean and normalize a single JSON entry according to the deterministic rules.

    Parameters:
    - entry (dict): A dictionary representing a single JSON entry.

    Returns:
    - dict: The cleaned and normalized JSON entry.
    """
    # Rule 1: Normalize `descriptions`
    descriptions = entry.get("descriptions", [])
    if not descriptions or all(isinstance(d, str) and d.strip() == "" for d in descriptions):
        figure_caption = entry.get("figure_caption", "")
        entry["descriptions"] = [figure_caption] if figure_caption else []
    else:
        # Remove exact duplicates
        descriptions = list(dict.fromkeys(descriptions))

        # Retain only the longest version of semantically similar sentences
        cleaned_descriptions = []
        for desc in descriptions:
            if not any(desc in other and len(desc) < len(other) for other in descriptions):
                cleaned_descriptions.append(desc)

        entry["descriptions"] = cleaned_descriptions

    # Rule 2: Normalize `quantum_problem`
    quantum_problem = entry.get("quantum_problem", "")
    if not quantum_problem or quantum_problem.strip() == "":
        entry["quantum_problem"] = "not identify"

    return entry

def clean_json_dataset(dataset: List[Dict[str, any]]) -> List[Dict[str, any]]:
    """
    Clean and normalize a dataset of JSON entries.

    Parameters:
    - dataset (list): A list of dictionaries representing the dataset.

    Returns:
    - list: The cleaned and normalized dataset.
    """
    return [clean_json_entry(entry) for entry in dataset]


if __name__ == "__main__":
    print("Quality Control module loaded successfully")
    print("Run validate_and_fix_dataset() to validate the dataset")
