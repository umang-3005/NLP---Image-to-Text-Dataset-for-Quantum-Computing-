"""
Dataset Export Module for the Quantum Circuit Dataset Pipeline.

This module handles:
1. Saving the final JSON dataset
2. Exporting statistics
3. Generating summary reports
4. Creating the final deliverables

Author: [Your Name]
Exam ID: 37
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent))

from config import CONFIG
from data_models import QuantumCircuitImage, DatasetStatistics
from quality_control import DatasetValidator
from utils.logging_utils import setup_logger


# Module logger
logger = setup_logger(__name__)


class DatasetExporter:
    """
    Exports the compiled dataset in the required format.
    
    This class ensures:
    1. Correct JSON schema
    2. Proper file organization
    3. Consistent naming conventions
    
    Attributes
    ----------
    output_dir : Path
        Directory for output files.
    images_dir : Path
        Directory containing images.
    exam_id : int
        Exam ID for file naming.
    
    Examples
    --------
    >>> exporter = DatasetExporter()
    >>> exporter.export(circuits, statistics)
    """
    
    def __init__(
        self,
        output_dir: Path = None,
        images_dir: Path = None,
        exam_id: int = None
    ):
        """
        Initialize the exporter.
        
        Parameters
        ----------
        output_dir : Path
            Output directory.
        images_dir : Path
            Images directory.
        exam_id : int
            Exam ID for naming.
        """
        self.output_dir = output_dir or CONFIG.paths.output_dir
        self.images_dir = images_dir or CONFIG.paths.images_dir
        self.exam_id = exam_id or CONFIG.exam_id
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    def export(
        self,
        circuits: List[QuantumCircuitImage],
        statistics: DatasetStatistics = None
    ) -> Dict[str, Path]:
        """
        Export the complete dataset.
        
        Parameters
        ----------
        circuits : List[QuantumCircuitImage]
            List of circuit data objects.
        statistics : DatasetStatistics
            Dataset statistics (optional).
        
        Returns
        -------
        Dict[str, Path]
            Dictionary mapping output types to file paths.
        """
        logger.info(f"Exporting dataset with {len(circuits)} circuits...")
        
        outputs = {}
        
        # Export main JSON dataset
        dataset_path = self.export_json_dataset(circuits)
        outputs['dataset'] = dataset_path
        
        # Export statistics if provided
        if statistics:
            stats_path = self.export_statistics(statistics)
            outputs['statistics'] = stats_path
        
        # Export summary report
        report_path = self.export_summary_report(circuits, statistics)
        outputs['report'] = report_path
        
        logger.info(f"Export complete. Files saved to: {self.output_dir}")
        return outputs
    
    def export_json_dataset(
        self,
        circuits: List[QuantumCircuitImage]
    ) -> Path:
        """
        Export the main JSON dataset.
        
        Parameters
        ----------
        circuits : List[QuantumCircuitImage]
            List of circuit data objects.
        
        Returns
        -------
        Path
            Path to the exported JSON file.
        
        Notes
        -----
        JSON structure:
        {
            "filename.png": {
                "arxiv_id": "2410.08073",
                "page_number": 3,
                "figure_number": 1,
                "quantum_gates": ["H", "CNOT"],
                "quantum_problem": "Bell State Preparation",
                "descriptions": ["..."],
                "text_positions": [[0, 100]]
            },
            ...
        }
        """
        dataset = {}

        for circuit in circuits:
            entry = circuit.to_dict()

            # Rule 1: Description fallback rule with text positions
            if not entry.get("descriptions") or all(not desc.strip() for desc in entry["descriptions"]):
                figure_caption = circuit.caption  # Use the caption from the QuantumCircuitImage object
                if figure_caption:
                    entry["descriptions"] = [figure_caption]
                    # Add the caption's position to text_positions
                    entry["text_positions"] = entry.get("text_positions", [])
                    entry["text_positions"].append(circuit.caption_position.to_tuple() if circuit.caption_position else [])

            # Rule 2: Quantum problem default rule
            if not entry.get("quantum_problem") or not entry["quantum_problem"].strip():
                entry["quantum_problem"] = "unknown"

            # Rule 3: Exact duplicate removal
            entry["descriptions"] = list(dict.fromkeys(entry["descriptions"]))

            # Rule 4: Partial vs full sentence resolution
            descriptions = entry["descriptions"]
            descriptions.sort(key=len, reverse=True)  # Sort by length (longest first)
            cleaned_descriptions = []
            for desc in descriptions:
                if not any(desc in longer_desc for longer_desc in cleaned_descriptions):
                    cleaned_descriptions.append(desc)
            entry["descriptions"] = cleaned_descriptions

            dataset[circuit.filename] = entry

        # Save to file
        output_path = self.output_dir / f"dataset_{self.exam_id}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved dataset JSON: {output_path}")
        return output_path
    
    def export_statistics(
        self,
        statistics: DatasetStatistics
    ) -> Path:
        """
        Export dataset statistics.
        
        Parameters
        ----------
        statistics : DatasetStatistics
            Statistics object.
        
        Returns
        -------
        Path
            Path to the statistics file.
        """
        output_path = self.output_dir / f"statistics_{self.exam_id}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(statistics.to_dict(), f, indent=2)
        
        logger.info(f"Saved statistics: {output_path}")
        return output_path
    
    def export_summary_report(
        self,
        circuits: List[QuantumCircuitImage],
        statistics: DatasetStatistics = None
    ) -> Path:
        """
        Export a human-readable summary report.
        
        Parameters
        ----------
        circuits : List[QuantumCircuitImage]
            Circuit data.
        statistics : DatasetStatistics
            Statistics (optional).
        
        Returns
        -------
        Path
            Path to the report file.
        """
        output_path = self.output_dir / f"summary_report_{self.exam_id}.txt"
        
        lines = [
            "=" * 60,
            "QUANTUM CIRCUIT DATASET SUMMARY REPORT",
            f"Exam ID: {self.exam_id}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            f"Total Images: {len(circuits)}",
        ]
        
        if statistics:
            lines.extend([
                f"Papers with Circuits: {statistics.papers_with_circuits}",
                f"Average Gates per Circuit: {statistics.avg_gates_per_circuit:.2f}",
                f"Average Descriptions per Image: {statistics.avg_descriptions_per_image:.2f}",
                "",
                "Gate Distribution:",
            ])
            
            for gate, count in sorted(
                statistics.gate_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            )[:15]:
                lines.append(f"  {gate}: {count}")
            
            lines.extend(["", "Algorithm Distribution:"])
            for algo, count in sorted(
                statistics.algorithm_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]:
                lines.append(f"  {algo}: {count}")
        
        lines.extend([
            "",
            "=" * 60,
            "Sample Entries:",
            "=" * 60,
        ])
        
        # Add a few sample entries
        for circuit in circuits[:3]:
            lines.extend([
                f"\nFilename: {circuit.filename}",
                f"  arXiv ID: {circuit.arxiv_id}",
                f"  Page: {circuit.page_number}, Figure: {circuit.figure_number}",
                f"  Gates: {', '.join(circuit.quantum_gates[:5])}",
                f"  Problem: {circuit.quantum_problem}",
            ])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Saved summary report: {output_path}")
        return output_path


class IncrementalExporter:
    """
    Handles incremental dataset export during processing.
    
    This class allows saving progress during the pipeline run,
    enabling recovery from crashes.
    
    Examples
    --------
    >>> exporter = IncrementalExporter()
    >>> exporter.add_circuit(circuit)
    >>> exporter.save_checkpoint()
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize the incremental exporter.
        
        Parameters
        ----------
        output_dir : Path
            Output directory for checkpoints.
        """
        self.output_dir = output_dir or CONFIG.paths.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.circuits: List[QuantumCircuitImage] = []
        self.checkpoint_path = self.output_dir / "checkpoint.json"
    
    def add_circuit(self, circuit: QuantumCircuitImage) -> None:
        """
        Add a circuit to the collection.
        
        Parameters
        ----------
        circuit : QuantumCircuitImage
            Circuit to add.
        """
        self.circuits.append(circuit)
        logger.debug(f"Added circuit: {circuit.filename}")
    
    def save_checkpoint(self) -> None:
        """Save current progress to checkpoint file."""
        data = {
            'count': len(self.circuits),
            'timestamp': datetime.now().isoformat(),
            'circuits': {c.filename: c.to_dict() for c in self.circuits}
        }
        
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Checkpoint saved: {len(self.circuits)} circuits")
    
    def load_checkpoint(self) -> int:
        """
        Load from checkpoint if exists.
        
        Returns
        -------
        int
            Number of circuits loaded.
        """
        if not self.checkpoint_path.exists():
            return 0
        
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for filename, circuit_data in data.get('circuits', {}).items():
                circuit = QuantumCircuitImage.from_dict(filename, circuit_data)
                self.circuits.append(circuit)
            
            logger.info(f"Loaded checkpoint: {len(self.circuits)} circuits")
            return len(self.circuits)
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return 0
    
    def finalize(self) -> List[QuantumCircuitImage]:
        """
        Finalize and return all circuits.
        
        Returns
        -------
        List[QuantumCircuitImage]
            All collected circuits.
        """
        # Remove checkpoint file
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        
        return self.circuits


def export_dataset(
    circuits: List[QuantumCircuitImage],
    statistics: DatasetStatistics = None
) -> Dict[str, Path]:
    """
    Convenience function to export the dataset.
    
    Parameters
    ----------
    circuits : List[QuantumCircuitImage]
        List of circuit data.
    statistics : DatasetStatistics
        Optional statistics.
    
    Returns
    -------
    Dict[str, Path]
        Mapping of output types to paths.
    """
    exporter = DatasetExporter()
    return exporter.export(circuits, statistics)


if __name__ == "__main__":
    print("Dataset Export module loaded successfully")
"""
Dataset Export Module for the Quantum Circuit Dataset Pipeline.

This module handles:
1. Saving the final JSON dataset
2. Exporting statistics
3. Generating summary reports
4. Creating the final deliverables

Author: [Your Name]
Exam ID: 37
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent))

from config import CONFIG
from data_models import QuantumCircuitImage, DatasetStatistics
from quality_control import DatasetValidator
from utils.logging_utils import setup_logger


# Module logger
logger = setup_logger(__name__)


class DatasetExporter:
    """
    Exports the compiled dataset in the required format.
    
    This class ensures:
    1. Correct JSON schema
    2. Proper file organization
    3. Consistent naming conventions
    
    Attributes
    ----------
    output_dir : Path
        Directory for output files.
    images_dir : Path
        Directory containing images.
    exam_id : int
        Exam ID for file naming.
    
    Examples
    --------
    >>> exporter = DatasetExporter()
    >>> exporter.export(circuits, statistics)
    """
    
    def __init__(
        self,
        output_dir: Path = None,
        images_dir: Path = None,
        exam_id: int = None
    ):
        """
        Initialize the exporter.
        
        Parameters
        ----------
        output_dir : Path
            Output directory.
        images_dir : Path
            Images directory.
        exam_id : int
            Exam ID for naming.
        """
        self.output_dir = output_dir or CONFIG.paths.output_dir
        self.images_dir = images_dir or CONFIG.paths.images_dir
        self.exam_id = exam_id or CONFIG.exam_id
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    def export(
        self,
        circuits: List[QuantumCircuitImage],
        statistics: DatasetStatistics = None
    ) -> Dict[str, Path]:
        """
        Export the complete dataset.
        
        Parameters
        ----------
        circuits : List[QuantumCircuitImage]
            List of circuit data objects.
        statistics : DatasetStatistics
            Dataset statistics (optional).
        
        Returns
        -------
        Dict[str, Path]
            Dictionary mapping output types to file paths.
        """
        logger.info(f"Exporting dataset with {len(circuits)} circuits...")
        
        outputs = {}
        
        # Export main JSON dataset
        dataset_path = self.export_json_dataset(circuits)
        outputs['dataset'] = dataset_path
        
        # Export statistics if provided
        if statistics:
            stats_path = self.export_statistics(statistics)
            outputs['statistics'] = stats_path
        
        # Export summary report
        report_path = self.export_summary_report(circuits, statistics)
        outputs['report'] = report_path
        
        logger.info(f"Export complete. Files saved to: {self.output_dir}")
        return outputs
    
    def export_json_dataset(
        self,
        circuits: List[QuantumCircuitImage]
    ) -> Path:
        """
        Export the main JSON dataset.
        
        Parameters
        ----------
        circuits : List[QuantumCircuitImage]
            List of circuit data objects.
        
        Returns
        -------
        Path
            Path to the exported JSON file.
        
        Notes
        -----
        JSON structure:
        {
            "filename.png": {
                "arxiv_id": "2410.08073",
                "page_number": 3,
                "figure_number": 1,
                "quantum_gates": ["H", "CNOT"],
                "quantum_problem": "Bell State Preparation",
                "descriptions": ["..."],
                "text_positions": [[0, 100]]
            },
            ...
        }
        """
        dataset = {}

        for circuit in circuits:
            entry = circuit.to_dict()

            # Rule 1: Description fallback rule with text positions
            if not entry.get("descriptions") or all(not desc.strip() for desc in entry["descriptions"]):
                figure_caption = circuit.caption  # Use the caption from the QuantumCircuitImage object
                if figure_caption:
                    entry["descriptions"] = [figure_caption]
                    # Add the caption's position to text_positions
                    entry["text_positions"] = entry.get("text_positions", [])
                    entry["text_positions"].append(circuit.caption_position.to_tuple() if circuit.caption_position else [])

            # Rule 2: Quantum problem default rule
            if not entry.get("quantum_problem") or not entry["quantum_problem"].strip():
                entry["quantum_problem"] = "unknown"

            # Rule 3: Exact duplicate removal
            entry["descriptions"] = list(dict.fromkeys(entry["descriptions"]))

            # Rule 4: Partial vs full sentence resolution
            descriptions = entry["descriptions"]
            descriptions.sort(key=len, reverse=True)  # Sort by length (longest first)
            cleaned_descriptions = []
            for desc in descriptions:
                if not any(desc in longer_desc for longer_desc in cleaned_descriptions):
                    cleaned_descriptions.append(desc)
            entry["descriptions"] = cleaned_descriptions

            dataset[circuit.filename] = entry

        # Save to file
        output_path = self.output_dir / f"dataset_{self.exam_id}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved dataset JSON: {output_path}")
        return output_path
    
    def export_statistics(
        self,
        statistics: DatasetStatistics
    ) -> Path:
        """
        Export dataset statistics.
        
        Parameters
        ----------
        statistics : DatasetStatistics
            Statistics object.
        
        Returns
        -------
        Path
            Path to the statistics file.
        """
        output_path = self.output_dir / f"statistics_{self.exam_id}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(statistics.to_dict(), f, indent=2)
        
        logger.info(f"Saved statistics: {output_path}")
        return output_path
    
    def export_summary_report(
        self,
        circuits: List[QuantumCircuitImage],
        statistics: DatasetStatistics = None
    ) -> Path:
        """
        Export a human-readable summary report.
        
        Parameters
        ----------
        circuits : List[QuantumCircuitImage]
            Circuit data.
        statistics : DatasetStatistics
            Statistics (optional).
        
        Returns
        -------
        Path
            Path to the report file.
        """
        output_path = self.output_dir / f"summary_report_{self.exam_id}.txt"
        
        lines = [
            "=" * 60,
            "QUANTUM CIRCUIT DATASET SUMMARY REPORT",
            f"Exam ID: {self.exam_id}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            f"Total Images: {len(circuits)}",
        ]
        
        if statistics:
            lines.extend([
                f"Papers with Circuits: {statistics.papers_with_circuits}",
                f"Average Gates per Circuit: {statistics.avg_gates_per_circuit:.2f}",
                f"Average Descriptions per Image: {statistics.avg_descriptions_per_image:.2f}",
                "",
                "Gate Distribution:",
            ])
            
            for gate, count in sorted(
                statistics.gate_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            )[:15]:
                lines.append(f"  {gate}: {count}")
            
            lines.extend(["", "Algorithm Distribution:"])
            for algo, count in sorted(
                statistics.algorithm_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]:
                lines.append(f"  {algo}: {count}")
        
        lines.extend([
            "",
            "=" * 60,
            "Sample Entries:",
            "=" * 60,
        ])
        
        # Add a few sample entries
        for circuit in circuits[:3]:
            lines.extend([
                f"\nFilename: {circuit.filename}",
                f"  arXiv ID: {circuit.arxiv_id}",
                f"  Page: {circuit.page_number}, Figure: {circuit.figure_number}",
                f"  Gates: {', '.join(circuit.quantum_gates[:5])}",
                f"  Problem: {circuit.quantum_problem}",
            ])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Saved summary report: {output_path}")
        return output_path


class IncrementalExporter:
    """
    Handles incremental dataset export during processing.
    
    This class allows saving progress during the pipeline run,
    enabling recovery from crashes.
    
    Examples
    --------
    >>> exporter = IncrementalExporter()
    >>> exporter.add_circuit(circuit)
    >>> exporter.save_checkpoint()
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize the incremental exporter.
        
        Parameters
        ----------
        output_dir : Path
            Output directory for checkpoints.
        """
        self.output_dir = output_dir or CONFIG.paths.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.circuits: List[QuantumCircuitImage] = []
        self.checkpoint_path = self.output_dir / "checkpoint.json"
    
    def add_circuit(self, circuit: QuantumCircuitImage) -> None:
        """
        Add a circuit to the collection.
        
        Parameters
        ----------
        circuit : QuantumCircuitImage
            Circuit to add.
        """
        self.circuits.append(circuit)
        logger.debug(f"Added circuit: {circuit.filename}")
    
    def save_checkpoint(self) -> None:
        """Save current progress to checkpoint file."""
        data = {
            'count': len(self.circuits),
            'timestamp': datetime.now().isoformat(),
            'circuits': {c.filename: c.to_dict() for c in self.circuits}
        }
        
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Checkpoint saved: {len(self.circuits)} circuits")
    
    def load_checkpoint(self) -> int:
        """
        Load from checkpoint if exists.
        
        Returns
        -------
        int
            Number of circuits loaded.
        """
        if not self.checkpoint_path.exists():
            return 0
        
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for filename, circuit_data in data.get('circuits', {}).items():
                circuit = QuantumCircuitImage.from_dict(filename, circuit_data)
                self.circuits.append(circuit)
            
            logger.info(f"Loaded checkpoint: {len(self.circuits)} circuits")
            return len(self.circuits)
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return 0
    
    def finalize(self) -> List[QuantumCircuitImage]:
        """
        Finalize and return all circuits.
        
        Returns
        -------
        List[QuantumCircuitImage]
            All collected circuits.
        """
        # Remove checkpoint file
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        
        return self.circuits


def export_dataset(
    circuits: List[QuantumCircuitImage],
    statistics: DatasetStatistics = None
) -> Dict[str, Path]:
    """
    Convenience function to export the dataset.
    
    Parameters
    ----------
    circuits : List[QuantumCircuitImage]
        List of circuit data.
    statistics : DatasetStatistics
        Optional statistics.
    
    Returns
    -------
    Dict[str, Path]
        Mapping of output types to paths.
    """
    exporter = DatasetExporter()
    return exporter.export(circuits, statistics)


if __name__ == "__main__":
    print("Dataset Export module loaded successfully")
