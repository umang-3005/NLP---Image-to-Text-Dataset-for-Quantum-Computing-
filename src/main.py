"""
Main Pipeline Orchestrator for the Quantum Circuit Dataset.

This is the main entry point that coordinates all pipeline stages:
1. Paper acquisition (download PDFs from arXiv)
2. PDF parsing and figure extraction
3. Quantum circuit detection (NLP-based)
4. Text extraction and figure-text linking
5. Quantum gate extraction
6. Quantum algorithm identification
7. Quality control and validation
8. Dataset export

"""

import sys
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Add src and project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from data_models import (
    PaperInfo, QuantumCircuitImage, DatasetStatistics,
    ProcessingStatus
)
from paper_acquisition import PaperAcquisitionPipeline
from pdf_extraction import (
    PDFParser, FigureExtractor, LayoutBasedFigureExtractor, 
    ExtractedFigure, save_figure_to_png
)
from text_extraction import (
    TextExtractor, FigureTextLinker, DescriptionExtractor,
    extract_figure_texts
)
from circuit_detection import (
    QuantumCircuitDetector, filter_circuit_figures
)
from gate_extraction import GateExtractor, GateCleaner
from algorithm_identification import AlgorithmIdentifier
from quality_control import DatasetValidator, DatasetCleaner, validate_and_fix_dataset
from dataset_export import DatasetExporter, IncrementalExporter, export_dataset
from utils.logging_utils import setup_logger, PipelineLogger

# Import the CNN-based circuit classifier (fine-tuned ConvNeXt model)
from cnn_classifier import CNNCircuitClassifier
from local_visual_extractor import LocalVisualExtractor

# Module logger
logger = setup_logger(__name__)


class QuantumCircuitDatasetPipeline:
    """
    Main pipeline orchestrator for dataset creation.
    
    This class coordinates all stages of the pipeline:
    1. Download PDFs from arXiv
    2. Extract figures from PDFs
    3. Detect quantum circuits (NLP-based)
    4. Extract text and link to figures
    5. Extract quantum gates
    6. Identify quantum algorithms
    7. Validate and export dataset
    
    Attributes
    ----------
    target_images : int
        Number of circuit images to collect.
    collected_circuits : List[QuantumCircuitImage]
        List of collected circuit data.
    incremental_exporter : IncrementalExporter
        For saving progress.
    
    Examples
    --------
    >>> pipeline = QuantumCircuitDatasetPipeline()
    >>> pipeline.run()
    >>> print(f"Collected {len(pipeline.collected_circuits)} circuits")
    """
    
    def __init__(self, target_images: int = None, phase1_only_limit: int = None):
        """
        Initialize the pipeline.
        
        Parameters
        ----------
        target_images : int
            Target number of images to collect (Phase 2 accepted circuits).
        phase1_only_limit : int
            When set, run Phase 1 only and stop after saving this many raw figures.
        """
        self.target_images = target_images or CONFIG.extraction.target_image_count
        self.phase1_only_limit = phase1_only_limit
        self.phase1_saved = 0
        self.collected_circuits: List[QuantumCircuitImage] = []
        
        # Initialize components
        self.circuit_detector = QuantumCircuitDetector()  # NLP-based (backup)
        self.cnn_classifier = CNNCircuitClassifier()      # CNN-based classifier (primary)
        self.gate_extractor = GateExtractor()
        self.algorithm_identifier = AlgorithmIdentifier()
        self.gate_cleaner = GateCleaner()
        
        # ADD THIS
        self.local_visual = LocalVisualExtractor()

        # Incremental saving
        self.incremental_exporter = IncrementalExporter()
        
        # Statistics tracking
        self.papers_processed = 0
        self.figures_extracted = 0
        self.circuits_detected = 0
        
        logger.info(f"Pipeline initialized. Target: {self.target_images} images")
    
    def run(self) -> List[QuantumCircuitImage]:
        """
        Run the complete pipeline.
        
        Returns
        -------
        List[QuantumCircuitImage]
            List of collected circuit data.
        """
        with PipelineLogger("Main Pipeline") as pl:
            pl.log(f"Starting pipeline run. Target: {self.target_images} images")
            
            # Try to resume from checkpoint
            loaded = self.incremental_exporter.load_checkpoint()
            if loaded > 0:
                self.collected_circuits = self.incremental_exporter.circuits
                pl.log(f"Resumed from checkpoint: {loaded} circuits already collected")
            
            # Stage 1: Paper Acquisition
            paper_pipeline = PaperAcquisitionPipeline(self.target_images)
            
            for paper, pdf_path in paper_pipeline.iterate_papers():
                # Check if we've reached targets (Phase 2 or Phase 1-only)
                if self.phase1_only_limit is None:
                    if len(self.collected_circuits) >= self.target_images:
                        pl.log(f"Target reached: {len(self.collected_circuits)} circuits")
                        break
                else:
                    if self.phase1_saved >= self.phase1_only_limit:
                        pl.log(f"Phase 1-only target reached: {self.phase1_saved} raw figures")
                        break
                
                # Process this paper
                circuits_from_paper = self._process_paper(paper, pdf_path)
                
                # Record results (Phase 2 mode only)
                if self.phase1_only_limit is None:
                    paper_pipeline.record_result(paper, len(circuits_from_paper))
                    
                    # Update collected circuits
                    self.collected_circuits.extend(circuits_from_paper)
                    
                    # Save checkpoint periodically
                    if len(self.collected_circuits) % 10 == 0:
                        for c in circuits_from_paper:
                            self.incremental_exporter.add_circuit(c)
                        self.incremental_exporter.save_checkpoint()
                
                # Progress reporting
                if self.phase1_only_limit is None:
                    pl.progress(
                        len(self.collected_circuits),
                        self.target_images,
                        f"Processed {paper.arxiv_id}"
                    )
                else:
                    pl.progress(
                        self.phase1_saved,
                        self.phase1_only_limit,
                        f"Phase1-only: Processed {paper.arxiv_id}"
                    )
            
            # Finalize
            if self.phase1_only_limit is None:
                pl.log("Pipeline complete. Running final validation and export...")
                
                # Stage 7: Quality Control
                self._run_quality_control()
                
                # Stage 8: Export
                self._export_results()
                
                pl.log(f"Final result: {len(self.collected_circuits)} circuits collected")
            else:
                pl.log(f"Phase 1-only run complete. Raw figures saved: {self.phase1_saved}")
        
        return self.collected_circuits
    
    def _process_paper(
        self,
        paper: PaperInfo,
        pdf_path: Path
    ) -> List[QuantumCircuitImage]:
        """
        Process a single paper.
        
        Parameters
        ----------
        paper : PaperInfo
            Paper information.
        pdf_path : Path
            Path to downloaded PDF.
        
        Returns
        -------
        List[QuantumCircuitImage]
            Circuits extracted from this paper.
        """
        circuits = []
        
        try:
            with PDFParser(pdf_path, paper.arxiv_id) as parser:
                # PHASE 1: Comprehensive Figure Extraction
                # Uses multiple methods: Layout Analysis + Embedded Images + Vector Graphics
                # Extracts ALL figures regardless of whether they have captions
                figure_extractor = LayoutBasedFigureExtractor(parser)
                figures = figure_extractor.extract_all_figures()
                self.figures_extracted += len(figures)
                
                # Get total page count for context
                total_pages = parser.page_count
                
                logger.info(f"[Phase 1] Paper {paper.arxiv_id}: {total_pages} pages, {len(figures)} figures extracted")
                
                if not figures:
                    logger.info(f"[Phase 1] No figures found in {paper.arxiv_id}")
                    return circuits
                
                # Save ALL figures to Phase 1 folder for traceability
                for fig in figures:
                    save_figure_to_png(fig, paper.arxiv_id, CONFIG.paths.phase1_images_dir)
                
                logger.info(f"[Phase 1] Saved {len(figures)} figures to phase1_raw_figures/")
                
                # Phase 1-only mode: stop at limit
                if self.phase1_only_limit is not None:
                    self.phase1_saved += len(figures)
                    if self.phase1_saved >= self.phase1_only_limit:
                        logger.info(f"Phase1-only limit {self.phase1_only_limit} reached")
                    self.papers_processed += 1
                    return []
                
                # Extract text and link to ALL figures (for metadata, not filtering)
                figures_with_text = extract_figure_texts(parser, figures)
                
                # PHASE 2: Raster Gatekeeper - Process ALL figures through visual filter
                for fig_data in figures_with_text:
                    # Strict break mid-paper if target is met
                    if len(self.collected_circuits) + len(circuits) >= self.target_images:
                        logger.info(f"Target {self.target_images} reached mid-paper. Stopping.")
                        break
                    
                    circuit = self._create_circuit_entry(
                        paper=paper,
                        figure=fig_data['figure'],
                        caption=fig_data.get('caption', ''),
                        contexts=fig_data.get('contexts', []),
                        descriptions=fig_data.get('descriptions', []),
                        text_positions=fig_data.get('text_positions', [])
                    )
                    
                    if circuit:
                        circuits.append(circuit)
                        self.circuits_detected += 1
                
                # Print rejection statistics
                total_rejected = self.figures_extracted - self.circuits_detected
                logger.info(f"Circuits rejected (Phase 2): {total_rejected}")
                
                self.papers_processed += 1
                logger.info(f"[Phase 2] Accepted {len(circuits)} circuits from {paper.arxiv_id}")
                
        except Exception as e:
            logger.error(f"Error processing {paper.arxiv_id}: {e}")
            paper.status = ProcessingStatus.FAILED
            paper.error_message = str(e)
        
        return circuits
    
    def _create_circuit_entry(
        self,
        paper: PaperInfo,
        figure: ExtractedFigure,
        caption: str,
        contexts: List[str],
        descriptions: List[str],
        text_positions: List[tuple]
    ) -> Optional[QuantumCircuitImage]:
        """
        Create a complete circuit entry with all extracted information.
        
        Parameters
        ----------
        paper : PaperInfo
            Paper information.
        figure : ExtractedFigure
            Extracted figure data.
        caption : str
            Figure caption.
        contexts : List[str]
            Context texts.
        descriptions : List[str]
            Description texts.
        text_positions : List[tuple]
            Text positions.
        
        Returns
        -------
        Optional[QuantumCircuitImage]
            Complete circuit data, or None if failed.
        """
        try:
            # PHASE 1: Save to raw extraction folder for traceability
            raw_filename = save_figure_to_png(
                figure,
                paper.arxiv_id,
                CONFIG.paths.phase1_images_dir
            )
            raw_path = CONFIG.paths.phase1_images_dir / raw_filename
            
            # PHASE 2: CNN Classifier Check (fine-tuned ConvNeXt model)
            detection_res = self.cnn_classifier.analyze(str(raw_path))
            cnn_score = detection_res['score']
            
            if cnn_score < 0.50:
                # REJECTED: Move to rejected folder
                rejected_path = CONFIG.paths.rejected_images_dir / raw_filename
                shutil.copy2(raw_path, rejected_path)
                logger.info(f"  [X] REJECTED: {raw_filename} (Score: {cnn_score:.2f}) - {detection_res['reason']}")
                return None
            
            logger.info(f"  [OK] ACCEPTED: {raw_filename} (Score: {cnn_score:.2f})")
            
            # FINAL: Copy to accepted dataset directory
            final_path = CONFIG.paths.images_dir / raw_filename
            shutil.copy2(raw_path, final_path)
            
            # AFTER CNN check passes:
            
            # 1. Run Local OCR
            ocr_result = self.local_visual.extract_gates_from_image(raw_path)
            visual_gates = ocr_result['detected_gates']
            
            # 2. Run Text Extraction
            gate_result = self.gate_extractor.extract(
                caption=caption,
                contexts=contexts
            )
            text_gates = gate_result.gates
            
            # 3. Combine Gates
            combined_gates = sorted(list(set(visual_gates + text_gates)))
            cleaned_gates = self.gate_cleaner.clean(combined_gates)
            
            # 4. Identify Algorithm
            algo_result = self.algorithm_identifier.identify(
                caption=caption,
                contexts=contexts
            )
            
            # Create the circuit entry
            circuit = QuantumCircuitImage(
                filename=raw_filename,
                arxiv_id=paper.arxiv_id,
                page_number=figure.page_number,
                figure_number=figure.figure_index,
                quantum_gates=cleaned_gates,  # <--- UPDATED to use combined gates
                quantum_problem=algo_result.algorithm,
                descriptions=descriptions if descriptions else [caption] if caption else [],
                text_positions=text_positions if text_positions else [(0, len(caption)) if caption else []],
                confidence_score=cnn_score,  # Use CNN score as primary confidence
                metadata={
                    'extraction_method': figure.extraction_method,
                    'gate_confidence': gate_result.confidence,
                    'algorithm_confidence': algo_result.confidence,
                    'algorithm_source': algo_result.source,
                    'cnn_score': cnn_score,
                    'cnn_reason': detection_res['reason'],
                    'cnn_features': detection_res.get('features', {})
                }
            )
            
            logger.debug(f"Created circuit entry: {raw_filename}")
            return circuit
            
        except Exception as e:
            logger.error(f"Error creating circuit entry: {e}")
            return None
    
    def _run_quality_control(self) -> None:
        """Run quality control on the collected circuits."""
        with PipelineLogger("Quality Control") as pl:
            # First, export current data
            exporter = DatasetExporter()
            exporter.export_json_dataset(self.collected_circuits)
            
            # Validate and fix
            is_valid, report = validate_and_fix_dataset()
            
            pl.log(f"Validation result: {'PASSED' if is_valid else 'FAILED'}")
            pl.log(f"Errors: {len(report.errors)}, Warnings: {len(report.warnings)}")
            
            if report.statistics:
                pl.log(f"Gates found: {len(report.statistics.gate_distribution)}")
                pl.log(f"Algorithms identified: {len(report.statistics.algorithm_distribution)}")
    
    def _export_results(self) -> None:
        """Export the final results."""
        with PipelineLogger("Export") as pl:
            # Calculate statistics
            stats = self._calculate_final_statistics()
            
            # Export
            outputs = export_dataset(self.collected_circuits, stats)
            
            for output_type, path in outputs.items():
                pl.log(f"Exported {output_type}: {path}")
            
            # Clean up checkpoint
            self.incremental_exporter.finalize()
    
    def _calculate_final_statistics(self) -> DatasetStatistics:
        """Calculate final dataset statistics."""
        from collections import Counter
        
        stats = DatasetStatistics()
        stats.total_papers_processed = self.papers_processed
        stats.total_images_extracted = self.figures_extracted
        stats.total_circuits_found = len(self.collected_circuits)
        
        # Count unique papers with circuits
        papers_with_circuits = set()
        all_gates = []
        all_algorithms = []
        total_descriptions = 0
        
        for circuit in self.collected_circuits:
            papers_with_circuits.add(circuit.arxiv_id)
            all_gates.extend(circuit.quantum_gates)
            all_algorithms.append(circuit.quantum_problem)
            total_descriptions += len(circuit.descriptions)
        
        stats.papers_with_circuits = len(papers_with_circuits)
        stats.gate_distribution = dict(Counter(all_gates))
        stats.algorithm_distribution = dict(Counter(all_algorithms))
        
        if self.collected_circuits:
            stats.avg_gates_per_circuit = len(all_gates) / len(self.collected_circuits)
            stats.avg_descriptions_per_image = total_descriptions / len(self.collected_circuits)
        
        return stats


def main():
    """
    Main entry point for the pipeline.
    
    This function initializes and runs the complete pipeline.
    """
    phase1_only_limit = None
    target_override = None

    # CLI usage:
    # python src/main.py                 -> full pipeline, default target
    # python src/main.py 200             -> full pipeline, target 200 accepted circuits
    # python src/main.py phase1-only 100 -> Phase 1 only, save 100 raw figures, no filtering
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "phase1-only":
            if len(sys.argv) > 2:
                try:
                    phase1_only_limit = int(sys.argv[2])
                except ValueError:
                    phase1_only_limit = 100
            else:
                phase1_only_limit = 100
        else:
            try:
                target_override = int(sys.argv[1])
            except ValueError:
                target_override = None

    target_display = target_override or CONFIG.extraction.target_image_count
    print("=" * 60)
    print("QUANTUM CIRCUIT DATASET PIPELINE")
    print(f"Exam ID: {CONFIG.exam_id}")
    if phase1_only_limit is None:
        print(f"Target: {target_display} images (Phase 2 accepted)")
    else:
        print(f"Mode: Phase 1 only | Raw figure limit: {phase1_only_limit}")
    print("=" * 60)
    
    # Create and run the pipeline
    pipeline = QuantumCircuitDatasetPipeline(
        target_images=target_override,
        phase1_only_limit=phase1_only_limit
    )
    
    try:
        circuits = pipeline.run()
        
        print("\n" + "=" * 60)
        if pipeline.phase1_only_limit is None:
            print("PIPELINE COMPLETE")
            print("=" * 60)
            print(f"Total circuits collected: {len(circuits)}")
            print(f"Papers processed: {pipeline.papers_processed}")
            print(f"Figures extracted (Phase 1): {pipeline.figures_extracted}")
            print(f"Circuits accepted (Phase 2): {pipeline.circuits_detected}")
            print(f"\nOutput directory: {CONFIG.paths.output_dir}")
            print(f"Raw figures (Phase 1): {CONFIG.paths.phase1_images_dir}")
            print(f"Accepted images: {CONFIG.paths.images_dir}")
            print(f"Rejected images: {CONFIG.paths.rejected_images_dir}")
        else:
            print("PHASE 1 ONLY RUN COMPLETE")
            print("=" * 60)
            print(f"Raw figures saved: {pipeline.phase1_saved}")
            print(f"Papers processed: {pipeline.papers_processed}")
            print(f"Figures extracted (Phase 1): {pipeline.figures_extracted}")
            print(f"Raw figures directory: {CONFIG.paths.phase1_images_dir}")
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        print("Progress has been saved to checkpoint.")
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
