"""
Run script for the Quantum Circuit Dataset Pipeline.

This script provides a simple entry point to run the pipeline
with various options.

Usage:
    python run_pipeline.py                    # Run full pipeline
    python run_pipeline.py --target 50        # Target 50 images
    python run_pipeline.py --validate-only    # Only validate existing dataset
    python run_pipeline.py --test             # Run test mode (5 images)

Author: [Your Name]
Exam ID: 37
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import CONFIG
from main import QuantumCircuitDatasetPipeline, main
from quality_control import validate_and_fix_dataset, DatasetValidator
from dataset_export import DatasetExporter

def run_full_pipeline(target: int = None):
    """
    Run the complete pipeline.
    
    Parameters
    ----------
    target : int
        Override target number of images.
    """
    if target:
        CONFIG.extraction.target_image_count = target
    
    main()


def run_validation_only():
    """Run validation on existing dataset."""
    print("=" * 60)
    print("VALIDATION MODE")
    print("=" * 60)
    
    validator = DatasetValidator()
    report = validator.validate()
    
    print(report.summary())
    
    if report.errors:
        print("\nErrors:")
        for error in report.errors[:10]:
            print(f"  [{error.severity}] {error.error_type}: {error.message}")
    
    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings[:10]:
            print(f"  [{warning.severity}] {warning.error_type}: {warning.message}")
    
    if report.statistics:
        print("\nStatistics:")
        print(f"  Total circuits: {report.statistics.total_circuits_found}")
        print(f"  Papers with circuits: {report.statistics.papers_with_circuits}")
        print(f"  Unique gates: {len(report.statistics.gate_distribution)}")


def run_test_mode():
    """Run pipeline in test mode (small target)."""
    print("=" * 60)
    print("TEST MODE - Target: 5 images")
    print("=" * 60)
    
    CONFIG.extraction.target_image_count = 5
    
    pipeline = QuantumCircuitDatasetPipeline(target_images=5)
    circuits = pipeline.run()
    
    print(f"\nTest complete. Collected {len(circuits)} circuits.")


def main_cli():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Quantum Circuit Dataset Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    Run full pipeline (250 images)
  python run_pipeline.py --target 50        Target 50 images
  python run_pipeline.py --validate-only    Validate existing dataset
  python run_pipeline.py --test             Test mode (5 images)
        """
    )
    
    parser.add_argument(
        "--target", "-t",
        type=int,
        default=None,
        help="Target number of images to collect"
    )
    
    parser.add_argument(
        "--validate-only", "-v",
        action="store_true",
        help="Only validate existing dataset"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (5 images)"
    )
    
    args = parser.parse_args()
    
    if args.validate_only:
        run_validation_only()
    elif args.test:
        run_test_mode()
    else:
        run_full_pipeline(args.target)


if __name__ == "__main__":
    main_cli()
