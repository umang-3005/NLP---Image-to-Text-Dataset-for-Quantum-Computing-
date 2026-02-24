# Quantum Circuit Extraction Pipeline

## 1. Project Overview

This project implements a robust, end-to-end pipeline for extracting quantum circuit diagrams from scientific research papers (PDFs) and converting them into a structured, semantically enriched dataset. The system is designed for document understanding and multimodal extraction, not just image scraping. It addresses the unique challenges of identifying, validating, and classifying quantum circuits in complex, noisy scientific documents, producing a dataset of quantum circuits with rich metadata and text alignment.

**Why is this problem hard?**

- Quantum circuits are visually diverse and often embedded in cluttered, multi-panel figures.
- Naive image extraction yields many irrelevant, partial, or low-quality images.
- Circuits must be distinguished from similar-looking non-circuit diagrams (e.g., graphs, plots, hardware photos).
- Textual context is essential for semantic labeling and downstream use.

A multi-stage pipeline is required to:

- Acquire and process PDFs
- Detect, validate, and extract only complete, high-quality circuit figures
- Classify images as quantum circuits or not
- Align extracted circuits with relevant text
- Output a dataset suitable for research and machine learning

## 2. High-Level Pipeline Architecture

**Pipeline Steps:**

**Input:** List of research paper identifiers (e.g., arXiv IDs)

**Paper Acquisition:** Download PDFs using APIs (arXiv, direct links), with rate limiting and error handling.

**PDF Handling:** Parse PDFs, extract layout, and identify figure regions.

**Figure Detection & Extraction:** Detect all candidate figures, compute bounding boxes, and extract images.

**Figure Validation:** Filter out partial, cropped, or irrelevant figures using geometric and content-based rules.

**Quantum Circuit Classification:** Use a CNN classifier and heuristics to distinguish quantum circuits from other images.

**False Positive/Negative Handling:** Log and review rejected/uncertain cases for debugging and improvement.

**Final Circuit Selection:** Accept only high-confidence, validated quantum circuits.

**Post-processing & Dataset Generation:** Align circuits with text, extract metadata, and output structured datasets.

**Why stages?**

- Each stage isolates a specific challenge (e.g., image quality, classification, text alignment), enabling targeted debugging and improvement.
- Intermediate outputs allow inspection and error analysis.

## 3. Directory Structure

- `src/` — All core source code modules
- `dataset/` — Training data for classifier (circuit/non-circuit images)
- `debug_output/` — Debug images: accepted, rejected, uncertain figures
- `images_37/` — Extracted images from the current batch
- `logs/` — Log files for pipeline runs
- `output/` — Final outputs: JSON datasets, reports, statistics
- `phase1_raw_figures/` — Raw figure crops from PDFs
- `rejected_figures/` — Figures rejected at any stage
- `temp/` — Temporary files (e.g., downloaded PDFs)
- `tests/` — Unit and integration tests
- `requirements.txt` — Python dependencies
- `run_pipeline.py` — Main entry point for running the pipeline
- `model.ipynb` — Model training and analysis notebook
- `circuit_classifier.pth` — Trained CNN weights
- `documentation/` — Project documentation (Markdown, LaTeX)

**Data at each stage:**

- PDFs: `temp/pdfs/`
- Raw figures: `phase1_raw_figures/`
- Debug images: `debug_output/`
- Final dataset: `output/`

## 4. Code Files — Detailed Explanation

### `src/`

- **`main.py`**: Orchestrates the full pipeline. Loads configs, manages workflow, and coordinates all modules.
- **`config.py`**: Centralizes configuration (paths, thresholds, model settings). All modules read from here.
- **`paper_acquisition.py`**: Handles paper list parsing, arXiv API queries, PDF downloads, and rate limiting. Inputs: paper list; Outputs: local PDFs.
- **`pdf_extraction.py`**: Parses PDFs, extracts layout, detects figure regions, and computes bounding boxes. Inputs: PDFs; Outputs: figure crops, bounding box metadata.
- **`circuit_detection.py`**: Applies figure validation rules (size, aspect, content checks) to filter out partial/cropped/irrelevant figures. Inputs: figure crops; Outputs: validated figures.
- **`cnn_classifier.py`**: Loads the trained CNN, runs inference to classify images as circuit/non-circuit. Inputs: validated figures; Outputs: circuit predictions, confidence scores.
- **`algorithm_identification.py`**: Uses NLP to analyze captions and nearby text, extracting algorithm/problem names and semantic info. Inputs: text, captions; Outputs: semantic labels.
- **`text_extraction.py`**: Extracts and preprocesses text from PDFs, aligns text with figures, and supports NLP modules.
- **`gate_extraction.py`**: (Optional) Analyzes circuit images to extract gate-level structure (if implemented).
- **`dataset_export.py`**: Aggregates accepted circuits, metadata, and text into final JSON/CSV datasets.
- **`quality_control.py`**: Performs post-hoc checks, logs errors, and ensures dataset integrity.
- **`data_models.py`**: Defines data structures for figures, circuits, and metadata.
- **`utils/`**: Utility functions (e.g., logging, file I/O, image processing).
- **`logging_utils.py`**: Standardizes logging across modules.

### Pipeline Flow Reference

- `main.py` calls: `paper_acquisition.py` → `pdf_extraction.py` → `circuit_detection.py` → `cnn_classifier.py` → `algorithm_identification.py`/`text_extraction.py` → `dataset_export.py` → `quality_control.py`

## 5. Figure Extraction Logic

- **Detection:** Figures are detected using PDF layout analysis (bounding boxes, font/graphic separation).
- **Bounding Boxes:** Computed to tightly crop figures, avoiding page margins and overlapping text.
- **Validation:**
  - Filters out full-page crops, blank/white images, and text-only images using pixel statistics and OCR.
  - Multi-panel figures (a, b, c, d) are merged and treated as a single figure if panels are contiguous.
  - Cropped/partial figures are detected by edge analysis and compared to expected figure boundaries.
- **Rules:** Only figures passing all checks are sent to classification.

## 6. Quantum Circuit Classification

- **Distinguishing Circuits:**
  - CNN classifier predicts circuit/non-circuit using visual features.
  - Heuristic rules (e.g., presence of wires, gates, circuit layout) supplement the model.
  - Caption/text signals (optional) can boost confidence if keywords are present.
- **False Positives:**
  - Graphs, hardware photos, and plots may resemble circuits; these are filtered by classifier and rules.
- **False Negatives:**
  - Some true circuits may be missed due to unusual layouts or poor image quality.
- **Prioritizing Recall:**
  - The system is tuned to maximize recall (find all possible circuits), then applies stricter precision filters.
- **Acceptance:**
  - Only images above a confidence threshold and passing all rules are accepted.
  - Uncertain or borderline cases are logged for review.

## 7. NLP and Text Understanding Components

- **Algorithm/Problem Identification:**
  - NLP models analyze captions and nearby text to extract algorithm names (e.g., QFT, Grover) and problem context.
- **Text Extraction:**
  - Text is extracted from PDFs, segmented by section/paragraph/sentence.
  - Semantic similarity and entity extraction align circuits with relevant text.
- **Alignment:**
  - Each circuit is linked to its textual description, section, and context for downstream use.
- **Models/Methods:**
  - Scientific transformers, embeddings, and rule-based entity extraction are used conceptually.

## 8. Failure Modes and Debugging Strategy

- **Common Failures:**
  - Cropped/partial figures, missed circuits, misclassified non-circuits, OCR/text extraction errors.
- **Detection & Logging:**
  - All rejected and uncertain figures are saved in `debug_output/` for inspection.
  - Logs record reasons for rejection (e.g., size, aspect, classifier confidence).
- **Debug Outputs:**
  - `accepted_circuits/`, `rejected_non_circuits/`, `uncertain/` folders contain categorized debug images.
- **Improvement:**
  - Developers can review debug outputs to adjust thresholds, retrain models, or refine rules to improve recall/precision.

## 9. Running the Pipeline

1. **Environment Setup:**
   - Install Python 3.8+
   - `pip install -r requirements.txt`
   - Ensure `circuit_classifier.pth` is present in the root directory
2. **Run Pipeline:**
   - `python run_pipeline.py --config src/config.py`
   - Input: list of paper IDs or PDFs (see config)
   - Intermediate outputs: `phase1_raw_figures/`, `debug_output/`, `logs/`
   - Final outputs: `output/` (JSON, images, reports)
3. **Configuration:**
   - All parameters (paths, thresholds, model options) are set in `src/config.py`
4. **Debugging:**
   - Inspect `debug_output/` and `logs/` for errors and rejected cases

## 10. Final Outputs and Dataset Format

- **Images:** Cropped, validated quantum circuit images (PNG/JPG)
- **Metadata:** JSON files with fields:
  - `paper_id`, `figure_id`, `bounding_box`, `caption`, `algorithm_name`, `text_section`, `confidence_score`, etc.
- **Gate/Algorithm/Text Links:**
  - Each circuit is linked to its algorithm/problem and relevant text
- **Usage:**
  - Dataset can be used for ML, benchmarking, or further scientific analysis

## 11. Design Philosophy and Constraints

- **Correctness & Completeness:**
  - The pipeline prioritizes extracting all true circuits, even at the cost of some false positives (which are filtered later).
- **Separation of Concerns:**
  - Extraction, classification, and text alignment are modular for easier debugging and extension.
- **Multimodal Alignment:**
  - Both image and text signals are required for robust circuit identification.
- **Assumptions Avoided:**
  - No hardcoded figure layouts, no reliance on perfect OCR, no single-modality shortcuts.

## 12. Future Extensions

- Improved figure detection (deep learning-based layout analysis)
- More advanced NLP for algorithm identification
- Active learning for classifier improvement
- Scaling to larger paper corpora and new domains
- Richer dataset formats (e.g., gate-level structure, graph representations)

---

For questions, debugging, or contributions, see `documentation/` and `QUICK_REFERENCE.md`. All code is modular and documented for research and engineering use.
