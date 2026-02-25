"""
PDF Parsing and Figure Extraction Module for the Quantum Circuit Dataset Pipeline.

This module handles:
1. Extracting figures/images from PDF documents using VISUAL LAYOUT ANALYSIS
2. Using LayoutParser with EfficientDet to detect figure regions
3. Properly excluding captions and body text from figure crops
4. Keeping subfigures together as single units
5. Converting extracted content to PNG format

Key insight: We use Document Layout Analysis (LayoutParser + PubLayNet model)
to visually detect figure regions on rendered pages. This solves:
- The "screenshot problem" (including captions/body text)
- The "fragment problem" (breaking vector graphics into pieces)

Author: [Your Name]
Exam ID: 37
"""

import fitz  
import io
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from PIL import Image
import hashlib
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent))

from config import CONFIG
from data_models import FigureInfo, PaperInfo
from utils.logging_utils import setup_logger, PipelineLogger


# Module logger
logger = setup_logger(__name__)

# Lazy-load LayoutParser to avoid import overhead when not needed
_layout_model = None

def get_layout_model():
    """
    Lazy-load the LayoutParser model for figure detection.
    Uses EfficientDet with PubLayNet weights trained on scientific documents.
    """
    global _layout_model
    if _layout_model is None:
        try:
            import layoutparser as lp
            logger.info("Loading LayoutParser EfficientDet model (PubLayNet)...")
            _layout_model = lp.models.EfficientDetLayoutModel(
                'lp://PubLayNet/tf_efficientdet_d0/config'
            )
            logger.info("LayoutParser model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LayoutParser model: {e}")
            _layout_model = None
    return _layout_model


@dataclass
class ExtractedFigure:
    """
    Represents an extracted figure from a PDF.
    
    Attributes
    ----------
    image_data : bytes
        Raw image data in PNG format.
    page_number : int
        Page number (1-indexed).
    figure_index : int
        Index of figure on the page (1-indexed).
    bbox : Tuple[float, float, float, float]
        Bounding box (x0, y0, x1, y1).
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    extraction_method : str
        Method used: 'vector_render', 'embedded_image', 'region_render'.
    caption_candidates : List[str]
        Potential caption texts found near the figure.
    """
    image_data: bytes
    page_number: int
    figure_index: int
    bbox: Tuple[float, float, float, float]
    width: int
    height: int
    extraction_method: str
    caption_candidates: List[str] = field(default_factory=list)


class PDFParser:
    """
    Parses PDF documents and extracts text and structure information using PyMuPDF.
    """
    
    def __init__(self, pdf_path: Path, arxiv_id: str):
        self.pdf_path = Path(pdf_path)
        self.arxiv_id = arxiv_id
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        
        logger.info(f"Opening PDF: {self.pdf_path}")
        self.doc = fitz.open(self.pdf_path)
        
        # Cache for extracted data
        self._full_text: Optional[str] = None
        self._text_blocks: Optional[List[Dict]] = None
        self._page_texts: Optional[Dict[int, str]] = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup."""
        self.close()
        return False
    
    def close(self) -> None:
        """Close the PDF document."""
        if self.doc:
            self.doc.close()
    
    @property
    def page_count(self) -> int:
        """Get total number of pages."""
        return len(self.doc)
    
    def get_page_text(self, page_num: int) -> str:
        """Get text from a specific page (1-indexed)."""
        if self._page_texts is None:
            self._page_texts = {}
        
        page_idx = page_num - 1 if page_num > 0 else 0
        if page_idx not in self._page_texts:
            if 0 <= page_idx < len(self.doc):
                page = self.doc[page_idx]
                self._page_texts[page_idx] = page.get_text("text")
            else:
                self._page_texts[page_idx] = ""
        
        return self._page_texts[page_idx]
    
    def get_full_text(self) -> str:
        """Get the complete text of the document with page markers."""
        if self._full_text is not None:
            return self._full_text
        
        logger.debug(f"Extracting full text from {self.arxiv_id}")
        text_parts = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            page_text = page.get_text("text")
            text_parts.append(f"[PAGE {page_num + 1}]\n{page_text}")
        self._full_text = "\n".join(text_parts)
        return self._full_text
    
    def get_text_blocks(self) -> List[Dict[str, Any]]:
        """Get text blocks with position information."""
        if self._text_blocks is not None:
            return self._text_blocks
        
        logger.debug(f"Extracting text blocks from {self.arxiv_id}")
        self._text_blocks = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    text_content = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_content += span.get("text", "")
                        text_content += "\n"
                    self._text_blocks.append({
                        'text': text_content.strip(),
                        'bbox': tuple(block["bbox"]),
                        'page': page_num + 1,
                        'block_type': 'text'
                    })
        return self._text_blocks
    
    def find_figure_captions(self) -> List[Dict[str, Any]]:
        """
        Find figure captions in the document.
        
        This method uses regex patterns to identify figure captions
        like "Figure 1:", "Fig. 2.", etc.
        
        Returns
        -------
        List[Dict]
            List of caption information with keys:
            - 'figure_number': Extracted figure number
            - 'caption_text': Full caption text
            - 'page': Page number
            - 'bbox': Bounding box of the caption
            - 'position': Character position in full text
        """
        logger.debug(f"Finding figure captions in {self.arxiv_id}")
        
        captions = []
        full_text = self.get_full_text()
        
        # Regex patterns for figure captions
        caption_patterns = [
            # "Figure 1:" or "Figure 1."
            r'(Figure\s+(\d+)[.:]\s*[^\n]+(?:\n(?![A-Z])[^\n]+)*)',
            # "Fig. 1:" or "Fig. 1."
            r'(Fig\.?\s+(\d+)[.:]\s*[^\n]+(?:\n(?![A-Z])[^\n]+)*)',
            # "FIG. 1:" uppercase
            r'(FIG\.?\s+(\d+)[.:]\s*[^\n]+(?:\n(?![A-Z])[^\n]+)*)',
        ]
        
        for pattern in caption_patterns:
            for match in re.finditer(pattern, full_text, re.IGNORECASE | re.MULTILINE):
                caption_text = match.group(1).strip()
                fig_num = int(match.group(2))
                
                # Find which page this caption is on
                page_num = self._find_page_for_position(match.start())
                
                captions.append({
                    'figure_number': fig_num,
                    'caption_text': caption_text,
                    'page': page_num,
                    'position': (match.start(), match.end()),
                    'bbox': None  # Could be enhanced with bbox detection
                })
        
        # Remove duplicates based on figure number and page
        seen = set()
        unique_captions = []
        for cap in captions:
            key = (cap['figure_number'], cap['page'])
            if key not in seen:
                seen.add(key)
                unique_captions.append(cap)
        
        logger.info(f"Found {len(unique_captions)} figure captions in {self.arxiv_id}")
        return unique_captions
    
    def _find_page_for_position(self, char_pos: int) -> int:
        """
        Find which page a character position belongs to.
        
        Parameters
        ----------
        char_pos : int
            Character position in the full text.
        
        Returns
        -------
        int
            Page number (1-indexed).
        """
        full_text = self.get_full_text()
        
        # Find page markers before this position
        page_markers = list(re.finditer(r'\[PAGE (\d+)\]', full_text[:char_pos + 50]))
        
        if page_markers:
            return int(page_markers[-1].group(1))
        return 1
    
    def find_figure_references(self) -> List[Dict[str, Any]]:
        """
        Find references to figures in the text (e.g., "see Fig. 3").
        
        Returns
        -------
        List[Dict]
            List of reference information with keys:
            - 'figure_number': Referenced figure number
            - 'context': Surrounding text (sentence or paragraph)
            - 'page': Page number
            - 'position': Character position in full text
        """
        logger.debug(f"Finding figure references in {self.arxiv_id}")
        
        references = []
        full_text = self.get_full_text()
        
        # Patterns for figure references
        ref_patterns = [
            r'(?:[Ss]ee\s+)?[Ff]ig(?:ure)?\.?\s*(\d+)',
            r'[Ff]ig(?:ure)?\.?\s*(\d+)\s+shows',
            r'[Aa]s\s+shown\s+in\s+[Ff]ig(?:ure)?\.?\s*(\d+)',
            r'[Ii]n\s+[Ff]ig(?:ure)?\.?\s*(\d+)',
            r'\([Ff]ig(?:ure)?\.?\s*(\d+)\)',
        ]
        
        for pattern in ref_patterns:
            for match in re.finditer(pattern, full_text):
                fig_num = int(match.group(1))
                
                # Extract surrounding context (roughly a sentence)
                start = max(0, match.start() - 100)
                end = min(len(full_text), match.end() + 200)
                context = full_text[start:end].strip()
                
                # Clean up context
                context = re.sub(r'\s+', ' ', context)
                
                page_num = self._find_page_for_position(match.start())
                
                references.append({
                    'figure_number': fig_num,
                    'context': context,
                    'page': page_num,
                    'position': (match.start(), match.end())
                })
        
        logger.info(f"Found {len(references)} figure references in {self.arxiv_id}")
        return references


class FigureExtractor:
    """
    Extracts figures/images from PDF documents using PyMuPDF.
    Combines embedded image extraction and vector region rendering with
    DBSCAN-inspired clustering to avoid fragmented figures.

    Enhanced to:
    - Ensure figures are fully extracted without cropping.
    - Exclude caption text or paragraph text from the output.
    - Combine subfigures into a single image.
    - Use the specified file naming format.
    """
    
    def __init__(self, parser: PDFParser, dpi: int = None):
        self.parser = parser
        self.dpi = dpi or CONFIG.extraction.image_dpi
        self.doc = parser.doc
        self.arxiv_id = parser.arxiv_id
    
    def extract_figures(self) -> List[ExtractedFigure]:
        """
        Extract figures from the PDF document.

        Returns:
            List[ExtractedFigure]: A list of extracted figures with metadata.
        """
        extracted_figures = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Save the image with the specified naming format
                figure_number = img_index + 1
                file_name = f"{self.arxiv_id}_{page_num + 1}_{figure_number}.png"

                # Create the ExtractedFigure object
                extracted_figure = ExtractedFigure(
                    image_data=image_bytes,
                    page_number=page_num + 1,
                    figure_index=figure_number,
                    bbox=None,  # Bounding box can be added if needed
                    width=base_image["width"],
                    height=base_image["height"],
                    extraction_method="embedded_image",
                    caption_candidates=[]  # Captions can be added if detected
                )

                extracted_figures.append(extracted_figure)

        return extracted_figures
    
    def extract_all_figures(self) -> List[ExtractedFigure]:
        logger.info(f"Extracting figures from {self.arxiv_id}")
        all_figures = []
        
        embedded = self._extract_embedded_images()
        all_figures.extend(embedded)
        logger.info(f"Found {len(embedded)} embedded images")
        
        vector_figures = self._extract_vector_figures()
        all_figures.extend(vector_figures)
        logger.info(f"Found {len(vector_figures)} vector figures")
        
        deduplicated = self._deduplicate_figures(all_figures)
        
        # Re-number figures sequentially across the entire paper (1, 2, 3, ...)
        for idx, fig in enumerate(deduplicated, start=1):
            fig.figure_index = idx
        
        logger.info(f"Total unique figures: {len(deduplicated)}")
        return deduplicated
    
    def _extract_embedded_images(self) -> List[ExtractedFigure]:
        figures = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    base_image = self.doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    img = Image.open(io.BytesIO(image_bytes))
                    if not self._check_size_constraints(img.width, img.height):
                        continue
                    png_buffer = io.BytesIO()
                    img.save(png_buffer, format='PNG')
                    png_data = png_buffer.getvalue()
                    bbox = (0, 0, img.width, img.height)
                    figures.append(ExtractedFigure(
                        image_data=png_data,
                        page_number=page_num + 1,
                        figure_index=img_idx + 1,
                        bbox=bbox,
                        width=img.width,
                        height=img.height,
                        extraction_method='embedded_image'
                    ))
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_idx} from page {page_num + 1}: {e}")
        return figures

    def _extract_vector_figures(self) -> List[ExtractedFigure]:
        figures = []
        captions = self.parser.find_figure_captions()
        page_captions: Dict[int, List[Dict[str, Any]]] = {}
        for cap in captions:
            page = cap['page']
            page_captions.setdefault(page, []).append(cap)
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            page_idx = page_num + 1
            drawings = page.get_drawings()
            if not drawings:
                continue
            
            # === SAFETY PATCH: SKIP PAGES WITH TOO MANY VECTORS ===
            # Normal diagrams have < 500 paths. Plots/Text-as-vectors have > 3000.
            # This prevents the O(N^2) clustering freeze.
            if len(drawings) > 2000:
                logger.warning(f"Skipping page {page_idx}: Too many vector paths ({len(drawings)}). Likely text-as-vectors or complex plot.")
                continue
            # ======================================================
            
            figure_regions = self._cluster_drawings_to_figures(drawings, page)
            for region_idx, region in enumerate(figure_regions):
                try:
                    clip_rect = fitz.Rect(region['bbox'])
                    zoom = self.dpi / 72
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
                    if not self._check_size_constraints(pix.width, pix.height):
                        continue
                    png_data = pix.tobytes("png")
                    fig_number = region_idx + 1
                    caption_texts = []
                    if page_idx in page_captions:
                        for cap in page_captions[page_idx]:
                            caption_texts.append(cap['caption_text'])
                            fig_number = cap['figure_number']
                    figures.append(ExtractedFigure(
                        image_data=png_data,
                        page_number=page_idx,
                        figure_index=fig_number,
                        bbox=tuple(clip_rect),
                        width=pix.width,
                        height=pix.height,
                        extraction_method='vector_render',
                        caption_candidates=caption_texts
                    ))
                except Exception as e:
                    logger.warning(f"Failed to render region {region_idx} on page {page_idx}: {e}")
        return figures

    def _cluster_drawings_to_figures(self, drawings: List[Dict], page: fitz.Page) -> List[Dict[str, Any]]:
        if not drawings:
            return []
        bboxes = []
        for d in drawings:
            if 'rect' in d:
                bboxes.append(fitz.Rect(d['rect']))
            elif 'items' in d:
                for item in d['items']:
                    if len(item) >= 2 and isinstance(item[1], (list, tuple)):
                        try:
                            rect = fitz.Rect(item[1])
                            if rect.is_valid and not rect.is_empty:
                                bboxes.append(rect)
                        except:
                            pass
        if not bboxes:
            return []
        clusters = self._cluster_bboxes(bboxes)
        figure_regions = []
        for cluster_bbox in clusters:
            width = cluster_bbox.width
            height = cluster_bbox.height
            if height > 0:
                aspect = width / height
                if CONFIG.extraction.min_aspect_ratio <= aspect <= CONFIG.extraction.max_aspect_ratio:
                    padded = cluster_bbox + (-10, -10, 10, 10)
                    padded = padded & page.rect
                    figure_regions.append({
                        'bbox': tuple(padded),
                        'area': padded.width * padded.height
                    })
        figure_regions.sort(key=lambda r: r['bbox'][1])
        return figure_regions

    def _cluster_bboxes(self, bboxes: List[fitz.Rect], distance_threshold: float = 50) -> List[fitz.Rect]:
        return self._cluster_elements_dbscan(bboxes, eps=distance_threshold)

    def _cluster_elements_dbscan(self, bboxes: List[fitz.Rect], eps: float = 40.0, min_samples: int = 1) -> List[fitz.Rect]:
        if not bboxes:
            return []
        clusters = []
        visited = [False] * len(bboxes)
        for i in range(len(bboxes)):
            if visited[i]:
                continue
            current_cluster = bboxes[i]
            visited[i] = True
            queue = [bboxes[i]]
            while queue:
                current_item = queue.pop(0)
                for j in range(len(bboxes)):
                    if not visited[j]:
                        expanded_item = current_item + (-eps, -eps, eps, eps)
                        if expanded_item.intersects(bboxes[j]):
                            visited[j] = True
                            current_cluster = current_cluster | bboxes[j]
                            queue.append(bboxes[j])
            clusters.append(current_cluster)
        return clusters
    
    def _check_size_constraints(self, width: int, height: int) -> bool:
        """
        Check if image dimensions are within acceptable bounds.
        
        Parameters
        ----------
        width : int
            Image width in pixels.
        height : int
            Image height in pixels.
        
        Returns
        -------
        bool
            True if size is acceptable.
        """
        cfg = CONFIG.extraction
        
        if width < cfg.min_image_width or height < cfg.min_image_height:
            return False
        if width > cfg.max_image_width or height > cfg.max_image_height:
            return False
        
        return True
    
    def _deduplicate_figures(
        self, 
        figures: List[ExtractedFigure]
    ) -> List[ExtractedFigure]:
        """
        Remove duplicate figures based on content hash AND spatial overlap.
        
        This handles:
        1. Exact duplicates (same content hash)
        2. Near-duplicates (overlapping bounding boxes on same page)
        3. Subfigures that should be merged into parent figure
        
        Parameters
        ----------
        figures : List[ExtractedFigure]
            List of all extracted figures.
        
        Returns
        -------
        List[ExtractedFigure]
            Deduplicated list with best version of each figure.
        """
        if not figures:
            return []
        
        # Step 1: Remove exact duplicates by content hash
        seen_hashes = set()
        hash_deduplicated = []
        
        for fig in figures:
            img_hash = hashlib.md5(fig.image_data).hexdigest()
            if img_hash not in seen_hashes:
                seen_hashes.add(img_hash)
                hash_deduplicated.append(fig)
        
        # Step 2: Remove overlapping figures on same page (keep larger one)
        by_page: Dict[int, List[ExtractedFigure]] = {}
        for fig in hash_deduplicated:
            page = fig.page_number
            if page not in by_page:
                by_page[page] = []
            by_page[page].append(fig)
        
        final_figures = []
        
        for page_num, page_figs in by_page.items():
            # Sort by area (largest first)
            page_figs_with_area = []
            for fig in page_figs:
                if fig.bbox:
                    area = (fig.bbox[2] - fig.bbox[0]) * (fig.bbox[3] - fig.bbox[1])
                else:
                    area = fig.width * fig.height
                page_figs_with_area.append((fig, area))
            
            page_figs_with_area.sort(key=lambda x: x[1], reverse=True)
            
            kept = []
            for fig, area in page_figs_with_area:
                is_duplicate = False
                
                for kept_fig in kept:
                    if self._figures_overlap(fig, kept_fig, overlap_threshold=0.5):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    kept.append(fig)
            
            final_figures.extend(kept)
        
        return final_figures
    
    def _figures_overlap(
        self, 
        fig1: ExtractedFigure, 
        fig2: ExtractedFigure, 
        overlap_threshold: float = 0.5
    ) -> bool:
        """Check if two figures overlap significantly."""
        if not fig1.bbox or not fig2.bbox:
            return False
        
        x1 = max(fig1.bbox[0], fig2.bbox[0])
        y1 = max(fig1.bbox[1], fig2.bbox[1])
        x2 = min(fig1.bbox[2], fig2.bbox[2])
        y2 = min(fig1.bbox[3], fig2.bbox[3])
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        intersection_area = (x2 - x1) * (y2 - y1)
        area1 = (fig1.bbox[2] - fig1.bbox[0]) * (fig1.bbox[3] - fig1.bbox[1])
        area2 = (fig2.bbox[2] - fig2.bbox[0]) * (fig2.bbox[3] - fig2.bbox[1])
        
        smaller_area = min(area1, area2)
        if smaller_area > 0:
            containment_ratio = intersection_area / smaller_area
            if containment_ratio > overlap_threshold:
                return True
        
        return False


class LayoutBasedFigureExtractor:
    """
    Extracts figures from PDF documents using Visual Layout Analysis.
    
    This extractor uses LayoutParser with a PubLayNet-trained model to:
    1. Render each PDF page as an image
    2. Detect figure regions using deep learning (EfficientDet)
    3. Crop only the figure bounding boxes, excluding captions and text
    4. Keep subfigures together as single visual units
    
    This solves the "screenshot problem" (including captions/body text)
    and the "fragment problem" (breaking vector graphics into pieces).
    
    Attributes
    ----------
    parser : PDFParser
        The PDF parser instance.
    dpi : int
        DPI for rendering PDF pages.
    confidence_threshold : float
        Minimum confidence for figure detection (0.0-1.0).
    """
    
    # PubLayNet label map
    LABEL_MAP = {
        0: 'Text',
        1: 'Title', 
        2: 'List',
        3: 'Table',
        4: 'Figure'
    }
    
    def __init__(
        self, 
        parser: PDFParser, 
        dpi: int = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the layout-based figure extractor.
        
        Parameters
        ----------
        parser : PDFParser
            The PDF parser instance.
        dpi : int, optional
            DPI for rendering pages. Higher = better quality but slower.
        confidence_threshold : float
            Minimum confidence for accepting detected figures (0.0-1.0).
        """
        self.parser = parser
        self.dpi = dpi or CONFIG.extraction.image_dpi
        self.doc = parser.doc
        self.arxiv_id = parser.arxiv_id
        self.confidence_threshold = confidence_threshold
        
        # Get the layout model (lazy-loaded singleton)
        self.model = get_layout_model()
        if self.model is None:
            logger.warning("LayoutParser model not available, falling back to legacy extraction")
    
    def extract_all_figures(self) -> List[ExtractedFigure]:
        """
        Extract ALL figures from the PDF using multiple methods.
        
        This combines:
        1. Visual Layout Analysis (LayoutParser) - detects figure regions visually
        2. Embedded Images - extracts raster images embedded in the PDF
        3. Vector Graphics Clusters - detects and renders vector drawings
        
        All figures are extracted regardless of whether they have captions.
        
        Returns
        -------
        List[ExtractedFigure]
            List of all extracted figures with metadata.
        """
        logger.info(f"Extracting ALL figures from {self.arxiv_id} (comprehensive extraction)")
        
        # Pre-cache captions to avoid repeated parsing
        self._cached_captions = self.parser.find_figure_captions()
        self._captions_by_page = {}
        for cap in self._cached_captions:
            page = cap['page']
            if page not in self._captions_by_page:
                self._captions_by_page[page] = []
            self._captions_by_page[page].append(cap)
        
        all_figures = []
        
        # Method 1: Layout Analysis (if model available)
        layout_figures = []
        if self.model is not None:
            for page_num in range(len(self.doc)):
                page_figures = self._extract_figures_from_page(page_num)
                layout_figures.extend(page_figures)
            logger.info(f"  Layout Analysis: Found {len(layout_figures)} figures")
        
        # Method 2: Embedded Images (raster images in PDF)
        embedded_figures = self._extract_embedded_images()
        logger.info(f"  Embedded Images: Found {len(embedded_figures)} figures")
        
        # Method 3: Vector Graphics Clusters (for TikZ/circuit diagrams)
        vector_figures = self._extract_vector_figures()
        logger.info(f"  Vector Graphics: Found {len(vector_figures)} figures")
        
        # Combine all methods
        all_figures.extend(layout_figures)
        all_figures.extend(embedded_figures)
        all_figures.extend(vector_figures)
        
        logger.info(f"  Total before deduplication: {len(all_figures)} figures")
        
        # Deduplicate based on content hash
        deduplicated = self._deduplicate_figures(all_figures)
        
        # Sort by page number first, then by vertical position (bbox y-coordinate)
        deduplicated.sort(key=lambda f: (f.page_number, f.bbox[1] if f.bbox else 0))
        
        # Re-number figures sequentially across the entire paper (1, 2, 3, ...)
        for idx, fig in enumerate(deduplicated, start=1):
            fig.figure_index = idx
        
        logger.info(f"Extracted {len(deduplicated)} unique figures (comprehensive extraction)")
        
        return deduplicated
    
    def _extract_embedded_images(self) -> List[ExtractedFigure]:
        """
        Extract embedded raster images from the PDF.
        
        Filters out:
        - Images that are too small (likely icons/fragments)
        - Images with extreme aspect ratios (likely decorative elements)
        
        Returns
        -------
        List[ExtractedFigure]
            List of embedded images.
        """
        figures = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    base_image = self.doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    img = Image.open(io.BytesIO(image_bytes))
                    
                    # Skip small images (likely fragments or icons)
                    if not self._check_size_constraints(img.width, img.height):
                        continue
                    
                    # Skip images with extreme aspect ratios (likely decorative)
                    if img.height > 0:
                        aspect = img.width / img.height
                        if aspect < 0.2 or aspect > 8.0:
                            logger.debug(f"Skipping embedded image with extreme aspect ratio: {aspect:.2f}")
                            continue
                    
                    # Skip very small area images (fragments)
                    area = img.width * img.height
                    if area < 10000:  # Less than ~100x100 pixels
                        logger.debug(f"Skipping small embedded image: {img.width}x{img.height}")
                        continue
                    
                    png_buffer = io.BytesIO()
                    img.save(png_buffer, format='PNG')
                    png_data = png_buffer.getvalue()
                    
                    # Try to get the actual position of the image on the page
                    bbox = self._get_image_position(page, xref)
                    if bbox is None:
                        bbox = (0, 0, img.width, img.height)
                    
                    figures.append(ExtractedFigure(
                        image_data=png_data,
                        page_number=page_num + 1,
                        figure_index=img_idx + 1,
                        bbox=bbox,
                        width=img.width,
                        height=img.height,
                        extraction_method='embedded_image'
                    ))
                except Exception as e:
                    logger.debug(f"Failed to extract embedded image {img_idx} from page {page_num + 1}: {e}")
        return figures
    
    def _get_image_position(self, page: fitz.Page, xref: int) -> Optional[Tuple[float, float, float, float]]:
        """
        Get the position of an embedded image on the page.
        
        Returns
        -------
        Optional[Tuple]
            Bounding box (x0, y0, x1, y1) or None if not found.
        """
        try:
            # Get all image instances on the page
            image_list = page.get_image_info(xrefs=True)
            for img_info in image_list:
                if img_info.get('xref') == xref:
                    bbox = img_info.get('bbox')
                    if bbox:
                        return tuple(bbox)
        except Exception:
            pass
        return None
        return figures
    
    def _extract_vector_figures(self) -> List[ExtractedFigure]:
        """
        Extract vector graphics clusters (for TikZ/circuit diagrams).
        
        Returns
        -------
        List[ExtractedFigure]
            List of rendered vector figure regions.
        """
        figures = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            page_idx = page_num + 1
            drawings = page.get_drawings()
            
            if not drawings:
                continue
            
            # Skip pages with too many vector paths (likely text-as-vectors)
            if len(drawings) > 2000:
                logger.debug(f"Skipping vector extraction on page {page_idx}: too many paths ({len(drawings)})")
                continue
            
            # Cluster drawings into figure regions
            figure_regions = self._cluster_drawings_to_figures(drawings, page)
            
            for region_idx, region in enumerate(figure_regions):
                try:
                    clip_rect = fitz.Rect(region['bbox'])
                    zoom = self.dpi / 72
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
                    
                    if not self._check_size_constraints(pix.width, pix.height):
                        continue
                    
                    png_data = pix.tobytes("png")
                    
                    figures.append(ExtractedFigure(
                        image_data=png_data,
                        page_number=page_idx,
                        figure_index=region_idx + 1,
                        bbox=tuple(clip_rect),
                        width=pix.width,
                        height=pix.height,
                        extraction_method='vector_render'
                    ))
                except Exception as e:
                    logger.debug(f"Failed to render vector region {region_idx} on page {page_idx}: {e}")
        
        return figures
    
    def _cluster_drawings_to_figures(self, drawings: List[Dict], page: fitz.Page) -> List[Dict[str, Any]]:
        """Cluster vector drawing elements into figure regions."""
        if not drawings:
            return []
        
        bboxes = []
        for d in drawings:
            if 'rect' in d:
                bboxes.append(fitz.Rect(d['rect']))
            elif 'items' in d:
                for item in d['items']:
                    if len(item) >= 2 and isinstance(item[1], (list, tuple)):
                        try:
                            rect = fitz.Rect(item[1])
                            if rect.is_valid and not rect.is_empty:
                                bboxes.append(rect)
                        except:
                            pass
        
        if not bboxes:
            return []
        
        clusters = self._cluster_bboxes(bboxes)
        figure_regions = []
        
        for cluster_bbox in clusters:
            width = cluster_bbox.width
            height = cluster_bbox.height
            if height > 0:
                aspect = width / height
                cfg = CONFIG.extraction
                if cfg.min_aspect_ratio <= aspect <= cfg.max_aspect_ratio:
                    padded = cluster_bbox + (-10, -10, 10, 10)
                    padded = padded & page.rect
                    figure_regions.append({
                        'bbox': tuple(padded),
                        'area': padded.width * padded.height
                    })
        
        figure_regions.sort(key=lambda r: r['bbox'][1])
        return figure_regions
    
    def _cluster_bboxes(self, bboxes: List[fitz.Rect], eps: float = 80.0) -> List[fitz.Rect]:
        """
        Cluster bounding boxes using DBSCAN-style algorithm.
        
        Uses larger eps (80px) to merge subfigures that belong together.
        """
        if not bboxes:
            return []
        
        clusters = []
        visited = [False] * len(bboxes)
        
        for i in range(len(bboxes)):
            if visited[i]:
                continue
            current_cluster = bboxes[i]
            visited[i] = True
            queue = [bboxes[i]]
            
            while queue:
                current_item = queue.pop(0)
                for j in range(len(bboxes)):
                    if not visited[j]:
                        expanded_item = current_item + (-eps, -eps, eps, eps)
                        if expanded_item.intersects(bboxes[j]):
                            visited[j] = True
                            current_cluster = current_cluster | bboxes[j]
                            queue.append(bboxes[j])
            
            clusters.append(current_cluster)
        
        # Merge clusters that are close to each other (subfigure merging)
        merged = self._merge_nearby_clusters(clusters, merge_distance=100.0)
        
        return merged
    
    def _merge_nearby_clusters(
        self, 
        clusters: List[fitz.Rect], 
        merge_distance: float = 100.0
    ) -> List[fitz.Rect]:
        """
        Merge clusters that are within merge_distance of each other.
        
        This helps combine subfigures (a), (b), (c) into a single figure.
        """
        if len(clusters) <= 1:
            return clusters
        
        merged = []
        used = [False] * len(clusters)
        
        for i in range(len(clusters)):
            if used[i]:
                continue
            
            current = clusters[i]
            used[i] = True
            
            # Keep merging until no more nearby clusters
            changed = True
            while changed:
                changed = False
                for j in range(len(clusters)):
                    if not used[j]:
                        expanded = current + (-merge_distance, -merge_distance, merge_distance, merge_distance)
                        if expanded.intersects(clusters[j]):
                            current = current | clusters[j]
                            used[j] = True
                            changed = True
            
            merged.append(current)
        
        return merged

    def _extract_figures_from_page(self, page_idx: int) -> List[ExtractedFigure]:
        """
        Extract figures from a single page using layout detection.
        
        Parameters
        ----------
        page_idx : int
            Zero-indexed page number.
        
        Returns
        -------
        List[ExtractedFigure]
            Figures detected on this page.
        """
        import layoutparser as lp
        
        page = self.doc[page_idx]
        page_num = page_idx + 1
        
        # Render page at high DPI for accurate detection
        zoom = self.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to PIL Image for LayoutParser
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_array = np.array(img)
        
        # Detect layout elements
        try:
            layout = self.model.detect(img_array)
        except Exception as e:
            logger.warning(f"Layout detection failed on page {page_num}: {e}")
            return []
        
        figures = []
        figure_idx = 0
        
        # Filter for Figure elements only
        for block in layout:
            # Check if this is a Figure (label index 4 in PubLayNet)
            if block.type == 'Figure' and block.score >= self.confidence_threshold:
                figure_idx += 1
                
                # Get bounding box coordinates (in pixel space)
                x1, y1, x2, y2 = block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2
                
                # Add small padding to avoid cutting edges
                padding = 5
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(pix.width, x2 + padding)
                y2 = min(pix.height, y2 + padding)
                
                # Check size constraints
                width = int(x2 - x1)
                height = int(y2 - y1)
                
                if not self._check_size_constraints(width, height):
                    logger.debug(f"Skipping figure on page {page_num}: size {width}x{height} out of bounds")
                    continue
                
                # Crop the figure region from the rendered page
                cropped = img.crop((int(x1), int(y1), int(x2), int(y2)))
                
                # Convert to PNG bytes
                png_buffer = io.BytesIO()
                cropped.save(png_buffer, format='PNG')
                png_data = png_buffer.getvalue()
                
                # Convert pixel bbox back to PDF coordinates
                pdf_bbox = (
                    x1 / zoom,
                    y1 / zoom,
                    x2 / zoom,
                    y2 / zoom
                )
                
                # Try to find associated caption
                caption_texts = self._find_caption_for_figure(
                    layout, block, page_num, page=page, zoom=zoom
                )
                
                figures.append(ExtractedFigure(
                    image_data=png_data,
                    page_number=page_num,
                    figure_index=figure_idx,
                    bbox=pdf_bbox,
                    width=width,
                    height=height,
                    extraction_method='layout_analysis',
                    caption_candidates=caption_texts
                ))
                
                logger.debug(
                    f"Detected figure on page {page_num}: "
                    f"{width}x{height}px, confidence={block.score:.2f}"
                )
        
        return figures
    
    def _find_caption_for_figure(
        self, 
        layout, 
        figure_block, 
        page_num: int,
        page = None,        # New Argument
        zoom: float = 1.0   # New Argument
    ) -> List[str]:
        """
        Find caption text associated with a detected figure.
        
        Looks for Text blocks directly below the figure that start with
        "Figure", "Fig.", etc.
        
        Parameters
        ----------
        layout : Layout
            The detected layout from LayoutParser.
        figure_block : TextBlock
            The detected figure block.
        page_num : int
            Page number for caption lookup.
        
        Returns
        -------
        List[str]
            Caption text candidates.
        """
        captions = []
        
        fig_bottom = figure_block.block.y_2
        fig_center_x = (figure_block.block.x_1 + figure_block.block.x_2) / 2
        
        # 1. Geometric Search
        for block in layout:
            if block.type == 'Text':
                text_top = block.block.y_1
                # Check if below figure (within 150px)
                if 0 < (text_top - fig_bottom) < 150:
                    text_center_x = (block.block.x_1 + block.block.x_2) / 2
                    # Check alignment
                    if abs(text_center_x - fig_center_x) < (figure_block.block.x_2 - figure_block.block.x_1) * 0.8:
                        
                        # CRITICAL FIX: Extract text pixels-to-PDF
                        if page:
                            r = fitz.Rect(
                                block.block.x_1 / zoom, block.block.y_1 / zoom,
                                block.block.x_2 / zoom, block.block.y_2 / zoom
                            )
                            extracted_text = page.get_text("text", clip=r).strip()
                            if extracted_text and re.match(r'^(FIG|Fig|Figure)', extracted_text, re.IGNORECASE):
                                captions.append(extracted_text)
                                
        # 2. Fallback (Keep your existing fallback code here)
        if not captions and hasattr(self, '_captions_by_page') and page_num in self._captions_by_page:
            for cap in self._captions_by_page[page_num]:
                captions.append(cap['caption_text'])
                
        return captions
    
    def _check_size_constraints(self, width: int, height: int) -> bool:
        """Check if image dimensions are within acceptable bounds."""
        cfg = CONFIG.extraction
        
        if width < cfg.min_image_width or height < cfg.min_image_height:
            return False
        if width > cfg.max_image_width or height > cfg.max_image_height:
            return False
        
        return True
    
    def _deduplicate_figures(
        self, 
        figures: List[ExtractedFigure]
    ) -> List[ExtractedFigure]:
        """
        Remove duplicate figures based on content hash AND spatial overlap.
        
        This handles:
        1. Exact duplicates (same content hash)
        2. Near-duplicates (overlapping bounding boxes on same page)
        3. Subfigures that should be merged into parent figure
        
        Parameters
        ----------
        figures : List[ExtractedFigure]
            List of all extracted figures.
        
        Returns
        -------
        List[ExtractedFigure]
            Deduplicated list with best version of each figure.
        """
        if not figures:
            return []
        
        # Step 1: Remove exact duplicates by content hash
        seen_hashes = set()
        hash_deduplicated = []
        
        for fig in figures:
            img_hash = hashlib.md5(fig.image_data).hexdigest()
            if img_hash not in seen_hashes:
                seen_hashes.add(img_hash)
                hash_deduplicated.append(fig)
        
        logger.debug(f"After hash deduplication: {len(hash_deduplicated)} figures (from {len(figures)})")
        
        # Step 2: Remove overlapping figures on same page (keep larger one)
        # Group by page
        by_page: Dict[int, List[ExtractedFigure]] = {}
        for fig in hash_deduplicated:
            page = fig.page_number
            if page not in by_page:
                by_page[page] = []
            by_page[page].append(fig)
        
        final_figures = []
        
        for page_num, page_figs in by_page.items():
            # Sort by area (largest first) so we keep the most complete figure
            page_figs_with_area = []
            for fig in page_figs:
                if fig.bbox:
                    area = (fig.bbox[2] - fig.bbox[0]) * (fig.bbox[3] - fig.bbox[1])
                else:
                    area = fig.width * fig.height
                page_figs_with_area.append((fig, area))
            
            page_figs_with_area.sort(key=lambda x: x[1], reverse=True)
            
            # Mark figures that are contained within or significantly overlap larger figures
            kept = []
            for fig, area in page_figs_with_area:
                is_duplicate = False
                
                for kept_fig in kept:
                    if self._figures_overlap(fig, kept_fig, overlap_threshold=0.5):
                        # This figure overlaps significantly with an already-kept (larger) figure
                        is_duplicate = True
                        logger.debug(f"Removing overlapping figure on page {page_num}: {fig.width}x{fig.height} overlaps with {kept_fig.width}x{kept_fig.height}")
                        break
                
                if not is_duplicate:
                    kept.append(fig)
            
            final_figures.extend(kept)
        
        logger.debug(f"After overlap deduplication: {len(final_figures)} figures")
        return final_figures
    
    def _figures_overlap(
        self, 
        fig1: ExtractedFigure, 
        fig2: ExtractedFigure, 
        overlap_threshold: float = 0.5
    ) -> bool:
        """
        Check if two figures overlap significantly.
        
        Parameters
        ----------
        fig1, fig2 : ExtractedFigure
            Figures to compare.
        overlap_threshold : float
            Minimum IoU or containment ratio to consider as overlap.
        
        Returns
        -------
        bool
            True if figures overlap significantly.
        """
        if not fig1.bbox or not fig2.bbox:
            return False
        
        # Calculate intersection
        x1 = max(fig1.bbox[0], fig2.bbox[0])
        y1 = max(fig1.bbox[1], fig2.bbox[1])
        x2 = min(fig1.bbox[2], fig2.bbox[2])
        y2 = min(fig1.bbox[3], fig2.bbox[3])
        
        if x2 <= x1 or y2 <= y1:
            return False  # No intersection
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate areas
        area1 = (fig1.bbox[2] - fig1.bbox[0]) * (fig1.bbox[3] - fig1.bbox[1])
        area2 = (fig2.bbox[2] - fig2.bbox[0]) * (fig2.bbox[3] - fig2.bbox[1])
        
        # Check if smaller figure is mostly contained within larger
        smaller_area = min(area1, area2)
        if smaller_area > 0:
            containment_ratio = intersection_area / smaller_area
            if containment_ratio > overlap_threshold:
                return True
        
        # Check IoU
        union_area = area1 + area2 - intersection_area
        if union_area > 0:
            iou = intersection_area / union_area
            if iou > overlap_threshold:
                return True
        
        return False


def save_figure_to_png(
    figure: ExtractedFigure, 
    arxiv_id: str, 
    output_dir: Path,
    figure_number: Optional[int] = None
) -> str:
    """
    Save an extracted figure to PNG file.
    
    Parameters
    ----------
    figure : ExtractedFigure
        The extracted figure data.
    arxiv_id : str
        arXiv identifier for naming.
    output_dir : Path
        Directory to save the image.
    figure_number : Optional[int]
        The figure number from caption. If None, leave blank in filename.
    
    Returns
    -------
    str
        Generated filename.
    
    Notes
    -----
    Filename format: {arxivID}_{pageNumber}_{figureNumber}.png
    Example: 2410.08073_3_1.png
    
    Edge case: If figure has no caption/number, figureNumber is left blank:
    Example: 2410.08073_3_.png
    """
    # Determine the figure number for filename
    if figure_number is not None:
        fig_num_str = str(figure_number)
    elif figure.figure_index is not None and figure.figure_index > 0:
        fig_num_str = str(figure.figure_index)
    else:
        fig_num_str = ""  # Leave blank for figures without caption/number
    
    # Generate filename: arxivID_pageNumber_figureNumber.png
    filename = f"{arxiv_id}_{figure.page_number}_{fig_num_str}.png"
    filepath = output_dir / filename
    
    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save image
    with open(filepath, 'wb') as f:
        f.write(figure.image_data)
    
    logger.debug(f"Saved figure: {filename}")
    return filename


if __name__ == "__main__":
    # Test the PDF parsing and figure extraction
    print("PDF Parser and Figure Extractor module loaded successfully")
    print("This module requires a PDF file to test.")
