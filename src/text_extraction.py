"""
Text Extraction and Figure-Text Linking Module.

This module handles:
1. Extracting structured text from PDFs
2. Linking figures to their captions
3. Finding paragraphs that reference each figure
4. Computing text positions for reproducibility

Author: [Your Name]
Exam ID: 37
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

import sys
sys.path.append(str(Path(__file__).parent))

from config import CONFIG
from data_models import TextPosition
from pdf_extraction import PDFParser, ExtractedFigure
from utils.logging_utils import setup_logger


# Module logger
logger = setup_logger(__name__)

@dataclass
class LinkedFigureText:
    """
    Text content linked to a specific figure.
    
    Attributes
    ----------
    figure_number : int
        The figure number this text is linked to.
    caption : str
        The figure caption.
    caption_position : TextPosition
        Position of caption in full text.
    referring_paragraphs : List[str]
        Paragraphs that reference this figure.
    paragraph_positions : List[TextPosition]
        Positions of referring paragraphs.
    page_number : int
        Page where the figure appears.
    """
    figure_number: int
    caption: str = ""
    caption_position: Optional[TextPosition] = None
    referring_paragraphs: List[str] = field(default_factory=list)
    paragraph_positions: List[TextPosition] = field(default_factory=list)
    page_number: int = 0


class TextExtractor:
    """
    Extracts and structures text from PDF documents.
    
    This class provides methods to extract text while preserving
    structure (sections, paragraphs, captions) for later analysis.
    
    Attributes
    ----------
    parser : PDFParser
        The PDF parser instance.
    full_text : str
        Complete extracted text.
    paragraphs : List[Dict]
        List of paragraph blocks with positions.
    
    Examples
    --------
    >>> with PDFParser("paper.pdf", "2410.08073") as parser:
    ...     extractor = TextExtractor(parser)
    ...     paragraphs = extractor.get_paragraphs()
    """
    
    def __init__(self, parser: PDFParser):
        """
        Initialize the text extractor.
        
        Parameters
        ----------
        parser : PDFParser
            The PDF parser with opened document.
        """
        self.parser = parser
        self.arxiv_id = parser.arxiv_id
        self._full_text: Optional[str] = None
        self._paragraphs: Optional[List[Dict]] = None
    
    @property
    def full_text(self) -> str:
        """Get the complete text of the document."""
        if self._full_text is None:
            self._full_text = self.parser.get_full_text()
        return self._full_text
    
    def get_paragraphs(self) -> List[Dict[str, Any]]:
        """
        Extract paragraphs with position information.
        
        Returns
        -------
        List[Dict]
            List of paragraphs with keys:
            - 'text': paragraph text
            - 'page': page number
            - 'position': TextPosition
            - 'type': 'paragraph', 'caption', 'title', etc.
        """
        if self._paragraphs is not None:
            return self._paragraphs
        
        logger.debug(f"Extracting paragraphs from {self.arxiv_id}")
        
        self._paragraphs = []
        text_blocks = self.parser.get_text_blocks()
        
        # Track position in concatenated full text
        current_pos = 0
        
        for block in text_blocks:
            text = block['text'].strip()
            if not text:
                continue
            
            # Find this text in the full document
            start_pos = self.full_text.find(text, current_pos)
            if start_pos == -1:
                start_pos = current_pos  # Fallback
            
            end_pos = start_pos + len(text)
            
            # Determine paragraph type
            para_type = self._classify_paragraph(text)
            
            self._paragraphs.append({
                'text': text,
                'page': block['page'],
                'position': TextPosition(start_pos, end_pos, para_type, block['page']),
                'type': para_type,
                'bbox': block.get('bbox')
            })
            
            current_pos = end_pos
        
        logger.info(f"Extracted {len(self._paragraphs)} paragraphs from {self.arxiv_id}")
        return self._paragraphs
    
    def _classify_paragraph(self, text: str) -> str:
        """
        Classify a paragraph by its type.
        
        Parameters
        ----------
        text : str
            Paragraph text.
        
        Returns
        -------
        str
            Type: 'caption', 'section_title', 'abstract', 'paragraph'
        """
        text_lower = text.lower().strip()
        
        # Check for caption
        if re.match(r'^(figure|fig\.?)\s*\d+', text_lower):
            return 'caption'
        
        # Check for section title (short, potentially numbered)
        if len(text) < 100 and re.match(r'^(\d+\.?\s+)?[A-Z]', text):
            if any(kw in text_lower for kw in ['introduction', 'method', 'result', 
                                                 'conclusion', 'abstract', 'reference',
                                                 'background', 'discussion', 'related work']):
                return 'section_title'
        
        # Check for abstract
        if 'abstract' in text_lower[:20]:
            return 'abstract'
        
        return 'paragraph'
    
    def find_text_position(self, search_text: str) -> Optional[TextPosition]:
        """
        Find the position of specific text in the document.
        
        Parameters
        ----------
        search_text : str
            Text to find.
        
        Returns
        -------
        Optional[TextPosition]
            Position if found, None otherwise.
        """
        full_text = self.full_text
        start = full_text.find(search_text)
        
        if start == -1:
            return None
        
        end = start + len(search_text)
        
        # Determine page
        page = self._find_page_for_position(start)
        
        return TextPosition(start, end, 'search', page)
    
    def _find_page_for_position(self, char_pos: int) -> int:
        """Find which page a character position belongs to."""
        full_text = self.full_text
        page_markers = list(re.finditer(r'\[PAGE (\d+)\]', full_text[:char_pos + 50]))
        
        if page_markers:
            return int(page_markers[-1].group(1))
        return 1


class FigureTextLinker:
    """
    Links figures to their associated text content.
    
    This class finds:
    1. Captions for each figure
    2. Paragraphs that reference each figure
    3. Section context for each figure
    
    Attributes
    ----------
    text_extractor : TextExtractor
        Text extraction component.
    captions : List[Dict]
        Extracted figure captions.
    
    Examples
    --------
    >>> linker = FigureTextLinker(parser)
    >>> linked = linker.link_figure(figure_number=1, page_number=3)
    """
    
    def __init__(self, parser: PDFParser):
        """
        Initialize the figure-text linker.
        
        Parameters
        ----------
        parser : PDFParser
            The PDF parser instance.
        """
        self.parser = parser
        self.text_extractor = TextExtractor(parser)
        self._captions: Optional[List[Dict]] = None
        self._references: Optional[List[Dict]] = None
    
    @property
    def captions(self) -> List[Dict]:
        """Get all figure captions."""
        if self._captions is None:
            self._captions = self.parser.find_figure_captions()
        return self._captions
    
    @property
    def references(self) -> List[Dict]:
        """Get all figure references."""
        if self._references is None:
            self._references = self.parser.find_figure_references()
        return self._references
    
    def link_figure(
        self, 
        figure_number: int = None,
        page_number: int = None
    ) -> LinkedFigureText:
        """
        Link a figure to its associated text.
        
        Parameters
        ----------
        figure_number : int
            The figure number to link.
        page_number : int
            The page where the figure appears.
        
        Returns
        -------
        LinkedFigureText
            Linked text content for the figure.
        """
        linked = LinkedFigureText(
            figure_number=figure_number or 0,
            page_number=page_number or 0
        )
        
        # Find caption
        caption_info = self._find_caption(figure_number, page_number)
        if caption_info:
            linked.caption = caption_info['caption_text']
            linked.caption_position = TextPosition(
                start=caption_info['position'][0],
                end=caption_info['position'][1],
                source='caption',
                page=caption_info['page']
            )
        
        # Find referring paragraphs
        refs = self._find_references(figure_number)
        for ref in refs:
            linked.referring_paragraphs.append(ref['context'])
            linked.paragraph_positions.append(TextPosition(
                start=ref['position'][0],
                end=ref['position'][1],
                source='reference',
                page=ref['page']
            ))
        
        return linked
    
    def _find_caption(
        self, 
        figure_number: int, 
        page_number: int = None
    ) -> Optional[Dict]:
        """
        Find caption for a specific figure.
        
        Parameters
        ----------
        figure_number : int
            Figure number to find.
        page_number : int
            Page number (for disambiguation).
        
        Returns
        -------
        Optional[Dict]
            Caption information if found.
        """
        for cap in self.captions:
            if cap['figure_number'] == figure_number:
                if page_number is None or cap['page'] == page_number:
                    return cap
        
        # If not found by number, try to find on same page
        if page_number:
            for cap in self.captions:
                if cap['page'] == page_number:
                    return cap
        
        return None
    
    def _find_references(self, figure_number: int) -> List[Dict]:
        """
        Find all paragraphs that reference a figure.
        """
        references = []
        
        # KEY CHANGE: Iterate over pre-parsed paragraphs, not raw text
        paragraphs = self.text_extractor.get_paragraphs()
        
        # Regex to find "Fig. X" or "Figure X" specifically
        ref_pattern = re.compile(rf'(?:Fig(?:ure)?\.?)\s*{figure_number}(?!\d)', re.IGNORECASE)
        
        for para in paragraphs:
            text = para['text']
            
            if ref_pattern.search(text):
                # Found a mention! Save the WHOLE paragraph.
                references.append({
                    'figure_number': figure_number,
                    'context': text,  # <--- This fixes the cut-off text bug
                    'page': para['page'],
                    'position': para['position'].to_tuple() if hasattr(para['position'], 'to_tuple') else None
                })
        
        return references
    
    def link_all_figures(
        self, 
        figures: List[ExtractedFigure]
    ) -> Dict[int, LinkedFigureText]:
        """
        Link all figures to their associated text.
        
        Parameters
        ----------
        figures : List[ExtractedFigure]
            List of extracted figures.
        
        Returns
        -------
        Dict[int, LinkedFigureText]
            Mapping from figure index to linked text.
        """
        linked_map = {}
        
        for idx, fig in enumerate(figures):
            linked = self.link_figure(
                figure_number=fig.figure_index,
                page_number=fig.page_number
            )
            linked_map[idx] = linked
        
        logger.info(f"Linked {len(linked_map)} figures to text")
        return linked_map
    
    def get_figure_context(
        self, 
        figure_number: int,
        context_window: int = 500
    ) -> List[str]:
        """
        Get extended context around a figure.
        
        This method extracts text around the figure caption
        and figure references to provide context.
        
        Parameters
        ----------
        figure_number : int
            Figure number.
        context_window : int
            Number of characters to include around each reference.
        
        Returns
        -------
        List[str]
            List of context strings.
        """
        contexts = []
        full_text = self.text_extractor.full_text
        
        # Get caption context
        caption_info = self._find_caption(figure_number)
        if caption_info:
            pos = caption_info['position']
            start = max(0, pos[0] - context_window // 2)
            end = min(len(full_text), pos[1] + context_window // 2)
            contexts.append(full_text[start:end])
        
        # Get reference contexts
        for ref in self._find_references(figure_number):
            contexts.append(ref['context'])
        
        return contexts


class DescriptionExtractor:
    """
    Extracts descriptive text for figures.
    
    This class identifies and extracts meaningful descriptions
    of figures from captions and surrounding text.
    
    Examples
    --------
    >>> extractor = DescriptionExtractor(parser)
    >>> descriptions = extractor.get_descriptions(figure_number=1)
    """
    
    def __init__(self, parser: PDFParser):
        """
        Initialize the description extractor.
        
        Parameters
        ----------
        parser : PDFParser
            The PDF parser instance.
        """
        self.linker = FigureTextLinker(parser)
        self.text_extractor = TextExtractor(parser)
    
    def get_descriptions(
        self, 
        figure_number: int,
        page_number: int = None
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Get descriptions for a figure with their positions.
        
        Parameters
        ----------
        figure_number : int
            Figure number.
        page_number : int
            Page number (optional).
        
        Returns
        -------
        Tuple[List[str], List[Tuple[int, int]]]
            (descriptions, text_positions)
        
        Notes
        -----
        Text positions are tuples (start, end) representing
        character offsets in the full text of the paper.
        This convention ensures reproducibility.
        """
        linked = self.linker.link_figure(figure_number, page_number)
        
        descriptions = []
        positions = []
        
        # Add caption as first description
        if linked.caption:
            descriptions.append(self._clean_description(linked.caption))
            if linked.caption_position:
                positions.append(linked.caption_position.to_tuple())
            else:
                positions.append((0, len(linked.caption)))
        
        # Add referring paragraphs
        for i, para in enumerate(linked.referring_paragraphs):
            cleaned = self._clean_description(para)
            if cleaned and len(cleaned) > 20:  # Skip very short references
                descriptions.append(cleaned)
                if i < len(linked.paragraph_positions):
                    positions.append(linked.paragraph_positions[i].to_tuple())
                else:
                    positions.append((0, 0))
        
        return descriptions, positions
    
    def _clean_description(self, text: str) -> str:
        """
        Clean a description text.
        
        Parameters
        ----------
        text : str
            Raw description text.
        
        Returns
        -------
        str
            Cleaned description.
        """
        # Remove page markers
        text = re.sub(r'\[PAGE \d+\]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text


def extract_figure_texts(
    parser: PDFParser,
    figures: List[ExtractedFigure]
) -> List[Dict[str, Any]]:
    """
    Extract all text information for a list of figures.
    
    Parameters
    ----------
    parser : PDFParser
        The PDF parser instance.
    figures : List[ExtractedFigure]
        List of extracted figures.
    
    Returns
    -------
    List[Dict]
        List of dictionaries with figure and text information:
        - 'figure': ExtractedFigure
        - 'caption': str
        - 'contexts': List[str]
        - 'descriptions': List[str]
        - 'text_positions': List[Tuple[int, int]]
    
    Examples
    --------
    >>> with PDFParser("paper.pdf", "2410.08073") as parser:
    ...     figures = FigureExtractor(parser).extract_all_figures()
    ...     texts = extract_figure_texts(parser, figures)
    """
    linker = FigureTextLinker(parser)
    desc_extractor = DescriptionExtractor(parser)
    
    results = []
    
    for fig in figures:
        # Get linked text
        linked = linker.link_figure(fig.figure_index, fig.page_number)
        
        # Get contexts
        contexts = linker.get_figure_context(fig.figure_index)
        
        # Get descriptions
        descriptions, positions = desc_extractor.get_descriptions(
            fig.figure_index, fig.page_number
        )
        
        results.append({
            'figure': fig,
            'caption': linked.caption,
            'contexts': contexts,
            'descriptions': descriptions,
            'text_positions': positions
        })
    
    logger.info(f"Extracted text for {len(results)} figures")
    return results


if __name__ == "__main__":
    print("Text Extraction and Figure-Text Linking module loaded successfully")
    print("This module requires a PDF file to test.")
