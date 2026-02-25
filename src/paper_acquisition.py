"""
Paper Acquisition Module for the Quantum Circuit Dataset Pipeline.

This module handles:
1. Reading the paper list file
2. Downloading PDFs from arXiv (using API for metadata + CloudScraper for files)
3. Managing the paper processing bookkeeping (CSV tracking)

The module ensures reproducibility by processing papers in the exact order
specified in the paper list file and stopping once the target number of
images is collected.

Author: [Your Name]
Exam ID: 37
"""

import csv
import time
import cloudscraper
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Generator
import re
import sys

# Add parent directory to path to import local modules
sys.path.append(str(Path(__file__).parent))

from config import CONFIG
from data_models import PaperInfo, ProcessingStatus
from utils.logging_utils import setup_logger, PipelineLogger


# Module logger
logger = setup_logger(__name__)


class PaperListReader:
    """
    Reads and manages the paper list file.
    
    This class provides an iterator over papers in the list,
    maintaining the exact order specified in the file.
    """
    
    def __init__(self, paper_list_path: Optional[Path] = None):
        self.paper_list_path = paper_list_path or CONFIG.paths.paper_list_file
        self.papers: List[PaperInfo] = []
        self._load_papers()
    
    def _load_papers(self) -> None:
        if not self.paper_list_path.exists():
            raise FileNotFoundError(f"Paper list not found: {self.paper_list_path}")
        
        logger.info(f"Loading papers from: {self.paper_list_path}")
        
        with open(self.paper_list_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    paper = PaperInfo.from_line(line)
                    self.papers.append(paper)
                except Exception as e:
                    logger.warning(f"Invalid line {line_num}: {line} - {e}")
        
        logger.info(f"Loaded {len(self.papers)} papers from the list")
    
    def __iter__(self) -> Generator[PaperInfo, None, None]:
        for paper in self.papers:
            yield paper
    
    def __len__(self) -> int:
        return len(self.papers)
    
    def __getitem__(self, index: int) -> PaperInfo:
        return self.papers[index]


class PDFDownloader:
    """
    Downloads PDFs from arXiv using the arXiv API and CloudScraper.
    """
    
    # arXiv API namespace
    ARXIV_NS = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }
    
    def __init__(
        self,
        download_dir: Optional[Path] = None,
        delay_between_requests: Optional[float] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None
    ):
        self.download_dir = download_dir or CONFIG.paths.pdfs_dir
        self.delay_between_requests = delay_between_requests or CONFIG.network.arxiv_request_delay
        self.max_retries = max_retries or CONFIG.network.max_retries
        self.timeout = timeout or CONFIG.network.request_timeout
        self.api_url = CONFIG.network.arxiv_api_url
        
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time = 0
        
        # Initialize CloudScraper to bypass Cloudflare/ detection
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            }
        )

    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limiting."""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < self.delay_between_requests:
            sleep_time = self.delay_between_requests - elapsed
            logger.debug(f"Rate limiting: waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()
    
    def _query_arxiv_api(self, arxiv_id: str) -> Optional[str]:
        """
        Query the arXiv API to get the official PDF URL for a paper.
        """
        # Clean arxiv_id (remove version for API query)
        clean_id = re.sub(r'v\d+$', '', arxiv_id)
        api_query_url = f"{self.api_url}?id_list={clean_id}"
        
        try:
            # Use scraper for API too, to avoid user-agent blocks
            response = self.scraper.get(api_query_url, timeout=self.timeout)
            
            if response.status_code != 200:
                logger.warning(f"arXiv API returned status {response.status_code} for {arxiv_id}")
                return None
            
            # Parse XML
            root = ET.fromstring(response.text)
            
            # Find entry
            entry = root.find('atom:entry', self.ARXIV_NS)
            if entry is None:
                logger.warning(f"No entry found in arXiv API response for {arxiv_id}")
                return None
            
            # Check for error (paper not found)
            id_elem = entry.find('atom:id', self.ARXIV_NS)
            if id_elem is not None and 'error' in id_elem.text.lower():
                logger.warning(f"Paper not found in arXiv: {arxiv_id}")
                return None
            
            # Find PDF link
            for link in entry.findall('atom:link', self.ARXIV_NS):
                if link.get('title') == 'pdf':
                    return link.get('href')
            
            # Fallback if no specific PDF link found
            if id_elem is not None:
                abs_url = id_elem.text
                paper_id = abs_url.split('/')[-1]
                return f"https://arxiv.org/pdf/{paper_id}.pdf"
            
            return None
            
        except Exception as e:
            logger.error(f"arXiv API query failed for {arxiv_id}: {e}")
            return None

    def download(self, paper: PaperInfo) -> Optional[Path]:
        """
        Download the PDF. First queries API for link, then downloads file.
        """
        pdf_path = self.download_dir / f"{paper.arxiv_id}.pdf"
        
        if pdf_path.exists():
            logger.info(f"Using cached PDF for {paper.arxiv_id}")
            return pdf_path
        
        logger.info(f"Processing {paper.arxiv_id} (waiting {self.delay_between_requests}s)")
        
        # 1. Query API for the URL
        self._wait_for_rate_limit()
        pdf_url = self._query_arxiv_api(paper.arxiv_id)
        
        if pdf_url is None:
            # Last resort fallback if API fails entirely
            clean_id = re.sub(r'v\d+$', '', paper.arxiv_id)
            pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
            logger.info(f"API failed, falling back to constructed URL: {pdf_url}")

        # 2. Download the actual file
        for attempt in range(1, self.max_retries + 1):
            try:
                self._wait_for_rate_limit()
                
                # Perform the download
                # We append .pdf to the URL if it comes from the API and lacks it (sometimes API returns /pdf/ID without .pdf)
                if not pdf_url.endswith('.pdf'):
                    pdf_url += ".pdf"

                response = self.scraper.get(pdf_url, timeout=self.timeout)
                
                # Check HTTP Status
                if response.status_code == 404:
                    logger.error(f"Paper not found (404): {paper.arxiv_id}")
                    paper.status = ProcessingStatus.FAILED
                    return None
                
                if response.status_code != 200:
                    logger.warning(f"HTTP {response.status_code} for {paper.arxiv_id}")
                    continue

                # Check Content Type
                content_type = response.headers.get('Content-Type', '').lower()
                if 'application/pdf' not in content_type:
                    logger.warning(f"Response content-type is {content_type}, not PDF. Retrying...")
                    continue

                # Save file
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Successfully downloaded: {paper.arxiv_id}")
                return pdf_path

            except Exception as e:
                logger.error(f"Error downloading {paper.arxiv_id}: {str(e)}")
                # Exponential backoff
                time.sleep(self.delay_between_requests * (2 ** attempt))

        paper.status = ProcessingStatus.FAILED
        logger.error(f"Failed to download {paper.arxiv_id} after {self.max_retries} attempts")
        return None


class PaperCountsTracker:
    """Tracks and saves the paper processing counts."""
    
    def __init__(self, papers: List[PaperInfo], csv_path: Optional[Path] = None):
        self.papers = papers
        self.csv_path = csv_path or CONFIG.paths.paper_counts_csv
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    def update(self, paper: PaperInfo, image_count: int) -> None:
        paper.image_count = image_count
        paper.status = ProcessingStatus.PROCESSED
        logger.debug(f"Updated count for {paper.arxiv_id}: {image_count} images")
    
    def save(self) -> None:
        logger.info(f"Saving paper counts to: {self.csv_path}")
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['arxiv_id', 'image_count'])
            for paper in self.papers:
                if paper.status == ProcessingStatus.NOT_VISITED:
                    writer.writerow([paper.full_id, ''])
                elif paper.image_count is not None:
                    writer.writerow([paper.full_id, paper.image_count])
                else:
                    writer.writerow([paper.full_id, 0])
        logger.info("Paper counts saved successfully")
    
    def get_statistics(self) -> dict:
        processed = [p for p in self.papers if p.status == ProcessingStatus.PROCESSED]
        with_images = [p for p in processed if p.image_count and p.image_count > 0]
        return {
            'total_papers': len(self.papers),
            'processed': len(processed),
            'with_images': len(with_images),
            'total_images': sum(p.image_count or 0 for p in processed),
            'not_visited': len([p for p in self.papers if p.status == ProcessingStatus.NOT_VISITED])
        }


class PaperAcquisitionPipeline:
    """Main orchestrator for paper acquisition."""
    
    def __init__(self, target_images: Optional[int] = None):
        self.target_images = target_images or CONFIG.extraction.target_image_count
        self.reader = PaperListReader()
        self.downloader = PDFDownloader()
        self.tracker = PaperCountsTracker(self.reader.papers)
        self.collected_images = 0
    
    def iterate_papers(self) -> Generator[tuple, None, None]:
        with PipelineLogger("Paper Acquisition") as pl:
            for idx, paper in enumerate(self.reader):
                if self.collected_images >= self.target_images:
                    pl.log(f"Target of {self.target_images} images reached. Stopping.")
                    break
                
                pl.progress(idx + 1, len(self.reader), f"Processing {paper.arxiv_id}")
                
                pdf_path = self.downloader.download(paper)
                
                if pdf_path is None:
                    paper.status = ProcessingStatus.FAILED
                    self.tracker.update(paper, 0)
                    continue
                
                yield paper, pdf_path
            
            self.tracker.save()
            stats = self.tracker.get_statistics()
            pl.log(f"Final statistics: {stats}")
    
    def record_result(self, paper: PaperInfo, image_count: int) -> None:
        self.tracker.update(paper, image_count)
        self.collected_images += image_count
        logger.info(f"Total images collected: {self.collected_images}/{self.target_images}")
    
    def save_progress(self) -> None:
        self.tracker.save()


if __name__ == "__main__":
    print("Testing Paper Acquisition Pipeline...")
    pipeline = PaperAcquisitionPipeline(target_images=10)
    for paper, pdf_path in pipeline.iterate_papers():
        print(f"Would process: {paper.arxiv_id} from {pdf_path}")
        pipeline.record_result(paper, 2)
    print("Test completed!")


