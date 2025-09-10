# app/services/ingestion.py
from __future__ import annotations
import os
import re
import hashlib
import logging
from typing import Iterable, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# LangChain Document import
try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    NLTKTextSplitter,
    SpacyTextSplitter
)

# Try to import advanced splitters
try:
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings
    HAS_SEMANTIC_CHUNKER = True
except Exception:
    HAS_SEMANTIC_CHUNKER = False

# PDF libraries
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except Exception:
    HAS_PDFPLUMBER = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class DocumentMetadata:
    """Structured metadata for documents"""
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    keywords: Optional[List[str]] = None
    sections: Optional[Dict[str, int]] = None  # section_name -> page_number
    conclusion: Optional[str] = None
    results: Optional[str] = None
    pdf_metadata: Optional[Dict] = None


# ------------------------- helpers -------------------------

def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _extract_authors_advanced(text: str) -> List[str]:
    """Extract authors using multiple strategies."""
    authors = []
    
    # Strategy 1: Look for author patterns near the beginning
    # Pattern for "FirstName LastName" with optional middle initial
    first_3000 = text[:3000]
    
    # Pattern 1: Names with superscripts (like in academic papers)
    pattern1 = r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s+)?[A-Z][a-z]+)\s*(?:[a-z,\d]+)?(?:\n|,|and)'
    matches1 = re.findall(pattern1, first_3000)
    
    # Pattern 2: Names between title and abstract
    title_end = re.search(r'\n\n', first_3000)
    abstract_start = re.search(r'Abstract', first_3000, re.IGNORECASE)
    if title_end and abstract_start:
        author_region = first_3000[title_end.end():abstract_start.start()]
        # Look for capitalized names
        pattern2 = r'([A-Z][a-z]+\s+(?:[A-Z]\.\s+)?[A-Z][a-z]+)'
        matches2 = re.findall(pattern2, author_region)
        authors.extend(matches2)
    
    # Pattern 3: Look for email patterns and extract names before them
    email_pattern = r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s+)?[A-Z][a-z]+)[^@\n]*@[a-zA-Z0-9.-]+\.[a-z]+'
    email_matches = re.findall(email_pattern, first_3000)
    authors.extend(email_matches)
    
    # Pattern 4: Look for "by Name Name" pattern
    by_pattern = r'by\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s+)?[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?\s+)?[A-Z][a-z]+)*)'
    by_matches = re.findall(by_pattern, first_3000, re.IGNORECASE)
    for match in by_matches:
        # Split by comma if multiple authors
        names = re.split(r',\s*|\s+and\s+', match)
        authors.extend(names)
    
    # Add matches from pattern 1
    authors.extend(matches1)
    
    # Clean and deduplicate
    cleaned_authors = []
    seen = set()
    for author in authors:
        author = author.strip()
        # Remove common false positives
        if (len(author) > 3 and 
            author not in seen and 
            not any(word in author.lower() for word in 
                   ['abstract', 'introduction', 'keywords', 'preprint', 'submitted', 
                    'arxiv', 'conference', 'journal', 'volume', 'pages', 'ieee', 'acm'])):
            cleaned_authors.append(author)
            seen.add(author.lower())
    
    return cleaned_authors[:10]  # Limit to max 10 authors


def _extract_sections(text: str) -> Dict[str, int]:
    """Extract section headers and their positions."""
    sections = {}
    
    # Common section patterns
    section_patterns = [
        r'^(\d+\.?\s+[A-Z][A-Za-z\s]+)$',  # 1. Introduction
        r'^([A-Z][A-Za-z\s]+):?\s*$',  # INTRODUCTION
        r'^\s*(?:Section\s+)?(\d+\.?\d*\.?\s*[A-Z][A-Za-z\s]+)',  # Section headers
    ]
    
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        if len(line) > 3 and len(line) < 100:  # Reasonable header length
            for pattern in section_patterns:
                if re.match(pattern, line):
                    # Check if it's likely a section header
                    if any(keyword in line.lower() for keyword in 
                          ['introduction', 'related', 'method', 'results', 'experiment',
                           'conclusion', 'discussion', 'abstract', 'background', 
                           'approach', 'evaluation', 'future', 'reference', 'appendix']):
                        sections[line] = i
                    break
    
    return sections


def _extract_document_metadata_advanced(text: str, pdf_metadata: Dict = None) -> DocumentMetadata:
    """Advanced metadata extraction with multiple strategies."""
    metadata = DocumentMetadata()
    
    # Extract title (more robust)
    lines = text.split('\n')
    non_empty_lines = [l.strip() for l in lines[:50] if l.strip()]
    
    # Title is usually the first substantial line
    for line in non_empty_lines[:10]:
        if (len(line) > 20 and len(line) < 200 and 
            not any(word in line.lower() for word in ['preprint', 'submitted', 'arxiv', 'accepted'])):
            metadata.title = line
            break
    
    # Extract authors
    metadata.authors = _extract_authors_advanced(text)
    
    # Extract abstract
    abstract_patterns = [
        r'Abstract\s*\n+([^\n]+(?:\n[^\n]+)*?)(?:\n\s*(?:Keywords?|1\.|Introduction|I\.))',
        r'ABSTRACT\s*\n+([^\n]+(?:\n[^\n]+)*?)(?:\n\s*(?:KEYWORDS?|1\.|INTRODUCTION))',
        r'Abstract[:\-—]\s*([^\n]+(?:\n[^\n]+)*?)(?:\n\s*\n)',
    ]
    
    for pattern in abstract_patterns:
        match = re.search(pattern, text[:5000], re.IGNORECASE | re.DOTALL)
        if match:
            metadata.abstract = match.group(1).strip()
            break
    
    # Extract keywords
    keyword_patterns = [
        r'Keywords?[:\-—]\s*([^\n]+)',
        r'Index Terms[:\-—]\s*([^\n]+)',
    ]
    
    for pattern in keyword_patterns:
        match = re.search(pattern, text[:5000], re.IGNORECASE)
        if match:
            keywords_text = match.group(1)
            metadata.keywords = [k.strip() for k in re.split(r'[,;]', keywords_text)]
            break
    
    # Extract sections
    metadata.sections = _extract_sections(text)
    
    # Extract conclusion (look in last 20% of document)
    text_lower = text.lower()
    conclusion_start = max(
        text_lower.rfind('conclusion'),
        text_lower.rfind('6. conclusion'),
        text_lower.rfind('7. conclusion'),
        text_lower.rfind('8. conclusion')
    )
    
    if conclusion_start > len(text) * 0.6:  # In last 40% of document
        conclusion_text = text[conclusion_start:conclusion_start + 3000]
        # Clean it up
        conclusion_text = re.sub(r'^\s*\d*\.?\s*conclusions?\s*', '', conclusion_text, flags=re.IGNORECASE)
        metadata.conclusion = conclusion_text[:1500].strip()
    
    # Extract results section
    results_patterns = [
        r'(?:results|experiments?|evaluation)\s*\n+(.*?)(?:\n\s*(?:\d+\.|conclusion|discussion|future))',
    ]
    
    for pattern in results_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            metadata.results = match.group(1)[:1500].strip()
            break
    
    # Store PDF metadata if available
    metadata.pdf_metadata = pdf_metadata
    
    return metadata


def _normalize_ws(text: str, preserve_structure: bool = True) -> str:
    """Normalize whitespace while preserving document structure."""
    # Remove hyphenation at line breaks
    text = re.sub(r'-\s*\r?\n\s*', '', text)
    
    # Normalize newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Collapse spaces within lines
    text = re.sub(r'[ \t\f\v]+', ' ', text)
    
    if preserve_structure:
        # Preserve paragraph structure
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Keep single newlines for readability
    else:
        text = re.sub(r'\n{2,}', '\n\n', text)
    
    return text.strip()


# ------------------------- PDF extraction -------------------------

def _extract_pdf_text_pymupdf(path: str) -> Tuple[str, Dict[str, Any]]:
    """Extract text and metadata from PDF using PyMuPDF."""
    doc = fitz.open(path)
    pdf_metadata = {}
    
    try:
        if doc.is_encrypted:
            try:
                doc.authenticate("")
            except Exception:
                raise RuntimeError(f"Encrypted PDF: {os.path.basename(path)}")
        
        # Extract PDF metadata
        pdf_info = doc.metadata
        if pdf_info:
            pdf_metadata = {
                'pdf_title': pdf_info.get('title', ''),
                'pdf_author': pdf_info.get('author', ''),
                'pdf_subject': pdf_info.get('subject', ''),
                'pdf_keywords': pdf_info.get('keywords', ''),
                'pdf_creator': pdf_info.get('creator', ''),
            }
        
        texts = []
        for page_num, page in enumerate(doc):
            # Get text preserving layout
            text = page.get_text("text")
            if text:
                texts.append(f"[Page {page_num + 1}]\n{text}")
            else:
                # Try blocks method
                blocks = page.get_text("blocks")
                blocks_sorted = sorted(blocks, key=lambda b: (round(b[1], 2), round(b[0], 2)))
                block_text = "\n".join(b[4] for b in blocks_sorted if len(b) >= 5)
                if block_text:
                    texts.append(f"[Page {page_num + 1}]\n{block_text}")
        
        full_text = "\n\n".join(texts)
        return _normalize_ws(full_text, preserve_structure=True), pdf_metadata
    finally:
        doc.close()


def _extract_pdf_text_pdfplumber(path: str) -> Tuple[str, Dict[str, Any]]:
    """Extract text from PDF using pdfplumber."""
    pdf_metadata = {}
    
    with pdfplumber.open(path) as pdf:
        if hasattr(pdf, 'metadata'):
            pdf_metadata = {
                'pdf_title': pdf.metadata.get('Title', ''),
                'pdf_author': pdf.metadata.get('Author', ''),
            }
        
        pages = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
            if text:
                pages.append(f"[Page {i + 1}]\n{text}")
        
        full_text = "\n\n".join(pages)
        return _normalize_ws(full_text, preserve_structure=True), pdf_metadata


def _extract_pdf_text(path: str) -> Tuple[str, Dict[str, Any]]:
    """Extract text and metadata from PDF."""
    if HAS_PYMUPDF:
        try:
            return _extract_pdf_text_pymupdf(path)
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}")
    
    if HAS_PDFPLUMBER:
        try:
            return _extract_pdf_text_pdfplumber(path)
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
    
    raise RuntimeError("No PDF extractor available")


# ------------------------- Document Processing -------------------------

def _create_metadata_chunks(metadata: DocumentMetadata, source_file: str) -> List[Document]:
    """Create special chunks for metadata that are easily retrievable."""
    chunks = []
    
    # Author chunk
    if metadata.authors:
        author_content = f"""Document: {metadata.title or source_file}
Authors: {', '.join(metadata.authors)}
Written by: {', '.join(metadata.authors)}
This document was authored by: {', '.join(metadata.authors)}
The authors of this paper are: {', '.join(metadata.authors)}"""
        
        chunks.append(Document(
            page_content=author_content,
            metadata={
                'type': 'authors_metadata',
                'authors': metadata.authors,
                'source': source_file
            }
        ))
    
    # Title and abstract chunk
    if metadata.title or metadata.abstract:
        title_abstract = f"""Title: {metadata.title or 'Unknown'}
        
Abstract: {metadata.abstract or 'Not available'}

Keywords: {', '.join(metadata.keywords) if metadata.keywords else 'Not specified'}"""
        
        chunks.append(Document(
            page_content=title_abstract,
            metadata={
                'type': 'title_abstract_metadata',
                'title': metadata.title,
                'source': source_file
            }
        ))
    
    # Conclusion chunk
    if metadata.conclusion:
        conclusion_content = f"""CONCLUSION of the document "{metadata.title or source_file}":

{metadata.conclusion}

This is the conclusion section where the authors summarize their findings and contributions."""
        
        chunks.append(Document(
            page_content=conclusion_content,
            metadata={
                'type': 'conclusion_metadata',
                'source': source_file
            }
        ))
    
    # Results chunk
    if metadata.results:
        results_content = f"""RESULTS of the document "{metadata.title or source_file}":

{metadata.results}

This section contains the experimental results and findings."""
        
        chunks.append(Document(
            page_content=results_content,
            metadata={
                'type': 'results_metadata',
                'source': source_file
            }
        ))
    
    return chunks


def load_files(paths: List[str]) -> List[Document]:
    """Load PDF files with advanced metadata extraction."""
    docs = []
    
    for path in _iter_paths(paths):
        try:
            text, pdf_metadata = _extract_pdf_text(path)
        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")
            continue
        
        if not text:
            logger.info(f"No text in: {path}")
            continue
        
        # Extract advanced metadata
        doc_metadata = _extract_document_metadata_advanced(text, pdf_metadata)
        
        abs_path = os.path.abspath(path)
        stat = os.stat(abs_path)
        source_name = os.path.basename(path)
        
        # Base metadata
        base_meta = {
            'source': source_name,
            'file_path': abs_path,
            'type': 'text',
            'extension': '.pdf',
            'size_bytes': stat.st_size,
            'sha256': _sha256_file(abs_path),
        }
        
        # Add extracted metadata
        if doc_metadata.title:
            base_meta['title'] = doc_metadata.title
        if doc_metadata.authors:
            base_meta['authors'] = doc_metadata.authors
            base_meta['author_names'] = ', '.join(doc_metadata.authors)
        if doc_metadata.keywords:
            base_meta['keywords'] = doc_metadata.keywords
        if doc_metadata.abstract:
            base_meta['abstract'] = doc_metadata.abstract[:500]
        
        # Add main document
        docs.append(Document(page_content=text, metadata=base_meta))
        
        # Add metadata chunks for better retrieval
        metadata_chunks = _create_metadata_chunks(doc_metadata, source_name)
        docs.extend(metadata_chunks)
        
        logger.info(f"Loaded {source_name} with {len(metadata_chunks)} metadata chunks")
        if doc_metadata.authors:
            logger.info(f"  Authors found: {doc_metadata.authors}")
    
    return docs


def _iter_paths(maybe_paths: Iterable[str]) -> Iterable[str]:
    """Yield existing PDF paths."""
    seen = set()
    for p in maybe_paths:
        if not isinstance(p, str):
            continue
        if not os.path.exists(p):
            logger.warning(f"Path not found: {p}")
            continue
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in files:
                    fp = os.path.join(root, fn)
                    if fp.lower().endswith('.pdf') and fp not in seen:
                        seen.add(fp)
                        yield fp
        elif p.lower().endswith('.pdf') and p not in seen:
            seen.add(p)
            yield p


def load_web(urls: List[str]) -> List[Document]:
    """Placeholder for web loading."""
    return []


# ------------------------- Smart Chunking -------------------------

def make_chunks(
    docs: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    use_semantic: bool = False,
    preserve_metadata_chunks: bool = True
) -> List[Document]:
    """
    Create smart chunks with multiple strategies.
    
    Args:
        docs: Documents to chunk
        chunk_size: Target size for chunks
        chunk_overlap: Overlap between chunks
        use_semantic: Use semantic chunking if available
        preserve_metadata_chunks: Keep metadata chunks intact
    """
    if not docs:
        return []
    
    # Separate metadata chunks from regular documents
    metadata_chunks = []
    regular_docs = []
    
    for doc in docs:
        if preserve_metadata_chunks and doc.metadata.get('type', '').endswith('_metadata'):
            metadata_chunks.append(doc)
        else:
            regular_docs.append(doc)
    
    # Use semantic chunker if available and requested
    if use_semantic and HAS_SEMANTIC_CHUNKER:
        try:
            from app.config import settings
            embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
            splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90
            )
        except Exception as e:
            logger.warning(f"Semantic chunking failed, using recursive: {e}")
            use_semantic = False
    
    if not use_semantic:
        # Use recursive splitter with smart separators
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=[
                "\n\n\n",  # Section breaks
                "\n\n",    # Paragraphs
                "\n",      # Lines
                ". ",      # Sentences
                "! ",
                "? ",
                "; ",
                ", ",
                " ",
                ""
            ],
            keep_separator=True
        )
    
    # Chunk regular documents
    chunks = splitter.split_documents(regular_docs)
    
    # Enrich chunk metadata
    enriched = []
    
    # Add metadata chunks first (they're important for retrieval)
    enriched.extend(metadata_chunks)
    
    # Process regular chunks
    for idx, chunk in enumerate(chunks):
        meta = dict(chunk.metadata or {})
        
        # Add chunk tracking
        content_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()[:16]
        meta.update({
            'chunk_index': idx,
            'chunk_id': f"{meta.get('sha256', '')[:16]}_{idx}_{content_hash}",
            'chunk_size': len(chunk.page_content),
        })
        
        # Detect important sections
        content_lower = chunk.page_content.lower()
        if idx < 5:  # Early chunks might have metadata
            meta['is_early_chunk'] = True
        if 'conclusion' in content_lower[:200]:
            meta['contains_conclusion'] = True
        if 'result' in content_lower[:200]:
            meta['contains_results'] = True
        if any(word in content_lower for word in ['author', '@', 'university', 'institute']):
            meta['may_contain_authors'] = True
        
        enriched.append(Document(page_content=chunk.page_content, metadata=meta))
    
    logger.info(f"Created {len(enriched)} chunks ({len(metadata_chunks)} metadata, {len(chunks)} content)")
    return enriched