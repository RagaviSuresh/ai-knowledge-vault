import os
import re
import time
import logging
from typing import Dict, Any, List
import fitz  # PyMuPDF

# Set up logging
logger = logging.getLogger("pdf_parser")
logger.setLevel(logging.INFO)
# Prevent duplicate logs if handlers already exist
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def clean_text(text: str) -> str:
    """
    Normalizes text according to the following rules:
    - Removes duplicate spaces (replaces multiple spaces/tabs with a single space)
    - Removes excessive blank lines (replaces 3+ consecutive newlines with 2 newlines)
    - Preserves paragraphs (double newlines are preserved)
    - Preserves bullet points
    - Preserves Unicode characters
    """
    if not text:
        return ""
    
    # Replace 3 or more consecutive newlines with exactly 2 newlines (preserves paragraph splits)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Process line-by-line to handle horizontal space normalization and preserve bullet points
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Replace multiple spaces or tabs with a single space
        cleaned_line = re.sub(r'[ \t]+', ' ', line)
        cleaned_line = cleaned_line.strip()
        cleaned_lines.append(cleaned_line)
    
    # Rejoin lines
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Final pass to ensure no vertical space creep
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text.strip()

def is_valid_heading(text: str) -> bool:
    """
    Validates if a piece of text is likely a heading based on length, 
    alphabetic presence, and exclusion of common non-heading patterns.
    """
    text = text.strip()
    if not text:
        return False
    
    # Must contain at least one alphabetic character
    if not re.search(r'[a-zA-Z]', text):
        return False
    
    # Headings are typically not extremely long or short
    if len(text) > 100 or len(text) < 3:
        return False
    
    # Headings typically do not contain more than 15 words
    words = text.split()
    if len(words) > 15:
        return False
    
    # Filter out page number indicators (e.g. "Page 1", "1 of 10")
    lower_text = text.lower()
    if re.match(r'^page\s+\d+$', lower_text):
        return False
    if re.match(r'^\d+\s+of\s+\d+$', lower_text):
        return False
    
    # Filter out common watermarks/links
    if "www." in lower_text or "http" in lower_text or ".com" in lower_text:
        return False
        
    return True

def parse_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Parses a PDF file using PyMuPDF (fitz).
    
    Extracts metadata, page-by-page text, full text, and counts,
    and runs custom heuristics to identify headings in a single pass.
    
    Returns a dictionary of results. Never crashes; returns a dict
    with success=False and the error message if parsing fails.
    """
    start_time = time.perf_counter()
    filename = os.path.basename(pdf_path)
    filepath = os.path.abspath(pdf_path)
    
    try:
        # Check file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
            
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF {filename}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "filename": filename
        }
        
    try:
        page_count = len(doc)
        
        # Extract Metadata
        meta = doc.metadata or {}
        title = meta.get("title") or ""
        author = meta.get("author") or ""
        subject = meta.get("subject") or ""
        keywords = meta.get("keywords") or ""
        creator = meta.get("creator") or ""
        producer = meta.get("producer") or ""
        creation_date = meta.get("creationDate") or ""
        modification_date = meta.get("modDate") or ""
        
        # Single Pass collection structures
        font_sizes = {}
        pages_data = []
        heading_candidates = []
        full_text_list = []
        
        for idx, page in enumerate(doc, 1):
            # 1. Plain text extraction
            raw_text = page.get_text("text")
            cleaned_page_text = clean_text(raw_text)
            word_count = len(cleaned_page_text.split())
            
            pages_data.append({
                "page_number": idx,
                "text": cleaned_page_text,
                "word_count": word_count
            })
            
            if cleaned_page_text:
                full_text_list.append(cleaned_page_text)
                
            # 2. Extract block dictionary once per page
            try:
                blocks = page.get_text("dict")["blocks"]
                for b in blocks:
                    if b.get("type") == 0:  # text block
                        for line in b["lines"]:
                            line_text = ""
                            line_sizes = []
                            is_bold = False
                            
                            for span in line["spans"]:
                                text_span = span.get("text", "").strip()
                                if not text_span:
                                    continue
                                line_text += " " + text_span
                                size_val = span.get("size", 10.0)
                                line_sizes.append(size_val)
                                
                                # Track font size frequencies weighted by character length
                                size_rounded = round(size_val, 1)
                                font_sizes[size_rounded] = font_sizes.get(size_rounded, 0) + len(text_span)
                                
                                # Check bold using flags (bit 4 is bold) or font name containing indicators
                                flags = span.get("flags", 0)
                                font_name = span.get("font", "").lower()
                                if (flags & 16) or any(x in font_name for x in ["bold", "heavy", "black", "semibold", "medium", "demi"]):
                                    is_bold = True
                                    
                            line_text = line_text.strip()
                            if not line_text:
                                continue
                                
                            avg_size = sum(line_sizes) / len(line_sizes) if line_sizes else 0
                            heading_candidates.append({
                                "text": line_text,
                                "size": avg_size,
                                "bold": is_bold,
                                "uppercase": line_text.isupper(),
                                "length": len(line_text)
                            })
            except Exception as e:
                logger.warning(f"Error parsing page {idx} details: {str(e)}")
                
        # Determine body font size
        body_font_size = 10.0
        if font_sizes:
            body_font_size = max(font_sizes, key=font_sizes.get)
            
        # Process full text
        full_text = "\n\n".join(full_text_list)
        total_words = len(full_text.split())
        total_characters = len(full_text)
        
        # Heuristics for Heading filtering
        headings = []
        seen_headings = set()
        
        for cand in heading_candidates:
            text_val = cand["text"]
            if not is_valid_heading(text_val):
                continue
                
            is_heading = False
            # Rule A: Significantly larger font size than body text
            if cand["size"] >= body_font_size + 1.5:
                is_heading = True
            # Rule B: Bold font and size is at least body font size
            elif cand["bold"] and cand["size"] >= body_font_size - 0.5:
                is_heading = True
            # Rule C: Short uppercase line
            elif cand["uppercase"] and cand["length"] < 60:
                is_heading = True
                
            if is_heading:
                cleaned_h = clean_text(text_val)
                if cleaned_h and cleaned_h not in seen_headings:
                    seen_headings.add(cleaned_h)
                    headings.append(cleaned_h)
                    
        result = {
            "success": True,
            "filename": filename,
            "filepath": filepath,
            "page_count": page_count,
            "title": title,
            "author": author,
            "subject": subject,
            "keywords": keywords,
            "creator": creator,
            "producer": producer,
            "creation_date": creation_date,
            "modification_date": modification_date,
            "headings": headings,
            "pages": pages_data,
            "full_text": full_text,
            "statistics": {
                "characters": total_characters,
                "words": total_words,
                "pages": page_count
            }
        }
        
        parsing_time = time.perf_counter() - start_time
        logger.info(
            f"Successfully parsed PDF: {filename} | "
            f"Pages: {page_count} | "
            f"Words: {total_words} | "
            f"Time: {parsing_time:.4f}s"
        )
        return result
        
    except Exception as e:
        logger.error(f"Unexpected error parsing PDF {filename}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "filename": filename
        }
    finally:
        doc.close()
